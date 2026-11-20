# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/test-time-scaling/ (ai://NEW-test-time-scaling).
"""Test-time scaling in PyTorch: sample a trained model many times and measure what more samples buy.

A small network learns a teacher's labelling from a few examples and is tested on new problems. For each test problem
it is sampled n times at a chosen temperature. From the samples come three curves against the number of samples k:
coverage (pass@k, any sample correct), majority voting, and best-of-k under a noisy verifier. All estimators work on
batches of problems at once.
"""

import torch
from torch import nn


def log_binom(n, k):
    """log C(n, k) for tensors, with -inf where k > n."""
    n, k = torch.as_tensor(n, dtype=torch.float64), torch.as_tensor(k, dtype=torch.float64)
    out = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return torch.where(k > n, torch.full_like(out, -float("inf")), out)


def pass_at_k(n, correct, k):
    """Unbiased pass@k, 1 - C(n - c, k) / C(n, k), for a batch of problems with c correct out of n."""
    correct = torch.as_tensor(correct, dtype=torch.float64)
    return 1.0 - torch.exp(log_binom(n - correct, torch.full_like(correct, k)) - log_binom(n, k))


def majority_vote(answers, num_classes):
    """Most frequent answer in each row of a (problems, n) tensor; ties go to the smallest answer."""
    counts = torch.zeros(answers.shape[0], num_classes, dtype=torch.long, device=answers.device)
    counts.scatter_add_(1, answers, torch.ones_like(answers))
    return counts.argmax(1)


def best_of_n(answers, scores):
    """Answer of the highest-scored sample in each row."""
    return answers.gather(1, scores.argmax(1, keepdim=True)).squeeze(1)


class Solver(nn.Module):
    """A two-layer network that maps a problem vector to logits over the answers."""

    def __init__(self, dim, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def teacher_task(n_train, n_test, dim, num_classes, generator):
    """Problems are Gaussian vectors; the answer is the argmax of a fixed random linear teacher.

    Trained on a few examples, a student solves some test problems reliably, some only sometimes and some never,
    which is the spread of difficulty that repeated sampling exploits.
    """
    w = torch.randn(dim, num_classes, generator=generator)
    x = torch.randn(n_train + n_test, dim, generator=generator)
    y = (x @ w).argmax(1)
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


def train(model, data, steps, lr=3e-3, weight_decay=1e-2, batch=64, generator=None):
    """Plain AdamW training with cross-entropy on random minibatches."""
    x, labels = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        idx = torch.randint(len(x), (batch,), generator=generator)
        loss = nn.functional.cross_entropy(model(x[idx]), labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample_answers(model, x, n, temperature, generator):
    """n sampled answers per problem; temperature 0 means greedy decoding repeated n times."""
    logits = model(x)
    if temperature == 0:
        return logits.argmax(1, keepdim=True).expand(-1, n)
    probs = torch.softmax(logits / temperature, dim=1)
    return torch.multinomial(probs, n, replacement=True, generator=generator)


def scaling_curves(answers, labels, ks, num_classes, verifier_noise, generator):
    """Coverage, majority-vote and best-of-k accuracy for each k, from one set of n samples per problem.

    Coverage uses the unbiased estimator over all n samples; voting and best-of-k use the first k samples.
    The verifier scores a correct sample 1 and a wrong one 0, plus Gaussian noise of scale verifier_noise.
    """
    n = answers.shape[1]
    right = answers == labels[:, None]
    correct = right.sum(1)
    scores = right.double() + verifier_noise * torch.randn(answers.shape, generator=generator, dtype=torch.float64)
    cov, vote, bon = [], [], []
    for k in ks:
        cov.append(pass_at_k(n, correct, k).mean().item())
        vote.append((majority_vote(answers[:, :k], num_classes) == labels).double().mean().item())
        bon.append((best_of_n(answers[:, :k], scores[:, :k]) == labels).double().mean().item())
    return {"coverage": cov, "vote": vote, "best_of_n": bon}


def per_sample_success(answers, labels):
    """Fraction of a problem's samples that are correct: an estimate of its per-sample success rate p."""
    return (answers == labels[:, None]).double().mean(1)
