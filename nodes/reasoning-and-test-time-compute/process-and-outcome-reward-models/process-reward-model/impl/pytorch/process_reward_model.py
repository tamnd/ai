# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/process-reward-model/ (ai://NEW-process-reward-model).
"""Process reward models, in PyTorch.

A PRM reads a whole solution once and predicts, at the last token of every step, the probability that the steps so
far are right. The file has:

* a toy step format in which a step is two tokens (a, b) and is right when b = a + 1 mod V, with a separator token
  after each step;
* PRM, a GRU encoder with a one-logit head read out at the separator positions, trained with binary cross-entropy
  on labels up to the first wrong step (later steps are masked out);
* solution scores from step scores (product, minimum, last) and best-of-N selection;
* implicit process rewards from a policy and a reference model: the per-step log-likelihood ratio and its partial
  sums, which add up to the outcome reward beta log pi(y)/pi_ref(y).
"""

import torch
from torch import nn
from torch.nn import functional as F


def make_solutions(n, steps, vocab, p_step, generator=None):
    """Token sequences (n, 3 steps) with a separator id `vocab` after each two-token step, step correctness
    (n, steps) and first-error labels (n, steps) with -1 after the first wrong step."""
    a = torch.randint(0, vocab, (n, steps), generator=generator)
    right = torch.rand(n, steps, generator=generator) < p_step
    wrong_b = (a + 1 + torch.randint(1, vocab, (n, steps), generator=generator)) % vocab
    b = torch.where(right, (a + 1) % vocab, wrong_b)
    sep = torch.full_like(a, vocab)
    tokens = torch.stack([a, b, sep], -1).reshape(n, 3 * steps)
    ok = torch.cumprod(right.long(), 1).bool()
    labels = ok.float()
    labels[torch.cumsum((~ok).long(), 1) > 1] = -1.0
    return tokens, ok, labels


class PRM(nn.Module):
    def __init__(self, vocab, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, tokens):
        """Step logits read at every third position (the separators): (batch, steps)."""
        h, _ = self.rnn(self.emb(tokens))
        return self.head(h[:, 2::3]).squeeze(-1)


def prm_loss(logits, labels):
    """Binary cross-entropy over the labelled steps only (labels >= 0)."""
    mask = labels >= 0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def train_prm(model, vocab, steps=6, iters=400, batch=256, p_step=0.85, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):
        tokens, _, labels = make_solutions(batch, steps, vocab, p_step, g)
        loss = prm_loss(model(tokens), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def aggregate(step_probs, how="min"):
    if how == "prod":
        return step_probs.prod(-1)
    if how == "min":
        return step_probs.min(-1).values
    if how == "last":
        return step_probs[..., -1]
    raise ValueError(how)


def implicit_step_rewards(logp_policy, logp_ref, step_ids, beta):
    """Per-step implicit rewards from token log-probs (batch, tokens) and the step index of every token.

    Returns rewards (batch, steps) = beta * sum of log ratios over each step's tokens, and their partial sums q^t.
    """
    ratio = beta * (logp_policy - logp_ref)
    n_steps = int(step_ids.max()) + 1
    r = torch.zeros(ratio.shape[0], n_steps, dtype=ratio.dtype).index_add_(1, step_ids, ratio)
    return r, r.cumsum(1)


class TinyLM(nn.Module):
    """A small causal model to stand in for a policy or a reference model."""

    def __init__(self, vocab, d=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.emb = nn.Embedding(vocab + 1, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.7)

    def token_logprobs(self, y):
        """log pi(y_t | y_<t) for every token of y (batch, length)."""
        x = torch.cat([torch.full((y.shape[0], 1), self.vocab, dtype=torch.long), y[:, :-1]], 1)
        h, _ = self.rnn(self.emb(x))
        return torch.log_softmax(self.out(h), -1).gather(2, y.unsqueeze(-1)).squeeze(-1)
