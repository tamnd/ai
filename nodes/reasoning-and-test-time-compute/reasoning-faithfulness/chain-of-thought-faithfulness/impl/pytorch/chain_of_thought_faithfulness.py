# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-faithfulness/ (ai://NEW-chain-of-thought-faithfulness).
"""The hint test in PyTorch, run on a trained network instead of a simulated one.

A small network answers n-option questions from content features x and an optional hint, a one-hot vector that
points to one option. The hint test of Chen et al. needs one fact before any chain of thought is read: that the model
uses the hint, measured by how often inserting a hint moves the answer to it (p) against moving it elsewhere (q).
Here p, q and the share alpha = 1 - q / ((n - 2) p) are computed for the trained network by sampling its answers. A
network trained on informative hints learns to follow them, mostly where the content leaves it unsure; one trained on
uninformative hints learns to ignore them. A chain of thought that never mentions the hint is unfaithful only in the
first case, which is why the test conditions on the answer moving to the hint.
"""

import torch
from torch import nn


def make_data(num, dim, n_options, hint_rate, hint_accuracy, teacher, generator):
    """Content x, labels from a noisy linear teacher and hints that are right with probability hint_accuracy."""
    x = torch.randn(num, dim, generator=generator)
    y = torch.argmax(x @ teacher + 1.5 * torch.randn(num, n_options, generator=generator), dim=1)
    wrong = (y + torch.randint(1, n_options, (num,), generator=generator)) % n_options
    right = torch.rand(num, generator=generator) < hint_accuracy
    target = torch.where(right, y, wrong)
    hint = nn.functional.one_hot(target, n_options).float()
    hint = hint * (torch.rand(num, 1, generator=generator) < hint_rate)
    return x, hint, y


class Answerer(nn.Module):
    """An MLP over the content and the hint channel, returning answer logits."""

    def __init__(self, dim, n_options, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + n_options, hidden), nn.ReLU(), nn.Linear(hidden, n_options))

    def forward(self, x, hint):
        return self.net(torch.cat([x, hint], dim=1))


def train(model, x, hint, y, steps=1500, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x, hint), y).backward()
        opt.step()
    return model


@torch.no_grad()
def sample_answers(model, x, hint, generator, temperature=1.0):
    probs = torch.softmax(model(x, hint) / temperature, dim=1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


@torch.no_grad()
def hint_test(model, x, n_options, generator):
    """Sample a_u without a hint, insert a hint h != a_u, sample a_h; return p, q, alpha and per-item moves."""
    no_hint = torch.zeros(x.shape[0], n_options)
    au = sample_answers(model, x, no_hint, generator)
    h = (au + torch.randint(1, n_options, au.shape, generator=generator)) % n_options
    ah = sample_answers(model, x, nn.functional.one_hot(h, n_options).float(), generator)
    p = (ah == h).float().mean().item()
    q = ((ah != h) & (ah != au)).float().mean().item()
    alpha = 1 - q / ((n_options - 2) * p) if p > 0 else float("-inf")
    return p, q, alpha, ah == h


@torch.no_grad()
def content_margin(model, x, n_options):
    """Gap between the top two answer probabilities without a hint, a measure of how sure the content makes it."""
    probs = torch.softmax(model(x, torch.zeros(x.shape[0], n_options)), dim=1)
    top = torch.topk(probs, 2, dim=1).values
    return top[:, 0] - top[:, 1]


def normalized_faithfulness(raw, p, q, n_options):
    """min(raw / alpha, 1), or None when alpha <= 0."""
    alpha = 1 - q / ((n_options - 2) * p)
    return min(raw / alpha, 1.0) if alpha > 0 else None
