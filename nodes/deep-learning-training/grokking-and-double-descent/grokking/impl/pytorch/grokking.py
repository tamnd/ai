# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://52PVZH).
"""Grokking in PyTorch: the modular-addition split, the Fourier algorithm, and a small embedding-plus-MLP model trained
with torch.optim.AdamW.

modular_addition_split builds all p * p pairs with label (a + b) mod p and splits them with a torch.Generator;
fourier_logits computes sum_k alpha_k cos(w_k (a + b - c)) through the angle-sum identities (eq-sum-angle) and the
read-off against cos(w_k c) and sin(w_k c) (eq-read-off); ModAddMLP is the model, make_optimizer the AdamW of the
original runs (learning rate 1e-3, weight decay 1, betas 0.9 and 0.98) and train_step one full-batch update; demo
returns the numbers of the worked example of the page."""

import math

import torch
import torch.nn.functional as F

KEY_FREQUENCIES = (14, 35, 41, 42, 52)


def modular_addition_split(p, train_fraction, generator=None):
    """All p * p pairs (a, b) with label (a + b) mod p, split at random; returns (train_x, train_y, test_x, test_y)."""
    if p < 2:
        raise ValueError("p must be at least 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)
    y = (x[:, 0] + x[:, 1]) % p
    order = torch.randperm(p * p, generator=generator)
    n_train = int(round(train_fraction * p * p))
    tr, te = order[:n_train], order[n_train:]
    return x[tr], y[tr], x[te], y[te]


def fourier_logits(a, b, p, freqs=KEY_FREQUENCIES, alpha=None):
    """Logits (n, p) with logit[c] = sum_k alpha_k cos(w_k (a + b - c)), w_k = 2 pi k / p, in float64."""
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1, 1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1, 1)
    w = 2.0 * math.pi * torch.tensor(freqs, dtype=torch.float64) / p
    alpha = torch.ones_like(w) if alpha is None else torch.as_tensor(alpha, dtype=torch.float64)
    cab = torch.cos(a * w) * torch.cos(b * w) - torch.sin(a * w) * torch.sin(b * w)  # eq-sum-angle
    sab = torch.sin(a * w) * torch.cos(b * w) + torch.cos(a * w) * torch.sin(b * w)
    c = torch.arange(p, dtype=torch.float64)
    cc, sc = torch.cos(torch.outer(w, c)), torch.sin(torch.outer(w, c))
    return (cab * alpha) @ cc + (sab * alpha) @ sc  # eq-read-off


def accuracy(logits, labels):
    """The fraction of rows whose largest logit is at the label."""
    return (logits.argmax(dim=1) == torch.as_tensor(labels)).double().mean().item()


class ModAddMLP(torch.nn.Module):
    """A shared embedding of both inputs, concatenated, one ReLU hidden layer and an unembedding; no biases."""

    def __init__(self, p, d=16, hidden=64, scale=1.0):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.randn(p, d) * scale)
        self.w1 = torch.nn.Parameter(torch.randn(2 * d, hidden) * scale / math.sqrt(2 * d))
        self.w2 = torch.nn.Parameter(torch.randn(hidden, p) * scale / math.sqrt(hidden))

    def forward(self, x):
        z = torch.cat([self.embed[x[:, 0]], self.embed[x[:, 1]]], dim=1)
        return torch.relu(z @ self.w1) @ self.w2


def make_optimizer(model, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)):
    """AdamW with the hyperparameters of the original grokking runs."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def train_step(model, opt, x, y):
    """One full-batch step on the cross-entropy loss; returns the loss before the step."""
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()


def weight_norm(model):
    """The l2 norm of all parameters together."""
    return math.sqrt(sum(float((q.detach() ** 2).sum()) for q in model.parameters()))


def demo(rng=None):
    """The worked example: the Fourier algorithm at p = 113 with the key frequencies, and pure weight decay."""
    p = 113
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    logits = fourier_logits(a, b, p)
    labels = (a + b) % p
    idx = torch.arange(p * p)
    correct = logits[idx, labels]
    wrong = logits.clone()
    wrong[idx, labels] = -math.inf
    w = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=1.0)
    for _ in range(1000):
        opt.zero_grad()
        w.grad = torch.zeros_like(w)
        opt.step()
    return {
        "p": p,
        "accuracy": accuracy(logits, labels),
        "correct_logit": correct.max().item(),
        "largest_wrong_logit": wrong.max().item(),
        "decay_1000_steps": w.item(),
    }
