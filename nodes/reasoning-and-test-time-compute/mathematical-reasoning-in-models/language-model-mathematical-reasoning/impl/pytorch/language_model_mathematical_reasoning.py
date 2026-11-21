# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-mathematical-reasoning/ (ai://NEW-language-model-mathematical-reasoning).
"""A network trained on multi-digit multiplication, read digit by digit.

Dziri et al. (2023) found that transformers trained on multiplication get the first and last digits of a product
right far more often than the middle ones, because those digits depend on only a few input digits. This file trains a
multilayer perceptron to map two D-digit numbers to the 2D digits of their product and measures the accuracy of each
output digit on pairs it was not trained on. With D = 2 the training sample covers every pair, so the network can
memorize the table; with D = 3 it sees under a fifth of the pairs and must generalize.
"""

import torch
from torch import nn


def digits(x, k):
    """The k lowest decimal digits of each entry of x, units first: shape (len(x), k)."""
    return torch.stack([(x // 10**i) % 10 for i in range(k)], 1)


def encode(a, b, d):
    """One-hot code of the d digits of a and of b, concatenated: shape (len(a), 20 d)."""
    one = nn.functional.one_hot(torch.cat([digits(a, d), digits(b, d)], 1), 10)
    return one.float().flatten(1)


def low_digits_rule(a, b, j):
    """The j lowest digits of a * b computed from the j lowest digits of a and b alone."""
    m = 10**j
    return (a % m) * (b % m) % m


class Multiplier(nn.Module):
    """A two-hidden-layer perceptron with one 10-way softmax per output digit."""

    def __init__(self, d, hidden=512):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(20 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 20 * d)
        )

    def forward(self, a, b):
        return self.net(encode(a, b, self.d)).view(-1, 2 * self.d, 10)


def sample(d, n, gen):
    """n random pairs of numbers below 10^d."""
    return torch.randint(0, 10**d, (n,), generator=gen), torch.randint(0, 10**d, (n,), generator=gen)


def train(d, n=200_000, epochs=15, batch=512, lr=1e-3, seed=0):
    """Train a Multiplier on n random pairs; returns the model and the training pairs."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    a, b = sample(d, n, gen)
    y = digits(a * b, 2 * d)
    model = Multiplier(d)
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            loss = nn.functional.cross_entropy(model(a[j], b[j]).reshape(-1, 10), y[j].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, (a, b)


@torch.no_grad()
def digit_accuracy(model, n=20_000, seed=1):
    """Accuracy of each output digit (units first) and of whole products on fresh random pairs."""
    d = model.d
    a, b = sample(d, n, torch.Generator().manual_seed(seed))
    hit = model(a, b).argmax(-1) == digits(a * b, 2 * d)
    return hit.float().mean(0), hit.all(1).float().mean().item()


def seen_fraction(train_pairs, d, n=20_000, seed=1):
    """Fraction of the fresh test pairs that also occur among the training pairs."""
    a, b = train_pairs
    seen = set((a * 10**d + b).tolist())
    ta, tb = sample(d, n, torch.Generator().manual_seed(seed))
    return sum(int(k) in seen for k in (ta * 10**d + tb).tolist()) / n
