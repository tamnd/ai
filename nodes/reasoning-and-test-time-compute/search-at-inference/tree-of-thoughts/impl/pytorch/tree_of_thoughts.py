# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/tree-of-thoughts/ (ai://NEW-tree-of-thoughts).
"""A learned state evaluator for tree search on the Game of 24.

Tree of Thoughts prompts the language model itself to value states. Here a small network plays that part: it is
trained to predict whether a multiset of numbers can still reach 24, and beam search over arithmetic steps keeps
the b states it values highest. Comparing it with a random evaluator isolates what the evaluator contributes.
"""

from fractions import Fraction
from functools import cache

import torch
from torch import nn

OPS = ("+", "-", "r-", "*", "/", "r/")


def _apply(a, b, op):
    return {
        "+": a + b,
        "-": a - b,
        "r-": b - a,
        "*": a * b,
        "/": a / b if b != 0 else None,
        "r/": b / a if a != 0 else None,
    }[op]


def moves(state):
    out = []
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            rest = state[:i] + state[i + 1 : j] + state[j + 1 :]
            for op in OPS:
                r = _apply(state[i], state[j], op)
                if r is not None:
                    out.append(tuple(sorted(rest + (r,))))
    return out


@cache
def solvable(state):
    return state[0] == 24 if len(state) == 1 else any(solvable(s) for s in moves(state))


def problem(numbers):
    return tuple(sorted(Fraction(int(x)) for x in numbers))


def features(state):
    """Per slot: present, x / 24 clipped to [-4, 4], sign, log(1 + |x|), integer, equal to 24; plus one-hot of how
    many numbers are left."""
    f = []
    for i in range(4):
        if i < len(state):
            x = float(state[i])
            f += [
                1.0,
                max(-4.0, min(4.0, x / 24)),
                float(x > 0) - float(x < 0),
                float(torch.log1p(torch.tensor(abs(x)))),
            ]
            f += [float(state[i].denominator == 1), float(x == 24)]
        else:
            f += [0.0] * 6
    return f + [float(len(state) == n) for n in (1, 2, 3, 4)]


def random_states(n, generator):
    """States met along random step sequences from random problems with numbers 1 to 13, with their labels."""
    xs, ys = [], []
    while len(xs) < n:
        s = problem(torch.randint(1, 14, (4,), generator=generator))
        while len(s) > 1 and len(xs) < n:
            xs.append(features(s))
            ys.append(float(solvable(s)))
            ms = moves(s)
            s = ms[int(torch.randint(len(ms), (1,), generator=generator))]
    return torch.tensor(xs), torch.tensor(ys)


class Evaluator(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model, x, y, steps=1500, batch=256, lr=3e-3, generator=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.randint(len(x), (batch,), generator=generator)
        opt.zero_grad()
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def beam_search(start, score, b):
    """Expand every state in the beam by every step, score the children, keep the best b, until two numbers are
    left; the last step is a goal test, so the search succeeds if a kept pair can make 24."""
    beam = [start]
    while len(beam[0]) > 2:
        cand = list(dict.fromkeys(c for s in beam for c in moves(s)))
        vals = score(cand)
        beam = [cand[i] for i in torch.argsort(vals, descending=True, stable=True)[:b].tolist()]
    return any(solvable(s) for s in beam)


def learned_score(model):
    return lambda states: model(torch.tensor([features(s) for s in states]))


def random_score(generator):
    return lambda states: torch.rand(len(states), generator=generator)
