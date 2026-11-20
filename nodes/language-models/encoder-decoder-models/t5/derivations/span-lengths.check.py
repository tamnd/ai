# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Numeric check of the derivation span-lengths: the lengths of inputs and targets under span corruption, and the
expected spans of i.i.d. corruption."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import t5  # noqa: E402

rng = np.random.default_rng(0)
for _ in range(300):
    n = int(rng.integers(20, 600))
    r = float(rng.choice([0.1, 0.15, 0.25, 0.5]))
    mu = int(rng.choice([2, 3, 5, 10]))
    N = min(max(round(n * r), 1), n - 1)
    m = max(round(N / mu), 1)
    if n - N < m:
        continue
    mask = t5.random_spans_mask(n, r, mu, rng)
    inputs, targets = t5.span_corrupt(np.arange(n), mask, list(range(10_000, 10_000 + m + 1)))
    assert len(inputs) == n - N + m and len(targets) == N + m + 1

assert (500 - 75 + 25, 75 + 25 + 1, 75 / 25) == (450, 101, 3)
mask = t5.random_spans_mask(500, 0.15, 3, rng)
assert mask.sum() == 75 and (mask & ~np.concatenate([[False], mask[:-1]])).sum() == 25

n, r = 512, 0.15
spans = r + (n - 1) * r * (1 - r)
assert round(spans, 2) == 65.30 and round(n * r + spans + 1, 1) == 143.1 and round(n - n * r + spans, 1) == 500.5
assert round(1 / (1 - r), 3) == 1.176
trials = rng.random((20_000, n)) < r
starts = trials & ~np.concatenate([np.zeros((20_000, 1), bool), trials[:, :-1]], axis=1)
assert abs(starts.sum(1).mean() - spans) < 0.1
lengths = trials.sum() / starts.sum()
assert abs(lengths - 1 / (1 - r)) < 0.01
N = round(n * r)
m = round(N / 3)
assert (N, m, N + m + 1, n - N + m) == (77, 26, 104, 461)
assert (256, round(256 / 3), 256 + 85 + 1) == (round(512 * 0.5), 85, 342)
print("ok")
