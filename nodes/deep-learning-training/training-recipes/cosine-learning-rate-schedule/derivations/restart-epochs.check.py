# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
# Check of derivations/restart-epochs.md: with run lengths T_0 T_mult^i, restart k is at T_0 (T_mult^k - 1) / (T_mult - 1),
# or k T_0 when T_mult = 1; for T_0 = 10 and T_mult = 2 the restarts are at 10, 30, 70, 150 and 310.
import math

import numpy as np


def restart_loop(k, t0, mult):
    start, length = 0, t0
    for _ in range(k):
        start += length
        length *= mult
    return start


def restart_closed(k, t0, mult):
    return k * t0 if mult == 1 else t0 * (mult**k - 1) // (mult - 1)


for t0 in range(1, 21):
    for mult in range(1, 5):
        for k in range(13):
            assert restart_loop(k, t0, mult) == restart_closed(k, t0, mult), (t0, mult, k)

# non-integer T_mult with float arithmetic
for t0, mult in [(1.0, 1.5), (3.0, 1.25), (7.0, 2.5)]:
    for k in range(13):
        a, b = restart_loop(k, t0, mult), t0 * (mult**k - 1) / (mult - 1)
        assert abs(a - b) < 1e-9 * max(1.0, a), (t0, mult, k, a, b)

assert [restart_closed(k, 10, 2) for k in range(1, 6)] == [10, 30, 70, 150, 310]


def position_loop(t, t0, mult):
    i, start, length = 0, 0.0, t0
    while t >= start + length:
        start += length
        length *= mult
        i += 1
    return i, t - start


def position_closed(t, t0, mult):
    if mult == 1:
        i = math.floor(t / t0)
    else:
        i = math.floor(math.log(1 + t * (mult - 1) / t0, mult) + 1e-12)
    start = i * t0 if mult == 1 else t0 * (mult**i - 1) / (mult - 1)
    return i, t - start


rng = np.random.default_rng(0)
for t in list(rng.uniform(0, 600, size=4_991)) + [0, 9.999, 10, 30, 70, 150, 310, 0.5, 29.9]:
    for t0, mult in [(10, 2), (1, 2), (50, 1), (3, 3)]:
        i1, c1 = position_loop(float(t), t0, mult)
        i2, c2 = position_closed(float(t), t0, mult)
        assert i1 == i2 and abs(c1 - c2) < 1e-9, (t, t0, mult, i1, i2, c1, c2)
print("ok")
