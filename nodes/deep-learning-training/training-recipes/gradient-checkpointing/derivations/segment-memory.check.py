# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://NEW-gradient-checkpointing).
# Check of derivations/segment-memory.md: a step-by-step simulation of the segment schedule gives the peak
# k + floor(n / k) and 2n - k layer forwards; enumeration over k finds the minimum 2q at k = q when n = q^2.
import math

import numpy as np


def simulate(n, k):
    """Replay the schedule on a set of saved indices; return (peak saved, layer forwards)."""
    q, r = divmod(n, k)
    bounds = np.cumsum([0] + [q + 1] * r + [q] * (k - r)).tolist()
    saved, peak, forwards = set(), 0, 0
    saved.add(0)
    for i in range(1, n + 1):  # forward pass: keep only the boundaries
        forwards += 1
        if i in bounds:
            saved.add(i)
        peak = max(peak, len(saved))
    for lo, hi in reversed(list(zip(bounds[:-1], bounds[1:], strict=True))):
        for i in range(lo + 1, hi):  # recompute the inside of the segment
            forwards += 1
            saved.add(i)
            peak = max(peak, len(saved))
        assert all(j in saved for j in range(lo, hi + 1))  # every layer of the segment has its input and output
        for i in range(lo + 1, hi + 1):
            saved.discard(i)
    assert saved == {0}
    return peak, forwards


for n in range(1, 81):
    for k in range(1, n + 1):
        peak, forwards = simulate(n, k)
        assert peak == k + n // k, (n, k, peak)
        assert forwards == 2 * n - k, (n, k, forwards)
        assert k + n / k >= 2 * math.sqrt(n) - 1e-12

for q in range(1, 41):
    n = q * q
    mem = [k + n // k for k in range(1, n + 1)]
    assert min(mem) == 2 * q and mem[q - 1] == 2 * q
    real = [k + n / k for k in np.linspace(1, n, 20001)]
    assert min(real) >= 2 * q - 1e-9

for n in range(1, 5001):
    assert min(k + n // k for k in range(1, n + 1)) == math.isqrt(4 * n + 1)

table = [(k, *simulate(9, k)) for k in range(1, 10)]
print("n = 9, (k, peak, forwards):", table)
print("n = 100, k with the least peak:", [k for k in range(1, 101) if k + 100 // k == 20])
print("ok")
