# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://1H67K0).
# Check of derivations/expert-capacity.md: with C = floor(k T c / N) slots per expert, a slot-by-slot dispatch keeps
# min(n_i, C) choices per expert, drops sum max(0, n_i - C) >= k T - N C >= k T (1 - c), pads N C - k T + D slots,
# drops nothing under even routing with c >= 1, and gives GShard's 2 N / (G E) per group for top-2.
import math

import numpy as np


def capacity(t, n, c, k=1):
    return int(math.floor(k * t * c / n + 1e-9))


def simulate(choices, n, cap):
    # choices: (T, k) distinct experts per token, taken rank by rank, then in token order.
    t, k = choices.shape
    count = np.zeros(n, dtype=int)
    kept = 0
    for r in range(k):
        for s in range(t):
            e = choices[s, r]
            if count[e] < cap:
                kept += 1
            count[e] += 1
    return kept, count


def main():
    rng = np.random.default_rng(20261116)

    for _ in range(3000):
        n = int(rng.integers(2, 17))
        k = int(rng.integers(1, min(3, n) + 1))
        t = int(rng.integers(1, 80))
        c = float(rng.uniform(0.25, 3.0))
        skew = rng.dirichlet(np.full(n, rng.uniform(0.1, 5.0)))
        choices = np.array([rng.choice(n, size=k, replace=False, p=skew) for _ in range(t)])
        cap = capacity(t, n, c, k)
        kept, loads = simulate(choices, n, cap)
        assert loads.sum() == k * t
        dropped = int(np.maximum(loads - cap, 0).sum())
        assert kept == int(np.minimum(loads, cap).sum())
        assert kept + dropped == k * t
        padding = n * cap - kept
        assert padding == n * cap - k * t + dropped and padding >= 0
        assert dropped >= k * t - n * cap
        assert k * t - n * cap >= k * t * (1 - c) - 1e-9

    # Item 2: even routing with c >= 1 drops nothing.
    for n in (2, 4, 8, 16):
        for k in (1, 2):
            for per in (1, 3, 8):
                t = n * per
                choices = np.array([[(s + r) % n for r in range(k)] for s in range(t)])
                for c in (1.0, 1.25, 1.5, 2.0):
                    cap = capacity(t, n, c, k)
                    kept, loads = simulate(choices, n, cap)
                    assert (loads == k * t // n).all()
                    assert kept == k * t
                    frac = 1 - k * t / (n * cap)
                    assert frac <= 1 - 1 / c + 1e-12

    # Item 3: everything to one expert.
    for cap_t in ((8, 4, 1.0), (8, 4, 3.0), (100, 10, 1.25)):
        t, n, c = cap_t
        cap = capacity(t, n, c)
        kept, _ = simulate(np.zeros((t, 1), dtype=int), n, cap)
        assert t - kept == t - min(t, cap)

    # Item 4: the worked example.
    loads_example = np.array([[0], [1], [0], [2], [0], [1], [3], [0]])
    for c, cap_expect, drop_expect, slots in ((1.0, 2, 2, 8), (1.5, 3, 1, 12), (0.5, 1, 4, 4)):
        cap = capacity(8, 4, c)
        assert cap == cap_expect
        kept, loads = simulate(loads_example, 4, cap)
        assert loads.tolist() == [4, 2, 1, 1]
        assert 8 - kept == drop_expect and 4 * cap == slots

    # Item 5: GShard's per-group capacity for top-2.
    for batch, groups, experts in ((2048, 8, 64), (4096, 16, 128), (1 << 20, 1024, 2048)):
        s = batch // groups
        assert capacity(s, experts, 1.0, k=2) == (2 * batch) // (groups * experts)
    print("ok: kept = sum min(n_i, C); D = sum max(0, n_i - C) >= kT - NC >= kT(1 - c); even routing drops nothing")


if __name__ == "__main__":
    main()
