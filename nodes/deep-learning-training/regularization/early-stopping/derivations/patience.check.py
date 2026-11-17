# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://NEW-early-stopping).
# Check of derivations/patience.md: the patience rule stops exactly p evaluations after its last improvement b;
# every evaluation before b is worse than the value at b, every one after is at least that value minus min_delta;
# on a curve that falls to m and then does not fall it returns m and stops at m + p; a flat stretch of p evaluations
# followed by a drop defeats patience p; and the kept value is biased low on a noisy flat curve.
import math

import numpy as np


def patience_rule(values, p, delta=0.0):
    """Return (stop epoch T or None, best epoch b, best value). Epochs count from 1."""
    best, b, wait = math.inf, None, 0
    for t, v in enumerate(values, start=1):
        if not math.isnan(v) and v < best - delta:
            best, b, wait = v, t, 0
        else:
            wait += 1
        if wait >= p:
            return t, b, best
    return None, b, best


def main():
    rng = np.random.default_rng(20261116)
    for _ in range(5000):
        n = int(rng.integers(2, 60))
        p = int(rng.integers(1, 8))
        delta = float(rng.choice([0.0, 0.01, 0.1]))
        vals = np.cumsum(rng.normal(scale=0.1, size=n)) + 1.0
        T, b, best = patience_rule(vals, p, delta)
        seen = vals if T is None else vals[:T]
        if T is not None:
            assert T == b + p, "the overshoot is exactly p"
        assert best == vals[b - 1]
        assert np.all(seen[: b - 1] > best), "everything before b is worse"
        assert np.all(seen[b:] >= best - delta), "everything after b is within delta"
        assert best <= seen.min() + delta
        if delta == 0.0:
            assert best == seen.min()
    print("ok: 5000 random walks: T = b + p, best <= min seen + delta, and best = min seen when delta = 0")

    for m in (1, 5, 30):
        for p in (1, 3, 10):
            vals = np.concatenate([np.linspace(1.0 + 0.1 * (m - 1), 1.0, m), 1.0 + np.cumsum(rng.uniform(0, 0.1, size=50))])
            assert patience_rule(vals, p) == (m + p, m, 1.0)
    print("ok: a curve that falls to epoch m and then does not fall gives (m + p, m)")

    for p in range(1, 21):
        vals = [1.0] + [1.0] * p + [0.5]
        T, b, _ = patience_rule(vals, p)
        assert (T, b) == (p + 1, 1) and min(vals) == 0.5, "the drop after the plateau is never seen"
    print("ok: for p = 1 .. 20 a plateau of p evaluations then a drop stops before the drop")

    sel = []
    for _ in range(20000):
        vals = 1.0 + 0.05 * rng.normal(size=100)
        _, _, best = patience_rule(vals, 10)
        sel.append(best)
    bias = float(np.mean(sel) - 1.0)
    assert bias < -0.05, "the kept value is biased low"
    print(f"ok: flat curve, noise sd 0.05, p = 10: mean kept value - true value = {bias:.4f}")

    T, b, best = patience_rule([math.nan, 1.0, math.nan, math.nan], 2)
    assert (T, b, best) == (4, 2, 1.0)
    print("ok: a NaN never improves")


if __name__ == "__main__":
    main()
