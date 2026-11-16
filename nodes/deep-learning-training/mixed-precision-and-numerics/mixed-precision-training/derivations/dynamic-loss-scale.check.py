# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://NEW-mixed-precision-training).
# Check of derivations/dynamic-loss-scale.md: the rule that halves the scale and skips the step on an overflow and
# doubles it after N steps in a row without one, run against a threshold T above which every step overflows. It
# skips floor(log2(S0 / T)) + 1 steps at the start when S0 >= T, then settles into a cycle of N + 1 steps with one
# skip, the scale at the largest S0 2^j below T on N of them; without overflows the scale doubles every N steps.
import math

import numpy as np


def run(s0, n, overflow, steps):
    """The scales used and the skip flags of `steps` steps; overflow(step, scale) tells if the step overflows."""
    s, good, scales, skips = s0, 0, [], []
    for t in range(steps):
        scales.append(s)
        bad = overflow(t, s)
        skips.append(bad)
        if bad:
            s, good = s * 0.5, 0
        else:
            good += 1
            if good == n:
                s, good = s * 2.0, 0
    return np.array(scales), np.array(skips)


def main():
    rng = np.random.default_rng(20261116)
    for _ in range(300):
        n = int(rng.choice([1, 2, 3, 7, 50, 2000]))
        s0 = 2.0 ** int(rng.integers(-4, 30))
        t_ = 2.0 ** rng.uniform(-10.0, 40.0)
        steps = 40 * (n + 1) + 200
        scales, skips = run(s0, n, lambda t, s, t_=t_: s >= t_, steps)
        lo = s0 * 2.0 ** math.floor(math.log2(t_ / s0))
        if lo == t_:
            lo /= 2.0
        assert t_ / 2 <= lo < t_
        # The start: one skip for each power S0 2^-j at or above T, in a row, if S0 >= T.
        first = math.floor(math.log2(s0 / t_)) + 1 if s0 >= t_ else 0
        if s0 >= t_ and s0 / 2.0**first == t_:
            first += 1
        assert skips[:first].all() and (first == len(skips) or not skips[first] or s0 < t_)
        if s0 >= t_:
            assert scales[first] == lo
            # Then the cycle: N steps at lo, one skipped step at 2 lo.
            tail_s, tail_k = scales[first:], skips[first:]
            m = (len(tail_k) // (n + 1)) * (n + 1)
            cyc_s = tail_s[:m].reshape(-1, n + 1)
            cyc_k = tail_k[:m].reshape(-1, n + 1)
            assert np.all(cyc_s[:, :n] == lo) and np.all(cyc_s[:, n] == 2 * lo)
            assert not cyc_k[:, :n].any() and cyc_k[:, n].all()
            assert np.mean(cyc_k) == 1 / (n + 1)
        else:
            # Below T the scale only grows, by 2 every N steps, until it first reaches T and that step is skipped.
            g = math.floor(math.log2(t_ / s0))
            if s0 * 2.0**g == t_:
                g -= 1
            j = (g + 1) * n
            if j < steps:
                assert not skips[:j].any() and skips[j]
                for i in range(j):
                    assert scales[i] == s0 * 2.0 ** (i // n)
    # The defaults of torch.amp.GradScaler, S0 = 2^16 and N = 2000: one skipped step in 2001 at steady state.
    scales, skips = run(2.0**16, 2000, lambda t, s: s >= 1000.0, 2001 * 5 + 7)
    assert skips[:7].all() and not skips[7] and scales[7] == 512.0
    assert np.isclose(skips[7:].mean(), 1 / 2001)
    # Overflows at random steps: every skip halves the scale and resets the count, so the scale is S0 2^(G - B) after
    # G doublings and B skips.
    n = 3
    mask = rng.random(500) < 0.1
    scales, skips = run(8.0, n, lambda t, s: bool(mask[t]), 500)
    assert np.array_equal(skips, mask)
    # G counts the completed blocks of N good steps inside each maximal run of good steps before step t.
    for t in range(500):
        runs = "".join("x" if b else "." for b in mask[:t]).split("x")
        doublings = sum(len(r) // n for r in runs)
        assert scales[t] == 8.0 * 2.0 ** (doublings - int(mask[:t].sum()))
    print("ok: the dynamic scale skips log2(S0/T) + 1 steps, then one step in N + 1, at the largest scale below T")


if __name__ == "__main__":
    main()
