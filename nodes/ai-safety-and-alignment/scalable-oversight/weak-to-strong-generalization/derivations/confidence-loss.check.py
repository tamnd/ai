# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://NEW-weak-to-strong-generalization).
# Check of derivations/confidence-loss.md: (1 - a) CE(p, q_w) + a CE(p, h) = CE(p, (1 - a) q_w + a h); a = 0 and a = 1
# give the CE to the weak labels and to the hardened predictions; for a fixed target the minimizer is the target and
# the logit gradient is p - target; for a > 1/2 the target sides with h; the adaptive threshold hardens half a batch.
import numpy as np


def ce(p, q):
    return -(q * np.log(p) + (1 - q) * np.log(1 - p))


rng = np.random.default_rng(5)
for _ in range(500):
    b = int(rng.integers(2, 20))
    p = rng.uniform(0.001, 0.999, size=b)
    qw = rng.uniform(size=b)
    s = np.sort(p)
    k = b // 2
    t = s[b - k - 1]
    h = (p > t).astype(float)
    assert h.sum() == k
    a = rng.uniform()
    two = (1 - a) * ce(p, qw) + a * ce(p, h)
    one = ce(p, (1 - a) * qw + a * h)
    assert np.allclose(two, one, atol=1e-12)
    assert np.allclose((1 - 0) * ce(p, qw) + 0 * ce(p, h), ce(p, qw))
    assert np.allclose(0 * ce(p, qw) + 1 * ce(p, h), ce(p, h))

grid = np.linspace(1e-5, 1 - 1e-5, 100001)
for _ in range(50):
    qw, a, h = rng.uniform(), rng.uniform(), float(rng.integers(0, 2))
    q = (1 - a) * qw + a * h
    if 1e-3 < q < 1 - 1e-3:
        assert abs(grid[np.argmin(ce(grid, q))] - q) < 2e-5
    z = rng.normal() * 2
    sig = lambda u: 1 / (1 + np.exp(-u))  # noqa: E731
    eps = 1e-6
    num = (ce(sig(z + eps), q) - ce(sig(z - eps), q)) / (2 * eps)
    assert abs(num - (sig(z) - q)) < 1e-6

for a in np.linspace(0, 1, 101):
    for qw in np.linspace(0, 1, 101):
        up = (1 - a) * qw + a * 1.0
        down = (1 - a) * qw + a * 0.0
        if a > 0.5:
            assert up > 0.5 and down < 0.5
        if qw < 0.5:
            assert (up > 0.5) == (a > (0.5 - qw) / (1 - qw) + 1e-12) or abs(
                a - (0.5 - qw) / (1 - qw)
            ) < 1e-9

assert abs(0.2 / 0.7 - 0.2857) < 1e-4
assert abs(0.5 * 0.3 + 0.5 * 1.0 - 0.65) < 1e-12
print("ok")
