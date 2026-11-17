# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://NEW-grokking).
# Check of derivations/weight-decay-shrinkage.md: AdamW with zero gradient from zero moments gives
# theta_t = (1 - lr * wd)^t theta_0; (1 - x)^t <= exp(-x t); the steps to shrink the norm from w0 to wc are
# ceil(ln(w0/wc) / -ln(1 - x)) <= ceil(ln(w0/wc) / x); at lr = 1e-3 and wd = 1, 1,000 steps give 0.3677 and 693 halve.
import math

import numpy as np


def adamw(theta, grad, m, v, t, lr, wd, b1=0.9, b2=0.98, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad * grad
    mhat, vhat = m / (1 - b1 ** (t + 1)), v / (1 - b2 ** (t + 1))
    return theta * (1 - lr * wd) - lr * mhat / (np.sqrt(vhat) + eps), m, v


rng = np.random.default_rng(1)
for lr, wd in [(1e-3, 1.0), (1e-2, 0.5), (3e-4, 0.1), (0.1, 2.0)]:
    theta0 = rng.normal(size=20)
    theta, m, v = theta0.copy(), np.zeros(20), np.zeros(20)
    for t in range(300):
        theta, m, v = adamw(theta, np.zeros(20), m, v, t, lr, wd)
    assert np.allclose(theta, (1 - lr * wd) ** 300 * theta0, rtol=1e-12)
    assert abs(np.linalg.norm(theta) - (1 - lr * wd) ** 300 * np.linalg.norm(theta0)) < 1e-12

for x in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]:
    for t in [1, 10, 100, 1000, 10000]:
        assert (1 - x) ** t <= math.exp(-x * t) * (1 + 1e-12)
    for ratio in [1.5, 2.0, 10.0, 100.0]:
        exact = math.ceil(math.log(ratio) / -math.log(1 - x))
        assert (1 - x) ** exact <= 1 / ratio * (1 + 1e-12) and (1 - x) ** (exact - 1) > 1 / ratio
        assert exact <= math.ceil(math.log(ratio) / x)
    if x <= 1e-3:
        assert abs(math.log(2) / -math.log(1 - x) - math.log(2) / x) / (math.log(2) / x) < 1e-3

assert abs(0.999**1000 - 0.36770) < 1e-5
assert math.ceil(math.log(2) / -math.log(0.999)) == 693
assert abs(math.log(2) / -math.log(0.999) - 692.8) < 0.05
print("weight-decay-shrinkage: ok")
