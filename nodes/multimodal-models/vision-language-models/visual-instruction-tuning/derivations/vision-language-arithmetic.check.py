"""Check every number in vision-language-arithmetic.md."""

import math

import numpy as np


def close(x, y, tol=5e-5):
    assert abs(x - y) <= tol * max(1.0, abs(y)), (x, y)


# 1. An always-yes model on a balanced split.
n = 3000
labels = np.array([1] * (n // 2) + [0] * (n // 2))
pred = np.ones(n, dtype=int)
tp = int(((pred == 1) & (labels == 1)).sum())
fp = int(((pred == 1) & (labels == 0)).sum())
fn = int(((pred == 0) & (labels == 1)).sum())
acc = float((pred == labels).mean())
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
close(acc, 0.5)
close(prec, 0.5)
close(rec, 1.0)
close(f1, 2 / 3)
assert round(100 * acc, 2) == 50.00 and round(100 * f1, 2) == 66.67

# 2. Visual tokens.
assert (224 // 14) ** 2 == 256 and (224 // 14) ** 2 + 1 == 257
assert (336 // 14) ** 2 == 576 and 336 // 14 == 24
assert 5 * 576 == 2880 and 5 * 32 == 160 and 5 * 64 == 320
assert (2880 // 160) ** 2 == 324


# 3. The gate: identity at init, gradient only on alpha.
rng = np.random.default_rng(0)
d = 8
y = rng.standard_normal(d)
x = rng.standard_normal((5, d))
Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
g = rng.standard_normal(d)  # upstream gradient


def block(alpha, Wv_):
    q, k, v = y @ Wq, x @ Wk, x @ Wv_
    s = k @ q / math.sqrt(d)
    p = np.exp(s - s.max())
    p /= p.sum()
    f = p @ v
    return y + math.tanh(alpha) * f, f


out, f = block(0.0, Wv)
assert np.allclose(out, y)
eps = 1e-6
da = (g @ block(eps, Wv)[0] - g @ block(-eps, Wv)[0]) / (2 * eps)
close(da, float(g @ f), 1e-6)
E = np.zeros_like(Wv)
E[2, 3] = 1.0
dw = (g @ block(0.0, Wv + eps * E)[0] - g @ block(0.0, Wv - eps * E)[0]) / (2 * eps)
assert abs(dw) < 1e-12
close(1 - math.tanh(0.0) ** 2, 1.0)

# 4. Trainable-parameter ratio.
close(10.2e9 / 188e6, 54.26, 1e-4)
assert round(10.2e9 / 188e6, 1) == 54.3

# 5. LLaVA-Instruct-158K.
assert 58 + 23 + 77 == 158
assert round(100 * 58 / 158) == 37 and round(100 * 23 / 158) == 15 and round(100 * 77 / 158) == 49

# 6. Training compute.
assert (6 + 20) * 8 == 208
assert 1536 * 15 * 24 == 552960
assert int(552960 / 208) == 2658

# 7. Projection parameters.
assert 1024 * 5120 == 5242880
assert 5120 * 5120 == 26214400
assert 1024 * 5120 + 5120 * 5120 == 31457280
assert round(31457280 / 5242880) == 6
close(100 * 31457280 / 13e9, 0.242, 1e-3)
print("ok")
