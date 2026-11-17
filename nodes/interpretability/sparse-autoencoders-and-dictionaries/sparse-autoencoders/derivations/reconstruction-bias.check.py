# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
# Check of derivations/reconstruction-bias.md: gamma = E|x_hat|^2 / E[x_hat . x] is 1 / (best rescale of x_hat),
# its polarization form agrees, one-latent L1 shrinkage gives gamma = 1 - lambda / (2 s), and FVU end points are 1 and 0.
import numpy as np

rng = np.random.default_rng(11)


def gamma(x, xh):
    return np.sum(xh * xh) / np.sum(xh * x)


for _ in range(50):
    x = rng.normal(size=(20, 5))
    xh = 0.7 * x + 0.2 * rng.normal(size=x.shape)
    g = gamma(x, xh)
    cs = np.linspace(0.2, 3.0, 28001)
    errs = [np.sum((c * xh - x) ** 2) for c in cs]
    assert abs(cs[int(np.argmin(errs))] - 1.0 / g) < 2e-4
    alt = 2 * np.sum(xh**2) / (np.sum(xh**2) + np.sum(x**2) - np.sum((xh - x) ** 2))
    assert abs(alt - g) < 1e-12

for _ in range(50):
    m = int(rng.integers(2, 6))
    d = rng.normal(size=m)
    d /= np.linalg.norm(d)
    lam = float(rng.uniform(0.1, 2.0))
    s = float(rng.uniform(lam / 2 + 0.05, 4.0))
    x = s * d
    fs = np.linspace(0.0, 5.0, 10001)
    loss = np.array([np.sum((x - f * d) ** 2) + lam * f for f in fs])
    f_grid = fs[int(np.argmin(loss))]
    f_star = s - lam / 2
    assert abs(f_grid - f_star) < 1e-3
    xh = f_star * d
    assert abs(gamma(x[None], xh[None]) - (1 - lam / (2 * s))) < 1e-12

x = np.array([[1.0, 1.0]])
assert abs(gamma(x, 0.5 * x) - 0.5) < 1e-12

X = rng.normal(size=(30, 4))
mean = X.mean(axis=0, keepdims=True)
den = np.sum((X - mean) ** 2)
assert abs(np.sum((X - np.repeat(mean, 30, axis=0)) ** 2) / den - 1.0) < 1e-12
assert np.sum((X - X) ** 2) / den == 0.0
# worked example: inputs (3, 1), (1, 3), identity decoder, tied encoder, no decoder bias
X = np.array([[3.0, 1.0], [1.0, 3.0]])
den = np.sum((X - X.mean(axis=0)) ** 2)
assert den == 4.0
cases = {
    "l1-lambda-1": (np.maximum(X - 0.5, 0.0), 2, 1.0, 0.25, 13 / 16),
    "l1-lambda-2": (np.maximum(X - 1.0, 0.0), 1, 4.0, 1.0, 2 / 3),
    "topk-1": (np.where(X == X.max(axis=1, keepdims=True), X, 0.0), 1, 2.0, 0.5, 1.0),
}
for name, (Z, l0, sse, f, g) in cases.items():
    assert np.all(np.count_nonzero(Z, axis=1) == l0), name
    assert abs(np.sum((X - Z) ** 2) - sse) < 1e-12, name
    assert abs(np.sum((X - Z) ** 2) / den - f) < 1e-12, name
    assert abs(gamma(X, Z) - g) < 1e-12, name
print("ok")
