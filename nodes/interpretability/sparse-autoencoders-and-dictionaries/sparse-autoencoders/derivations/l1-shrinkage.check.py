# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
# Check of derivations/l1-shrinkage.md: the one-latent minimizer of ||x - f d||^2 + lam f over f >= 0 is
# max(0, <d, x> - lam/2), a ReLU with bias -lam/2 computes it, x = d with lam = 1 gives 1/2, and rescaling the decoder
# by s and the encoder by 1/s keeps the reconstruction and divides the penalty by s.
import numpy as np

rng = np.random.default_rng(11)
grid = np.linspace(0.0, 8.0, 800001)
for _ in range(200):
    m = int(rng.integers(1, 7))
    d = rng.normal(size=m)
    d /= np.linalg.norm(d)
    x = rng.normal(size=m) * 2.0
    lam = float(rng.uniform(0.0, 4.0))
    g = np.sum(x * x) - 2 * grid * (d @ x) + grid**2 + lam * grid
    f_star = max(0.0, d @ x - lam / 2)
    assert abs(grid[np.argmin(g)] - f_star) < 2e-5
    direct = np.sum((x - f_star * d) ** 2) + lam * f_star
    assert abs(direct - g.min()) < 1e-6
    if d @ x > lam / 2:
        assert abs(max(0.0, d @ x) - f_star - lam / 2) < 1e-12
    assert f_star == max(0.0, float(d @ x) + (-lam / 2))  # ReLU(<d, x> + b) with b = -lam/2

# the one-latent example: x = d, lam = 1
d = np.array([1.0])
f = max(0.0, 1.0 - 0.5)
assert f == 0.5 and np.sum((d - f * d) ** 2) == 0.25

# scale symmetry with several latents, an encoder W (n, m), bias b and decoder D (m, n)
for _ in range(50):
    m, n = 4, 9
    W, b, D = rng.normal(size=(n, m)), rng.normal(size=n), rng.normal(size=(m, n))
    x = rng.normal(size=(20, m))
    z = np.maximum(x @ W.T + b, 0.0)
    for s in (0.5, 3.0, 1e4):
        zs = np.maximum(x @ (W / s).T + b / s, 0.0)
        assert np.allclose(zs @ (D * s).T, z @ D.T)
        assert np.isclose(zs.sum(), z.sum() / s)
        assert np.array_equal(zs > 0, z > 0)
print("ok")
