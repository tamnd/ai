# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://TD3H76).
# Check of derivations/inference-fold.md: batch normalization in inference mode is an affine map, it folds into the
# linear layer before it, and the two momentum conventions give the same running average.
import numpy as np


def main():
    rng = np.random.default_rng(20261114)
    c, k, n, eps = 6, 4, 10, 1e-5
    gamma, beta = rng.uniform(0.5, 2.0, size=c), rng.normal(size=c)
    mu, var = rng.normal(size=c), rng.uniform(0.2, 3.0, size=c)
    w, b = rng.normal(size=(c, k)), rng.normal(size=c)
    z = rng.normal(size=(n, k))
    x = z @ w.T + b
    bn = gamma * (x - mu) / np.sqrt(var + eps) + beta
    a = gamma / np.sqrt(var + eps)
    assert np.allclose(bn, a * x + (beta - a * mu), rtol=1e-12, atol=1e-12)
    w2, b2 = np.diag(a) @ w, a * (b - mu) + beta
    assert np.allclose(bn, z @ w2.T + b2, rtol=1e-12, atol=1e-12)
    for alpha in [0.1, 0.01, 0.5]:
        r1 = r2 = rng.normal(size=c)
        for _ in range(5):
            t = rng.normal(size=c)
            r1 = (1 - alpha) * r1 + alpha * t
            rho = 1 - alpha
            r2 = rho * r2 + (1 - rho) * t
        assert np.allclose(r1, r2, rtol=1e-12, atol=1e-12), alpha
    print("ok: inference mode is affine, folds into the linear layer, and the momentum conventions agree")


if __name__ == "__main__":
    main()
