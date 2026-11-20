# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""Numeric check of the derivation linear-autoencoder-and-pca."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoencoder as ae  # noqa: E402

rng = np.random.default_rng(0)
s = np.array([5.0, 3.0, 1.0, 0.4, 0.1])  # the eigenvalues of S
D, C, N = 5, 2, 400
Q = np.linalg.qr(rng.normal(size=(D, D)))[0]
Z = rng.normal(size=(N, D))
Z -= Z.mean(0)
Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z / N)).T
X = 2.0 + Z * np.sqrt(s) @ Q.T  # a mean of 2 and a covariance of exactly Q diag(s) Q^T
Xc = X - X.mean(0)
np.testing.assert_allclose(Xc.T @ Xc / N, Q * s @ Q.T, atol=1e-12)


def err(R):
    return np.sum((X - R) ** 2) / N


# statement 1: PCA reaches the discarded variance, random rank-C maps with the best bias do worse
assert abs(err(ae.pca_reconstruct(X, C)) - s[C:].sum()) < 1e-10
for _ in range(200):
    A, B = rng.normal(size=(D, C)), rng.normal(size=(C, D))
    R = X.mean(0) + Xc @ A @ B  # the best b_2 for this A and B
    assert err(R) >= s[C:].sum() - 1e-12
# a trained linear autoencoder with biases reaches it, and its decoder rows lie in the principal subspace
cfg = ae.from_widths([D, C], code_act="linear", out="linear")
p = ae.init(rng, cfg)
for _ in range(30000):
    _, g = ae.objective(p, cfg, X)
    p = {k: [W - 0.01 * G for W, G in zip(p[k], g[k])] for k in p}
assert abs(err(ae.reconstruct(p, cfg, X)) - s[C:].sum()) < 1e-9
P = Q[:, :C] @ Q[:, :C].T
np.testing.assert_allclose(p["dec_W"][0] @ P, p["dec_W"][0], atol=1e-4)
np.testing.assert_allclose(p["enc_W"][0] @ p["dec_W"][0], P, atol=1e-4)  # A B is the projection

# statement 2: the GL(C) symmetry, and a decoder whose singular vectors are not the principal directions
Ginv = np.array([[2.0, 1.0], [0.0, 1.0]])
B = Ginv @ Q[:, :C].T
A = np.linalg.pinv(B)
np.testing.assert_allclose(A @ B, P, atol=1e-12)  # still a minimum
U = np.linalg.svd(B.T, full_matrices=False)[0]
assert np.abs(U.T @ Q[:, :C]).max(0).min() < 0.95  # a left singular vector off every principal direction
for _ in range(20):
    G = rng.normal(size=(C, C))
    q = {"enc_W": [p["enc_W"][0] @ G], "enc_b": [p["enc_b"][0] @ G], "dec_W": [np.linalg.solve(G, p["dec_W"][0])],
         "dec_b": p["dec_b"]}  # fmt: skip
    np.testing.assert_allclose(ae.reconstruct(q, cfg, X), ae.reconstruct(p, cfg, X), atol=1e-8)

# statements 3 and 4: gradient descent on E_lambda from several starts, on the centred data (step 1)
for lam in (0.2, 2.0):
    for seed in range(3):
        r = np.random.default_rng(100 + seed)
        p = ae.init(r, ae.from_widths([D, 3], code_act="linear", out="linear"))
        cfg3 = ae.from_widths([D, 3], code_act="linear", out="linear")
        for _ in range(12000):
            _, g = ae.objective(p, cfg3, Xc, weight_decay=lam)
            p = {k: [W - 0.04 * G for W, G in zip(p[k], g[k])] for k in p}
        np.testing.assert_allclose(p["enc_W"][0], p["dec_W"][0].T, atol=1e-6)
        U, sv, _ = np.linalg.svd(p["dec_W"][0].T, full_matrices=False)
        want = np.clip(1.0 - lam / s[:3], 0.0, None)
        np.testing.assert_allclose(sv**2, want, atol=1e-5)
        for i in range(3):
            if want[i] > 0:
                assert abs(abs(U[:, i] @ Q[:, i]) - 1.0) < 1e-5
# the scalar case of Kunin et al.: w2 w1 = 1 - lambda / x^2 with w1 = w2
x2, lam = 4.0, 1.0
w = np.sqrt(1 - lam / x2)
grad = lambda w1, w2: (2 * w2 * (w2 * w1 - 1) * x2 + 2 * lam * w1, 2 * w1 * (w2 * w1 - 1) * x2 + 2 * lam * w2)  # noqa: E731
np.testing.assert_allclose(grad(w, w), (0.0, 0.0), atol=1e-15)
print("linear-autoencoder-and-pca: ok")
