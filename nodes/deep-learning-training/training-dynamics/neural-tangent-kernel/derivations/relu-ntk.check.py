# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://3SJCNH).
# Check of derivations/relu-ntk.md: the empirical NTK of f(x) = (1/sqrt(m)) sum_j a_j relu(w_j . x) at width m tends
# to the arc-cosine form, the first-layer term x.x' (pi - phi)/(2 pi) plus the second-layer term
# |x||x'| (sin phi + (pi - phi) cos phi)/(2 pi), with a root-mean-square error proportional to 1/sqrt(m).
import numpy as np

rng = np.random.default_rng(1)
n, d = 6, 4
X = rng.standard_normal((n, d)) * rng.uniform(0.5, 2.0, size=(n, 1))


def analytic(X):
    nr = np.linalg.norm(X, axis=1)
    cos = np.clip(X @ X.T / np.outer(nr, nr), -1, 1)
    phi = np.arccos(cos)
    first = X @ X.T * (np.pi - phi) / (2 * np.pi)
    second = np.outer(nr, nr) * (np.sin(phi) + (np.pi - phi) * cos) / (2 * np.pi)
    return first, second


def empirical(X, m, seed):
    r = np.random.default_rng(seed)
    W, a = r.standard_normal((m, X.shape[1])), r.standard_normal(m)
    U = X @ W.T
    first = (((U > 0) * a**2) @ (U > 0).T) * (X @ X.T) / m
    second = np.maximum(U, 0) @ np.maximum(U, 0).T / m
    return first, second


A1, A2 = analytic(X)
A = A1 + A2
# The diagonal is |x|^2: half from each term.
assert np.allclose(np.diag(A1), np.sum(X**2, axis=1) / 2) and np.allclose(np.diag(A2), np.sum(X**2, axis=1) / 2)

# Monte Carlo of the two Gaussian integrals with one million draws of w.
w = rng.standard_normal((1_000_000, d))
Uw = X @ w.T
N = w.shape[0]
for name, Z, target in (
    ("first", (Uw > 0).astype(float), A1 / (X @ X.T)),
    ("second", np.maximum(Uw, 0), A2),
):
    mean = Z @ Z.T / N
    se = np.sqrt(np.maximum((Z**2) @ (Z**2).T / N - mean**2, 0) / N)
    z = np.abs(mean - target).max() / se.max()
    print(
        f"Monte Carlo of the {name} integral, 1e6 draws: max error {np.abs(mean - target).max():.2e}, {z:.2f} standard errors"
    )
    assert np.all(np.abs(mean - target) <= 5 * se + 1e-12)

widths = [2**k for k in range(6, 17, 2)]
rms = []
for m in widths:
    sq = []
    for s in range(30):
        F, S = empirical(X, m, 100 + s)
        sq.append(np.mean((F + S - A) ** 2))
    rms.append(np.sqrt(np.mean(sq)))
    print(
        f"m={m:<6} root-mean-square error of the empirical NTK: {rms[-1]:.5f}  sqrt(m)*error: {np.sqrt(m) * rms[-1]:.4f}"
    )
slope = np.polyfit(np.log(widths), np.log(rms), 1)[0]
print(f"fitted slope of log error against log m: {slope:.4f} (the derivation gives -1/2)")
assert abs(slope + 0.5) < 0.05
print("ok")
