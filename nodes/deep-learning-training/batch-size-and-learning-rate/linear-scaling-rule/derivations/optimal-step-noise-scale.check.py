# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://NEW-linear-scaling-rule).
# Check of derivations/optimal-step-noise-scale.md: on a quadratic L(theta) = theta.H theta / 2 + c with per-example
# gradients of mean G and covariance Sigma, the expected loss after the step theta - eps G_est with a batch of B is
# L - eps |G|^2 + eps^2 (G.HG + tr(H Sigma) / B) / 2; its minimizer is eps_max / (1 + B_noise / B) with
# eps_max = |G|^2 / G.HG and B_noise = tr(H Sigma) / G.HG, the best improvement is dL_max / (1 + B_noise / B), a step
# larger than 2 eps_opt raises the expected loss, and with H = h I the noise scale is B_simple = tr(Sigma) / |G|^2.
# Monte Carlo over sampled batches confirms each formula.
import numpy as np

rng = np.random.default_rng(1)
d = 6
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
H = Q @ np.diag(np.geomspace(0.2, 5.0, d)) @ Q.T
theta = rng.normal(size=d)
G = H @ theta
A = rng.normal(size=(d, d))
Sigma = 4.0 * A @ A.T / d  # per-example gradient covariance


def L(t):
    return 0.5 * np.einsum("...i,ij,...j->...", t, H, t)


eps_max = G @ G / (G @ H @ G)
b_noise = np.trace(H @ Sigma) / (G @ H @ G)
dl_max = 0.5 * (G @ G) ** 2 / (G @ H @ G)
Lc = np.linalg.cholesky(Sigma)

for B in [1, 4, 16, 64, 256]:
    M = 200_000
    # the mean of B per-example gradients has covariance Sigma / B; draw it directly
    g_est = G + (rng.normal(size=(M, d)) @ Lc.T) / np.sqrt(B)
    eps_opt = eps_max / (1 + b_noise / B)
    grid = eps_opt * np.linspace(0.5, 1.5, 41)
    mc = np.array([L(theta - e * g_est).mean() for e in grid])
    closed = L(theta) - grid * (G @ G) + 0.5 * grid**2 * (G @ H @ G + np.trace(H @ Sigma) / B)
    assert np.allclose(mc, closed, rtol=0, atol=2e-2 * abs(closed).max()), B
    # the minimizer of the Monte Carlo curve is eps_opt; fit a parabola to it
    a2, a1, _ = np.polyfit(grid, mc, 2)
    assert abs(-a1 / (2 * a2) / eps_opt - 1) < 0.03, (B, -a1 / (2 * a2), eps_opt)
    gain = L(theta) - L(theta - eps_opt * g_est).mean()
    assert abs(gain / (dl_max / (1 + b_noise / B)) - 1) < 0.03, B
    # twice the optimal step gives zero expected gain; more than twice makes the loss worse
    assert abs(L(theta - 2 * eps_opt * g_est).mean() - L(theta)) < 0.03 * dl_max
    assert L(theta - 2.3 * eps_opt * g_est).mean() > L(theta)

# small B: eps_opt grows linearly in B; large B: it saturates at eps_max
e = lambda B: eps_max / (1 + b_noise / B)
assert abs(e(0.02 * b_noise) / e(0.01 * b_noise) - 2) < 0.02
assert e(1000 * b_noise) / eps_max > 0.999
assert abs(e(b_noise) / eps_max - 0.5) < 1e-12

# H = h I: B_noise equals B_simple = tr(Sigma) / |G|^2
h = 2.5
Hi = h * np.eye(d)
Gi = Hi @ theta
assert np.isclose(np.trace(Hi @ Sigma) / (Gi @ Hi @ Gi), np.trace(Sigma) / (Gi @ Gi))
# and the expected squared error of the batch gradient relative to |G|^2 is B_simple / B (Monte Carlo)
B = 8
g_est = G + (rng.normal(size=(200_000, d)) @ Lc.T) / np.sqrt(B)
ratio = np.mean(np.sum((g_est - G) ** 2, axis=1)) / (G @ G)
assert abs(ratio / (np.trace(Sigma) / (G @ G) / B) - 1) < 0.02
print("ok")
