# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://NEW-critical-batch-size).
# Check of derivations/one-step-improvement.md. A finite data set of N examples has the per-example losses
# L_i(theta) = (theta - c_i).H(theta - c_i) / 2, so the full loss is an exact quadratic. We take real SGD steps
# theta - eps * G_B, where G_B is the mean gradient of B examples drawn with replacement, and average the loss after the
# step over many sampled batches. The average must match
#   E[L] = L(theta) - eps |G|^2 + eps^2 (G.HG + tr(H Sigma) / B) / 2,
# its minimizer must be eps_max / (1 + B_noise / B), and the gain at the minimizer dL_max / (1 + B_noise / B).
import numpy as np

rng = np.random.default_rng(7)
d, N = 8, 64
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
H = Q @ np.diag(np.geomspace(0.3, 3.0, d)) @ Q.T
# skewed, non-Gaussian example offsets, so the batch gradient is not Gaussian at small B
c = rng.exponential(1.0, size=(N, d)) * rng.choice([-1.0, 1.0], size=(N, d))
# a point near the minimum c_bar, where the noise dominates and B_noise is large
theta = c.mean(axis=0) + 0.6 * rng.normal(size=d)


def loss(t):
    diff = t[..., None, :] - c  # (..., N, d)
    return 0.5 * np.einsum("...ni,ij,...nj->...", diff, H, diff) / N


g = (theta - c) @ H  # (N, d) per-example gradients at theta
G = g.mean(axis=0)
Sigma = (g - G).T @ (g - G) / N  # population covariance: batches are drawn with replacement
gHg = G @ H @ G
eps_max = G @ G / gHg
b_noise = np.trace(H @ Sigma) / gHg
dl_max = 0.5 * (G @ G) ** 2 / gHg
L0 = loss(theta)
print(f"B_noise = {b_noise:.3f}, B_simple = {np.trace(Sigma) / (G @ G):.3f}, eps_max = {eps_max:.4f}")

M = 200_000
for B in [1, 4, 16, 64, 256, 1024]:
    counts = rng.multinomial(B, np.full(N, 1.0 / N), size=M)  # (M, N) how often each example is in the batch
    g_b = counts @ g / B  # (M, d) batch gradients of real sampled batches
    eps_opt = eps_max / (1.0 + b_noise / B)
    grid = eps_opt * np.linspace(0.25, 1.75, 31)
    # the loss is quadratic, so the average of L(theta - e g_b) over the M batches is exact from their first two
    # sample moments; the direct average over the batches is checked below for one B
    m1 = g_b.mean(axis=0)
    m2 = g_b.T @ g_b / M
    mc = L0 - grid * (G @ m1) + 0.5 * grid**2 * np.trace(H @ m2)  # exact average over the M sampled batches
    closed = L0 - grid * (G @ G) + 0.5 * grid**2 * (gHg + np.trace(H @ Sigma) / B)
    assert np.allclose(mc, closed, rtol=0.0, atol=0.01 * dl_max), B
    a2, a1, _ = np.polyfit(grid, mc, 2)
    eps_hat = -a1 / (2 * a2)
    gain = L0 - (L0 - eps_opt * (G @ m1) + 0.5 * eps_opt**2 * np.trace(H @ m2))
    print(f"B={B:4d}  eps_opt={eps_opt:.4f}  MC argmin={eps_hat:.4f}  gain={gain:.5f}  "
          f"predicted={dl_max / (1 + b_noise / B):.5f}")
    assert abs(eps_hat / eps_opt - 1) < 0.02, (B, eps_hat, eps_opt)
    assert abs(gain / (dl_max / (1 + b_noise / B)) - 1) < 0.02, B

# a direct check that the per-batch loss average (not only the moments) agrees, at B = 4
B = 4
counts = rng.multinomial(B, np.full(N, 1.0 / N), size=20_000)
g_b = counts @ g / B
e = eps_max / (1 + b_noise / B)
direct = loss(theta - e * g_b).mean()
assert abs((L0 - direct) / (dl_max / (1 + b_noise / B)) - 1) < 0.03, direct

# the regimes: linear in B well below B_noise, flat well above, half of eps_max at B = B_noise
f = lambda B: eps_max / (1 + b_noise / B)
assert abs(f(b_noise / 100) / f(b_noise / 200) - 2) < 0.01
assert f(100 * b_noise) / eps_max > 0.99
assert np.isclose(f(b_noise), eps_max / 2)

# a step beyond twice eps_opt raises the loss, on average over sampled batches; a step just under it lowers it
B = 16
counts = rng.multinomial(B, np.full(N, 1.0 / N), size=M)
g_b = counts @ g / B
m1, m2 = g_b.mean(axis=0), g_b.T @ g_b / M
after = lambda e: L0 - e * (G @ m1) + 0.5 * e**2 * np.trace(H @ m2)
assert after(2.2 * f(B)) > L0 and after(1.8 * f(B)) < L0
# the relative error of the batch gradient is B_simple / B
b_simple = np.trace(Sigma) / (G @ G)
rel = np.mean(np.sum((g_b - G) ** 2, axis=1)) / (G @ G)
assert abs(rel / (b_simple / B) - 1) < 0.02, (rel, b_simple / B)
# with H a multiple of the identity the two noise scales agree
H2 = 1.7 * np.eye(d)
g2 = (theta - c) @ H2
G2 = g2.mean(axis=0)
S2 = (g2 - G2).T @ (g2 - G2) / N
assert np.isclose(np.trace(H2 @ S2) / (G2 @ H2 @ G2), np.trace(S2) / (G2 @ G2))
print("ok")
