# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://JDMWH4).
# Check of derivations/k-small-steps.md: on least squares, k SGD steps with batches B_0..B_{k-1} of size n and rate eta
# differ from one step with their union and rate k*eta by exactly eta^2 sum_j H_j sum_{i<j} g_i(w_{t+i}), whose norm is at
# most (k - 1) eta lambda_max / 2 times the size of the large step; the gap shrinks in proportion to eta; any rate other
# than k*eta leaves a first-order gap; and the rule fails once k*eta*lambda_max > 2, where the large-batch run diverges
# while the k small steps still converge.
import numpy as np

rng = np.random.default_rng(0)
N, d, n, k = 4096, 8, 32, 8
X = rng.normal(size=(N, d)) * np.linspace(0.3, 1.5, d)
w_true = rng.normal(size=d)
y = X @ w_true + 0.5 * rng.normal(size=N)


def grad(w, idx):
    Xb = X[idx]
    return Xb.T @ (Xb @ w - y[idx]) / len(idx)


def hess(idx):
    Xb = X[idx]
    return Xb.T @ Xb / len(idx)


def loss(w):
    r = X @ w - y
    return 0.5 * float(r @ r) / N


lam_max = float(np.linalg.eigvalsh(X.T @ X / N).max())

# 1. the exact identity and the bound, for several rates
w0 = rng.normal(size=d)
for eta in [1e-3, 1e-2, 5e-2]:
    batches = [rng.choice(N, n, replace=False) for _ in range(k)]
    w, gs = w0.copy(), []
    for b in batches:
        gs.append(grad(w, b))
        w = w - eta * gs[-1]
    union = np.concatenate(batches)
    w_big = w0 - k * eta * grad(w0, union)
    pred = eta**2 * sum(hess(batches[j]) @ sum(gs[:j]) for j in range(1, k))
    assert np.allclose(w - w_big, pred, atol=1e-12), (w - w_big, pred)
    lam_b = max(float(np.linalg.eigvalsh(hess(b)).max()) for b in batches)
    bound = (k - 1) * eta * lam_b / 2 * max(np.linalg.norm(g) for g in gs) * k * eta
    assert np.linalg.norm(w - w_big) <= bound + 1e-15

# 2. the gap is second order: halving eta about quarters it, so the relative gap halves
gaps = []
for eta in [4e-3, 2e-3, 1e-3]:
    batches = [np.arange(j * n, (j + 1) * n) for j in range(k)]
    w = w0.copy()
    for b in batches:
        w = w - eta * grad(w, b)
    w_big = w0 - k * eta * grad(w0, np.arange(k * n))
    gaps.append(np.linalg.norm(w - w_big) / np.linalg.norm(w_big - w0))
assert 1.8 < gaps[0] / gaps[1] < 2.2 and 1.8 < gaps[1] / gaps[2] < 2.2, gaps

# 3. only eta_hat = k eta matches to first order: other rates leave a gap that does not shrink with eta
for factor in [np.sqrt(k), k / 2, 2 * k]:
    eta = 1e-3
    w = w0.copy()
    for b in batches:
        w = w - eta * grad(w, b)
    w_other = w0 - factor * eta * grad(w0, np.arange(k * n))
    w_lin = w0 - k * eta * grad(w0, np.arange(k * n))
    assert np.linalg.norm(w - w_other) > 20 * np.linalg.norm(w - w_lin)


# 4. whole runs: T small steps against T/k scaled steps, with small and with large eta
def run(eta, batch, steps, seed):
    r = np.random.default_rng(seed)
    w = np.zeros(d)
    for _ in range(steps):
        w = w - eta * grad(w, r.choice(N, batch, replace=False))
        if not np.isfinite(w).all() or np.linalg.norm(w) > 1e6:
            return np.inf
    return loss(w)


T = 800
best = loss(np.linalg.lstsq(X, y, rcond=None)[0])
eta_small = 0.05 / lam_max
small = run(eta_small, n, T, 1)
large = run(k * eta_small, k * n, T // k, 2)
assert abs(small - large) < 0.01 * best and large < 1.01 * best, (small, large, best)
# here k eta lambda_max = 0.4, inside the stable range of the large step

eta_big = 1.0 / lam_max  # stable for the small batch (eta lambda < 2) ...
small = run(eta_big, n, T, 3)
large = run(k * eta_big, k * n, T // k, 4)  # ... but k eta lambda_max = 8 > 2
assert np.isfinite(small) and small < 1.5 * best, (small, best)
assert large == np.inf
print("ok")
