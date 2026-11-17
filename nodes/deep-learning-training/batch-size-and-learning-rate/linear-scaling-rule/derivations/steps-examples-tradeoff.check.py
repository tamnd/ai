# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://NEW-linear-scaling-rule).
# Check of derivations/steps-examples-tradeoff.md: if each step at batch B lowers the loss by dL_max / (1 + B_noise / B),
# the steps to a target are S = S_min + E_min / B with S_min the steps at infinite batch and E_min = the integral of
# B_noise over those steps, so E = B S, B_crit = E_min / S_min is the step-average of B_noise, and
# (S / S_min - 1)(E / E_min - 1) = 1 exactly, for any profile of dL_max and B_noise along the way; at B = B_crit both
# factors are 1. Then noisy SGD with the optimal step on an ill-conditioned quadratic follows the curve within 30% over more than two decades of steps.
import numpy as np

# 1. the continuous model, with a B_noise that grows as the loss falls
L0, L1 = 10.0, 0.01
Ls = np.geomspace(L0, L1, 200_001)
dL = -np.diff(Ls)
Lm = 0.5 * (Ls[1:] + Ls[:-1])
dl_max = 0.05 * Lm  # the best progress per step falls with the loss
b_noise = 30.0 / np.sqrt(Lm)  # and the noise scale grows, as McCandlish et al. observe


def steps(B):
    return float(np.sum(dL * (1 + b_noise / B) / dl_max))


S_min = float(np.sum(dL / dl_max))
E_min = float(np.sum(dL * b_noise / dl_max))
B_crit = E_min / S_min
w = dL / dl_max  # steps spent at each loss level by the infinite-batch run
assert np.isclose(B_crit, np.sum(w * b_noise) / np.sum(w))
assert b_noise.min() < B_crit < b_noise.max()
for B in [1.0, 10.0, B_crit, 300.0, 1e4]:
    S = steps(B)
    E = B * S
    assert np.isclose(S, S_min + E_min / B)
    assert np.isclose((S / S_min - 1) * (E / E_min - 1), 1.0)
S = steps(B_crit)
assert np.isclose(S, 2 * S_min) and np.isclose(B_crit * S, 2 * E_min)
# small batches: E -> E_min (each example counts); large: S -> S_min (each step counts)
assert steps(1e-3) * 1e-3 / E_min < 1.001 and steps(1e7) / S_min < 1.001

# 2. noisy SGD on a quadratic with the optimal step of derivations/optimal-step-noise-scale.md at every step
rng = np.random.default_rng(2)
d = 20
lam = np.geomspace(0.01, 1.0, d)  # H = diag(lam)
sig = np.full(d, 0.3)  # per-example gradient variance, Sigma = diag(sig^2)
theta0 = np.ones(d)
target = 1e-3 * 0.5 * np.sum(lam * theta0**2)


def run(B, seed):
    r = np.random.default_rng(seed)
    th = theta0.copy()
    s, bn = 0, []
    while 0.5 * np.sum(lam * th**2) > target:
        G = lam * th
        GHG = np.sum(lam * G**2)
        bn.append(np.sum(lam * sig**2) / GHG)
        eps = (G @ G / GHG) / (1 + bn[-1] / B)
        th = th - eps * (G + sig * r.normal(size=d) / np.sqrt(B))
        s += 1
    return s, np.mean(bn)


Bs = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384])
S = np.array([np.mean([run(B, 100 * i + j)[0] for j in range(6)]) for i, B in enumerate(Bs)])
# fit log S = log S_min + log(1 + B_crit / B) in log space, as McCandlish et al. fit their Pareto fronts
grid = np.geomspace(1.0, 1e5, 5001)
best = None
for bc in grid:
    off = np.log(S) - np.log1p(bc / Bs)
    res = np.sum((off - off.mean()) ** 2)
    if best is None or res < best[0]:
        best = (res, bc, np.exp(off.mean()))
_, B_crit, S_min = best
E_min = B_crit * S_min
fit = S_min * (1 + B_crit / Bs)
assert np.all(np.abs(fit / S - 1) < 0.30), fit / S
assert S[0] / S[-1] > 300  # the runs span more than two decades of steps
_, bn_avg = run(1e9, 7)  # the noise scale averaged over the steps of a near-noiseless run
assert 0.5 < B_crit / bn_avg < 2.0, (B_crit, bn_avg)
print("ok", round(S_min, 1), round(E_min, 1), round(B_crit, 1), round(bn_avg, 1))
