# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://NEW-critical-batch-size).
# Check of derivations/step-example-tradeoff.md, in two parts.
# (a) The model itself: when each step at batch B gains dL_max(L) / (1 + B_noise(L) / B), the steps to a target are
#     S(B) = S_min + E_min / B with S_min = int dL / dL_max and E_min = int B_noise dL / dL_max, and then
#     (S / S_min - 1)(E / E_min - 1) = 1 with E = B S. We step a loss down with small steps and compare.
# (b) A noisy quadratic trained by SGD at the optimal step of each point: the steps to a target loss, measured at many
#     batch sizes, fall on the hyperbola, and B_crit = E_min / S_min is close to the noise scale averaged over the
#     steps of a full-batch run.
import numpy as np

# (a) dL_max(L) = k L and B_noise(L) = b0 / L: the noise scale grows as the loss falls, as McCandlish et al. expect
k, b0, L_start, L_end = 2e-3, 50.0, 1.0, 0.1
s_min = np.log(L_start / L_end) / k  # int_{L_end}^{L_start} dL / (k L)
e_min = b0 * (1 / L_end - 1 / L_start) / k  # int b0 / L * dL / (k L)


def steps(B):
    L, n = L_start, 0
    while L > L_end:
        L -= k * L / (1 + b0 / (L * B))
        n += 1
    return n


for B in [4, 16, 64, 256, 1024, 4096, 16384]:
    S = steps(B)
    E = B * S
    assert abs(S / (s_min + e_min / B) - 1) < 2e-3, (B, S, s_min + e_min / B)
    prod = (S / s_min - 1) * (E / e_min - 1)
    # far above B_crit, S / S_min - 1 is small and one step of rounding moves it; compare there with a looser bound
    assert abs(prod - 1) < (0.02 if B <= 1024 else 0.1), (B, prod)
b_crit = e_min / s_min
print(f"(a) S_min = {s_min:.1f}, E_min = {e_min:.0f}, B_crit = {b_crit:.2f}")
# at B = B_crit the run takes twice the fewest steps and twice the fewest examples
assert abs(steps(b_crit) / (2 * s_min) - 1) < 2e-3

# (b) SGD on L(theta) = theta.H theta / 2 with per-example gradients H theta + noise of covariance Sigma.
# The Hessian spreads over a factor of 33 and the start puts equal loss in every direction, so the run is long enough
# (tens of full-batch steps) for the per-step model to average out.
d = 30
h = np.geomspace(0.03, 1.0, d)  # diagonal Hessian
sig = 4.0 * h  # diagonal Sigma, the same at every theta
theta0 = 1.0 / np.sqrt(h)
target = 0.01 * 0.5 * np.sum(h * theta0**2)


def run(B, seed, record=False):
    r = np.random.default_rng(seed)
    th, n, noise = theta0.copy(), 0, []
    while 0.5 * np.sum(h * th**2) > target:
        G = h * th
        gHg = np.sum(h * G**2)
        bn = np.sum(h * sig) / gHg  # B_noise = tr(H Sigma) / G.HG at this point
        eps = (G @ G / gHg) / (1 + bn / B)  # the optimal step for this batch size
        th = th - eps * (G + r.normal(size=d) * np.sqrt(sig / B))
        n += 1
        noise.append(bn)
    return (n, np.array(noise)) if record else n


Bs = 2.0 ** np.arange(1, 17)
S = np.array([np.mean([run(B, s) for s in range(6)]) for B in Bs])
# fit S = S_min + E_min / B by least squares on the relative error
A = np.stack([np.ones(len(Bs)), 1.0 / Bs], 1)
(fs, fe), *_ = np.linalg.lstsq(A / S[:, None], np.ones_like(S), rcond=None)
ratio = S / (fs + fe / Bs)
n_full, noise = run(1e15, 0, record=True)
print(f"(b) fit S_min = {fs:.1f}, E_min = {fe:.0f}, B_crit = {fe / fs:.1f}; full-batch steps = {n_full}, "
      f"B_noise averaged over them = {noise.mean():.1f}")
for B, s_b, r_b in zip(Bs, S, ratio):
    print(f"    B = {int(B):6d}  S = {s_b:8.1f}  S / fit = {r_b:.3f}")
assert np.all(np.abs(ratio - 1) < 0.15)
assert abs(fs / n_full - 1) < 0.15
assert 1 / 1.5 < (fe / fs) / noise.mean() < 1.5
print("ok")
