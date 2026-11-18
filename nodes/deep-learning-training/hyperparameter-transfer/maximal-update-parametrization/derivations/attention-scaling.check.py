# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://4VJECR).
# Check of derivations/attention-scaling.md: for independent q and k with iid zero-mean coordinates, q.k has mean 0
# and standard deviation sqrt(d) s^2, so q.k / sqrt(d) is Theta(1) and q.k / d goes to 0; for correlated q and k
# (k = rho q + sqrt(1 - rho^2) z), q.k / d goes to rho s^2 and q.k / sqrt(d) grows like sqrt(d); the muP scale
# alpha sqrt(d0) / d equals the SP scale alpha / sqrt(d) at d = d0.
import numpy as np

rng = np.random.default_rng(1)
s, rho, trials = 1.5, 0.6, 400
dims = np.array([16, 64, 256, 1024, 4096])
rows = []
for d in dims:
    q = s * rng.standard_normal((trials, d))
    z = s * rng.standard_normal((trials, d))
    k_ind = s * rng.standard_normal((trials, d))
    k_al = rho * q + np.sqrt(1 - rho**2) * z
    dot_ind = np.sum(q * k_ind, axis=1)
    dot_al = np.sum(q * k_al, axis=1)
    rows.append(
        [
            np.std(dot_ind / np.sqrt(d)),
            np.mean(np.abs(dot_ind / d)),
            np.mean(dot_al / d),
            np.mean(dot_al / np.sqrt(d)),
        ]
    )
rows = np.array(rows)

# independent: q.k / sqrt(d) has standard deviation s^2 at every d; q.k / d shrinks like 1 / sqrt(d)
assert np.allclose(rows[:, 0], s**2, rtol=0.12)
assert np.polyfit(np.log(dims), np.log(rows[:, 1]), 1)[0] < -0.4
# correlated: q.k / d tends to rho s^2; q.k / sqrt(d) grows like sqrt(d)
assert np.allclose(rows[1:, 2], rho * s**2, rtol=0.05)
assert abs(np.polyfit(np.log(dims), np.log(rows[:, 3]), 1)[0] - 0.5) < 0.05
# at the base head dimension the two scales agree
for d0 in (8, 64, 128):
    alpha = 0.7
    assert np.isclose(alpha * np.sqrt(d0) / d0, alpha / np.sqrt(d0))
    assert np.isclose((alpha * np.sqrt(d0) / (4 * d0)) / (alpha / np.sqrt(4 * d0)), 0.5)
print("ok")
