# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation schedules."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

K = 800_000
assert cm.cm_schedule(0, K) == (2, 0.9)
N, mu = cm.cm_schedule(K, K)
assert N == 151 and math.isclose(mu, 0.9 ** (2 / 151)) and round(mu, 5) == 0.99861
ns = [cm.cm_schedule(k, K)[0] for k in range(0, K + 1, 1000)]
assert all(b >= a for a, b in zip(ns, ns[1:], strict=False)) and ns[0] == 2 and ns[-1] == 151

steps = [cm.ict_steps(k, K) for k in range(0, K + 1, 100)]
assert sorted(set(steps)) == [10 * 2**j + 1 for j in range(8)]
assert cm.ict_steps(K // 8 - 1, K) == 11 and cm.ict_steps(K // 8, K) == 21 and cm.ict_steps(K, K) == 1281

# uniform index: density of log sigma is sigma^(1/rho) / (rho (b - a))
rho, smin, smax = cm.RHO, cm.EPS, cm.T_MAX
a, b = smin ** (1 / rho), smax ** (1 / rho)
ls = np.log(cm.karras_steps(100_000))
hist, edges = np.histogram(ls, bins=40, density=True)
mid = 0.5 * (edges[1:] + edges[:-1])
assert np.allclose(hist, np.exp(mid / rho) / (rho * (b - a)), rtol=0.02)
assert math.isclose(np.exp(np.median(ls)), ((a + b) / 2) ** rho, rel_tol=1e-3)
assert 2.5 < ((a + b) / 2) ** rho < 2.52

# lognormal: interval masses of the normal of log sigma truncated to [sigma_min, sigma_max]
pm, ps = -1.1, 2.0
for N in (11, 161, 1281):
    ts = cm.karras_steps(N)
    p = cm.lognormal_probs(ts)
    assert p.shape == (N - 1,) and np.all(p > 0) and math.isclose(p.sum(), 1.0)

    def cdf(x):
        return 0.5 * (1 + math.erf((x - pm) / (math.sqrt(2) * ps)))

    tot = cdf(math.log(smax)) - cdf(math.log(smin))
    want = np.array([(cdf(math.log(hi)) - cdf(math.log(lo))) / tot for lo, hi in zip(ts[:-1], ts[1:], strict=True)])
    assert np.allclose(p, want, rtol=1e-9)
    # half of the mass lies below the median exp(-1.1) of the untruncated lognormal, up to the truncation
    below = p[ts[1:] <= math.exp(pm)].sum()
    assert abs(below - (0.5 - cdf(math.log(smin))) / tot) < (0.06 if N == 11 else 0.01)
    w = cm.ict_weight(ts[:-1], ts[1:])
    assert np.all(np.diff(w) < 0)
print("schedules: ok")
