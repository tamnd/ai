# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation two-stage-bound."""

import math

import numpy as np


def ent(v):
    return 0.5 * math.log(2 * math.pi * math.e * v)


def ce(m, v, pm, pv):
    """CE(N(m, v) || N(pm, pv))."""
    return 0.5 * math.log(2 * math.pi * pv) + (v + (m - pm) ** 2) / (2 * pv)


def kl(m, v, pm, pv):
    return 0.5 * (math.log(pv / v) + (v + (m - pm) ** 2) / pv - 1.0)


rng = np.random.default_rng(5)
a, s2 = 1.7, 0.3
# The KL term as negative entropy plus cross entropy, closed form and Monte Carlo.
for x, pm, pv in ((0.4, 0.0, 1.0), (-1.2, 0.5, 2.5)):
    m = a * x
    assert abs(kl(m, s2, pm, pv) - (-ent(s2) + ce(m, s2, pm, pv))) < 1e-12
    z = m + math.sqrt(s2) * rng.standard_normal(400000)
    logq = -0.5 * (z - m) ** 2 / s2 - 0.5 * math.log(2 * math.pi * s2)
    logp = -0.5 * (z - pm) ** 2 / pv - 0.5 * math.log(2 * math.pi * pv)
    assert abs(-logq.mean() - ent(s2)) < 0.01
    assert abs(-logp.mean() - ce(m, s2, pm, pv)) < 0.01
    assert abs((logq - logp).mean() - kl(m, s2, pm, pv)) < 0.01
# The data-averaged cross entropy is that of the aggregate N(0, a^2 + s2), smallest at the aggregate.
xs = rng.standard_normal(200000)
agg = a * a + s2
for pm, pv in ((0.0, 1.0), (0.3, agg), (0.0, agg)):
    avg = np.mean([ce(a * x, s2, pm, pv) for x in xs[:20000]])
    assert abs(avg - ce(0.0, agg, pm, pv)) < 0.03
best = min((ce(0.0, agg, pm, pv), pm, pv) for pm in np.linspace(-1, 1, 41) for pv in np.linspace(0.5, 6, 111))
assert best[1] == 0.0 and abs(best[2] - agg) < 0.03
assert abs(ce(0.0, agg, 0.0, agg) - ent(agg)) < 1e-12
# A mixture prior, by quadrature on a fine grid.
zg = np.linspace(-15, 15, 300001)
dz = zg[1] - zg[0]
q = np.exp(-0.5 * (zg - 0.7) ** 2 / s2) / math.sqrt(2 * math.pi * s2)
pr = 0.3 * np.exp(-0.5 * (zg + 1) ** 2) / math.sqrt(2 * math.pi) + 0.7 * np.exp(-0.5 * (zg - 2) ** 2 / 0.5) / math.sqrt(
    math.pi
)
H = -(q * np.log(q + 1e-300)).sum() * dz
CE = -(q * np.log(pr)).sum() * dz
KL = (q * (np.log(q + 1e-300) - np.log(pr))).sum() * dz
assert abs(H - ent(s2)) < 1e-6 and abs(KL - (-H + CE)) < 1e-9 and KL > 0
