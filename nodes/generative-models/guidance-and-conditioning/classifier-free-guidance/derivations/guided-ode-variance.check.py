# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation guided-ode-variance."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

m1, v1, m0, v0 = 1.0, 0.25, 0.0, 1.25
abar = cg.alpha_bars(1000)
sT = float(cg.edm_sigma(abar[-1]))
assert abs(sT - 157.4) < 0.05


def finite(s, sT=sT):
    return (1 + sT**2) * (v1 / (v1 + sT**2)) ** s * (v0 / (v0 + sT**2)) ** (1 - s)


def rk4_flow(s, x, sT=sT, n=40000, lo=1e-7):
    """Integrate dx/dsigma = eps(x, sigma) of the guided pair from sT down to lo, as an RK4 in u = log sigma."""

    def f(u, x):
        sg = np.exp(u)
        e = s * sg * (x - m1) / (v1 + sg**2) + (1 - s) * sg * (x - m0) / (v0 + sg**2)
        return sg * e

    us = np.linspace(np.log(sT), np.log(lo), n + 1)
    for u, un in zip(us[:-1], us[1:], strict=True):
        h = un - u
        k1 = f(u, x)
        k2 = f(u + h / 2, x + h / 2 * k1)
        k3 = f(u + h / 2, x + h / 2 * k2)
        k4 = f(un, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# DDIM in x = z / sqrt(abar), sigma = sqrt((1 - abar) / abar) is Euler: x' = x + (sigma' - sigma) eps
z, e = np.array([[0.3], [-1.2]]), np.array([[0.7], [0.1]])
a, a2 = abar[500], abar[400]
z2 = cg.ddim_sample(lambda zz, t: e, z, abar, [500, 400])
x_eul = z / np.sqrt(a) + (cg.edm_sigma(a2) - cg.edm_sigma(a)) * e
x0 = (z - np.sqrt(1 - a) * e) / np.sqrt(a)
assert np.allclose(np.sqrt(a2) * x0 + np.sqrt(1 - a2) * e, np.sqrt(a2) * x_eul)
assert z2.shape == z.shape

means = {}
for s in (0.0, 0.5, 1.0, 2.0, 3.0):

    def eps(zz, t, s=s):
        a = np.full(len(zz), abar[t])
        return cg.guide(cg.gaussian_eps(zz, a, m1, v1), cg.gaussian_eps(zz, a, m0, v0), s)

    x = cg.ddim_sample(eps, np.array([[0.0], [1.0]]), abar, list(range(999, -1, -1)))
    var_ddim = float((x[1, 0] - x[0, 0]) ** 2)
    xr = rk4_flow(s, np.array([0.0, 1.0]) * np.sqrt(1 + sT**2))
    var_rk = float((xr[1] - xr[0]) ** 2)
    assert abs(var_rk - finite(s)) < 1e-6 * finite(s)
    assert abs(var_ddim / finite(s) - 1) < 0.02
    assert abs(finite(s) / cg.guided_ode_variance(v1, v0, s) - 1) < 1e-3
    means[s] = (float(xr[0]), float(x[0, 0]))
    t = cg.tilted_gaussian(m1, v1, m0, v0, s)
    assert t is not None
    if s > 1:
        assert cg.guided_ode_variance(v1, v0, s) < t[1]
    elif 0 < s < 1:
        assert cg.guided_ode_variance(v1, v0, s) > t[1]
    else:
        assert np.isclose(cg.guided_ode_variance(v1, v0, s), t[1])
assert np.isclose(cg.guided_ode_variance(v1, v0, 2.0), 0.05) and np.isclose(cg.guided_ode_variance(v1, v0, 3.0), 0.01)
# s = 0 and s = 1 recover the two models; the start from mean 0 leaves an offset of sqrt(v1 / (v1 + sT^2)) m1
assert abs(means[1.0][0] - m1) < 0.005 and abs(means[0.0][0] - m0) < 1e-12
assert round(means[2.0][0], 2) == 1.32 and round(means[2.0][1], 2) == 1.32
assert round(means[3.0][0], 2) == 1.40 and round(means[3.0][1], 2) == 1.40
# the tilted means for comparison: 1.11 and 1.15
assert round(cg.tilted_gaussian(m1, v1, m0, v0, 2.0)[0], 2) == 1.11
# concavity of log: for s outside [0, 1] the geometric extrapolation of precisions exceeds the linear one
rng = np.random.default_rng(5)
u1, u0 = rng.uniform(0.1, 10, (2, 1000))
for s in (1.5, 3.0, 10.0, -0.5):
    lin = s * u1 + (1 - s) * u0
    geo = u1**s * u0 ** (1 - s)
    ok = lin > 0
    assert np.all(geo[ok] >= lin[ok] * (1 - 1e-12))
