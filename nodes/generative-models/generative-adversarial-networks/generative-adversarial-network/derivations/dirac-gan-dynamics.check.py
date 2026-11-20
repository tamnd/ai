# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation dirac-gan-dynamics."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402

FP0 = 0.5  # f'(0) for f(t) = -log(1 + e^-t)


def field(u, gamma=0.0):
    th, ps = u
    fp = gan.sigmoid(-ps * th)
    return np.array([-fp * ps, fp * th - gamma * ps])


def jac(fn, u, e=1e-6):
    return np.stack([(fn(u + e * d) - fn(u - e * d)) / (2 * e) for d in np.eye(2)], 1)


zero = np.zeros(2)
ev = np.linalg.eigvals(jac(field, zero))  # statement 1
assert np.allclose(sorted(ev.imag), [-FP0, FP0], atol=1e-8) and np.allclose(ev.real, 0, atol=1e-8)

u = np.array([1.0, 0.3])  # statement 2: RK4 with small steps keeps theta^2 + psi^2
r0 = u @ u
for _ in range(20000):
    k1 = field(u)
    k2 = field(u + 0.0005 * k1)
    k3 = field(u + 0.0005 * k2)
    k4 = field(u + 0.001 * k3)
    u = u + 0.001 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
assert abs(u @ u - r0) < 1e-9, u @ u - r0

for h in (0.05, 0.1, 0.5):  # statement 3, on the reference's update operator

    def step(v, h=h):
        return gan.dirac_gan(v[0], v[1], h, 1)[-1]

    lam = np.linalg.eigvals(jac(step, zero))
    assert np.allclose(np.abs(lam), math.sqrt(1 + h**2 * FP0**2), atol=1e-8)
    it = gan.dirac_gan(1e-4, 0.0, h, 50)
    ratio = np.linalg.norm(it[1:], axis=1) / np.linalg.norm(it[:-1], axis=1)
    assert np.allclose(ratio, math.sqrt(1 + h**2 * FP0**2), rtol=1e-6)

for gamma in (0.1, 0.5, 1.0, 1.0001, 3.0):  # statement 4; gamma_critical = 2 f'(0) = 1
    lam = np.linalg.eigvals(jac(lambda v, g=gamma: field(v, g), zero))
    want = -gamma / 2 + np.array([1, -1]) * np.sqrt(complex(gamma**2 / 4 - FP0**2))
    assert np.allclose(sorted(lam, key=lambda c: (c.real, c.imag)), sorted(want, key=lambda c: (c.real, c.imag)),
                       atol=1e-6)  # fmt: skip
    assert np.all(lam.real < 0)
    assert np.linalg.norm(gan.dirac_gan(1.0, 0.0, 0.1, 3000, gamma)[-1]) < 1e-2
print("dirac-gan-dynamics: ok")
