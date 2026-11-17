# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://17Q1PD).
# Check of derivations/self-stabilization-ode.md: along X' = X Y, Y' = alpha - beta X^2 / 2 the quantity
# g = h(beta X^2 / (2 alpha)) + Y^2 / alpha with h(z) = z - log z - 1 is conserved; g >= 0 with equality only at the
# fixed point (sqrt(2 alpha / beta), 0); and Y changes sign along every other orbit.
import numpy as np


def g(X, Y, a, b):
    z = b * X * X / (2 * a)
    return z - np.log(z) - 1 + Y * Y / a


for a, b, X0, Y0 in ((1.0, 1.0, 0.1, 0.0), (1.3, 0.7, 2.0, 0.5), (0.2, 3.0, 0.05, -0.3)):

    def f(s):
        return np.array([s[0] * s[1], a - b * s[0] ** 2 / 2])

    s = np.array([X0, Y0])
    g0 = g(X0, Y0, a, b)
    ys, h = [], 1e-3
    for _ in range(30000):
        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(s[1])
        assert s[0] > 0
    assert abs(g(s[0], s[1], a, b) - g0) < 1e-7 * max(1, g0)
    assert min(ys) < 0 < max(ys)
    d = np.sqrt(2 * a / b)
    assert np.allclose(f(np.array([d, 0.0])), 0) and abs(g(d, 0.0, a, b)) < 1e-14
    X, Y = np.meshgrid(np.linspace(0.05, 3 * d, 50), np.linspace(-2, 2, 41))
    G = g(X, Y, a, b)
    assert np.all(G >= -1e-14)
    assert np.all(G[(np.abs(X - d) > 1e-6) | (np.abs(Y) > 1e-6)] > 0)
print("self-stabilization-ode: ok")
