# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Numeric check of the derivation discretization: ZOH is exact for held inputs, the closed forms, stability, the z^3/12 gap."""

import math

import numpy as np

rng = np.random.default_rng(20211101)


def expm(m):
    w, v = np.linalg.eig(m)
    return (v @ np.diag(np.exp(w)) @ np.linalg.inv(v)).real


n, dt, steps = 4, 0.2, 25
m = rng.normal(size=(n, n))
A = -(m @ m.T) / n - 0.3 * np.eye(n) + (m - m.T)
B = rng.normal(size=n)
u = rng.normal(size=steps)

# Statement 1: ZOH equals the exact solution of the ODE when u is held over each step (RK4 with 400 substeps).
Ab = expm(dt * A)
Bb = np.linalg.solve(A, (Ab - np.eye(n)) @ B)
x_zoh, x_ode = np.zeros(n), np.zeros(n)
h = dt / 400
for k in range(steps):
    x_zoh = Ab @ x_zoh + Bb * u[k]
    for _ in range(400):
        f = lambda x: A @ x + B * u[k]  # noqa: E731
        k1 = f(x_ode); k2 = f(x_ode + h / 2 * k1); k3 = f(x_ode + h / 2 * k2); k4 = f(x_ode + h * k3)  # noqa: E702
        x_ode = x_ode + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(x_zoh, x_ode, atol=1e-10)

# The closed form of Bb against the top-right block of exp(dt [[A, B], [0, 0]]) (Van Loan).
big = np.zeros((n + 1, n + 1)); big[:n, :n] = A; big[:n, n] = B  # noqa: E702
E = expm(dt * big)
assert np.allclose(E[:n, :n], Ab) and np.allclose(E[:n, n], Bb)

# Scalar forms.
a, b = -0.7, 1.3
assert math.isclose(math.exp(dt * a), expm(np.array([[dt * a]]))[0, 0])
assert math.isclose((math.exp(dt * a) - 1) / a * b, sum(math.exp(a * (i + 0.5) * dt / 10**5) for i in range(10**5)) * dt / 10**5 * b, rel_tol=1e-9)

# Statement 2: the bilinear update satisfies the trapezoid equation.
Abl = np.linalg.solve(np.eye(n) - dt / 2 * A, np.eye(n) + dt / 2 * A)
Bbl = np.linalg.solve(np.eye(n) - dt / 2 * A, dt * B)
x0 = rng.normal(size=n)
x1 = Abl @ x0 + Bbl * 0.8
assert np.allclose(x1 - x0, dt / 2 * A @ (x0 + x1) + dt * B * 0.8)

# Statement 3: the image of stable, marginal and unstable eigenvalues, for many step sizes.
lam = rng.normal(size=2000) * 5 + 1j * rng.normal(size=2000) * 50
for d in (1e-3, 0.1, 1.0, 10.0, 1e4):
    z = d * lam / 2
    mu = (1 + z) / (1 - z)
    assert np.all((np.abs(mu) < 1) == (lam.real < 0))
    assert np.allclose(np.abs((1 + 1j * d * lam.imag / 2) / (1 - 1j * d * lam.imag / 2)), 1.0)
    with np.errstate(over="ignore"):
        assert np.all((np.abs(np.exp(d * lam)) < 1) == (lam.real < 0))
for _ in range(50):
    m = rng.normal(size=(5, 5))
    S = -(m @ m.T) - 1e-3 * np.eye(5) + 4 * (m - m.T)
    for d in (1e-3, 1.0, 1e4):
        assert np.abs(np.linalg.eigvals(np.linalg.solve(np.eye(5) - d / 2 * S, np.eye(5) + d / 2 * S))).max() < 1

# Statement 4: (1 + z/2)/(1 - z/2) - e^z is z^3/12 to leading order.
a = -1.0 + 3j
for d in (0.02, 0.01, 0.005):
    z = d * a
    gap = (1 + z / 2) / (1 - z / 2) - np.exp(z)
    assert abs(gap / (z**3 / 12) - 1) < 0.1
print("discretization: ZOH matches RK4 under held input over 25 steps; Van Loan block agrees; bilinear stability holds for 5 step sizes; gap ~ z^3/12")
