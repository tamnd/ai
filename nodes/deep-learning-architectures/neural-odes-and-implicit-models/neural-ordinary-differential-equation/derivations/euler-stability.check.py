# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""Numeric check of the derivation euler-stability."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import neural_ode as node  # noqa: E402

rng = np.random.default_rng(4)
for _ in range(10):
    M = rng.normal(size=(4, 4))
    K = M - M.T  # antisymmetric
    omega = np.abs(np.linalg.eigvals(K).imag).max()
    h0 = rng.normal(size=(1, 4))
    dt, n = 0.1, 50

    # Statement 2: the exact flow keeps the norm, Euler steps grow by at most sqrt(1 + dt^2 omega^2) per step,
    # and by exactly that factor along the fastest eigenvector.
    exact = node.rk4(lambda h, t: h @ K.T, h0, 0.0, dt * n, 2000)
    assert abs(np.linalg.norm(exact) - np.linalg.norm(h0)) < 1e-9
    eul = node.euler(lambda h, t: h @ K.T, h0, 0.0, dt * n, n)
    factor = np.sqrt(1 + dt**2 * omega**2)
    assert np.linalg.norm(eul) > np.linalg.norm(h0)
    assert np.linalg.norm(eul) <= factor**n * np.linalg.norm(h0) * (1 + 1e-9)
    step = np.eye(4) + dt * K
    assert abs(np.abs(np.linalg.eigvals(step)).max() - factor) < 1e-12

    # Statement 3: with K - gamma I the steps contract if and only if dt (gamma^2 + omega^2) <= 2 gamma.
    for gamma in (0.05, 0.5, 2.0):
        ok = dt * (gamma**2 + omega**2) <= 2 * gamma
        radius = np.abs(np.linalg.eigvals(np.eye(4) + dt * (K - gamma * np.eye(4)))).max()
        assert (radius <= 1 + 1e-12) == ok, (gamma, omega, radius)
        eul = node.euler(lambda h, t: h @ (K - gamma * np.eye(4)).T, h0, 0.0, dt * 400, 400)
        assert (np.linalg.norm(eul) <= np.linalg.norm(h0)) == ok or abs(radius - 1) < 1e-3
print("euler-stability: ok")
