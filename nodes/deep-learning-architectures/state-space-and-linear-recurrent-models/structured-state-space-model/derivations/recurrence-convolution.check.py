# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Numeric check of the derivation recurrence-convolution: recurrence = convolution, causality, impulse, Vandermonde, conjugation."""

import numpy as np

rng = np.random.default_rng(20220601)


def expm(m):
    w, v = np.linalg.eig(m)
    return v @ np.diag(np.exp(w)) @ np.linalg.inv(v)


def disc(A, B, dt, rule):
    n = len(A)
    if rule == "zoh":
        Ab = expm(dt * A)
        return Ab, np.linalg.solve(A, (Ab - np.eye(n)) @ B)
    lhs = np.eye(n) - dt / 2 * A
    return np.linalg.solve(lhs, np.eye(n) + dt / 2 * A), np.linalg.solve(lhs, dt * B)


def recur(Ab, Bb, C, D, u):
    x, ys = np.zeros(len(Ab), dtype=Ab.dtype), []
    for uk in u:
        x = Ab @ x + Bb * uk
        ys.append(C @ x + D * uk)
    return np.array(ys)


def kern(Ab, Bb, C, L):
    return np.array([C @ np.linalg.matrix_power(Ab, i) @ Bb for i in range(L)])


n, L = 5, 64
m = rng.normal(size=(n, n))
A = -(m @ m.T) / n - 0.2 * np.eye(n) + (m - m.T)
B, C, D = rng.normal(size=n), rng.normal(size=n), 0.7
u = rng.normal(size=L)
for rule in ("zoh", "bilinear"):
    Ab, Bb = disc(A, B, 0.1, rule)
    K = kern(Ab, Bb, C, L)
    y = recur(Ab, Bb, C, D, u)
    direct = np.array([sum(K[k - j] * u[j] for j in range(k + 1)) for k in range(L)]) + D * u
    fft = np.fft.irfft(np.fft.rfft(u, 2 * L) * np.fft.rfft(K.real, 2 * L), 2 * L)[:L] + D * u
    assert np.allclose(y, direct) and np.allclose(y.real, fft)  # statement 1
    v = u.copy(); v[40:] = rng.normal(size=L - 40)  # noqa: E702
    assert np.allclose(recur(Ab, Bb, C, D, v)[:40], y[:40])  # statement 2
    e = np.zeros(L); e[0] = 1  # noqa: E702
    assert np.allclose(recur(Ab, Bb, C, D, e), K + D * e)  # statement 3
    V = rng.normal(size=(n, n)) + 2 * np.eye(n)
    Vi = np.linalg.inv(V)
    Ab2, Bb2 = disc(Vi @ A @ V, Vi @ B, 0.1, rule)
    assert np.allclose(kern(Ab2, Bb2, C @ V, L), K)  # statement 5
    # Statement 4 through the eigendecomposition: K_l = sum_n (C W)_n (W^-1 Bb)_n abar_n^l.
    w, W = np.linalg.eig(Ab)
    Cn, Bn = C @ W, np.linalg.solve(W, Bb)
    vander = w[:, None] ** np.arange(L)
    assert np.allclose((Cn * Bn) @ vander, K)
    # The real system has conjugate pairs: keep one of each pair and take twice the real part.
    keep = w.imag > 1e-12
    real = np.abs(w.imag) <= 1e-12
    half = 2 * ((Cn * Bn)[keep] @ vander[keep]).real + ((Cn * Bn)[real] @ vander[real]).real
    assert np.allclose(half, K)
print(f"recurrence-convolution: recurrence, direct and FFT convolution agree (N={n}, L={L}); causality, impulse, Vandermonde and conjugation hold for ZOH and bilinear")
