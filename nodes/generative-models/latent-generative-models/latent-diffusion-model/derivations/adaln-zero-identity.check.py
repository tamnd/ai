# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation adaln-zero-identity: a small adaLN-Zero block in NumPy."""

import numpy as np

rng = np.random.default_rng(6)
D, C, Hf = 5, 3, 8
P = {
    "Wg": rng.standard_normal((C, D)),
    "Wb": rng.standard_normal((C, D)),
    "Wa": np.zeros((C, D)),
    "bg": rng.standard_normal(D),
    "bb": rng.standard_normal(D),
    "ba": np.zeros(D),
    "F1": rng.standard_normal((D, Hf)),
    "F2": rng.standard_normal((Hf, D)),
}
x, c, target = rng.standard_normal((4, D)), rng.standard_normal((4, C)), rng.standard_normal((4, D))


def block(P, x):
    g, b, a = c @ P["Wg"] + P["bg"], c @ P["Wb"] + P["bb"], c @ P["Wa"] + P["ba"]
    ln = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-6)
    F = np.tanh(((1 + g) * ln + b) @ P["F1"]) @ P["F2"]
    return x + a * F, F


def loss(P, x):
    return 0.5 * ((block(P, x)[0] - target) ** 2).sum()


y, F = block(P, x)
assert np.array_equal(y, x)
g = y - target
h = 1e-6
for k, v in P.items():
    fd = np.zeros_like(v)
    for i in np.ndindex(v.shape):
        d = np.zeros_like(v)
        d[i] = h
        fd[i] = (loss(dict(P, **{k: v + d}), x) - loss(dict(P, **{k: v - d}), x)) / (2 * h)
    if k == "Wa":
        assert np.allclose(fd, c.T @ (g * F), atol=1e-6) and np.abs(fd).max() > 1e-2
    elif k == "ba":
        assert np.allclose(fd, (g * F).sum(0), atol=1e-6)
    else:
        assert np.abs(fd).max() < 1e-8, k
fdx = np.zeros_like(x)
for i in np.ndindex(x.shape):
    d = np.zeros_like(x)
    d[i] = h
    fdx[i] = (loss(P, x + d) - loss(P, x - d)) / (2 * h)
assert np.allclose(fdx, g, atol=1e-6)
