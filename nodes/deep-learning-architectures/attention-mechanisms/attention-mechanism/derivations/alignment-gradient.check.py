# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Numeric check of the derivation alignment-gradient: the gradient of the loss at the energies."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import attention  # noqa: E402

rng = np.random.default_rng(0)


def context(e, H, mask):
    a = attention.softmax(e, mask)
    return a, a @ H


for _ in range(20):
    S, k = (int(v) for v in rng.integers([2, 1], [9, 6]))
    H, e, G = rng.normal(size=(S, k)), rng.normal(size=S), rng.normal(size=k)  # the loss is G . c
    mask = rng.random(S) < 0.7
    mask[int(rng.integers(S))] = True
    a, c = context(e, H, mask)
    gamma = H @ G
    formula = a * (gamma - a @ gamma)
    num = np.zeros(S)
    for j in range(S):
        d = np.zeros(S)
        d[j] = 1e-6
        num[j] = (G @ context(e + d, H, mask)[1] - G @ context(e - d, H, mask)[1]) / 2e-6
    assert np.allclose(num, formula, atol=1e-8)
    assert abs(formula.sum()) < 1e-12 and np.all(formula[~mask] == 0.0)
    same = np.tile(rng.normal(size=k), (S, 1))
    a, _ = context(e, same, mask)
    g2 = same @ G
    assert np.allclose(a * (g2 - a @ g2), 0.0, atol=1e-14)

# inside RNNsearch, at Bahdanau et al.'s initialization v_a = 0: the energies are all 0, so the weights are uniform,
# but their gradients are not, and they reach v_a through tanh(W_a s + U_a h) while W_a and U_a, whose gradients
# carry a factor v_a, get none until v_a has moved
cfg = attention.config()
p = attention.init(rng, cfg, 0.5)
p["v_a"][:] = 0.0
src, tgt = rng.integers(1, 7, (2, 5)), rng.integers(0, 6, (2, 4))
logits, alpha, cache = attention.forward(p, cfg, src, tgt)
assert np.allclose(alpha, 0.2)
g = attention.backward(p, cfg, cache)
assert np.all(g["W_a"] == 0.0) and np.all(g["U_a"] == 0.0) and np.abs(g["v_a"]).sum() > 0
print("alignment-gradient: ok")
