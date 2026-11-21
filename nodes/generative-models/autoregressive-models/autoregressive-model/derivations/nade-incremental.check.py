# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Numeric check of the derivation nade-incremental."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoregressive as ar  # noqa: E402

rng = np.random.default_rng(2)
for D, H in ((7, 5), (5, 12), (2, 1)):
    q = {k: rng.normal(0.0, 1.0, s) for k, s in ar.nade_shapes(D, H).items()}
    X = ar.all_binary(D)
    a = ar.nade_logits(q, X)
    assert np.allclose(a, ar.nade_logits_naive(q, X), rtol=1e-12, atol=1e-12)
    assert abs(np.exp(np.sum(X * a - ar.softplus(a), 1)).sum() - 1.0) < 1e-12
    # the first conditional sees no input: its logit is the same for every x
    assert np.allclose(a[:, 0], a[0, 0])
    # the recursion z_{i+1} = z_i + x_i W_i
    x = X[-1]
    z = q["c"].copy()
    for i in range(D):
        assert np.allclose(z, q["c"] + x[:i] @ q["W"][:i])
        z = z + x[i] * q["W"][i]
# multiply-adds of the pre-activations: incremental H per variable, naive H (i - 1) for conditional i
D, H = 784, 500
incremental, naive = D * H, sum(H * (i - 1) for i in range(1, D + 1))
assert naive == H * D * (D - 1) // 2 and naive // incremental == (D - 1) // 2
print("nade-incremental: ok")
