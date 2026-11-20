# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation cd-k-limit."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402


def g_k(p, data, T, k):
    X = eb.all_binary(T.shape[0].bit_length() - 1)
    emp = np.array([np.mean(np.all(data == v, 1)) for v in X])
    w = emp @ np.linalg.matrix_power(T, k)
    sp, ph = eb.stats(p, data), eb.p_h(p, X)
    g = {"W": sp["W"] - (X * w[:, None]).T @ ph, "b": sp["b"] - w @ X, "c": sp["c"] - w @ ph}
    return np.concatenate([v.ravel() for v in g.values()])


rng = np.random.default_rng(3)
base = {k: rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(4, 3)).items()}
rhos = []
for scale in (0.5, 1.0, 2.0, 3.0):
    p = {k: scale * v for k, v in base.items()}
    T = eb.gibbs_transition(p)
    pm = eb.model_probs(p)
    assert (T > 0).all()
    np.testing.assert_allclose(T.sum(1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(pm @ T, pm, rtol=1e-10, atol=1e-15)
    F = pm[:, None] * T
    np.testing.assert_allclose(F, F.T, rtol=1e-9, atol=1e-15)  # detailed balance
    X = eb.all_binary(4)
    data = X[[0, 1, 5, 6, 9, 15, 15, 12]]
    ex = eb.exact_grad(p, data)
    exact = np.concatenate([ex[k].ravel() for k in ("W", "b", "c")])
    rho = np.sort(np.abs(np.linalg.eigvals(T)))[-2]
    err = np.array([np.abs(g_k(p, data, T, k) - exact).max() for k in range(0, 80)])
    assert err[1] > 1e-6
    assert all(a > b for a, b in zip(err[1:12], err[2:13], strict=False))
    assert err[79] < 1e-12
    rate = (err[12] / err[4]) ** (1 / 8)  # the geometric rate of the bias, at most rho
    assert abs(rate / rho - 1) < 0.05
    rhos.append(rho)
assert all(a < b for a, b in zip(rhos, rhos[1:], strict=False)), rhos  # larger weights mix more slowly
