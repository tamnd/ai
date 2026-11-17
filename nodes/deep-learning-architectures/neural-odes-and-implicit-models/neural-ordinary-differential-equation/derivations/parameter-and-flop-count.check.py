# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""Numeric check of the derivation parameter-and-flop-count."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import neural_ode as node  # noqa: E402


class Counted(np.ndarray):
    """An array that adds the multiply-adds of each matrix product it takes part in to a counter."""

    macs = 0

    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return (a @ b).view(Counted)

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        Counted.macs += a.shape[0] * a.shape[1] * b.shape[1]
        return (a @ b).view(Counted)


rng = np.random.default_rng(5)
for d, D, H, N, B, method in [(2, 3, 4, 3, 5, "rk4"), (3, 5, 7, 2, 4, "euler"), (1, 1, 1, 1, 1, "rk4")]:
    S = {"euler": 1, "rk4": 4}[method]
    cfg = node.config(d, H, d_state=D, n_steps=N, method=method)
    params = node.init(rng, cfg)
    # Statement 1.
    assert sum(v.size for v in params.values()) == node.param_count(cfg) == 2 * D * H + 2 * H + D
    counted = {k: v.view(Counted) for k, v in params.items()}
    # Statement 2.
    Counted.macs = 0
    y = node.forward(counted, rng.normal(size=(B, d)), cfg)
    fwd = 2 * Counted.macs
    assert fwd == node.flops(cfg, B) == 4 * S * N * B * D * H
    # Statement 3.
    Counted.macs = 0
    node.adjoint_backward(counted, np.asarray(y).view(Counted), rng.normal(size=(B, D)).view(Counted), cfg)
    assert 2 * Counted.macs == 3 * fwd

# Statement 4, and the counts of shapes.json: the state of the adjoint solve and the stages kept by backprop.
shapes = json.loads((pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "shapes.json").read_text())
for c in shapes["configs"]:
    k, m = c["config"], c["model"]
    cfg = node.config(k["d"], k["d_hidden"], d_state=k["d_state"], n_steps=k["n_steps"])
    assert m["params"] == node.param_count(cfg) and m["flops_forward"] == node.flops(cfg, k["B"])
    assert m["flops_backward"] == 3 * m["flops_forward"]
    assert m["memory_adjoint"] == 2 * k["B"] * k["d_state"] + node.param_count(cfg)
    assert m["memory_backprop"] == 4 * k["n_steps"] * k["B"] * k["d_state"]
print("parameter-and-flop-count: ok")
