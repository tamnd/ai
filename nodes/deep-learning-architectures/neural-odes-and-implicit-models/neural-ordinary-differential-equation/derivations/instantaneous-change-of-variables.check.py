# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""Numeric check of the derivation instantaneous-change-of-variables."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import neural_ode as node  # noqa: E402

rng = np.random.default_rng(2)
eps = 1e-6

# Statement 2: the trace by the H diagonal entries of W1 W2 equals the trace of the Jacobian.
cfg = node.config(4, 6)
params = node.init(rng, cfg)
params["b1"] = rng.normal(0, 0.3, 6)
h = rng.normal(size=(5, 4))
for b in range(5):
    J = np.zeros((4, 4))
    for j in range(4):
        hp, hm = h[b:b + 1].copy(), h[b:b + 1].copy()
        hp[0, j] += eps
        hm[0, j] -= eps
        J[:, j] = (node.field(params, hp, 0.7) - node.field(params, hm, 0.7))[0] / (2 * eps)
    assert abs(node.trace_jacobian(params, h[b:b + 1], 0.7)[0] - np.trace(J)) < 1e-8

# Statement 1: the integral of -tr equals -log |det| of the Jacobian of the flow map.
cfg = node.config(3, 5, n_steps=100)
params = node.init(rng, cfg, 1.5)
x = rng.normal(size=(4, 3))
y, dlogp = node.log_det_flow(params, x, cfg)
for b in range(4):
    J = np.zeros((3, 3))
    for j in range(3):
        xp, xm = x[b:b + 1].copy(), x[b:b + 1].copy()
        xp[0, j] += eps
        xm[0, j] -= eps
        J[:, j] = (node.forward(params, xp, cfg) - node.forward(params, xm, cfg))[0] / (2 * eps)
    assert abs(dlogp[b] + np.log(abs(np.linalg.det(J)))) < 1e-7

# For a linear field the flow map is exp(T A), and log det exp(T A) = T tr A.
A = rng.normal(size=(3, 3))
Phi = node.rk4(lambda v, t: v @ A.T, np.eye(3), 0.0, 1.0, 400).T
assert abs(np.log(np.linalg.det(Phi)) - np.trace(A)) < 1e-10
print("instantaneous-change-of-variables: ok")
