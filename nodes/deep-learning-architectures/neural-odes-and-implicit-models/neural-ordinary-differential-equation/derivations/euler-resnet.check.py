# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""Numeric check of the derivation euler-resnet."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import neural_ode as node  # noqa: E402


def expm(A, terms=40):
    k = max(0, int(np.ceil(np.log2(max(np.abs(A).sum(1).max(), 1e-300)))) + 1)
    M, E, term = A / 2.0**k, np.eye(len(A)), np.eye(len(A))
    for i in range(1, terms):
        term = term @ M / i
        E = E + term
    for _ in range(k):
        E = E @ E
    return E


rng = np.random.default_rng(0)

# Statement 1: an Euler step is a residual block with the branch dt f.
cfg = node.config(3, 5, n_steps=1, T=0.4, method="euler")
params, h = node.init(rng, cfg), rng.normal(size=(4, 3))
np.testing.assert_allclose(node.forward(params, h, cfg), h + 0.4 * node.field(params, h, 0.0), rtol=0, atol=1e-15)

# Statement 2: on h' = A h the RK4 step is the degree-4 Taylor polynomial of exp(dt A), and the orders are 1 and 4.
A = rng.normal(size=(3, 3))
dt = 0.1
M = dt * A
taylor = sum(np.linalg.matrix_power(M, j) / math.factorial(j) for j in range(5))
h0 = rng.normal(size=(2, 3))
np.testing.assert_allclose(node.rk4(lambda v, t: v @ A.T, h0, 0.0, dt, 1), h0 @ taylor.T, rtol=1e-14, atol=1e-15)
exact = h0 @ expm(A).T
for method, order in (("euler", 1), ("rk4", 4)):
    errs = [np.abs(node.SOLVERS[method](lambda v, t: v @ A.T, h0, 0.0, 1.0, n) - exact).max() for n in (64, 128, 256)]
    for e1, e2 in zip(errs, errs[1:]):
        assert abs(math.log2(e1 / e2) - order) < 0.15, (method, errs)
print("euler-resnet: ok")
