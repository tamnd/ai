# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""Numeric check of the derivation adjoint."""

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


rng = np.random.default_rng(1)
cfg = node.config(2, 4, d_state=3, n_steps=200)
params = node.init(rng, cfg)
params["b1"], params["b2"] = rng.normal(0, 0.3, 4), rng.normal(0, 0.3, 3)
x, c = rng.normal(size=(3, 2)), rng.normal(size=(3, 3))


def L(p, xx):
    return float((node.forward(p, xx, cfg) * c).sum())


# Statements 1 and 2: the adjoint gradient equals the gradient of the solution, up to the error of the solver.
dx, g = node.adjoint_backward(params, node.forward(params, x, cfg), c, cfg)
eps = 1e-6
for b in range(3):
    for j in range(2):
        xp, xm = x.copy(), x.copy()
        xp[b, j] += eps
        xm[b, j] -= eps
        assert abs(dx[b, j] - (L(params, xp) - L(params, xm)) / (2 * eps)) < 1e-7
for k in node.NAMES:
    for idx in np.ndindex(params[k].shape):
        hi = {q: v.copy() for q, v in params.items()}
        lo = {q: v.copy() for q, v in params.items()}
        hi[k][idx] += eps
        lo[k][idx] -= eps
        assert abs(g[k][idx] - (L(hi, x) - L(lo, x)) / (2 * eps)) < 1e-7, (k, idx)

# Statement 1 on a linear field f(h) = A h: a(0) = a(T) exp(T A).
A = rng.normal(size=(3, 3))
aT = rng.normal(size=(1, 3))
a0 = node.rk4(lambda a, t: -a @ A, aT, 1.0, 0.0, 400)
np.testing.assert_allclose(a0, aT @ expm(A), rtol=1e-10, atol=1e-12)

# Statement 3: the backward solve carries 2 B D + P numbers, for any number of steps.
seen = []
solve = node.SOLVERS["rk4"]
node.SOLVERS["rk4"] = lambda f, z, t0, t1, n: (seen.append(z.size), solve(f, z, t0, t1, n))[1]
for n in (1, 10, 100):
    c2 = node.config(2, 4, d_state=3, n_steps=n)
    y = node.forward(params, x, c2)
    seen.clear()
    node.adjoint_backward(params, y, c, c2)
    assert seen == [2 * 3 * 3 + node.param_count(c2)]
node.SOLVERS["rk4"] = solve
print("adjoint: ok")
