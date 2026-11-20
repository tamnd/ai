# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation planar-determinant."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402

rng = np.random.default_rng(2)
for D in range(2, 9):
    w, u, b = rng.standard_normal(D), rng.standard_normal(D), rng.normal(0.0, 0.5)
    uh = flow.u_hat(w, u)
    assert abs(w @ uh - (-1 + math.log1p(math.exp(w @ u)))) < 1e-12 and w @ uh > -1
    perp = lambda v: v - (v @ w) * w / (w @ w)  # noqa: E731
    assert np.allclose(perp(uh), perp(u), atol=1e-12)
    for z in rng.standard_normal((4, D)):
        eps = 1e-6
        J = np.stack([(flow.planar((z + eps * e)[None], w, uh, b)[0][0] - flow.planar((z - eps * e)[None], w, uh, b)[0][0])
                      / (2 * eps) for e in np.eye(D)], 1)  # fmt: skip
        assert abs(flow.planar(z[None], w, uh, b)[1][0] - math.log(abs(np.linalg.det(J)))) < 1e-7

    alpha = np.linspace(-8, 8, 4001)
    good = alpha + (w @ uh) * np.tanh(alpha + b)
    assert np.all(np.diff(good) > 0)
    bad_u = u - (w @ u + 3.0) * w / (w @ w)  # w.u = -3
    bad = alpha + (w @ bad_u) * np.tanh(alpha + b)
    assert np.any(np.diff(bad) < 0)

    z = rng.standard_normal(D)  # invert by bisection on alpha = w.z
    y = flow.planar(z[None], w, uh, b)[0][0]
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid + (w @ uh) * math.tanh(mid + b) < y @ w:
            lo = mid
        else:
            hi = mid
    z_back = y - uh * math.tanh(lo + b)
    assert np.allclose(z_back, z, atol=1e-9)
print("planar-determinant: ok")
