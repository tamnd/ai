# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation coupling-jacobian."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402

rng = np.random.default_rng(1)
for D, H in ((2, 3), (5, 4), (6, 16)):
    cfg = flow.config(D, H, 1)
    p = {k: rng.normal(0.0, 0.7, s) for k, s in flow.param_shapes(cfg).items()}
    b = flow.masks(cfg)[0]
    K, T = np.flatnonzero(b == 1), np.flatnonzero(b == 0)
    for x in rng.standard_normal((5, D)):
        eps = 1e-6
        J = np.stack([(flow.forward(p, cfg, (x + eps * e)[None])[0][0] - flow.forward(p, cfg, (x - eps * e)[None])[0][0])
                      / (2 * eps) for e in np.eye(D)], 1)  # fmt: skip
        assert np.allclose(J[np.ix_(K, K)], np.eye(len(K)), atol=1e-8)
        assert np.allclose(J[np.ix_(K, T)], 0.0, atol=1e-8)
        _, _, _, s, _ = flow._layer(p, 0, b, x[None])
        assert np.allclose(J[np.ix_(T, T)], np.diag(np.exp(s[0, T])), atol=1e-7)
        y, ld, _ = flow.forward(p, cfg, x[None])
        assert abs(ld[0] - s[0, T].sum()) < 1e-12
        assert abs(ld[0] - np.linalg.slogdet(J)[1]) < 1e-7
        assert np.allclose(flow.inverse(p, cfg, y)[0], x, atol=1e-12)

cfg = flow.config(4, 3, 1)  # additive coupling: s = 0 gives a unit determinant
p = {k: rng.normal(0.0, 0.7, s) for k, s in flow.param_shapes(cfg).items()}
p["c0_Ws"][:] = 0.0
p["c0_bs"][:] = 0.0
assert np.allclose(flow.forward(p, cfg, rng.standard_normal((7, 4)))[1], 0.0)
print("coupling-jacobian: ok")
