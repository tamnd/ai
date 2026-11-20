# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation rbm-free-energy-and-conditionals."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

rng = np.random.default_rng(2)
for D, H in ((2, 1), (3, 4), (5, 3)):
    p = {k: 1.5 * rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(D, H)).items()}
    X, Hs = eb.all_binary(D), eb.all_binary(H)
    E = np.array([eb.energy(p, np.repeat(v[None], len(Hs), 0), Hs) for v in X])  # (2^D, 2^H)
    np.testing.assert_allclose(np.log(np.exp(-E).sum(1)), -eb.free_energy(p, X), rtol=1e-12)
    joint = np.exp(-E) / np.exp(-E).sum()
    # p(h | v) is the product of the sigmoids
    ph = eb.p_h(p, X)
    cond = joint / joint.sum(1, keepdims=True)
    np.testing.assert_allclose(cond, np.prod(np.where(Hs[None] == 1, ph[:, None], 1 - ph[:, None]), -1), rtol=1e-11)
    pv = eb.p_v(p, Hs)
    cond_v = (joint / joint.sum(0, keepdims=True)).T
    np.testing.assert_allclose(cond_v, np.prod(np.where(X[None] == 1, pv[:, None], 1 - pv[:, None]), -1), rtol=1e-11)
    # conditional independence: p(h_j = 1 | v, h_other) is the same for every value of the other hidden units
    if H > 1:
        for vi in range(len(X)):
            for j in range(H):
                on, off = Hs[:, j] == 1, Hs[:, j] == 0
                ratio = joint[vi, on] / (joint[vi, on] + joint[vi, off])
                np.testing.assert_allclose(ratio, ph[vi, j], rtol=1e-11)
