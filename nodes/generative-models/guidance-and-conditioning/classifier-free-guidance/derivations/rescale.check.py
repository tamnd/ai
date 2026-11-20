# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation rescale."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

rng = np.random.default_rng(6)
e_c, e_u = rng.standard_normal((2, 5, 4, 8, 8))
e_u = 0.8 * e_u + 0.6 * e_c
ax = (1, 2, 3)
for s in (1.0, 3.0, 7.5):
    cfg = cg.guide(e_c, e_u, s)
    for phi in (0.0, 0.5, 0.7, 1.0):
        out = cg.rescale(e_c, cfg, phi)
        for ddof in (0, 1):
            sp, sc = e_c.std(axis=ax, ddof=ddof, keepdims=True), cfg.std(axis=ax, ddof=ddof, keepdims=True)
            two_step = phi * cfg * sp / sc + (1 - phi) * cfg
            assert np.allclose(out, two_step)
            assert np.allclose(out.std(axis=ax, ddof=ddof, keepdims=True), phi * sp + (1 - phi) * sc)
        if phi == 0.0:
            assert np.allclose(out, cfg)
        if phi == 1.0:
            assert np.allclose(out.std(axis=ax), e_c.std(axis=ax))
# the guided output grows with s, and the rescale with phi = 1 removes the growth
growth = [cg.guide(e_c, e_u, s).std(axis=ax).mean() for s in (1.0, 3.0, 7.5)]
assert growth[0] < growth[1] < growth[2]
