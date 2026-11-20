# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation guidance-conventions."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402

rng = np.random.default_rng(1)
z, e_c, e_u = rng.standard_normal((3, 6, 4))
abar = rng.uniform(0.01, 0.99, 6)
for w in (0.0, 0.3, 4.0, 6.5):
    s = 1.0 + w
    assert np.allclose(cg.guide_ho(e_c, e_u, w), cg.guide(e_c, e_u, s))
    assert np.allclose(w * e_c + (1 - w) * e_u, cg.guide(e_c, e_u, w))
    g = cg.guide(e_c, e_u, s)
    for conv in (cg.eps_to_x0, cg.eps_to_v):
        assert np.allclose(cg.guide(conv(z, e_c, abar), conv(z, e_u, abar), s), conv(z, g, abar))
    score = -g / np.sqrt(1 - abar)[:, None]
    assert np.allclose(cg.guide(-e_c / np.sqrt(1 - abar)[:, None], -e_u / np.sqrt(1 - abar)[:, None], s), score)
    assert np.allclose(cg.x0_to_eps(z, cg.eps_to_x0(z, g, abar), abar), g)
assert np.allclose(cg.guide(e_c, e_u, 1.0), e_c) and np.allclose(cg.guide_ho(e_c, e_u, 0.0), e_c)
# a combination whose weights do not sum to 1 is not invariant
x_c, x_u = cg.eps_to_x0(z, e_c, abar), cg.eps_to_x0(z, e_u, abar)
assert not np.allclose(2.0 * x_c, cg.eps_to_x0(z, 2.0 * e_c, abar))
assert not np.allclose(cg.rescale(x_c, cg.guide(x_c, x_u, 5.0), 0.7),
                       cg.eps_to_x0(z, cg.rescale(e_c, cg.guide(e_c, e_u, 5.0), 0.7), abar))  # fmt: skip
