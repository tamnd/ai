# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Numeric check of the derivation parameterization."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import consistency_model as cm  # noqa: E402

rng = np.random.default_rng(1)
cfg = cm.config(5, 9, 6)
for scale in (1.0, 1e3):
    p = {k: scale * rng.standard_normal(v.shape) for k, v in cm.init(rng, cfg).items()}
    x = rng.standard_normal((7, 5))
    assert np.array_equal(cm.forward(p, cfg, x, cm.EPS), x)
assert cm.c_skip(cm.EPS) == 1.0 and cm.c_out(cm.EPS) == 0.0
t = np.geomspace(cm.EPS, cm.T_MAX, 2000)
assert np.all(np.diff(cm.c_skip(t)) < 0) and np.all(np.diff(cm.c_out(t)) > 0)
assert np.isclose(cm.c_skip(cm.T_MAX), 0.25 / (79.998**2 + 0.25)) and cm.c_skip(cm.T_MAX) < 4e-5
assert abs(cm.c_out(1e8) - cm.SIGMA_DATA) < 1e-8
sd, eps = cm.SIGMA_DATA, cm.EPS
h = 1e-6
for tt in (0.01, 0.5, 3.0, 40.0):
    fd = (cm.c_out(tt + h) - cm.c_out(tt - h)) / (2 * h)
    assert np.isclose(fd, sd * (sd**2 + eps * tt) / (sd**2 + tt**2) ** 1.5, rtol=1e-7)
s = np.geomspace(1e-3, 100, 50)
assert np.allclose(cm.c_skip(s, eps=0.0), sd**2 / (s**2 + sd**2))
assert np.allclose(cm.c_out(s, eps=0.0), s * sd / np.sqrt(s**2 + sd**2))
print("parameterization: ok")
