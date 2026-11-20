# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Numeric check of the derivation change-of-variables."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import flow  # noqa: E402

rng = np.random.default_rng(0)
cfg = flow.config(2, 6, 4)
p = {k: rng.normal(0.0, 0.3, s) for k, s in flow.param_shapes(cfg).items()}

h = 0.05  # the density integrates to 1
g = np.arange(-20, 20, h) + h / 2
X, Y = np.meshgrid(g, g)
grid = np.stack([X.ravel(), Y.ravel()], 1)
dens = np.exp(flow.log_prob(p, cfg, grid))
assert abs(dens.sum() * h * h - 1) < 1e-6, dens.sum() * h * h

s = flow.sample(p, cfg, 400000, rng)  # samples of f^-1(z) follow it
edges = np.linspace(-3, 3, 13)
hist, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[edges, edges])
w = edges[1] - edges[0]
sub = 10
fine = np.linspace(-3, 3, 12 * sub + 1)
fc = (fine[:-1] + fine[1:]) / 2
FX, FY = np.meshgrid(fc, fc, indexing="ij")
fd = np.exp(flow.log_prob(p, cfg, np.stack([FX.ravel(), FY.ravel()], 1))).reshape(12 * sub, 12 * sub)
mass = fd.reshape(12, sub, 12, sub).sum((1, 3)) * (w / sub) ** 2
big = mass > 0.01
assert np.max(np.abs(hist[big] / len(s) - mass[big]) / mass[big]) < 0.05

for x in rng.standard_normal((10, 2)):  # the composition: the sum of the layers' log-determinants
    eps = 1e-6
    J = np.stack([(flow.forward(p, cfg, (x + eps * e)[None])[0][0] - flow.forward(p, cfg, (x - eps * e)[None])[0][0])
                  / (2 * eps) for e in np.eye(2)], 1)  # fmt: skip
    assert abs(flow.forward(p, cfg, x[None])[1][0] - np.log(abs(np.linalg.det(J)))) < 1e-7
print("change-of-variables: ok")
