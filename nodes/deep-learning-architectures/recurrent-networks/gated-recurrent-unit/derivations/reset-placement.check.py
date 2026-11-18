# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Numeric check of the derivation reset-placement: when the two candidates agree, and the flipped update gate."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gru  # noqa: E402

rng = np.random.default_rng(1078)
n_in, H = 3, 5
x, h0 = rng.normal(size=(3, 8, n_in)), rng.uniform(-1, 1, size=(3, H))


def gap(params):
    return np.abs(gru.forward(params, x, h0, "after")[0] - gru.forward(params, x, h0, "before")[0]).max()


params = {k: 2 * v for k, v in gru.init(rng, gru.config(n_in, H)).items()}
assert gap(params) > 1e-2
diag = params["U"].copy()
diag[:, 2 * H :] = np.diag(rng.normal(size=H))
b_h = params["b_h"].copy()
b_h[2 * H :] = 0.0
assert gap(params | {"U": diag, "b_h": b_h}) < 1e-14
assert gap(params | {"U": diag}) > 1e-3  # b_hn != 0
off = diag.copy()
off[0, 2 * H + 1] = 0.5
assert gap(params | {"U": off, "b_h": b_h}) > 1e-3  # U_n not diagonal

# The cost: the same multiply-adds, one product of H x 3H after, two dependent products before.
for reset, products in (("after", [(H, 3 * H)]), ("before", [(H, 2 * H), (H, H)])):
    assert sum(k * n for k, n in products) == 3 * H * H, reset

# The convention of Chung et al.: h_t = (1 - z') h_{t-1} + z' n_t, with z' computed from the update-gate weights.
flip = {k: v.copy() for k, v in params.items()}
for k in ("W", "U"):
    flip[k][:, H : 2 * H] *= -1
for k in ("b_i", "b_h"):
    flip[k][H : 2 * H] *= -1
want, _ = gru.forward(params, x, h0, "before")
h = h0
for t in range(8):
    _, s = gru.cell(flip, x[:, t], h, "before")  # its z is the z' of Chung et al.
    h = (1 - s["z"]) * h + s["z"] * s["n"]
    assert np.allclose(h, want[:, t], atol=1e-13)
print("reset-placement: ok")
