# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://NEW-autoencoder).
"""Numeric check of the derivation contractive-penalty."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import autoencoder as ae  # noqa: E402

rng = np.random.default_rng(0)
for act in ("sigmoid", "tanh", "linear"):
    cfg = ae.from_widths([5, 7], code_act=act)
    p = ae.init(rng, cfg)
    p["enc_b"] = [rng.normal(size=7)]
    W, b = p["enc_W"][0], p["enc_b"][0]
    for x in rng.normal(0, 2, (6, 5)):
        h = ae.act(act, x @ W + b)
        J_fd = np.stack([(ae.encode(p, cfg, (x + 1e-6 * e)[None])[0] - ae.encode(p, cfg, (x - 1e-6 * e)[None])[0]) / 2e-6
                         for e in np.eye(5)], axis=1)  # fmt: skip
        J = ae.jacobian(p, cfg, x)
        np.testing.assert_allclose(J, J_fd, rtol=1e-6, atol=1e-9)
        sp = ae.act_grad(act, x @ W + b, h)
        np.testing.assert_allclose(np.sum(J**2), np.sum(sp**2 * np.sum(W**2, 0)), rtol=1e-12)  # statement 1
        if act == "sigmoid":
            np.testing.assert_allclose(np.sum(J**2), np.sum((h * (1 - h)) ** 2 * np.sum(W**2, 0)), rtol=1e-12)
        if act == "linear":
            np.testing.assert_allclose(np.sum(J**2), np.sum(W**2), rtol=1e-12)  # statement 3

# statement 4: the gradients of the batch penalty
cfg = ae.from_widths([4, 3], code_act="sigmoid")
p = ae.init(rng, cfg)
p["enc_b"] = [rng.normal(size=3)]
x = rng.normal(size=(5, 4))
value, g = ae.contractive_penalty(p, cfg, x)
np.testing.assert_allclose(value, np.mean([np.sum(ae.jacobian(p, cfg, xi) ** 2) for xi in x]), rtol=1e-12)
for key, arr in (("W", p["enc_W"][0]), ("b", p["enc_b"][0])):
    num = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        arr[idx] = old + 1e-6
        up = ae.contractive_penalty(p, cfg, x)[0]
        arr[idx] = old - 1e-6
        down = ae.contractive_penalty(p, cfg, x)[0]
        arr[idx] = old
        num[idx] = (up - down) / 2e-6
    np.testing.assert_allclose(g[key], num, rtol=1e-6, atol=1e-10)
# saturation: pushing the pre-activations far from 0 sends the sigmoid penalty to 0 with W fixed
p["enc_b"] = [np.full(3, 30.0)]
assert ae.contractive_penalty(p, cfg, x)[0] < 1e-20
print("contractive-penalty: ok")
