# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Numeric check of the derivation parameter-and-flop-count."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import latent_diffusion as ld  # noqa: E402

shapes = json.loads((pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "shapes.json").read_text())
rng = np.random.default_rng(7)
for c in shapes["configs"]:
    k = c["config"]
    N, ch, H, E, M, d, C, B = (k[n] for n in ("positions", "channels", "hidden", "temb", "ctx_tokens", "attn",
                                               "classes", "B"))  # fmt: skip
    cfg = ld.config(N, ch, H, E, M, d, C)
    P = N * ch * H + E * H + H + (C + 1) * M * d + H * d + d * d + d * H + H * N * ch + N * ch
    F = 2 * B * (N * ch * H + E * H + H * d + M * d * d + M * d * H + M * d + M * H + H * N * ch)
    assert ld.param_count(cfg) == P == c["model"]["params"]
    assert ld.flops_forward(cfg, B) == F == c["model"]["flops_forward"]
    if P < 10**5:
        p = ld.init(rng, cfg)
        assert sum(v.size for v in p.values()) == P
        Bs = 3
        _, (zf, te, h, q, Cx, K, V, w, h2) = ld._forward(p, cfg, rng.standard_normal((Bs, N, ch)), 5, np.zeros(Bs, int))
        mac = (zf.shape[1] * H + te.shape[1] * H + H * q.shape[1] + K.shape[1] * d * K.shape[2]
               + V.shape[1] * d * V.shape[2] + w.shape[1] * d + w.shape[1] * H + h2.shape[1] * zf.shape[1])  # fmt: skip
        assert 2 * Bs * mac == ld.flops_forward(cfg, Bs)
