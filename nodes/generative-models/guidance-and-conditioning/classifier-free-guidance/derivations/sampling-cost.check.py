# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Numeric check of the derivation sampling-cost."""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl" / "numpy"))
import classifier_free_guidance as cg  # noqa: E402


def formula(D, H, E, C):
    return D * H + E * H + (C + 1) * H + H + H * D + D


for c in json.loads((HERE.parent / "fixtures" / "shapes.json").read_text())["configs"]:
    k = c["config"]
    cfg = cg.config(k["dim"], k["hidden"], k["temb"], k["classes"])
    p = cg.init(np.random.default_rng(0), cfg)
    P = formula(k["dim"], k["hidden"], k["temb"], k["classes"])
    assert sum(v.size for v in p.values()) == cg.param_count(cfg) == P == c["model"]["params"]
    D, H, E, B = k["dim"], k["hidden"], k["temb"], k["B"]
    assert cg.flops_forward(cfg, B) == 2 * (B * D * H + B * E * H + B * H * D) == c["model"]["flops_forward"]
    assert p["emb"].shape == (k["classes"] + 1, H)
assert formula(2, 64, 16, 3) == 1602 and cg.flops_forward(cg.config(2, 64, 16, 3), 256) == 655360
assert formula(4096, 1024, 256, 1000) == 9680896 and cg.flops_forward(cg.config(4096, 1024, 256, 1000), 1) == 17301504
abar = cg.alpha_bars(1000)
steps = [int(v) for v in np.linspace(999, 0, 32).round()]
counts = {}
for interval in (None, (0.28, 2.9), (1e9, 2e9)):
    eps, count = cg.guided_sampler(lambda z, t: 0.5 * z, lambda z, t: 0 * z, 2.0, abar, interval=interval)
    cg.ddim_sample(eps, np.zeros((1, 2)), abar, steps)
    counts[interval] = count["nfe"]
assert counts[None] == 64 and counts[(1e9, 2e9)] == 32 and counts[(0.28, 2.9)] == 44
assert round(1 - 44 / 64, 2) == 0.31
