# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Numeric check of the derivation parameter-and-flop-count."""

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "impl" / "numpy"))
import diffusion as dm  # noqa: E402

for c in json.loads((HERE.parent / "fixtures" / "shapes.json").read_text())["configs"]:
    k = c["config"]
    D, H, E, B = k["d_data"], k["d_hidden"], k["d_time"], k["B"]
    cfg = dm.config(D, H, E, k["n_steps"])
    P = (D + E + 1) * H + (H + 1) * D
    assert (
        P
        == dm.param_count(cfg)
        == c["model"]["params"]
        == sum(v.size for v in dm.init(np.random.default_rng(0), cfg).values())
    )
    assert 2 * B * H * (2 * D + E) == dm.flops_forward(cfg, B) == c["model"]["flops_forward"]
assert dm.param_count(dm.config()) == 58 and dm.flops_forward(dm.config(), 4) == 400
two = dm.config(2, 64, 16, 100)
assert dm.param_count(two) == 1346 and dm.flops_forward(two, 256) == 655_360
assert 100 * dm.flops_forward(two, 256) == 65_536_000
print("parameter-and-flop-count: ok")
