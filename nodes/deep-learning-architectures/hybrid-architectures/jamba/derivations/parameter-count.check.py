# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Numeric check of the derivation parameter-count: the 52B and 12B of Table 1 from the shapes."""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import jamba  # noqa: E402

cfg = jamba.config(vocab=65536, d_model=4096, n_blocks=4, block_len=8, moe_every=2, n_experts=16, top_k=2,
                   d_ff=14336, n_heads=32, n_kv=8, d_state=16, d_conv=4, expand=2, dt_rank=256)  # fmt: skip
shapes = jamba.param_shapes(cfg)
kinds = {"attn": ("wq", "wk", "wv", "wo"), "experts": ("e_gate", "e_up", "e_down"), "router": ("router",),
         "mlp": ("w_gate", "w_up", "w_down"), "norm": ("norm1", "norm2"), "io": ("embed", "head", "norm_f")}  # fmt: skip
total = {k: 0 for k in (*kinds, "mamba")}
for name, s in shapes.items():
    key = name.split("_", 1)[1] if name.startswith("l") and name[1].isdigit() else name
    kind = next((k for k, names in kinds.items() if key in names), "mamba")
    total[kind] += math.prod(s)
D, F = 4096, 14336
assert total == {"attn": 167_772_160, "experts": 45_097_156_608, "router": 1_048_576, "mlp": 2_818_572_288,
                 "norm": 262_144, "io": 536_875_008, "mamba": 2_948_636_544}, total  # fmt: skip
P = sum(total.values())
assert P == jamba.param_count(cfg) == 51_570_323_328
assert jamba.param_count(cfg, active=True) == P - 224 * 3 * D * F == 12_110_311_296
assert (round(P / 1e9), round(jamba.param_count(cfg, active=True) / 1e9)) == (52, 12)
assert round(100 * total["experts"] / P) == 87
print("parameter-count: ok")
