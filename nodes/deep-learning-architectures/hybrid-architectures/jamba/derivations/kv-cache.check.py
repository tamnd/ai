# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Numeric check of the derivation kv-cache: the numbers of Table 1 and the constant state of the Mamba layers."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import jamba  # noqa: E402

GiB, T = 2**30, 2**18
assert jamba.kv_cache_bytes(4, 8, 128, T) == 4 * GiB == 2**32
assert jamba.kv_cache_bytes(32, 8, 128, T) == 32 * GiB
assert jamba.kv_cache_bytes(32, 32, 128, T) == 128 * GiB

cfg = jamba.config(vocab=65536, d_model=4096, n_blocks=4, block_len=8, moe_every=2, n_experts=16, top_k=2,
                   d_ff=14336, n_heads=32, n_kv=8, d_state=16, d_conv=4, expand=2, dt_rank=256)  # fmt: skip
kinds = [m for m, _ in jamba.layers(cfg)]
assert kinds.count("attn") == 4 and kinds.count("mamba") == 28
state = 28 * cfg["d_inner"] * (cfg["d_conv"] - 1 + cfg["d_state"])
assert state == 4_358_144
assert jamba.cache_numbers(cfg, 1, T) == 2**31 + state
assert round(100 * state / 2**31, 1) == 0.2
crossover = cfg["d_inner"] * (cfg["d_conv"] - 1 + cfg["d_state"]) / (2 * cfg["n_kv"] * cfg["head_dim"])
assert crossover == 76

# Generation with a small model: the attention arrays grow by 2 G A per token, the Mamba arrays stay the same.
small = jamba.config(n_blocks=2)
rng = np.random.default_rng(0)
p = jamba.init(rng, small)
st = jamba.init_state(small, 1)
for t in range(1, 6):
    _, st = jamba.step(p, small, rng.integers(0, small["vocab"], 1), st)
    for (m, _), (a, b) in zip(jamba.layers(small), st):
        size = a.size + b.size
        if m == "attn":
            assert size == 2 * small["n_kv"] * small["head_dim"] * t
        else:
            assert size == small["d_inner"] * (small["d_conv"] - 1 + small["d_state"])
print("kv-cache: ok")
