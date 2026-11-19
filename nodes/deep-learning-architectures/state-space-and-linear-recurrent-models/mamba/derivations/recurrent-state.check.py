# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Numeric check of the derivation recurrent-state: the state carried in generation, against a KV cache."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import mamba  # noqa: E402

rng = np.random.default_rng(3)
for _ in range(5):
    D, L, N, K = (int(v) for v in rng.integers([2, 1, 1, 2], [9, 4, 6, 5]))
    cfg = mamba.config(d_model=D, n_layers=L, d_state=N, d_conv=K, disc=("simple", "zoh")[int(rng.integers(2))])
    p = mamba.init(rng, cfg, scale=0.3)
    tok = rng.integers(0, cfg["vocab"], (2, 12))
    logits, _ = mamba.forward(p, cfg, tok)
    state, out = mamba.init_state(cfg, 2), []
    for t in range(12):
        o, state = mamba.step(p, cfg, tok[:, t], state)
        assert sum(a.size + b.size for a, b in state) == 2 * L * cfg["d_inner"] * (K - 1 + N)
        out.append(o)
    assert np.allclose(np.stack(out, 1), logits, atol=1e-12)

L, D, K, N = 64, 2560, 4, 16
assert L * 2 * D * (K - 1 + N) == 6_225_920
crossover = [T for T in range(1, 100) if L * D * T > L * 2 * D * (K - 1 + N)][0]
assert crossover == 39 == 2 * (K - 1 + N) + 1
print("recurrent-state: ok")
