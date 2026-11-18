# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Numeric check of the derivation temporal-rows-and-empty-memories."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import memn2n  # noqa: E402

rng = np.random.default_rng(17)
cfg = memn2n.config(vocab=10, dim=5, hops=2, mem=4, sent_len=3)
params = memn2n.init(rng, cfg, sigma=0.5)
story = rng.integers(1, 10, (2, 4, 3))
story[1, 2] = 0  # an empty slot
query = rng.integers(1, 10, (2, 3))
_, attn = memn2n.forward(params, story, query, cfg, trace=True)
# Statement 1: a common shift of T0, the T_A of hop 1, leaves the attention of hop 1 unchanged.
shifted = dict(params, T0=params["T0"] + rng.normal(size=(1, 5)))
_, attn2 = memn2n.forward(shifted, story, query, cfg, trace=True)
assert np.allclose(attn[0], attn2[0])
# A shift of one row only does change it.
one = dict(params, T0=params["T0"].copy())
one["T0"][1] += 1.0
_, attn3 = memn2n.forward(one, story, query, cfg, trace=True)
assert not np.allclose(attn[0], attn3[0])
# Statement 2: the empty slot is its temporal rows and still has a positive weight.
lw = memn2n.position_encoding(3, 5)
m = memn2n.encode(params["E0"], story, lw) + params["T0"]
assert np.allclose(m[1, 2], params["T0"][2])
assert attn[0, 1, 2] > 0 and attn[1, 1, 2] > 0
u = memn2n.encode(params["E0"], query, lw)
s = np.einsum("bd,bnd->bn", u, m)
assert np.isclose(attn[0, 1, 2], np.exp(u[1] @ params["T0"][2]) / np.exp(s[1]).sum())
print("temporal-rows-and-empty-memories: ok")
