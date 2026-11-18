# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Numeric check of the derivation linear-start-is-linear."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import memn2n  # noqa: E402

rng = np.random.default_rng(11)
for tying in ("adjacent", "layerwise"):
    cfg = memn2n.config(vocab=10, dim=5, hops=3, mem=4, sent_len=3, tying=tying)
    params = memn2n.init(rng, cfg, sigma=0.4)
    story = rng.integers(1, 10, (1, 4, 3))
    query = rng.integers(1, 10, (1, 3))
    lw = memn2n.position_encoding(3, 5)
    q_name, hops, w_name = memn2n.plan(cfg)

    def vectors(pair):
        e, t = pair
        return memn2n.encode(params[e], story, lw)[0] + params[t]

    u1 = memn2n.encode(params[q_name], query, lw)[0]
    G = np.eye(5)
    for in_pair, out_pair in hops:
        m, c = vectors(in_pair), vectors(out_pair)
        Hk = params["H"] if tying == "layerwise" else np.eye(5)
        G = (Hk + c.T @ m) @ G
    want = params[w_name] @ (G @ u1)
    got = memn2n.forward(params, story, query, cfg, linear=True)[0]
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12)
    # The same map applied to 2 u^1 gives twice the logits; with the softmax it does not.
    assert np.allclose(params[w_name] @ (G @ (2 * u1)), 2 * got)
    soft = memn2n.forward(params, story, query, cfg)[0]
    doubled = dict(params, **{q_name: 2 * params[q_name]})
    if tying == "layerwise":  # B is used only by the question
        assert np.allclose(memn2n.forward(doubled, story, query, cfg, linear=True)[0], 2 * got)
        assert not np.allclose(memn2n.forward(doubled, story, query, cfg)[0], 2 * soft)
print("linear-start-is-linear: ok")
