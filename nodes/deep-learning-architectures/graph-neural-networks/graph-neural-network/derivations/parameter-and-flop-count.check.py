# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""Numeric check of the derivation parameter-and-flop-count: the formulas against counted arrays and products."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import graph_neural_network as gnn  # noqa: E402


def formula_params(F, D, L, K):
    return (2 * F * D + D) + (L - 1) * (2 * D * D + D) + D * K + K


def formula_flops(F, D, L, K, B, N):
    return 4 * B * N * F * D + (L - 1) * 4 * B * N * D * D + L * 2 * B * N * N * D + 2 * B * D * K


def counted_flops(params, B, N):
    # Replay forward's products and add 2 m k n for each.
    total, width = 0, params["layers"][0]["W_self"].shape[0]
    for layer in params["layers"]:
        D = layer["W_self"].shape[1]
        total += 2 * (2 * B * N * width * D)  # H W_self and H W_nbr
        total += 2 * B * N * N * D  # A (H W_nbr)
        width = D
    return total + 2 * B * width * params["head"]["W"].shape[1]


rng = np.random.default_rng(228)
assert formula_params(3, 8, 2, 4) == 228 and formula_flops(3, 8, 2, 4, 2, 6) == 6656
for _ in range(100):
    F, D, L, K, B, N = (int(v) for v in rng.integers(1, 9, size=6))
    cfg = gnn.config(d_in=F, d_hidden=D, n_layers=L, classes=K)
    params = gnn.init(rng, cfg)
    leaves = [v for layer in params["layers"] for v in layer.values()] + list(params["head"].values())
    assert sum(v.size for v in leaves) == formula_params(F, D, L, K) == gnn.param_count(cfg)
    assert counted_flops(params, B, N) == formula_flops(F, D, L, K, B, N) == gnn.flops(cfg, B, N)
    logits, _ = gnn.forward(params, rng.normal(size=(B, N, F)), np.zeros((B, N, N)), cfg)
    assert logits.shape == (B, K)
print("parameter-and-flop-count: ok")
