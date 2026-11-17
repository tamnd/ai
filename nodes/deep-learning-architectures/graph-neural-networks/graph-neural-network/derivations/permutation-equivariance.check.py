# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://NEW-graph-neural-network).
"""Numeric check of the derivation permutation-equivariance: layers equivariant, sum readout invariant."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import graph_neural_network as gnn  # noqa: E402

rng = np.random.default_rng(20170404)
cfg = gnn.config(d_in=3, d_hidden=5, n_layers=3, classes=2)
for _ in range(50):
    n = int(rng.integers(2, 10))
    upper = np.triu(rng.random((n, n)) < rng.random(), 1)
    adj = (upper | upper.T).astype(float)[None]
    x = rng.normal(size=(1, n, 3))
    P = np.eye(n)[rng.permutation(n)]
    adj_p, x_p = P @ adj @ P.T, P @ x
    for how in ("sum", "mean", "max"):  # steps 1 to 3
        assert np.allclose(gnn.aggregate(adj_p, x_p, how), P @ gnn.aggregate(adj, x, how))
    params = gnn.init(rng, cfg)
    for layer in params["layers"]:
        layer["b"] = rng.normal(size=layer["b"].shape)
    logits, cache = gnn.forward(params, x, adj, cfg)
    logits_p, cache_p = gnn.forward(params, x_p, adj_p, cfg)
    for h, h_p in zip(cache["h"], cache_p["h"], strict=True):  # step 4
        assert np.allclose(h_p, P @ h)
    assert np.allclose(cache_p["graph"], cache["graph"]) and np.allclose(logits_p, logits)  # step 5
    a_hat, a_hat_p = gnn.normalized_adjacency(adj), gnn.normalized_adjacency(adj_p)
    assert np.allclose(a_hat_p, P @ a_hat @ P.T)  # step 6
    W = rng.normal(size=(3, 4))
    assert np.allclose(gnn.gcn_layer(W, x_p, a_hat_p), P @ gnn.gcn_layer(W, x, a_hat))

# Assumption 2 matters: a bias that differs per node position is not equivariant.
n = 5
adj = np.ones((1, n, n)) - np.eye(n)
x = rng.normal(size=(1, n, 3))
P = np.eye(n)[[1, 2, 3, 4, 0]]
p = {"W_self": rng.normal(size=(3, 4)), "W_nbr": rng.normal(size=(3, 4)), "b": rng.normal(size=(n, 4))}
assert not np.allclose(gnn.mp_layer(p, P @ x, P @ adj @ P.T), P @ gnn.mp_layer(p, x, adj))
print("permutation-equivariance: ok")
