# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://NEW-graph-neural-network).
"""Numeric check of the derivation wl-regular-graphs: d-regular pairs with equal features get equal readouts."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import graph_neural_network as gnn  # noqa: E402


def graph(n, edges):
    a = np.zeros((n, n))
    for u, v in edges:
        a[u, v] = a[v, u] = 1.0
    return a


triangles = graph(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
cycle = graph(6, [(i, (i + 1) % 6) for i in range(6)])
path = graph(6, [(i, i + 1) for i in range(5)])
prism = graph(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])  # 3-regular
k33 = graph(6, [(u, v) for u in range(3) for v in range(3, 6)])  # 3-regular, bipartite, no triangle
assert not np.array_equal(np.linalg.eigvalsh(triangles), np.linalg.eigvalsh(cycle))  # not isomorphic
assert not np.array_equal(np.linalg.eigvalsh(prism), np.linalg.eigvalsh(k33))

rng = np.random.default_rng(1810)
for g1, g2 in ((triangles, cycle), (prism, k33)):
    hist = gnn.wl_histograms([g1, g2], rounds=3)  # step 4
    assert hist[0] == hist[1]
    for _ in range(20):
        c = rng.normal(size=4)
        x = np.broadcast_to(c, (2, 6, 4)).copy()  # assumption 2
        adj = np.stack([g1, g2])
        cfg = gnn.config(d_in=4, d_hidden=6, n_layers=int(rng.integers(1, 4)), classes=3)
        params = gnn.init(rng, cfg)
        for layer in params["layers"]:
            layer["b"] = rng.normal(size=layer["b"].shape)
        logits, cache = gnn.forward(params, x, adj, cfg)  # steps 1 to 3, sum aggregation
        assert np.allclose(cache["h"][-1][0], cache["h"][-1][0][0]) and np.allclose(
            cache["h"][-1][0], cache["h"][-1][1]
        )
        assert np.allclose(logits[0], logits[1])
        for how in ("sum", "mean", "max"):  # step 2 for each aggregation, with a shared update
            h = x
            for layer in params["layers"]:
                h = gnn.relu(h @ layer["W_self"] + gnn.aggregate(adj, h, how) @ layer["W_nbr"] + layer["b"])
            assert np.allclose(gnn.readout(h)[0], gnn.readout(h)[1])

# The 6-path is not regular: 1-WL separates it from the 6-cycle, and so does a random network.
hist = gnn.wl_histograms([path, cycle], rounds=3)
assert hist[0][1] != hist[1][1]
cfg = gnn.config(d_in=1, d_hidden=6, n_layers=1, classes=3)
params = gnn.init(rng, cfg)
logits, _ = gnn.forward(params, np.ones((2, 6, 1)), np.stack([path, cycle]), cfg)
assert not np.allclose(logits[0], logits[1])
print("wl-regular-graphs: ok")
