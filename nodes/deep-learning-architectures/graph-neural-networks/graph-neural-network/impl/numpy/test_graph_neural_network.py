# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""Tests of the NumPy reference: the worked example, the shape contract, the fixture, and the symmetries of the page."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graph_neural_network import (
    aggregate,
    config,
    flops,
    forward,
    gcn_layer,
    init,
    mp_layer,
    normalized_adjacency,
    param_count,
    readout,
    wl_histograms,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["n_layers"], k["classes"])


def random_graph(rng, n, p=0.4):
    """A random undirected graph without self-loops, as a dense (N, N) matrix."""
    upper = np.triu(rng.random((n, n)) < p, 1)
    return (upper | upper.T).astype(float)


def cycle(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1.0
    return a


def two_triangles():
    a = np.zeros((6, 6))
    a[:3, :3] = cycle(3)
    a[3:, 3:] = cycle(3)
    return a


def path(n):
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def unflatten(flat):
    layers = {}
    for k, v in flat.items():
        parts = k.split("/")
        if parts[0] == "layers":
            layers.setdefault(int(parts[1]), {})[parts[2]] = v
    head = {"W": flat["head/W"], "b": flat["head/b"]}
    return {"layers": [layers[i] for i in sorted(layers)], "head": head}


def test_worked_example():
    # The path 1 - 2 - 3 with the scalar features (1, 2, 3), as on the page.
    a = path(3)[None]  # (B, N, N)
    x = np.array([[[1.0], [2.0], [3.0]]])  # (B, N, F)
    assert np.allclose(aggregate(a, x, "sum")[0, :, 0], [2.0, 4.0, 2.0])
    assert np.allclose(aggregate(a, x, "mean")[0, :, 0], [2.0, 2.0, 2.0])
    assert np.allclose(aggregate(a, x, "max")[0, :, 0], [2.0, 3.0, 2.0])
    a_hat = normalized_adjacency(a)[0]
    want = np.array(
        [
            [1 / 2, 1 / 6**0.5, 0],
            [1 / 6**0.5, 1 / 3, 1 / 6**0.5],
            [0, 1 / 6**0.5, 1 / 2],
        ]
    )
    assert np.allclose(a_hat, want)
    z = gcn_layer(np.ones((1, 1)), x, a_hat[None], activation=False)[0, :, 0]
    assert np.allclose(z, [1.3165, 2.2997, 2.3165], atol=5e-5)
    p = {
        "W_self": np.ones((1, 1)),
        "W_nbr": np.full((1, 1), 0.5),
        "b": np.array([-3.0]),
    }
    h = mp_layer(p, x, a)[0, :, 0]  # relu(x + 0.5 * sum of neighbours - 3)
    assert np.allclose(h, [0.0, 1.0, 1.0])
    assert readout(h[None, :, None])[0, 0] == 2.0


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    x = rng.normal(size=(k["B"], k["N"], k["F"]))
    adj = np.stack([random_graph(rng, k["N"]) for _ in range(k["B"])])
    logits, cache = forward(params, x, adj, cfg)
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["graph"].shape) == blocks["readout"]["outputs"]["graph"]
    assert list(cache["h"][0].shape) == blocks["layer_first"]["outputs"]["h"]
    for h in cache["h"][1:]:
        assert list(h.shape) == blocks["layer_rest"]["outputs"]["h"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k = cfg_of(c), c["config"]
    assert param_count(cfg) == c["model"]["params"]
    assert flops(cfg, k["B"], k["N"]) == c["model"]["flops_forward"]
    params = init(np.random.default_rng(0), cfg)
    n = sum(v.size for p in params["layers"] for v in p.values()) + sum(v.size for v in params["head"].values())
    assert n == c["model"]["params"]


def test_shared_fixture():
    params = unflatten(dict(np.load(FIXTURES / "params-small.npz")))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    logits, cache = forward(params, inp["x"], inp["adj"], cfg_of(SHAPES[0]))
    assert np.allclose(logits, want["logits"], rtol=1e-12, atol=1e-12)
    assert np.allclose(cache["graph"], want["graph"], rtol=1e-12, atol=1e-12)
    for i, h in enumerate(cache["h"]):
        assert np.allclose(h, want[f"h/{i}"], rtol=1e-12, atol=1e-12)
    a_hat = normalized_adjacency(inp["adj"])
    assert np.allclose(a_hat, want["adj_hat"], rtol=1e-12, atol=1e-12)
    assert np.allclose(gcn_layer(inp["W_gcn"], inp["x"], a_hat), want["gcn"], rtol=1e-12, atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(1, 9),
    how=st.sampled_from(["sum", "mean", "max"]),
)
def test_layers_are_permutation_equivariant(seed, n, how):
    rng = np.random.default_rng(seed)
    a = random_graph(rng, n)[None]  # (B, N, N)
    h = rng.normal(size=(1, n, 4))  # (B, N, D)
    P = np.eye(n)[rng.permutation(n)]  # a permutation matrix
    pa, ph = P @ a @ P.T, P @ h
    assert np.allclose(aggregate(pa, ph, how), P @ aggregate(a, h, how))
    p = init(rng, config(4, 3, 1, 2))["layers"][0]
    assert np.allclose(mp_layer(p, ph, pa), P @ mp_layer(p, h, a))
    W = rng.normal(size=(4, 3))
    assert np.allclose(
        gcn_layer(W, ph, normalized_adjacency(pa)),
        P @ gcn_layer(W, h, normalized_adjacency(a)),
    )


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 9))
def test_readout_is_permutation_invariant(seed, n):
    rng = np.random.default_rng(seed)
    cfg = config(3, 5, 3, 2)
    params = init(rng, cfg)
    for p in params["layers"]:
        p["b"] = rng.normal(size=p["b"].shape)
    a = random_graph(rng, n)[None]
    x = rng.normal(size=(1, n, 3))
    P = np.eye(n)[rng.permutation(n)]
    y, _ = forward(params, x, a, cfg)
    y_perm, _ = forward(params, P @ x, P @ a @ P.T, cfg)
    assert np.allclose(y, y_perm)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 12))
def test_normalized_adjacency_spectrum(seed, n):
    rng = np.random.default_rng(seed)
    a = random_graph(rng, n, p=rng.random())
    a_hat = normalized_adjacency(a)
    assert np.allclose(a_hat, a_hat.T)
    lam = np.linalg.eigvalsh(a_hat)
    assert lam.min() >= -1 - 1e-10 and lam.max() <= 1 + 1e-10
    assert abs(lam.max() - 1) < 1e-10  # D~^(1/2) 1 is an eigenvector with eigenvalue 1
    s = np.sqrt((a + np.eye(n)).sum(axis=1))
    assert np.allclose(a_hat @ s, s)
    # Without the self-loops, I + D^(-1/2) A D^(-1/2) of Kipf and Welling's Eq. 7 has its eigenvalues in [0, 2].
    d = a.sum(axis=1)
    inv = np.divide(1.0, np.sqrt(d), out=np.zeros(n), where=d > 0)
    mu = np.linalg.eigvalsh(np.eye(n) + inv[:, None] * a * inv[None, :])
    assert mu.min() >= -1e-10 and mu.max() <= 2 + 1e-10


def loop_forward(params, x, adj):
    """The model of forward, node by node and edge by edge, for one graph."""
    h = x
    for p in params["layers"]:
        new = np.zeros((h.shape[0], p["b"].shape[0]))
        for v in range(h.shape[0]):
            m = np.zeros(p["b"].shape[0])
            for u in range(h.shape[0]):
                if adj[v, u]:
                    m += h[u] @ p["W_nbr"]
            new[v] = np.maximum(h[v] @ p["W_self"] + m + p["b"], 0.0)
        h = new
    g = sum(h[v] for v in range(h.shape[0]))
    return g @ params["head"]["W"] + params["head"]["b"]


def test_dense_matches_loops():
    rng = np.random.default_rng(1)
    cfg = config(3, 4, 3, 2)
    params = init(rng, cfg)
    for p in params["layers"]:
        p["b"] = rng.normal(size=p["b"].shape)
    adj = np.stack([random_graph(rng, 7) for _ in range(3)])
    x = rng.normal(size=(3, 7, 3))
    y, _ = forward(params, x, adj, cfg)
    for b in range(3):
        assert np.allclose(y[b], loop_forward(params, x[b], adj[b]), rtol=1e-12, atol=1e-12)
    # GCN: the dense product against the sum over the neighbours and the node itself, with 1 / sqrt(d_v d_u).
    W = rng.normal(size=(3, 4))
    z = gcn_layer(W, x, normalized_adjacency(adj), activation=False)
    for b in range(3):
        deg = adj[b].sum(axis=1) + 1
        for v in range(7):
            want = sum((x[b, u] @ W) / np.sqrt(deg[v] * deg[u]) for u in range(7) if adj[b, v, u] or u == v)
            assert np.allclose(z[b, v], want)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), how=st.sampled_from(["sum", "mean", "max"]))
def test_wl_pair_has_equal_readouts(seed, how):
    # Two triangles and the 6-cycle: 1-WL gives the same colours, so every message-passing network gives the same output.
    rng = np.random.default_rng(seed)
    adj = np.stack([two_triangles(), cycle(6)])  # (B, N, N)
    assert wl_histograms(list(adj), 3)[0] == wl_histograms(list(adj), 3)[1]
    cfg = config(2, 5, 3, 3)
    params = init(rng, cfg)
    for p in params["layers"]:
        p["b"] = rng.normal(size=p["b"].shape)
    x = np.broadcast_to(rng.normal(size=(1, 1, 2)), (2, 6, 2))  # the same features on every node
    y, cache = forward(params, x, adj, cfg)
    assert np.allclose(y[0], y[1]) and np.allclose(cache["graph"][0], cache["graph"][1])
    h = x
    for p in params["layers"]:
        h = np.maximum(h @ p["W_self"] + aggregate(adj, h @ p["W_nbr"], how) + p["b"], 0.0)
        assert np.allclose(h[0], h[1])
    g = normalized_adjacency(adj)
    assert np.allclose(
        gcn_layer(params["layers"][0]["W_self"], x, g)[0],
        gcn_layer(params["layers"][0]["W_self"], x, g)[1],
    )


def test_wl_separates_and_so_can_the_model():
    # The 6-path and the 6-cycle: 1-WL separates them after one round, and so does the model with random weights.
    adj = np.stack([path(6), cycle(6)])
    hist = wl_histograms(list(adj), 2)
    assert hist[0][0] == hist[1][0] and hist[0][1] != hist[1][1]
    rng = np.random.default_rng(3)
    cfg = config(1, 6, 2, 2)
    y, _ = forward(init(rng, cfg), np.ones((2, 6, 1)), adj, cfg)
    assert not np.allclose(y[0], y[1])


def test_sum_separates_where_mean_and_max_do_not():
    # A centre with 2 neighbours against a centre with 3, all with the same feature: mean and max give the same vector.
    a2 = np.zeros((4, 4))
    a2[0, 1:3] = a2[1:3, 0] = 1.0
    a3 = np.zeros((4, 4))
    a3[0, 1:4] = a3[1:4, 0] = 1.0
    adj = np.stack([a2, a3])
    h = np.ones((2, 4, 3)) * np.array([0.5, -1.0, 2.0])
    for how in ("mean", "max"):
        out = aggregate(adj, h, how)
        assert np.allclose(out[0, 0], out[1, 0])
    out = aggregate(adj, h, "sum")
    assert np.allclose(out[0, 0] * 3, out[1, 0] * 2)


def test_isolated_node_gets_zero():
    adj = np.zeros((1, 3, 3))
    h = np.arange(6.0).reshape(1, 3, 2)
    for how in ("sum", "mean", "max"):
        assert np.array_equal(aggregate(adj, h, how), np.zeros((1, 3, 2)))
    assert np.allclose(normalized_adjacency(adj)[0], np.eye(3))  # only the self-loop is left


def test_repeated_gcn_propagation_oversmooths():
    # On a connected graph, A_hat^m x tends to a multiple of sqrt(d~): the nodes keep only their degree.
    rng = np.random.default_rng(4)
    a = path(5) + np.diag(np.ones(3), 2) + np.diag(np.ones(3), -2)
    s = np.sqrt((a + np.eye(5)).sum(axis=1))
    z = rng.normal(size=5)
    g = normalized_adjacency(a)
    for _ in range(200):
        z = g @ z
    assert np.allclose(z / s, (z / s)[0], rtol=1e-8)


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        config(0, 4, 1, 2)
    cfg = config(3, 4, 2, 2)
    with pytest.raises(ValueError):
        forward(
            init(np.random.default_rng(0), cfg),
            np.zeros((1, 2, 5)),
            np.zeros((1, 2, 2)),
            cfg,
        )
    with pytest.raises(ValueError):
        aggregate(np.zeros((1, 2, 2)), np.zeros((1, 2, 3)), "median")


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    adj = np.stack([random_graph(rng, k["N"]) for _ in range(k["B"])])
    return forward, (
        init(rng, cfg),
        rng.normal(size=(k["B"], k["N"], k["F"])),
        adj,
        cfg,
    )
