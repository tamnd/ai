# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""Tests of the PyTorch version: the shared fixture, the shape contract, the gradient, an edge-list oracle and learning."""

import json
import pathlib

import numpy as np
import pytest
import torch

from graph_neural_network import (
    GNN,
    config,
    gcn_layer,
    normalized_adjacency,
    param_count,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["n_layers"], k["classes"])


def random_adj(gen, B, N):
    upper = torch.triu((torch.rand(B, N, N, generator=gen) < 0.4).double(), 1)
    return upper + upper.transpose(-1, -2)


def test_shared_fixture():
    flat = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    model = GNN.from_params(flat, cfg_of(SHAPES[0]))
    logits, cache = model(inp["x"], inp["adj"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["graph"], want["graph"], rtol=1e-10, atol=1e-12)
    for i, h in enumerate(cache["h"]):
        torch.testing.assert_close(h, want[f"h/{i}"], rtol=1e-10, atol=1e-12)
    a_hat = normalized_adjacency(inp["adj"])
    torch.testing.assert_close(a_hat, want["adj_hat"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gcn_layer(inp["W_gcn"], inp["x"], a_hat), want["gcn"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    gen = torch.Generator().manual_seed(0)
    model = GNN(cfg_of(c), gen)
    logits, cache = model(
        torch.randn(k["B"], k["N"], k["F"], generator=gen),
        random_adj(gen, k["B"], k["N"]),
    )
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["graph"].shape) == blocks["readout"]["outputs"]["graph"]
    assert list(cache["h"][0].shape) == blocks["layer_first"]["outputs"]["h"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    assert sum(p.numel() for p in GNN(cfg).parameters()) == c["model"]["params"]


def test_gradient():
    # Autograd through the module against finite differences, via a functional copy that takes the weights as inputs.
    gen = torch.Generator().manual_seed(1)
    model = GNN(config(3, 4, 2, 2), gen)
    with torch.no_grad():  # nonzero biases keep the ReLU inputs of isolated nodes off the kink at 0
        for layer in model.layers:
            layer.b.normal_(generator=gen)
    adj = random_adj(gen, 2, 5)
    x = torch.randn(2, 5, 3, generator=gen, requires_grad=True)
    names = [f"layers.{i}.{n}" for i in range(2) for n in ("W_self", "W_nbr", "b")] + [
        "W",
        "b",
    ]
    ps = [dict(model.named_parameters())[n].detach().clone().requires_grad_(True) for n in names]

    def g(x, *ps):
        h = x
        for i in range(2):
            W_self, W_nbr, b = ps[3 * i : 3 * i + 3]
            h = torch.relu(h @ W_self + adj @ (h @ W_nbr) + b)
        return h.sum(-2) @ ps[6] + ps[7]

    torch.testing.assert_close(model(x, adj)[0], g(x, *ps))
    assert torch.autograd.gradcheck(g, (x, *ps))
    model(x, adj)[0].sum().backward()
    g(x, *ps).sum().backward()
    for n, q in zip(names, ps, strict=True):
        torch.testing.assert_close(dict(model.named_parameters())[n].grad, q.grad)
    W = torch.randn(3, 4, generator=gen, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, W: gcn_layer(W, x, normalized_adjacency(adj)), (x, W))


def test_agrees_with_edge_list_scatter():
    # A second oracle: the messages are summed edge by edge with index_add_, as sparse graph libraries do.
    gen = torch.Generator().manual_seed(2)
    model = GNN(config(3, 5, 3, 2), gen)
    with torch.no_grad():
        for layer in model.layers:
            layer.b.normal_(generator=gen)
    adj = random_adj(gen, 1, 8)
    x = torch.randn(1, 8, 3, generator=gen)
    dst, src = adj[0].nonzero(as_tuple=True)
    h = x[0]
    for layer in model.layers:
        m = torch.zeros(8, layer.b.shape[0]).index_add_(0, dst, h[src] @ layer.W_nbr)
        h = torch.relu(h @ layer.W_self + m + layer.b)
    torch.testing.assert_close(model(x, adj)[0][0], h.sum(0) @ model.W + model.b)


def test_permutation_equivariance():
    gen = torch.Generator().manual_seed(3)
    model = GNN(config(3, 5, 2, 2), gen)
    adj = random_adj(gen, 1, 7)
    x = torch.randn(1, 7, 3, generator=gen)
    P = torch.eye(7)[torch.randperm(7, generator=gen)]
    y, cache = model(x, adj)
    y_p, cache_p = model(P @ x, P @ adj @ P.T)
    torch.testing.assert_close(cache_p["h"][-1], P @ cache["h"][-1])
    torch.testing.assert_close(y_p, y)


def test_learns():
    # Fit cycles against paths of 4 to 7 nodes, with a constant feature: a sum readout of one layer sees the degrees.
    adjs, labels = [], []
    for n in range(4, 8):
        for closed in (0, 1):
            a = torch.zeros(7, 7)
            for i in range(n - 1 + closed):
                a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1.0
            adjs.append(a)
            labels.append(closed)
    adj = torch.stack(adjs)
    x = torch.zeros(8, 7, 1)
    for i, n in enumerate(range(4, 8)):
        x[2 * i : 2 * i + 2, :n] = 1.0  # padded nodes have feature 0 and no edges
    y = torch.tensor(labels)
    torch.manual_seed(0)
    model = GNN(config(1, 8, 2, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x, adj)[0], y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
