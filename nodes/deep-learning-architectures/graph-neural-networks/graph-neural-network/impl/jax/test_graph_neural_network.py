# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://NEW-graph-neural-network).
"""Tests of the JAX version: the shared fixture, jit, the shape contract, the gradient, a segment-sum oracle and learning."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from graph_neural_network import (
    apply,
    config,
    from_flat,
    gcn_layer,
    init,
    normalized_adjacency,
    param_count,
)
from jax.flatten_util import ravel_pytree

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["n_layers"], k["classes"])


def random_adj(rng, B, N):
    upper = np.triu(rng.random((B, N, N)) < 0.4, 1)
    return jnp.asarray((upper | upper.transpose(0, 2, 1)).astype(float))


def test_shared_fixture():
    params = from_flat(dict(np.load(FIXTURES / "params-small.npz")))
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    logits, cache = apply(params, inp["x"], inp["adj"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cache["graph"], want["graph"], rtol=1e-10, atol=1e-12)
    for i, h in enumerate(cache["h"]):
        np.testing.assert_allclose(h, want[f"h/{i}"], rtol=1e-10, atol=1e-12)
    a_hat = normalized_adjacency(inp["adj"])
    np.testing.assert_allclose(a_hat, want["adj_hat"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gcn_layer(inp["W_gcn"], inp["x"], a_hat), want["gcn"], rtol=1e-10, atol=1e-12)


def test_jit_gives_the_same_logits():
    params = from_flat(dict(np.load(FIXTURES / "params-small.npz")))
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    eager = apply(params, inp["x"], inp["adj"])[0]
    np.testing.assert_allclose(jax.jit(apply)(params, inp["x"], inp["adj"])[0], eager, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(jax.random.PRNGKey(0), cfg_of(c))
    logits, cache = apply(
        params,
        jnp.asarray(rng.normal(size=(k["B"], k["N"], k["F"]))),
        random_adj(rng, k["B"], k["N"]),
    )
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["graph"].shape) == blocks["readout"]["outputs"]["graph"]
    assert list(cache["h"][0].shape) == blocks["layer_first"]["outputs"]["h"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    leaves = jax.tree_util.tree_leaves(init(jax.random.PRNGKey(0), cfg))
    assert sum(v.size for v in leaves) == c["model"]["params"]


def test_gradient():
    # jax.grad against central finite differences on every parameter entry; the noise moves the zero biases off
    # the ReLU kink that isolated nodes would otherwise sit on.
    rng = np.random.default_rng(1)
    params = init(jax.random.PRNGKey(1), config(3, 4, 2, 2))
    params = jax.tree_util.tree_map(lambda v: v + 0.1 * jnp.asarray(rng.normal(size=v.shape)), params)
    adj = random_adj(rng, 2, 5)
    x = jnp.asarray(rng.normal(size=(2, 5, 3)))
    t = jnp.asarray(rng.normal(size=(2, 2)))

    def loss(p):
        return jnp.sum(apply(p, x, adj)[0] * t)

    g = jax.grad(loss)(params)
    loss_j = jax.jit(loss)
    flat, unravel = ravel_pytree(params)
    gflat, _ = ravel_pytree(g)
    eps = 1e-6
    for i in range(flat.size):
        e = jnp.zeros_like(flat).at[i].set(eps)
        fd = (loss_j(unravel(flat + e)) - loss_j(unravel(flat - e))) / (2 * eps)
        assert abs(fd - gflat[i]) <= 1e-6 * max(abs(fd), abs(gflat[i]), 1e-8) + 1e-8, i


def test_agrees_with_segment_sum():
    # A second oracle: the messages are summed edge by edge with jax.ops.segment_sum.
    rng = np.random.default_rng(2)
    params = init(jax.random.PRNGKey(2), config(3, 5, 3, 2))
    adj = random_adj(rng, 1, 8)
    x = jnp.asarray(rng.normal(size=(1, 8, 3)))
    dst, src = np.nonzero(np.asarray(adj[0]))
    h = x[0]
    for p in params["layers"]:
        m = jax.ops.segment_sum(h[src] @ p["W_nbr"], dst, num_segments=8)
        h = jax.nn.relu(h @ p["W_self"] + m + p["b"])
    np.testing.assert_allclose(
        apply(params, x, adj)[0][0],
        h.sum(0) @ params["head"]["W"] + params["head"]["b"],
    )


def test_learns():
    # Fit cycles against paths of 4 to 7 nodes, as in the PyTorch test.
    adj = np.zeros((8, 7, 7))
    x = np.zeros((8, 7, 1))
    y = np.zeros(8, dtype=int)
    for i, n in enumerate(range(4, 8)):
        for closed in (0, 1):
            j = 2 * i + closed
            for v in range(n - 1 + closed):
                adj[j, v, (v + 1) % n] = adj[j, (v + 1) % n, v] = 1.0
            x[j, :n] = 1.0
            y[j] = closed
    adj, x, y = jnp.asarray(adj), jnp.asarray(x), jnp.asarray(y)
    params = init(jax.random.PRNGKey(0), config(1, 8, 2, 2))

    def loss(p):
        logits = apply(p, x, adj)[0]
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(8), y])

    @jax.jit
    def step(p, m, v, t):  # Adam, lr 0.01, written out
        g = jax.grad(loss)(p)
        m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
        upd = lambda p, m, v: p - 0.01 * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        return jax.tree_util.tree_map(upd, p, m, v), m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for t in range(1, 501):
        params, m, v = step(params, m, v, t)
    assert loss(params) < 0.05
