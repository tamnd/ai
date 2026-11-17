# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Tests of the JAX reference: the shared fixture, the shape contract, jit, vmap, the gradient against finite
differences, and a few steps of training that lower the loss."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from memn2n import config, cross_entropy, forward, init, param_count, zero_null  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("vocab", "dim", "hops", "mem", "sent_len")


def cfg_of(c, **kw):
    return config(**{k: c["config"][k] for k in KEYS}, **kw)


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, np.load(FIXTURES / "expected-small.npz")


def test_shared_fixture():
    params, inp, want = fixture()
    cfg = cfg_of(SHAPES[0])
    logits, attn = forward(params, inp["story"], inp["query"], cfg, trace=True)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(attn, want["attention"], rtol=1e-10, atol=1e-12)
    assert float(cross_entropy(logits, inp["answer"])) == pytest.approx(float(want["loss"]), rel=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    story = jnp.zeros((k["B"], k["N"], k["J"]), dtype=jnp.int32)
    query = jnp.zeros((k["B"], k["J"]), dtype=jnp.int32)
    logits, attn = forward(params, story, query, cfg, trace=True)
    assert list(logits.shape) == c["blocks"]["answer"]["outputs"]["logits"]
    assert list(attn.shape[1:]) == c["blocks"]["hop"]["outputs"]["attention"]


def test_jit_and_vmap_agree():
    params, inp, want = fixture()
    cfg = cfg_of(SHAPES[0])
    fast = jax.jit(lambda p, s, q: forward(p, s, q, cfg))
    np.testing.assert_allclose(fast(params, inp["story"], inp["query"]), want["logits"], rtol=1e-10, atol=1e-12)
    one = jax.vmap(lambda s, q: forward(params, s[None], q[None], cfg)[0])
    np.testing.assert_allclose(one(inp["story"], inp["query"]), want["logits"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("tying", ["adjacent", "layerwise"])
def test_gradient_matches_finite_differences(tying):
    _, inp, _ = fixture()
    cfg = config(vocab=12, dim=6, hops=3, mem=5, sent_len=4, tying=tying)
    params = init(jax.random.PRNGKey(1), cfg, sigma=0.5)

    def loss(p):
        return cross_entropy(forward(p, inp["story"], inp["query"], cfg), inp["answer"])

    g = jax.grad(loss)(params)
    rng = np.random.default_rng(0)
    for name in sorted(params):
        d = jnp.asarray(rng.normal(size=params[name].shape))
        eps = 1e-6
        plus = loss(dict(params, **{name: params[name] + eps * d}))
        minus = loss(dict(params, **{name: params[name] - eps * d}))
        assert float(jnp.vdot(g[name], d)) == pytest.approx(float(plus - minus) / (2 * eps), rel=1e-5, abs=1e-8)


def test_training_lowers_the_loss():
    _, inp, _ = fixture()
    cfg = cfg_of(SHAPES[0])
    params = init(jax.random.PRNGKey(2), cfg)

    def loss(p):
        return cross_entropy(forward(p, inp["story"], inp["query"], cfg), inp["answer"])

    step = jax.jit(lambda p: jax.tree_util.tree_map(lambda w, g: w - 0.05 * g, p, jax.grad(loss)(p)))
    start = float(loss(params))
    for _ in range(50):
        params = zero_null(step(params), cfg)
    assert float(loss(params)) < 0.5 * start
