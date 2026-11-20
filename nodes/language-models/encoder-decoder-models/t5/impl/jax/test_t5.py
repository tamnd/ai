# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Tests of the JAX version: the shared fixture and jax.grad against the hand-written gradients, the shape contract,
the fixed-size cache under jit, greedy decoding through lax.scan, causality through the Jacobian, and the encoder
without position biases."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from t5 import (  # noqa: E402
    KEYS,
    bucket,
    config,
    cross_keys_values,
    decode,
    decode_step,
    decoder,
    encode,
    forward,
    generate,
    init,
    init_cache,
    loss,
    param_count,
    rms_norm,
    shift_right,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    return config(*(c["config"][k] for k in KEYS), **kw)


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    cfg = config()
    src, tgt = inp["src"], inp["tgt"]
    np.testing.assert_allclose(forward(params, cfg, src, tgt), want["logits"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(encode(params, cfg, src), want["e"], rtol=1e-12, atol=1e-13)
    value, grads = jax.value_and_grad(loss)(params, cfg, src, tgt)
    assert float(value) == pytest.approx(float(want["loss"]), rel=1e-12)
    for name, g in grads.items():
        np.testing.assert_allclose(g, want["d" + name], rtol=1e-10, atol=1e-12, err_msg=name)


def test_cache_and_greedy_decoding_match_the_fixture():
    params, inp, want = fixture()
    cfg = config()
    src = inp["src"]
    e = encode(params, cfg, src)
    cross = cross_keys_values(params, cfg, e)
    ids = shift_right(inp["tgt"])
    cache = init_cache(cfg, 2, ids.shape[1])
    traces = []

    def traced_step(c, tok, t):
        traces.append(t)  # runs once per compilation, not per call
        return decode_step(params, cfg, cross, c, tok, t)

    step = jax.jit(traced_step)
    rows = []
    for t in range(ids.shape[1]):
        logits, cache = step(cache, ids[:, t], t)
        rows.append(logits)
    assert len(traces) == 1  # one compilation for every position
    np.testing.assert_allclose(jnp.stack(rows, 1), want["logits_cached"], rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(generate(params, cfg, src, 6), want["greedy"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shape_contract(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        k = c["config"]
        p = init(jax.random.PRNGKey(0), cfg)
        assert sum(v.size for v in p.values()) == c["model"]["params"]
        src = jnp.zeros((k["B"], k["S"]), jnp.int64)
        tgt = jnp.zeros((k["B"], k["T"]), jnp.int64)
        assert list(forward(p, cfg, src, tgt).shape) == c["blocks"]["head"]["outputs"]["logits"]
        assert list(encode(p, cfg, src).shape) == c["blocks"]["cross"]["inputs"]["e"]


@pytest.mark.parametrize("pos", ["relative", "none"])
def test_cache_matches_the_full_pass(pos):
    cfg = config(vocab=9, d_model=8, n_heads=4, d_kv=2, d_ff=12, n_layers=3, pos=pos)
    p = init(jax.random.PRNGKey(1), cfg)
    src = jax.random.randint(jax.random.PRNGKey(2), (2, 6), 0, 9)
    ids = jax.random.randint(jax.random.PRNGKey(3), (2, 9), 0, 9)
    e = encode(p, cfg, src)
    full = decode(p, cfg, e, ids)
    cross, cache = cross_keys_values(p, cfg, e), init_cache(cfg, 2, 12)  # room to spare: the tail stays masked
    step = jax.jit(lambda c, tok, t: decode_step(p, cfg, cross, c, tok, t))
    rows = []
    for t in range(9):
        logits, cache = step(cache, ids[:, t], t)
        rows.append(logits)
    np.testing.assert_allclose(jnp.stack(rows, 1), full, rtol=1e-12, atol=1e-13)


def test_causality_through_the_jacobian():
    cfg = config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2)
    p = init(jax.random.PRNGKey(4), cfg)
    e = encode(p, cfg, jnp.array([[1, 2, 3, 4]]))
    ids = jnp.array([[0, 5, 6, 7, 8]])

    def logits_of(y):  # the decoder from its input embeddings; the output layer keeps the embedding as it is
        return decoder(p, cfg, e, y[None])[0]

    J = jax.jacfwd(logits_of)(p["emb"][ids[0]])  # (T, V, T, M): output position, logit, input position, feature
    reach = jnp.abs(J).sum((1, 3))
    np.testing.assert_array_equal(np.triu(np.asarray(reach), 1), 0)
    assert bool((jnp.diag(reach) > 0).all())


def test_without_positions_the_encoder_sees_a_set():
    for pos, equivariant in (("none", True), ("relative", False)):
        cfg = config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2, pos=pos)
        p = init(jax.random.PRNGKey(5), cfg)
        src = jax.random.randint(jax.random.PRNGKey(6), (2, 7), 0, 9)
        perm = jax.random.permutation(jax.random.PRNGKey(7), 7)
        same = np.allclose(encode(p, cfg, src[:, perm]), encode(p, cfg, src)[:, perm], rtol=1e-12, atol=1e-13)
        assert same == equivariant


def test_buckets_and_norm():
    got = bucket(jnp.arange(-20, 21), True, 8, 16)
    want = [3] * 15 + [2] * 4 + [1, 0, 5] + [6] * 4 + [7] * 15
    np.testing.assert_array_equal(got, want)
    x = jnp.array([[3.0, 3.0, 3.0, 3.0]])
    np.testing.assert_allclose(rms_norm(x, jnp.ones(4)), 3 / jnp.sqrt(9 + 1e-6), rtol=1e-15)
