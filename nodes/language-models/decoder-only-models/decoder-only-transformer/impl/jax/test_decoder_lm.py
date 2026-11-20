# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Tests of the JAX version: the shared fixture and jax.grad against the hand-written gradients, the shape contract,
the fixed-size cache under jit, greedy decoding through lax.scan, and causality."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from decoder_lm import (  # noqa: E402
    attend,
    config,
    decode_step,
    forward,
    generate,
    init,
    init_cache,
    keys_values,
    layer_norm,
    loss,
    mlp,
    param_count,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["vocab"], k["n_ctx"], k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], **kw)


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    cfg = config()
    ids = inp["ids"]
    np.testing.assert_allclose(forward(params, cfg, ids), want["logits"], rtol=1e-12, atol=1e-13)
    value, grads = jax.value_and_grad(loss)(params, cfg, ids)
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for k, g in grads.items():
        np.testing.assert_allclose(g, want["d" + k], rtol=1e-10, atol=1e-12, err_msg=k)


def test_cache_and_greedy_decoding_match_the_fixture():
    params, inp, want = fixture()
    cfg = config()
    ids = inp["ids"]
    traces = []

    def traced_step(c, tok, t):
        traces.append(t)  # runs once per compilation, not per call
        return decode_step(params, cfg, c, tok, t)

    step = jax.jit(traced_step)
    cache, rows = init_cache(cfg, 2, 8), []
    for t in range(8):
        logits, cache = step(cache, ids[:, t], t)
        rows.append(logits)
    np.testing.assert_allclose(jnp.stack(rows, 1), want["logits_cached"], rtol=1e-12, atol=1e-13)
    assert len(traces) == 1  # one compilation for every position
    greedy = jax.jit(lambda p, prompt: generate(p, cfg, prompt, 5))(params, ids[:, :3])
    np.testing.assert_array_equal(greedy, want["greedy"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shape_contract(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        B = c["config"]["B"]
        params = init(jax.random.PRNGKey(0), cfg)
        ids = jnp.zeros((B, cfg["n_ctx"]), jnp.int32)
        assert list(forward(params, cfg, ids).shape) == c["blocks"]["head"]["outputs"]["logits"]
        assert init_cache(cfg, B, cfg["n_ctx"]).size == c["model"]["kv_cache"]


@pytest.mark.parametrize("kw", [{}, {"tied": False}, {"pos": "none"}], ids=["tied", "untied", "no-pos"])
def test_cache_matches_the_full_pass(kw):
    cfg = config(vocab=9, n_ctx=10, d_model=8, n_heads=4, d_ff=12, n_layers=3, **kw)
    params = init(jax.random.PRNGKey(1), cfg, scale=0.5)
    ids = jax.random.randint(jax.random.PRNGKey(2), (3, 10), 0, 9)
    full = forward(params, cfg, ids)
    step = jax.jit(lambda c, tok, t: decode_step(params, cfg, c, tok, t))
    cache, rows = init_cache(cfg, 3, 10), []
    for t in range(10):
        logits, cache = step(cache, ids[:, t], t)
        rows.append(logits)
    np.testing.assert_allclose(jnp.stack(rows, 1), full, rtol=1e-11, atol=1e-12)
    seq = ids[:, :4]
    for _ in range(5):
        seq = jnp.concatenate([seq, forward(params, cfg, seq)[:, -1].argmax(-1)[:, None]], 1)
    np.testing.assert_array_equal(generate(params, cfg, ids[:, :4], 5), seq[:, 4:])


def test_causality_through_the_jacobian():
    cfg = config(vocab=9, n_ctx=6, d_model=8, n_heads=2, d_ff=12, n_layers=2)
    params = init(jax.random.PRNGKey(3), cfg, scale=0.5)
    ids = jax.random.randint(jax.random.PRNGKey(4), (1, 6), 0, 9)
    x = params["wte"][ids]  # differentiate the logits with respect to the input states

    def logits_of(x):
        p = params
        z = x + p["wpe"][:6]
        allowed = jnp.tril(jnp.ones((6, 6), bool))
        for li in range(2):
            h = layer_norm(z, p[f"l{li}_ln1_g"], p[f"l{li}_ln1_b"])
            k, v = keys_values(p, li, h, 2)
            z = mlp(p, li, z + attend(p, li, h, k, v, allowed, 2))
        return layer_norm(z, p["ln_f_g"], p["ln_f_b"]) @ p["wte"].T

    np.testing.assert_allclose(logits_of(x), forward(params, cfg, ids), rtol=1e-13)
    J = jax.jacfwd(logits_of)(x)[0, :, :, 0, :, :]  # (out position, V, in position, D)
    reach = np.abs(np.asarray(J)).max(axis=(1, 3)) > 0
    np.testing.assert_array_equal(reach, np.tril(np.ones((6, 6), bool)))


def test_context_limit():
    cfg = config()
    params = init(jax.random.PRNGKey(5), cfg)
    with pytest.raises(ValueError):
        forward(params, cfg, jnp.zeros((1, 9), jnp.int32))
