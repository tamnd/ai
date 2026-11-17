# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Tests of the JAX Mamba: the shared fixture and its gradients, the associative scan against the sequential one,
the step mode, jit and the parameter count."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mamba import combine, config, forward, init, init_state, loss, param_count, param_shapes, step  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]
SIZES = ("vocab", "d_model", "n_layers", "d_state", "d_conv", "expand", "dt_rank")


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIX / "params-small.npz").items()}
    return params, jnp.asarray(np.load(FIX / "input-small.npz")["tokens"]), np.load(FIX / "expected-small.npz")


@pytest.mark.parametrize("scan", ["associative", "sequential"])
def test_fixture(scan):
    params, tok, exp = fixture()
    cfg = config()
    logits, hss = jax.jit(lambda p, t: forward(p, cfg, t, scan, states=True))(params, tok[:, :-1])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.allclose(hss[-1][:, -1], exp["last_state"], atol=1e-12)
    value, grads = jax.jit(jax.value_and_grad(lambda p, t: loss(p, cfg, t, scan)))(params, tok)
    assert np.isclose(value, exp["loss"], atol=1e-12)
    for k, g in grads.items():
        assert np.allclose(g, exp["d" + k], atol=1e-11), k


def test_combine_is_associative():
    ks = jax.random.split(jax.random.PRNGKey(0), 6)
    e = [jax.random.normal(k, (3,)) for k in ks]
    a, b, c = (e[0], e[1]), (e[2], e[3]), (e[4], e[5])
    for u, v in zip(combine(combine(a, b), c), combine(a, combine(b, c))):
        assert np.allclose(u, v, atol=1e-14)


@pytest.mark.parametrize("disc", ["simple", "zoh"])
def test_step_mode_equals_forward(disc):
    cfg = config(disc=disc, d_conv=4)
    p = init(jax.random.PRNGKey(1), cfg, scale=0.4)
    tok = jax.random.randint(jax.random.PRNGKey(2), (2, 11), 0, cfg["vocab"])
    want = forward(p, cfg, tok)
    one = jax.jit(lambda p, t, s: step(p, cfg, t, s))
    state, out = init_state(cfg, 2), []
    for t in range(tok.shape[1]):
        o, state = one(p, tok[:, t], state)
        out.append(o)
    assert np.allclose(jnp.stack(out, 1), want, atol=1e-12)


def test_jit_and_long_sequences():
    cfg = config(d_model=16, d_state=8)
    p = init(jax.random.PRNGKey(3), cfg, scale=0.3)
    tok = jax.random.randint(jax.random.PRNGKey(4), (1, 129), 0, cfg["vocab"])
    a = jax.jit(lambda p, t: forward(p, cfg, t))(p, tok)
    b = jax.jit(lambda p, t: forward(p, cfg, t, "sequential"))(p, tok)
    assert np.allclose(a, b, atol=1e-10)
    assert np.isfinite(jax.jit(jax.grad(lambda p, t: loss(p, cfg, t)))(p, tok)["l0_A_log"]).all()


def test_param_count():
    for kw in ({}, {"d_model": 20, "d_state": 7, "d_conv": 4}, {"vocab": 50280, "d_model": 2560, "n_layers": 64,
                                                               "d_state": 16, "d_conv": 4}):  # fmt: skip
        cfg = config(**kw)
        assert param_count(cfg) == sum(int(np.prod(s)) for s in param_shapes(cfg).values())
    assert param_count(config(vocab=50280, d_model=2560, n_layers=64, d_state=16, d_conv=4)) == 2_768_345_600


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    k = c["config"]
    cfg = config(**{n: k[n] for n in SIZES})
    assert param_count(cfg) == c["model"]["params"]
    assert sum(int(a.size + b.size) for a, b in init_state(cfg, k["B"])) == c["model"]["recurrent_state"]
    if c["kind"] != "named":
        p = init(jax.random.PRNGKey(0), cfg)
        tok = jax.random.randint(jax.random.PRNGKey(1), (k["B"], k["T"]), 0, cfg["vocab"])
        assert list(forward(p, cfg, tok).shape) == c["blocks"]["head"]["outputs"]["logits"]


def test_default_config_is_the_example():
    assert config() == config(**{n: SHAPES[0]["config"][n] for n in SIZES})
