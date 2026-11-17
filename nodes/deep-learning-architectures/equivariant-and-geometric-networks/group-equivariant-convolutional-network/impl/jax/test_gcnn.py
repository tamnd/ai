# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://1REXCC).
"""Tests of the JAX reference: the shared fixture, the shape contract, equivariance, jit and vmap."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gcnn import NAMES, config, forward, gconv, init, lift, loss_fn, param_count, rotate, rotate_p4  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["c_in"], k["c1"], k["c2"], k["k"], k["n_classes"], **kw)


def test_shared_fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    x, labels = jnp.asarray(inp["x"]), jnp.asarray(inp["labels"])
    np.testing.assert_allclose(forward(params, x, cfg), want["logits"], rtol=1e-12, atol=1e-14)
    assert float(loss_fn(params, x, labels, cfg)) == pytest.approx(float(want["loss"]), rel=1e-12)
    gp, gx = jax.grad(loss_fn, argnums=(0, 1))(params, x, labels, cfg)
    np.testing.assert_allclose(gx, want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(gp[k], want[f"d{k}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    x = jnp.zeros((k["B"], k["c_in"], k["N"], k["N"]))
    h1 = lift(x, params["W1"])
    assert list(h1.shape) == c["blocks"]["lift"]["outputs"]["h1"]
    assert list(gconv(h1, params["W2"]).shape) == c["blocks"]["gconv"]["outputs"]["h2"]
    assert list(forward(params, x, cfg).shape) == c["blocks"]["head"]["outputs"]["logits"]
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())


@pytest.mark.parametrize("mode", ["zero", "circular"])
def test_layers_are_equivariant_and_logits_invariant(mode):
    cfg = config(2, 3, 2, 3, 4, mode=mode)
    params = init(jax.random.PRNGKey(1), cfg)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 6, 6))
    h1 = lift(x, params["W1"], mode)
    for r in range(1, 4):
        np.testing.assert_allclose(lift(rotate(x, r), params["W1"], mode), rotate_p4(h1, r), atol=1e-12)
        np.testing.assert_allclose(gconv(rotate_p4(h1, r), params["W2"], mode),
                                   rotate_p4(gconv(h1, params["W2"], mode), r), atol=1e-12)
        np.testing.assert_allclose(forward(params, rotate(x, r), cfg), forward(params, x, cfg), atol=1e-12)


def test_jit_and_vmap_agree():
    cfg = config(1, 2, 3, 3, 2)
    params = init(jax.random.PRNGKey(3), cfg)
    x = jax.random.normal(jax.random.PRNGKey(4), (5, 1, 6, 6))
    batched = jax.jit(lambda p, x: forward(p, x, cfg))(params, x)
    rows = jax.vmap(lambda xi: forward(params, xi[None], cfg)[0])(x)
    np.testing.assert_allclose(batched, rows, rtol=1e-12, atol=1e-13)


def test_gradient_of_the_shared_weights_sums_the_copies():
    # The bank holds each weight 4 times, so the gradient of a weight is the sum of the gradients of its 4 copies.
    from gcnn import gconv_bank
    W = jax.random.normal(jax.random.PRNGKey(5), (2, 3, 4, 3, 3))
    G = jax.random.normal(jax.random.PRNGKey(6), (8, 12, 3, 3))
    gW = jax.grad(lambda W: jnp.sum(gconv_bank(W) * G))(W)
    copies = jax.grad(lambda W: jnp.sum(gconv_bank(W) * G))(jnp.zeros_like(W))
    np.testing.assert_allclose(gW, copies)
    assert float(jnp.sum(gW)) == pytest.approx(float(jnp.sum(G)))
