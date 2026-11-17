# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""Tests of the JAX reference: the shared fixture, the shape contract and the adjoint backward rule."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neural_ode import NAMES, config, forward, forward_adjoint, init, mse_loss, param_count

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return config(k["d"], k["d_hidden"], k["d_state"], k["n_steps"])


def test_shared_fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    x, t = jnp.asarray(inp["x"]), jnp.asarray(inp["t"])

    def loss(params, x):
        return mse_loss(forward(params, x, cfg)[:, :cfg["d"]], t)

    np.testing.assert_allclose(forward(params, x, cfg), want["y"], rtol=1e-12, atol=1e-14)
    assert float(loss(params, x)) == pytest.approx(float(want["loss"]), rel=1e-12)
    gp, gx = jax.grad(loss, argnums=(0, 1))(params, x)
    np.testing.assert_allclose(gx, want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(gp[k], want[f"d{k}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    y = forward(params, jnp.zeros((k["B"], k["d"])), cfg)
    assert list(y.shape) == c["blocks"]["solve"]["outputs"]["y"]
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())


def test_adjoint_forward_equals_direct():
    cfg = config(2, 4, d_state=3, n_steps=5)
    params = init(jax.random.PRNGKey(3), cfg)
    x = jax.random.normal(jax.random.PRNGKey(4), (4, 2))
    np.testing.assert_array_equal(forward_adjoint(params, x, cfg), forward(params, x, cfg))


def test_adjoint_gradient_converges_to_autodiff():
    # The adjoint gradient differs from autodiff through the steps by the error of the solver, which shrinks with dt.
    params = init(jax.random.PRNGKey(7), config(2, 4, d_state=3))
    x = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    c = jax.random.normal(jax.random.PRNGKey(5), (3, 3))
    gaps = []
    for n in (2, 4, 8):
        cfg = config(2, 4, d_state=3, n_steps=n)
        grads = [jax.grad(lambda p, x: jnp.sum(fn(p, x, cfg) * c), argnums=(0, 1))(params, x)
                 for fn in (forward, forward_adjoint)]
        leaves = [jax.tree_util.tree_leaves(g) for g in grads]
        gaps.append(max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(*leaves)))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


def test_jit_and_vmap_agree():
    cfg = config(2, 4, d_state=3, n_steps=3)
    params = init(jax.random.PRNGKey(9), cfg)
    x = jax.random.normal(jax.random.PRNGKey(10), (6, 2))
    batched = jax.jit(lambda p, x: forward(p, x, cfg))(params, x)
    rows = jax.vmap(lambda xi: forward(params, xi[None], cfg)[0])(x)
    np.testing.assert_allclose(batched, rows, rtol=1e-13, atol=1e-14)


def test_training_reduces_loss():
    cfg = config(2, 8, d_state=3, n_steps=4)
    params = init(jax.random.PRNGKey(8), cfg)
    x = jax.random.normal(jax.random.PRNGKey(11), (16, 2))
    t = jnp.stack([x[:, 0] * x[:, 1], jnp.sin(x[:, 0])], 1)
    loss = jax.jit(lambda p: mse_loss(forward_adjoint(p, x, cfg)[:, :2], t))
    grad = jax.jit(jax.grad(lambda p: mse_loss(forward_adjoint(p, x, cfg)[:, :2], t)))
    first = float(loss(params))
    for _ in range(200):
        params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grad(params))
    assert float(loss(params)) < 0.5 * first
