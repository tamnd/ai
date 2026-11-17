# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://NEW-multilayer-perceptron).
"""Tests of the JAX reference: the shared fixture, the shape contract, check_grads, XOR, jit and training."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from multilayer_perceptron import apply, config, init, mse_loss, param_count, sgd_step  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
X_XOR = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T_XOR = jnp.array([[0.0], [1.0], [1.0], [0.0]])


def cfg_of(c, activation="tanh"):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["d_out"], k["n_hidden"], activation)


def test_shared_fixture():
    raw = dict(np.load(FIXTURES / "params-small.npz"))
    params = {k: [jnp.asarray(raw[f"{k}/{i}"]) for i in range(3)] for k in ("W", "b")}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    x, t = jnp.asarray(inp["x"]), jnp.asarray(inp["t"])
    np.testing.assert_allclose(apply(params, x, cfg), want["y"], rtol=1e-10, atol=1e-12)
    loss, (gp, gx) = jax.value_and_grad(mse_loss, argnums=(0, 1))(params, x, t, cfg)
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-10)
    np.testing.assert_allclose(gx, want["dx"], rtol=1e-10, atol=1e-12)
    for k in ("W", "b"):
        for i in range(3):
            np.testing.assert_allclose(gp[k][i], want[f"d{k}/{i}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    y = apply(params, jnp.zeros((k["B"], k["d_in"])), cfg)
    assert list(y.shape) == c["blocks"]["output"]["outputs"]["y"]
    assert param_count(cfg) == c["model"]["params"] == sum(a.size for a in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu"])
def test_check_grads(activation):
    cfg = config(3, 4, 2, 2, activation)
    params = init(jax.random.PRNGKey(1), cfg)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
    t = jax.random.normal(jax.random.PRNGKey(3), (4, 2))
    check_grads(lambda p, xx: mse_loss(p, xx, t, cfg), (params, x), order=1, modes=["rev"])


def test_xor_network():
    params = {"W": [jnp.array([[1.0, 1.0], [1.0, 1.0]]), jnp.array([[1.0, -2.0]])], "b": [jnp.array([0.0, -1.0]), jnp.array([0.0])]}
    np.testing.assert_array_equal(apply(params, X_XOR, config(2, 2, 1, 1, "relu")), T_XOR)


def test_linear_collapse():
    cfg = config(3, 5, 2, 2, "identity")
    p = init(jax.random.PRNGKey(4), cfg)
    p["b"] = [jax.random.normal(jax.random.PRNGKey(5 + i), b.shape) for i, b in enumerate(p["b"])]
    x = jax.random.normal(jax.random.PRNGKey(9), (6, 3))
    W = p["W"][2] @ p["W"][1] @ p["W"][0]
    b = p["W"][2] @ (p["W"][1] @ p["b"][0] + p["b"][1]) + p["b"][2]
    np.testing.assert_allclose(apply(p, x, cfg), x @ W.T + b, rtol=1e-12, atol=1e-12)


def test_jit_and_batch_independence():
    cfg = config(3, 5, 2, 2, "gelu")
    p = init(jax.random.PRNGKey(6), cfg)
    x = jax.random.normal(jax.random.PRNGKey(7), (5, 3))
    y = jax.jit(lambda pp, xx: apply(pp, xx, cfg))(p, x)
    np.testing.assert_allclose(y, jnp.concatenate([apply(p, x[i : i + 1], cfg) for i in range(5)]), rtol=1e-12, atol=1e-14)


def test_training_reduces_loss():
    cfg = config(2, 4, 1, 1, "tanh")
    params = init(jax.random.PRNGKey(0), cfg)
    grad = jax.jit(jax.value_and_grad(lambda p: mse_loss(p, X_XOR, T_XOR, cfg)))
    first = float(grad(params)[0])
    for _ in range(500):
        _, g = grad(params)
        params = sgd_step(params, g, 0.5)
    assert float(mse_loss(params, X_XOR, T_XOR, cfg)) < 0.01 * first


def test_config_errors():
    with pytest.raises(ValueError):
        config(3, 4, 0, 1)
    with pytest.raises(ValueError):
        config(3, 4, 2, 1, "softplus")
