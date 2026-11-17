# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Tests of the JAX reference: the shared fixture and its gradients, the shape contract, jit, vmap, and the
gradient against finite differences."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rnn import NONLINEARITIES, cell, config, forward, init, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, np.load(FIXTURES / "expected-small.npz")


def loss_fn(params, x, h0, r, r_h, nl):
    h, cache = forward(params, x, h0, nl)
    return jnp.sum(h * r) + jnp.sum(cache["h_T"] * r_h)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_fixture(nl):
    params, inp, want = fixture()
    sfx = "" if nl == "tanh" else "_relu"
    h, cache = forward(params, inp["x"], inp["h0"], nl)
    np.testing.assert_allclose(h, want["h" + sfx], rtol=1e-10, atol=1e-12)
    for name in ("h_T", "a"):
        np.testing.assert_allclose(cache[name], want[name + sfx], rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_fixture_gradients(nl):
    params, inp, want = fixture()
    sfx = "" if nl == "tanh" else "_relu"
    dp, dx, dh0 = jax.grad(loss_fn, argnums=(0, 1, 2))(params, inp["x"], inp["h0"], inp["r"], inp["r_h"], nl)
    got = {"dx": dx, "dh0": dh0} | {"d" + k: v for k, v in dp.items()}
    for name, v in got.items():
        np.testing.assert_allclose(v, want[name + sfx], rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shape_contract(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    params = init(jax.random.PRNGKey(0), cfg)
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    if k["H"] > 300:
        return
    h, cache = forward(params, jnp.zeros((k["B"], k["T"], k["I"])))
    assert list(h.shape) == blocks["layer"]["outputs"]["h"]
    assert list(cache["h_T"].shape) == blocks["layer"]["outputs"]["h_T"]
    assert list(cache["a"].shape) == blocks["step"]["internals"]["a"]


def test_jit_and_vmap_agree():
    params, inp, want = fixture()
    fast = jax.jit(lambda p, x, h0: forward(p, x, h0)[0])
    np.testing.assert_allclose(fast(params, inp["x"], inp["h0"]), want["h"], rtol=1e-10, atol=1e-12)
    # vmap over the batch, one sequence at a time with a batch axis of 1, gives the batched result.
    one = jax.vmap(lambda x, h0: forward(params, x[None], h0[None])[0][0])
    np.testing.assert_allclose(one(inp["x"], inp["h0"]), want["h"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_gradient_against_finite_differences(nl):
    params = init(jax.random.PRNGKey(2), config(2, 3))
    kx, kh, kr, ks = jax.random.split(jax.random.PRNGKey(3), 4)
    x, h0 = jax.random.normal(kx, (2, 4, 2)), 0.5 * jax.random.normal(kh, (2, 3))
    r, r_h = jax.random.normal(kr, (2, 4, 3)), jax.random.normal(ks, (2, 3))
    assert float(jnp.min(jnp.abs(forward(params, x, h0, nl)[1]["a"]))) > 1e-4  # away from the kink of relu
    grads = jax.grad(loss_fn)(params, x, h0, r, r_h, nl)
    for name, arr in params.items():
        flat = np.asarray(arr).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            eps = 1e-6
            up, down = flat.copy(), flat.copy()
            up[i] += eps
            down[i] -= eps
            f = lambda v: float(loss_fn(params | {name: jnp.asarray(v.reshape(arr.shape))}, x, h0, r, r_h, nl))  # noqa: E731
            num[i] = (f(up) - f(down)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(grads[name]).ravel(), num, rtol=1e-6, atol=1e-8, err_msg=name)


def test_worked_example():
    p = {"W": jnp.array([[1.0]]), "U": jnp.array([[0.9]]), "b_i": jnp.zeros(1), "b_h": jnp.zeros(1)}
    x = jnp.zeros((1, 10, 1)).at[0, 0, 0].set(1.0)
    for nl, h10, grad in (("tanh", 0.1892358120, 0.0419487706), ("relu", 0.9**9, 0.9**9)):
        _, cache = forward(p, x, jnp.zeros((1, 1)), nl)
        np.testing.assert_allclose(cache["h_T"][0, 0], h10, rtol=0, atol=1e-10)
        g = jax.grad(lambda x: forward(p, x, jnp.zeros((1, 1)), nl)[1]["h_T"][0, 0])(x)
        np.testing.assert_allclose(g[0, 0, 0], grad, rtol=0, atol=1e-10)


def test_identity_jacobian_of_the_irnn():
    # With U = I, zero biases, no input and a positive state, d h_T / d h_0 is the identity for any T.
    H = 5
    p = {"W": jnp.zeros((1, H)), "U": jnp.eye(H), "b_i": jnp.zeros(H), "b_h": jnp.zeros(H)}
    h0 = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (H,))) + 0.1
    J = jax.jacobian(lambda h0: forward(p, jnp.zeros((1, 100, 1)), h0[None], "relu")[1]["h_T"][0])(h0)
    np.testing.assert_array_equal(J, np.eye(H))


def test_unknown_nonlinearity_is_an_error():
    params = init(jax.random.PRNGKey(0), config())
    with pytest.raises(ValueError):
        cell(params, jnp.zeros((1, 3)), jnp.zeros((1, 4)), "gelu")
