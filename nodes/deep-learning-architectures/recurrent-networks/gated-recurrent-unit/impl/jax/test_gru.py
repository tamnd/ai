# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://XAGPFN).
"""Tests of the JAX reference: the shared fixture and its gradients, the shape contract, jit, vmap, and the
gradient against finite differences."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gru import cell, config, forward, init, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RESETS = ("after", "before")


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, np.load(FIXTURES / "expected-small.npz")


def loss_fn(params, x, h0, r, r_h, reset):
    h, cache = forward(params, x, h0, reset)
    return jnp.sum(h * r) + jnp.sum(cache["h_T"] * r_h)


@pytest.mark.parametrize("reset", RESETS)
def test_shared_fixture(reset):
    params, inp, want = fixture()
    sfx = "" if reset == "after" else "_before"
    h, cache = forward(params, inp["x"], inp["h0"], reset)
    np.testing.assert_allclose(h, want["h" + sfx], rtol=1e-10, atol=1e-12)
    for name in ("h_T", "r", "z", "n"):
        np.testing.assert_allclose(cache[name], want[name + sfx], rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("reset", RESETS)
def test_shared_fixture_gradients(reset):
    params, inp, want = fixture()
    sfx = "" if reset == "after" else "_before"
    dp, dx, dh0 = jax.grad(loss_fn, argnums=(0, 1, 2))(params, inp["x"], inp["h0"], inp["r"], inp["r_h"], reset)
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
    for name, shape in blocks["cell"]["internals"].items():
        assert list(cache[name].shape) == shape, name


def test_jit_and_vmap_agree():
    params, inp, want = fixture()
    fast = jax.jit(lambda p, x, h0: forward(p, x, h0)[0])
    np.testing.assert_allclose(fast(params, inp["x"], inp["h0"]), want["h"], rtol=1e-10, atol=1e-12)
    # vmap over the batch, one sequence at a time with a batch axis of 1, gives the batched result.
    one = jax.vmap(lambda x, h0: forward(params, x[None], h0[None])[0][0])
    np.testing.assert_allclose(one(inp["x"], inp["h0"]), want["h"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("reset", RESETS)
def test_gradient_against_finite_differences(reset):
    params = init(jax.random.PRNGKey(2), config(2, 3))
    key = jax.random.PRNGKey(3)
    kx, kh, kr, ks = jax.random.split(key, 4)
    x, h0 = jax.random.normal(kx, (2, 4, 2)), 0.5 * jax.random.normal(kh, (2, 3))
    r, r_h = jax.random.normal(kr, (2, 4, 3)), jax.random.normal(ks, (2, 3))
    grads = jax.grad(loss_fn)(params, x, h0, r, r_h, reset)
    for name, arr in params.items():
        flat = np.asarray(arr).ravel()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            eps = 1e-6
            up, down = flat.copy(), flat.copy()
            up[i] += eps
            down[i] -= eps
            f = lambda v: float(loss_fn(params | {name: jnp.asarray(v.reshape(arr.shape))}, x, h0, r, r_h, reset))  # noqa: E731
            num[i] = (f(up) - f(down)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(grads[name]).ravel(), num, rtol=1e-6, atol=1e-8, err_msg=name)


def test_worked_example_leaky_integrator():
    p = {
        "W": jnp.array([[0.0, 0.0, 1.0]]),
        "U": jnp.zeros((1, 3)),
        "b_i": jnp.array([0.0, 2.0, 0.0]),
        "b_h": jnp.zeros(3),
    }
    h, _ = forward(p, jnp.ones((1, 10, 1)), jnp.zeros((1, 1)))
    np.testing.assert_allclose(h[0, 0, 0], 0.0907842488, rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[0, 7, 0], 0.4857076084, rtol=0, atol=1e-10)
    # d h_10 / d h_0 = z^10 through jax.grad of the scan.
    g = jax.grad(lambda h0: forward(p, jnp.ones((1, 10, 1)), h0)[1]["h_T"][0, 0])(jnp.zeros((1, 1)))
    np.testing.assert_allclose(g[0, 0], 0.2810338623, rtol=0, atol=1e-10)


def test_unknown_reset_is_an_error():
    params = init(jax.random.PRNGKey(0), config())
    with pytest.raises(ValueError):
        cell(params, jnp.zeros((1, 3)), jnp.zeros((1, 4)), "never")
