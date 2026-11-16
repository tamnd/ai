# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://NEW-convolutional-neural-network).
"""Tests of the JAX version: the worked example, the shared fixture, equivariance and the gradient."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from cnn import apply, config, conv2d

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
NAMES = ["conv1", "pool1", "conv2", "pool2"]


def cfg_of(c):
    k = c["config"]
    return config(k["channels"], k["c1"], k["c2"], k["f1"], k["f2"], k["classes"], k["size"])


def test_worked_example():
    x = jnp.arange(9.0).reshape(1, 1, 3, 3)
    w = jnp.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
    np.testing.assert_array_equal(conv2d(x, w)[0, 0], [[27.0, 37.0], [57.0, 67.0]])
    np.testing.assert_array_equal(conv2d(x, w, pad=1, stride=2)[0, 0], [[0.0, 11.0], [30.0, 67.0]])


def test_shared_fixture():
    flat = dict(np.load(FIXTURES / "params-small.npz"))
    inputs, expected = dict(np.load(FIXTURES / "input-small.npz")), dict(np.load(FIXTURES / "expected-small.npz"))
    params = {}
    for path, v in flat.items():
        layer, leaf = path.split("/")
        params.setdefault(layer, {})[leaf] = jnp.asarray(v)
    cfg = cfg_of(next(c for c in SHAPES if c["kind"] == "example"))
    logits, cache = apply(params, jnp.asarray(inputs["x"]), cfg)
    np.testing.assert_allclose(logits, expected["logits"], rtol=1e-10, atol=1e-12)
    for name in NAMES:
        np.testing.assert_allclose(cache[name], expected[name], rtol=1e-10, atol=1e-12)
    lx, lw, lb = (jnp.asarray(inputs[k]) for k in ("layer_x", "layer_w", "layer_b"))
    np.testing.assert_allclose(conv2d(lx, lw, lb, 2, 1), expected["layer_zeros"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(conv2d(lx, lw, lb, 2, 1, "circular"), expected["layer_circular"], rtol=1e-10, atol=1e-12)


def test_equivariance_circular():
    kx, kw = jax.random.split(jax.random.PRNGKey(0))
    x, w = jax.random.normal(kx, (1, 2, 8, 8)), jax.random.normal(kw, (3, 2, 3, 3))
    for t in [(1, 0), (2, -3), (7, 5)]:
        np.testing.assert_allclose(conv2d(jnp.roll(x, t, (2, 3)), w, pad=1, mode="circular"),
                                   jnp.roll(conv2d(x, w, pad=1, mode="circular"), t, (2, 3)), rtol=1e-12, atol=1e-12)


def test_gradient_numeric():
    kx, kw = jax.random.split(jax.random.PRNGKey(1))
    x, w = jax.random.normal(kx, (1, 2, 6, 6)), jax.random.normal(kw, (3, 2, 3, 3))
    loss = lambda w_: jnp.sum(jnp.sin(conv2d(x, w_, stride=2, pad=1)))  # noqa: E731
    g = jax.grad(loss)(w)
    eps, idx = 1e-6, (1, 0, 2, 1)
    e = jnp.zeros_like(w).at[idx].set(eps)
    np.testing.assert_allclose(g[idx], (loss(w + e) - loss(w - e)) / (2 * eps), rtol=1e-6)


def test_shape_contract():
    for c in SHAPES:
        cfg = cfg_of(c)
        x = jnp.zeros(c["blocks"]["conv1"]["inputs"]["x"])
        params = {}
        for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
            b = c["blocks"][name]
            if name.startswith("conv"):
                d, cin = b["outputs"]["y"][1], b["inputs"]["x"][1]
                params[name] = {"w": jnp.zeros((d, cin, 5, 5)), "b": jnp.zeros(d)}
            else:
                n_in, n_out = b["inputs"]["x"][1], list(b["outputs"].values())[0][1]
                params[name] = {"w": jnp.zeros((n_in, n_out)), "b": jnp.zeros(n_out)}
        assert sum(v.size for p in params.values() for v in p.values()) == c["model"]["params"]
        logits, cache = apply(params, x, cfg)
        assert list(logits.shape) == c["blocks"]["fc3"]["outputs"]["logits"]
        for name in NAMES:
            assert list(cache[name].shape) == c["blocks"][name]["outputs"]["y"], (c["name"], name)
