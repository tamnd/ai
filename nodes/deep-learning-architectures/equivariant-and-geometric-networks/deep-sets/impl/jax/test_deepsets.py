# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://S263A8).
"""Tests of the JAX version: the shared fixture and its gradients, the shape contract, jit, vmap over sets, the
invariance and the equivariance, and the padding."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from deepsets import POOLS, config, features, forward, init, loss, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d_in"], k["width"], k["n_layers"], k["n_classes"], **kw)


def test_shared_fixture_and_gradients():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    np.testing.assert_allclose(forward(params, cfg, inp["x"], inp["mask"]), want["logits"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(features(params, cfg, inp["x"], inp["mask"]), want["features"], rtol=1e-12, atol=1e-13)
    value, grads = jax.value_and_grad(loss, argnums=(0, 2))(params, cfg, inp["x"], inp["labels"], inp["mask"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for name, g in grads[0].items():
        np.testing.assert_allclose(g, want["d" + name], rtol=1e-10, atol=1e-12, err_msg=name)
    np.testing.assert_allclose(grads[1], want["dx"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shape_contract(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        k = c["config"]
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (k["B"], k["set_size"], k["d_in"]))
        assert list(forward(init(key, cfg), cfg, x).shape) == c["blocks"]["readout"]["outputs"]["logits"]


@pytest.mark.parametrize("head", ["set", "element"])
def test_jit_and_vmap(head):
    cfg = config(head=head)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    params = init(k1, cfg, scale=0.5)
    x = jax.random.normal(k2, (4, 6, 3))
    eager = forward(params, cfg, x)
    np.testing.assert_allclose(jax.jit(lambda p, x: forward(p, cfg, x))(params, x), eager, rtol=1e-12)
    one = jax.vmap(lambda s: forward(params, cfg, s[None])[0])(x)
    np.testing.assert_allclose(one, eager, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("p", POOLS)
def test_invariance_equivariance_and_padding(p):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(2), 4)
    x = jax.random.normal(k2, (2, 7, 3))
    perm = jax.random.permutation(k3, 7)
    cfg = config(3, 6, 3, 4, pool=p, readout=p)
    params = init(k1, cfg, 0.5)
    np.testing.assert_allclose(forward(params, cfg, x[:, perm]), forward(params, cfg, x), rtol=1e-12, atol=1e-13)
    pad = jnp.concatenate([x, jax.random.normal(k4, (2, 3, 3))], 1)
    mask = jnp.concatenate([jnp.ones((2, 7)), jnp.zeros((2, 3))], 1)
    np.testing.assert_allclose(forward(params, cfg, pad, mask), forward(params, cfg, x), rtol=1e-12, atol=1e-13)
    cfg = config(3, 6, 3, pool=p, head="element")
    params = init(k1, cfg, 0.5)
    s = forward(params, cfg, x)
    np.testing.assert_allclose(forward(params, cfg, x[:, perm]), s[:, perm], rtol=1e-12, atol=1e-13)
