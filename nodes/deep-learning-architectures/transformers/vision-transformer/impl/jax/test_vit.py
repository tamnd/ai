# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Tests of the JAX version: the shared fixture and its gradients, the shape contract, jit, vmap, and the bag of
patches without position embeddings."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vit import config, forward, init, loss, n_tokens, param_count, patchify  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["image_size"], k["patch_size"], k["channels"], k["d_model"], k["n_heads"], k["d_ff"],
                  k["n_layers"], k["n_classes"], **kw)  # fmt: skip


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    cfg = config()
    np.testing.assert_allclose(forward(params, cfg, inp["images"]), want["logits"], rtol=1e-12, atol=1e-13)
    value, grads = jax.value_and_grad(loss, argnums=(0, 2))(params, cfg, inp["images"], inp["labels"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for name, g in grads[0].items():
        np.testing.assert_allclose(g, want["d" + name], rtol=1e-10, atol=1e-12, err_msg=name)
    np.testing.assert_allclose(grads[1], want["dimages"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shape_contract(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    assert n_tokens(cfg) == c["model"]["tokens"]
    if c["kind"] != "named":
        k = c["config"]
        key = jax.random.PRNGKey(0)
        params = init(key, cfg)
        images = jax.random.normal(key, (k["B"], k["S"], k["S"], k["C"]))
        assert list(forward(params, cfg, images).shape) == c["blocks"]["head"]["outputs"]["logits"]
        assert list(patchify(images, k["P"]).shape) == c["blocks"]["embed"]["internals"]["patches"]


@pytest.mark.parametrize("kw", [{}, {"pool": "gap", "pos": "none", "head_hidden": 3}], ids=["cls", "gap-no-pos"])
def test_jit_and_vmap(kw):
    cfg = config(**kw)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    params = init(k1, cfg, scale=0.5)
    images = jax.random.normal(k2, (4, 8, 8, 3))
    eager = forward(params, cfg, images)
    np.testing.assert_allclose(jax.jit(lambda p, x: forward(p, cfg, x))(params, images), eager, rtol=1e-12)
    one = jax.vmap(lambda x: forward(params, cfg, x[None])[0])(images)
    np.testing.assert_allclose(one, eager, rtol=1e-12, atol=1e-13)


def test_no_positions_means_a_bag_of_patches():
    cfg = config(6, 2, 2, 8, 2, 12, 2, 4, pos="none")
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    params = init(k1, cfg, scale=0.5)
    images = jax.random.normal(k2, (3, 6, 6, 2))
    perm = jax.random.permutation(k3, 9)
    grid = images.reshape(3, 3, 2, 3, 2, 2).transpose(0, 1, 3, 2, 4, 5).reshape(3, 9, 2, 2, 2)[:, perm]
    shuffled = grid.reshape(3, 3, 3, 2, 2, 2).transpose(0, 1, 3, 2, 4, 5).reshape(3, 6, 6, 2)
    np.testing.assert_allclose(forward(params, cfg, shuffled), forward(params, cfg, images), rtol=1e-11, atol=1e-13)
