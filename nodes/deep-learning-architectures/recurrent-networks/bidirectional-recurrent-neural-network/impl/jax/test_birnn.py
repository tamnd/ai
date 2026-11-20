# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://Z7Z0J3).
"""Tests of the JAX version: the shared fixture and its gradients, the shape contract, jit and vmap."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from birnn import config, forward, init, loss, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return config(k["input_size"], k["hidden_size"], k["n_layers"], k["n_classes"], "lstm", k["directions"] == 2)


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_shared_fixture_and_gradients():
    # The masked reverse scan gives the outputs of the reference, which reverses the real steps of each sequence.
    params, inp, want = fixture()
    cfg = config()
    logits = forward(params, cfg, inp["x"], inp["lengths"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-13)
    value, grads = jax.value_and_grad(loss, argnums=(0, 2))(params, cfg, inp["x"], inp["labels"], inp["lengths"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for k, v in grads[0].items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-10, atol=1e-12, err_msg=k)
    np.testing.assert_allclose(grads[1], want["dx"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        params = init(jax.random.PRNGKey(0), cfg)
        assert sum(v.size for v in params.values()) == c["model"]["params"]
        k = c["config"]
        logits = forward(params, cfg, jnp.ones((k["B"], k["T"], k["input_size"])))
        assert list(logits.shape) == c["blocks"]["output"]["outputs"]["logits"]


@pytest.mark.parametrize("cell", ["lstm", "tanh"])
def test_jit_and_vmap(cell):
    cfg = config(cell=cell)
    params = init(jax.random.PRNGKey(1), cfg, scale=0.5)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 5, 3))
    lengths = jnp.array([5, 3, 1, 4])
    want = forward(params, cfg, x, lengths)
    got = jax.jit(lambda p, xs, n: forward(p, cfg, xs, n))(params, x, lengths)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    one = jax.vmap(lambda xi, n: forward(params, cfg, xi[None], n[None])[0])(x, lengths)
    np.testing.assert_allclose(one, want, rtol=1e-12, atol=1e-13)
    # The outputs of a sequence do not depend on what its padding holds.
    y = x.at[1, 3:].set(1e3)
    np.testing.assert_allclose(forward(params, cfg, y, lengths)[1, :3], want[1, :3], rtol=1e-12, atol=1e-13)
