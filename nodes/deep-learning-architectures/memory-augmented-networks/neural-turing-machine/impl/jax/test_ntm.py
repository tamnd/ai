# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""Tests of the JAX reference: the shared fixture, the shape contract, jit, vmap and the gradient against finite
differences."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ntm import bits_per_sequence, config, cosine, forward, init, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("x_dim", "y_dim", "hidden", "n_loc", "width", "n_read", "n_write", "shift")


def cfg_of(c):
    return config(**{k: c["config"][k] for k in KEYS})


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, np.load(FIXTURES / "expected-small.npz")


def test_shared_fixture():
    params, inp, want = fixture()
    cfg = cfg_of(SHAPES[0])
    logits, ws, mem = forward(params, inp["x"], cfg, trace=True)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ws, want["weights"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mem, want["memory"], rtol=1e-10, atol=1e-12)
    loss = bits_per_sequence(logits, inp["targets"], inp["mask"])
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    logits, ws, mem = forward(params, jnp.zeros((k["B"], k["T"], k["X"])), cfg, trace=True)
    assert list(logits.shape) == [k["B"], k["T"], k["Y"]]
    assert list(ws.shape) == [k["T"], k["R"] + k["W"]] + c["blocks"]["address"]["outputs"]["w"]
    assert list(mem.shape) == c["blocks"]["write"]["outputs"]["memory"]


def test_jit_and_vmap_agree():
    params, inp, want = fixture()
    cfg = cfg_of(SHAPES[0])
    jitted = jax.jit(lambda p, x: forward(p, x, cfg))
    np.testing.assert_allclose(jitted(params, inp["x"]), want["logits"], rtol=1e-10, atol=1e-12)
    one = jax.vmap(lambda x: forward(params, x[None], cfg)[0])(inp["x"])  # one sequence at a time
    np.testing.assert_allclose(one, want["logits"], rtol=1e-10, atol=1e-12)


def test_gradient_against_finite_differences():
    params, inp, _ = fixture()
    cfg = cfg_of(SHAPES[0])

    def loss(p):
        return bits_per_sequence(forward(p, inp["x"], cfg), inp["targets"], inp["mask"])

    grads = jax.grad(loss)(params)
    rng = np.random.default_rng(0)
    for name in params:
        direction = jnp.asarray(rng.normal(size=params[name].shape))
        h = 1e-6
        plus = loss({**params, name: params[name] + h * direction})
        minus = loss({**params, name: params[name] - h * direction})
        assert float((plus - minus) / (2 * h)) == pytest.approx(
            float((grads[name] * direction).sum()), rel=1e-5, abs=1e-8
        )


def test_cosine_gradient_of_a_blank_row_is_finite():
    g = jax.grad(lambda m: cosine(m, jnp.array([[1.0, -1.0]])).sum())(jnp.zeros((1, 3, 2)))
    assert bool(jnp.isfinite(g).all())
