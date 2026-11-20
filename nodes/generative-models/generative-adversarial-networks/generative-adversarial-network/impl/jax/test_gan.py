# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Tests of the JAX version: the shared fixture and jax.grad of the five losses, the counts of shapes.json, jit, and
the Dirac-GAN of Mescheder et al.: simultaneous steps spiral out and the R1 penalty makes them converge."""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gan import DIS, GEN, LOSSES, config, critic, d_grads, dirac_gan, g_grads, generate, init, input_grad, param_count

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, dict(np.load(FIXTURES / "expected-small.npz"))


def test_shared_fixture_forward():
    p, inp, want = fixture()
    fake = generate(p, inp["z"])
    np.testing.assert_allclose(fake, want["fake"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(critic(p, inp["x"]), want["a_real"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(critic(p, fake), want["a_fake"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(input_grad(p, inp["x"]), want["grad_real"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", LOSSES)
def test_shared_fixture_gradients(kind):
    p, inp, want = fixture()
    value, grads = jax.jit(d_grads, static_argnums=3)(p, inp["x"], inp["z"], kind, inp["alpha"], 10.0, 1.0)
    np.testing.assert_allclose(value, want[f"{kind}/d/loss"], rtol=1e-12)
    assert set(grads) == set(DIS)
    for k, g in grads.items():
        np.testing.assert_allclose(g, want[f"{kind}/d/d{k}"], rtol=1e-10, atol=1e-12, err_msg=k)
    value, grads = g_grads(p, inp["z"], kind)
    np.testing.assert_allclose(value, want[f"{kind}/g/loss"], rtol=1e-12)
    assert set(grads) == set(GEN)
    for k, g in grads.items():
        np.testing.assert_allclose(g, want[f"{kind}/g/d{k}"], rtol=1e-10, atol=1e-12, err_msg=k)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_noise"], k["d_hidden"], k["d_data"], k["d_critic"])
    assert param_count(cfg) == (mod["params_gen"], mod["params_dis"])
    p = init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    z = jnp.zeros((k["B"], k["d_noise"]))
    assert generate(p, z).shape == (k["B"], k["d_data"])
    assert critic(p, generate(p, z)).shape == (k["B"],)


def test_dirac_gan():
    h = 0.1
    n = np.linalg.norm(np.asarray(dirac_gan(1.0, 0.0, h, 2000)), axis=1)
    assert np.all(np.diff(n) > 0) and n[-1] > 2.5
    near = np.linalg.norm(np.asarray(dirac_gan(1e-3, 0.0, h, 100)), axis=1)
    assert near[-1] / near[0] == pytest.approx(math.sqrt(1 + h**2 / 4) ** 100, rel=1e-4)
    assert np.linalg.norm(np.asarray(dirac_gan(1.0, 0.0, h, 500, gamma=1.0))[-1]) < 1e-3
