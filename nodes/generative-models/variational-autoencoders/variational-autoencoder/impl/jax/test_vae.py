# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Tests of the JAX version: the shared fixture and jax.grad for both decoders and the three objectives, the counts of
shapes.json, jit and vmap, the KL term against a jax.random average, and the bound tightening with more samples."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vae import DECODERS, OBJECTIVES, config, forward, init, kl, log_weights, loss, objective, param_count, sample

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture(decoder):
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    if decoder == "bernoulli":
        params = {k: v for k, v in params.items() if not k.startswith("olv_")}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, dict(np.load(FIXTURES / "expected-small.npz"))


@pytest.mark.parametrize("decoder", DECODERS)
def test_shared_fixture_and_grad(decoder):
    p, inp, want = fixture(decoder)
    cfg = config(decoder=decoder)
    out = forward(p, cfg, inp["x"], inp["eps"])
    for k in ("mu", "lv", "z", "a") + (("olv",) if decoder == "gaussian" else ()):
        np.testing.assert_allclose(out[k], want[f"{decoder}/{k}"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(log_weights(cfg, inp["x"], out, inp["eps"]), want[f"{decoder}/log_w"], rtol=1e-12)
    for kind in OBJECTIVES:
        value, grads = jax.value_and_grad(objective)(p, cfg, inp["x"], inp["eps"], kind)
        np.testing.assert_allclose(value, want[f"{decoder}/{kind}/loss"], rtol=1e-12)
        for k, g in grads.items():
            np.testing.assert_allclose(g, want[f"{decoder}/{kind}/d{k}"], rtol=1e-10, atol=1e-12, err_msg=k)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    for decoder, key in (("bernoulli", "params"), ("gaussian", "params_gaussian")):
        cfg = config(k["d_in"], k["d_hidden"], k["d_latent"], decoder)
        assert param_count(cfg) == mod[key]
        if k["d_in"] < 100:
            assert sum(v.size for v in init(jax.random.PRNGKey(0), cfg).values()) == mod[key]


def test_jit_and_vmap():
    p, inp, _ = fixture("gaussian")
    cfg = config(decoder="gaussian")
    for kind in OBJECTIVES:
        f = jax.jit(lambda p, x, e, kind=kind: objective(p, cfg, x, e, kind))
        np.testing.assert_allclose(f(p, inp["x"], inp["eps"]), objective(p, cfg, inp["x"], inp["eps"], kind), rtol=1e-12)
        # the batch objective is the mean of the objectives of the single examples
        each = jax.vmap(lambda x, e: objective(p, cfg, x[None], e[:, None], kind), in_axes=(0, 1))(inp["x"], inp["eps"])
        np.testing.assert_allclose(jnp.mean(each), objective(p, cfg, inp["x"], inp["eps"], kind), rtol=1e-12)


def test_closed_form_kl_is_the_mean_of_the_sampled_one():
    key = jax.random.PRNGKey(3)
    mu, lv = jnp.array([[0.5, -1.0, 0.0]]), jnp.array([[0.3, -0.7, 1.2]])
    eps = jax.random.normal(key, (400_000, 1, 3))
    z = mu + jnp.exp(0.5 * lv) * eps
    sampled = jnp.mean(jnp.sum(-0.5 * (lv + eps**2) + 0.5 * z**2, -1))
    np.testing.assert_allclose(sampled, kl(mu, lv)[0], atol=0.01)


def test_more_samples_tighten_the_bound():
    p, inp, _ = fixture("bernoulli")
    cfg = config()
    keys = jax.random.split(jax.random.PRNGKey(4), 400)
    bound = {S: -float(jnp.mean(jax.jit(jax.vmap(lambda k, S=S: loss(p, cfg, inp["x"], k, S, "iwae")))(keys)))
             for S in (1, 4, 16)}  # fmt: skip
    assert bound[1] < bound[4] < bound[16]
    assert sample(p, cfg, keys[0], 5).shape == (5, 6)
