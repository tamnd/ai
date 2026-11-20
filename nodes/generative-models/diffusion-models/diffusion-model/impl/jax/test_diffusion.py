# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Tests of the JAX version: the shared fixture and jax.grad of the simple loss, the counts of shapes.json, the reverse
mean at the true noise as the posterior mean, the jitted deterministic sampler against a Python loop of DDIM steps,
and DDIM with eta = 1 as the posterior step."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffusion import (
    config,
    ddim_sample,
    ddim_step,
    ddpm_step,
    eps_model,
    flops_forward,
    guided_eps,
    init,
    param_count,
    param_shapes,
    posterior,
    q_sample,
    reverse_mean,
    schedule,
    simple_loss,
    time_embedding,
)

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def random_params(key, cfg, scale=0.5):
    p = {}
    for k, s in param_shapes(cfg).items():
        key, sub = jax.random.split(key)
        p[k] = scale * jax.random.normal(sub, s)
    return p


def test_shared_fixture():
    p = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    x0, t, eps, z = inp["x0"], inp["t"], inp["eps"], inp["z"]
    np.testing.assert_allclose(time_embedding(t, cfg["d_time"]), want["emb"], rtol=1e-12, atol=1e-14)
    xt = q_sample(cfg, x0, t, eps)
    np.testing.assert_allclose(xt, want["xt"], rtol=1e-12, atol=1e-14)
    out = eps_model(p, cfg, xt, t)
    np.testing.assert_allclose(out, want["eps_pred"], rtol=1e-12, atol=1e-14)
    loss, g = jax.value_and_grad(simple_loss)(p, cfg, x0, t, eps)
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    pm, pv = posterior(cfg, xt, x0, t)
    np.testing.assert_allclose(pm, want["post_mean"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pv, want["post_var"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(reverse_mean(cfg, xt, t, out), want["mu"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ddpm_step(cfg, xt, t, out, z), want["x_prev"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        ddim_step(cfg, xt, t, t // 2, out, z, 0.5),
        want["x_ddim"],
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["d_time"], k["n_steps"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out = eps_model(p, cfg, jnp.zeros((k["B"], k["d_data"])), jnp.ones(k["B"], int))
    assert out.shape == (k["B"], k["d_data"])


def test_reverse_mean_at_the_true_noise_is_the_posterior_mean():
    cfg = config()
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    x0, eps = jax.random.normal(k1, (5, 3)), jax.random.normal(k2, (5, 3))
    t = jnp.array([2, 4, 6, 8, 10])
    xt = q_sample(cfg, x0, t, eps)
    np.testing.assert_allclose(
        reverse_mean(cfg, xt, t, eps),
        posterior(cfg, xt, x0, t)[0],
        rtol=1e-12,
        atol=1e-13,
    )


def test_jitted_sampler_matches_a_python_loop():
    cfg = config()
    p = random_params(jax.random.PRNGKey(2), cfg)
    xT = jax.random.normal(jax.random.PRNGKey(3), (4, 3))
    got = jax.jit(lambda q, x: ddim_sample(q, cfg, x))(p, xT)
    x = xT
    for t in range(cfg["n_steps"], 0, -1):
        tt = jnp.full((4,), t)
        x = ddim_step(cfg, x, tt, tt - 1, eps_model(p, cfg, x, tt), jnp.zeros_like(x))
    np.testing.assert_allclose(got, x, rtol=1e-10, atol=1e-12)


def test_ddim_with_eta_one_is_the_posterior_step_and_guidance():
    cfg = config()
    ks = jax.random.split(jax.random.PRNGKey(4), 3)
    xt, eps, z = (jax.random.normal(k, (4, 3)) for k in ks)
    t = jnp.array([2, 5, 8, 10])
    s = schedule(cfg)
    want = reverse_mean(cfg, xt, t, eps) + jnp.sqrt(s["beta_tilde"][t - 1])[:, None] * z
    np.testing.assert_allclose(ddim_step(cfg, xt, t, t - 1, eps, z, 1.0), want, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(guided_eps(eps, z, 0.0), eps)
