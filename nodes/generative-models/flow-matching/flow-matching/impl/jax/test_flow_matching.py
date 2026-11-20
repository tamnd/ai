# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Tests of the JAX version: the shared fixture and jax.grad of the loss, the counts of shapes.json, the target as
jax.jacfwd of the path in t, the conditional field on the path, the jitted samplers against Python loops, the jacfwd
divergence against the closed form for the network, the log-likelihood of a Gaussian, and a short training run on two
Gaussian modes."""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flow_matching import (
    adam_init,
    cfm_loss,
    cond_field,
    config,
    divergence,
    euler_sample,
    flops_forward,
    init,
    log_likelihood,
    logit_normal,
    make_step,
    midpoint_sample,
    param_count,
    param_shapes,
    path,
    target,
    time_features,
    velocity,
)

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
MODES = jnp.array([[-2.0, 0.0], [2.0, 0.0]])


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
    cfg = config(sigma_min=float(inp["sigma_min"]))
    x0, x1, t = inp["x0"], inp["x1"], inp["t"]

    def v_fn(x, s):
        return velocity(p, cfg, x, s)

    def close(a, b, rtol=1e-12, atol=1e-14):
        np.testing.assert_allclose(np.asarray(a), b, rtol=rtol, atol=atol)

    close(time_features(t, cfg["d_time"]), want["feat"])
    xt = path(cfg, x0, x1, t)
    close(xt, want["xt"])
    close(target(cfg, x0, x1), want["u"])
    close(v_fn(xt, t), want["v"])
    close(divergence(v_fn, xt, t), want["div"], 1e-10, 1e-12)
    loss, g = jax.value_and_grad(cfm_loss)(p, cfg, x0, x1, t)
    close(loss, want["loss"])
    for k in p:
        close(g[k], want[f"d{k}"], 1e-10, 1e-13)
    close(euler_sample(v_fn, x0, 4), want["x_euler"], 1e-12, 1e-13)
    close(midpoint_sample(v_fn, x0, 2), want["x_midpoint"], 1e-12, 1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["d_time"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    assert velocity(p, cfg, jnp.zeros((k["B"], k["d_data"])), jnp.zeros(k["B"])).shape == (k["B"], k["d_data"])


@pytest.mark.parametrize("sigma_min", [0.0, 0.1])
def test_target_is_the_time_derivative_of_the_path(sigma_min):
    cfg = config(sigma_min=sigma_min)
    k0, k1 = jax.random.split(jax.random.PRNGKey(1))
    x0, x1 = jax.random.normal(k0, (5, 3)), jax.random.normal(k1, (5, 3))
    t = jnp.linspace(0.05, 0.95, 5)
    d = jax.vmap(lambda a, b, s: jax.jacfwd(lambda r: path(cfg, a[None], b[None], r[None])[0])(s))(x0, x1, t)
    np.testing.assert_allclose(d, target(cfg, x0, x1), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        cond_field(cfg, path(cfg, x0, x1, t), x1, t), target(cfg, x0, x1), rtol=1e-10, atol=1e-12
    )


def test_jitted_samplers_match_python_loops():
    cfg = config()
    p = random_params(jax.random.PRNGKey(2), cfg)

    def v_fn(x, s):
        return velocity(p, cfg, x, s)

    x = jax.random.normal(jax.random.PRNGKey(3), (4, 3))
    e, m = x, x
    for i in range(5):
        t = jnp.full(4, i / 5)
        e = e + v_fn(e, t) / 5
        m = m + v_fn(m + 0.1 * v_fn(m, t), t + 0.1) / 5
    np.testing.assert_allclose(jax.jit(lambda z: euler_sample(v_fn, z, 5))(x), e, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(jax.jit(lambda z: midpoint_sample(v_fn, z, 5))(x), m, rtol=1e-12, atol=1e-13)


def test_jacfwd_divergence_matches_the_closed_form():
    cfg = config()
    p = random_params(jax.random.PRNGKey(4), cfg)
    x, t = jax.random.normal(jax.random.PRNGKey(5), (6, 3)), jnp.linspace(0.0, 1.0, 6)
    inp = jnp.concatenate([x, time_features(t, cfg["d_time"])], 1)
    h = jnp.tanh(inp @ p["W1"] + p["b1"])
    want = (1.0 - h**2) @ jnp.einsum("ij,ji->j", p["W1"][:3], p["W2"])
    np.testing.assert_allclose(divergence(lambda z, s: velocity(p, cfg, z, s), x, t), want, rtol=1e-12, atol=1e-13)


def test_log_likelihood_of_a_gaussian_path():
    s = 0.5

    def v_fn(x, t):
        t = t[:, None]
        return (s**2 * t - (1 - t)) / ((1 - t) ** 2 + t**2 * s**2) * x

    def log_p0(x):
        return -0.5 * jnp.sum(x**2, 1) - 0.5 * x.shape[1] * math.log(2 * math.pi)

    x1 = jax.random.normal(jax.random.PRNGKey(6), (7, 2))
    want = -0.5 * jnp.sum(x1**2, 1) / s**2 - 2 * math.log(s * math.sqrt(2 * math.pi))
    np.testing.assert_allclose(log_likelihood(v_fn, x1, 200, log_p0), want, rtol=0, atol=1e-4)


def test_learns_two_modes():
    cfg = config(2, 64, 16)
    key = jax.random.PRNGKey(7)
    key, sub = jax.random.split(key)
    p = init(sub, cfg)
    st, step, trace = adam_init(p), make_step(cfg), []
    for _ in range(1500):
        key, a, b, c, d = jax.random.split(key, 5)
        x1 = MODES[jax.random.randint(a, (256,), 0, 2)] + 0.3 * jax.random.normal(b, (256, 2))
        p, st, loss = step(p, st, jax.random.normal(c, (256, 2)), x1, logit_normal(d, 256))
        trace.append(float(loss))
    assert np.mean(trace[-100:]) < 0.8 * np.mean(trace[:10])
    xs = midpoint_sample(lambda x, s: velocity(p, cfg, x, s), jax.random.normal(key, (2000, 2)), 20)
    dist = jnp.min(jnp.linalg.norm(xs[:, None] - MODES[None], axis=-1), 1)
    assert 0.4 < float(jnp.mean(xs[:, 0] > 0)) < 0.6
    assert float(jnp.mean(dist < 1.0)) > 0.9
