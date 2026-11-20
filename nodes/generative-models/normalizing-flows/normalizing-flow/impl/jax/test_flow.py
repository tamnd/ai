# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Tests of the JAX version: the shared fixture and jax.grad of the negative log-likelihood, the counts of
shapes.json, the log-determinant against the determinant of jax.jacfwd, the inverse under jit, the planar flow's
determinant and its fix, and the autoregressive layer's d-pass sampling."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flow import (
    ar_density,
    ar_sample,
    config,
    flops_forward,
    forward,
    init,
    inverse,
    log_prob,
    nll,
    param_count,
    param_shapes,
    planar,
    u_hat,
)

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, dict(np.load(FIXTURES / "expected-small.npz"))


def random_params(key, cfg, scale=0.5):
    p = {}
    for k, s in param_shapes(cfg).items():
        key, sub = jax.random.split(key)
        p[k] = scale * jax.random.normal(sub, s)
    return p


def test_shared_fixture():
    p, inp, want = fixture()
    cfg = config()
    z, ld = forward(p, cfg, inp["x"])
    np.testing.assert_allclose(z, want["z"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(ld, want["logdet"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(log_prob(p, cfg, inp["x"]), want["log_prob"], rtol=1e-12)
    np.testing.assert_allclose(inverse(p, cfg, inp["u"]), want["x_inv"], rtol=1e-12, atol=1e-13)
    flow_p = {k: v for k, v in p.items() if k in param_shapes(cfg)}
    value, grads = jax.value_and_grad(nll)(flow_p, cfg, inp["x"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for k, g in grads.items():
        np.testing.assert_allclose(g, want[f"d{k}"], rtol=1e-10, atol=1e-12, err_msg=k)
    np.testing.assert_allclose(jax.grad(nll, argnums=2)(flow_p, cfg, inp["x"]), want["dx"], rtol=1e-10, atol=1e-12)
    w, u, b = p["planar_w"], p["planar_u"], p["planar_b"][0]
    f, pld = planar(inp["x"], w, u_hat(w, u), b)
    np.testing.assert_allclose(f, want["planar_f"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(pld, want["planar_logdet"], rtol=1e-12, atol=1e-13)
    q = {k[3:]: v for k, v in p.items() if k.startswith("ar_")}
    au, ald = ar_density(q, inp["x"])
    np.testing.assert_allclose(au, want["ar_u"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(ald, want["ar_logdet"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(ar_sample(q, inp["u"]), want["ar_x"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["n_layers"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    assert sum(v.size for v in init(jax.random.PRNGKey(0), cfg).values()) == mod["params"]


def test_logdet_against_jacfwd_and_inverse_under_jit():
    cfg = config(4, 6, 3)
    p = random_params(jax.random.PRNGKey(1), cfg)
    x = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    z, ld = jax.jit(lambda q, v: forward(q, cfg, v))(p, x)
    J = jax.vmap(jax.jacfwd(lambda v: forward(p, cfg, v[None])[0][0]))(x)
    np.testing.assert_allclose(ld, jnp.linalg.slogdet(J)[1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(jax.jit(lambda q, v: inverse(q, cfg, v))(p, z), x, rtol=1e-12, atol=1e-12)


def test_zero_init_is_the_identity():
    cfg = config(5, 7, 4)
    x = jax.random.normal(jax.random.PRNGKey(3), (10, 5))
    z, ld = forward(init(jax.random.PRNGKey(4), cfg), cfg, x)
    np.testing.assert_array_equal(z, x)
    np.testing.assert_array_equal(ld, 0.0)


def test_planar_determinant_and_fix():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(5), 3)
    w, u = jax.random.normal(k1, (3,)), jax.random.normal(k2, (3,))
    u = u - (w @ u + 3.0) * w / (w @ w)
    uh = u_hat(w, u)
    assert float(w @ uh) > -1
    z = jax.random.normal(k3, (5, 3))
    J = jax.vmap(jax.jacfwd(lambda v: planar(v[None], w, uh, 0.3)[0][0]))(z)
    np.testing.assert_allclose(planar(z, w, uh, 0.3)[1], jnp.log(jnp.abs(jnp.linalg.det(J))), rtol=1e-10)
    J_bad = jax.vmap(jax.jacfwd(lambda v: planar(v[None], w, u, 0.3)[0][0]))(-0.3 * w[None] / (w @ w))
    assert float(jnp.linalg.det(J_bad[0])) < 0  # at w.z + b = 0 the unfixed map reverses orientation


def test_autoregressive_passes():
    ks = jax.random.split(jax.random.PRNGKey(6), 5)
    D = 5
    q = {"Wm": 0.7 * jax.random.normal(ks[0], (D, D)), "bm": jax.random.normal(ks[1], (D,)),
         "Wa": 0.4 * jax.random.normal(ks[2], (D, D)), "ba": 0.3 * jax.random.normal(ks[3], (D,))}  # fmt: skip
    u = jax.random.normal(ks[4], (8, D))
    x = jax.jit(ar_sample)(q, u)
    np.testing.assert_allclose(ar_density(q, x)[0], u, rtol=1e-12, atol=1e-12)
    assert float(jnp.abs(ar_sample(q, u, D - 1) - x).max()) > 1e-3
