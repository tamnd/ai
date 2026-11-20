# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Tests of the JAX implementation: the shared fixture with jax.grad, the boundary condition, the zero total
derivative of the Gaussian consistency function by jax.jvp, the student-only gradient of the toy of Proposition 1,
the progressive-distillation target, and a model distilled by one scan of Adam steps and one trained from scratch."""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import consistency_model as cm

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=msg)


def test_shared_fixture():
    P = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    p = {k: v for k, v in P.items() if not k.startswith("minus_")}
    pm = {k[6:]: v for k, v in P.items() if k.startswith("minus_")}
    cfg = cm.config(*(int(v) for v in inp["cfg"]))
    ts, n = cm.karras_steps(18), inp["n"]
    t_lo, t_hi = ts[n], ts[n + 1]
    x_hi, x_lo = cm.cd_pair(
        inp["x0"], inp["z"], t_lo, t_hi, lambda x, t: cm.score(x, t, inp["means"], 0.1, inp["weights"])
    )
    close(x_lo, want["x_lo"])
    close(cm.forward(p, cfg, x_hi, t_hi), want["f"])
    assert float(cm.consistency_loss(p, pm, cfg, x_hi, t_hi, x_lo, t_lo)) == pytest.approx(
        float(want["cd_loss"]), rel=1e-12
    )
    for k, g in cm.consistency_loss_grad(p, pm, cfg, x_hi, t_hi, x_lo, t_lo).items():
        close(g, want[f"cd_d{k}"], rtol=1e-10, msg=k)
    xc_hi, xc_lo = cm.ct_pair(inp["x0"], inp["z"], t_lo, t_hi)
    args = (cfg, xc_hi, t_hi, xc_lo, t_lo, 1.0 / (t_hi - t_lo), 0.00054 * math.sqrt(3.0))
    assert float(cm.consistency_loss(p, p, *args)) == pytest.approx(float(want["ct_loss"]), rel=1e-12)
    for k, g in jax.grad(lambda q: cm.consistency_loss(q, q, *args))(p).items():
        close(g, want[f"ct_d{k}"], rtol=1e-10, msg=k)
    close(cm.ema(pm, p, 0.95)["W1"], want["ema"])
    close(cm.sample(lambda x, t: cm.forward(p, cfg, x, t), inp["xT"], [0.8], inp["noise"]), want["sample"])
    (a_t, s_t), (a_s, s_s), (a_s2, s_s2) = (cm.vp(v) for v in inp["lam"])
    z = inp["z_pd"]
    z_s2 = cm.ddim_step(cm.ddim_step(z, jnp.tanh(z), a_t, s_t, a_s, s_s), 0.5 * jnp.tanh(z), a_s, s_s, a_s2, s_s2)
    close(cm.pd_target(z, z_s2, a_t, s_t, a_s2, s_s2), want["pd_target"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cm.config(k["dim"], k["hidden"], k["temb"])
    assert cm.param_count(cfg) == mod["params"]
    assert cm.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = cm.init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    assert cm.forward(p, cfg, jnp.zeros((k["B"], k["dim"])), 3.0).shape == (k["B"], k["dim"])


def test_boundary_condition():
    cfg = cm.config(4, 7, 6)
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    p = jax.tree.map(lambda v: 5.0 * v + 1.0, cm.init(k1, cfg))
    x = jax.random.normal(k2, (5, 4))
    assert bool((cm.forward(p, cfg, x, cm.EPS) == x).all())


def test_gaussian_total_derivative_is_zero():
    mu, s2 = jnp.array([[0.5, -1.0]]), 0.2

    def sf(x, t):
        return cm.score(x, t, mu, s2, jnp.ones(1))

    for t in (0.01, 1.0, 50.0):
        x = t * jax.random.normal(jax.random.PRNGKey(int(t * 100)), (6, 2))
        d = cm.total_derivative(lambda xx, tt: cm.gaussian_consistency(xx, tt, mu, s2), x, t, sf)
        close(d, jnp.zeros((6, 2)), atol=1e-12)
        close(cm.solve_ode(x, t, cm.EPS, sf, 400), cm.gaussian_consistency(x, t, mu, s2), rtol=1e-4, atol=1e-4)


def test_toy_student_only_gradient():
    ts, z, xi = jnp.linspace(0.1, 1.0, 4001), jax.random.normal(jax.random.PRNGKey(5), (1, 2000)), 0.7
    smin, lo, hi = ts[0], ts[:-1], ts[1:]
    for theta in (-0.4, 1.3):
        g = jax.grad(lambda th: cm.toy_ct_loss(th, th, xi, ts, z))(theta)
        want = (2 * smin * (1 / lo - 1 / hi) * (1 - smin / hi) * (theta - xi)).mean()
        assert float(g) == pytest.approx(float(want), rel=1e-9)
        assert float(cm.toy_ct_loss(theta, 0.0, xi, ts, z)) == pytest.approx(
            float(((1 - smin / hi) ** 2).mean()) * theta**2, rel=1e-2
        )


def near_props(x, means):
    d = jnp.sqrt(((x[:, None] - means[None]) ** 2).sum(-1))
    return float((d.min(1) < 0.3).mean()), np.bincount(np.asarray(d.argmin(1)), minlength=3) / len(x)


def test_distillation_and_training():
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(jax.random.PRNGKey(7), 8)
    ang = jnp.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    means, s2, w = 0.8 * jnp.stack([jnp.cos(ang), jnp.sin(ang)], 1), 0.01, jnp.full(3, 1 / 3)
    x0 = means[jax.random.randint(k1, (6000,), 0, 3)] + 0.1 * jax.random.normal(k2, (6000, 2))

    def sf(x, t):
        return cm.score(x, t, means, s2, w)

    cfg = cm.config(2, 64, 8)
    xT = 80.0 * jax.random.normal(k3, (2000, 2))
    ref = cm.solve_ode(xT, cm.T_MAX, cm.EPS, sf, 200)
    _, pm, _ = jax.jit(lambda p, k: cm.train_cd(p, cfg, x0, sf, 3000, k, lr=3e-3))(cm.init(k4, cfg), k5)
    one = cm.sample(lambda x, t: cm.forward(pm, cfg, x, t), xT)
    near, props = near_props(one, means)
    assert (
        float(jnp.sqrt(((one - ref) ** 2).sum(1)).mean()) < 0.2 and near > 0.65 and np.abs(props - 1 / 3).max() < 0.06
    )

    p, _ = cm.train_ct(cm.init(k6, cfg), cfg, x0, 3000, k7, lr=3e-3)
    noise = jax.random.normal(k8, (1, 2000, 2))
    (n1, props1) = near_props(cm.sample(lambda x, t: cm.forward(p, cfg, x, t), xT), means)
    n2, _ = near_props(cm.sample(lambda x, t: cm.forward(p, cfg, x, t), xT, [0.8], noise), means)
    assert n1 > 0.25 and n2 > max(n1, 0.6) and np.abs(props1 - 1 / 3).max() < 0.1
