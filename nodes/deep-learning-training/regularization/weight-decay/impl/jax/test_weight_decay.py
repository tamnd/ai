# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
"""Tests of the JAX code: the worked example, the shared fixture, jit, the mask, jax.grad of the penalty and of the
negative log posterior, and scale invariance with the schedule of Li and Arora. optax is not installed here, so the
fixture holds fixed numbers that the PyTorch tests check against torch.optim.SGD; the optax test is skipped."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from weight_decay import (decay_mask, effective_lr, exp_lr_schedule, l2_penalty, map_ridge, neg_log_posterior,
                          sgd_init, sgd_update)

jax.config.update("jax_enable_x64", True)

# The shared fixture of the NumPy tests.
P = {"w": [[0.5, -1.5], [2.0, 0.25]], "b": [0.25, -0.5]}
G = [{"w": [[1.0, -2.0], [0.5, 0.0]], "b": [0.5, 1.0]},
     {"w": [[0.5, 1.0], [-0.25, 1.5]], "b": [-1.0, 0.0]},
     {"w": [[-1.5, 0.5], [0.0, -1.0]], "b": [2.0, -0.5]}]
CONFIGS = {"l2": {"lr": 0.1, "weight_decay": 0.1},
           "momentum": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1},
           "nesterov": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "nesterov": True},
           "sgdw": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "decoupled": True}}
AFTER3 = {
    "l2": {"w": [[0.4876395, -1.4084285], [1.916343, 0.19407474999999996]], "b": [0.09256975000000003, -0.5331595]},
    "momentum": {"w": [[0.2604795, -1.1211484999999999], [1.802403, 0.05259474999999999]],
                 "b": [0.09148975000000001, -0.6893994999999999]},
    "nesterov": {"w": [[0.27808138050000003, -1.0763897315], [1.741800087, 0.019138485250000017]],
                 "b": [-0.04922180975000001, -0.6991175605000001]},
    "sgdw": {"w": [[0.2725395, -1.1582284999999999], [1.8537929999999998, 0.05907474999999996]],
             "b": [0.09751974999999999, -0.7032594999999999]},
}
AFTER3_MASKED_B = [0.10450000000000001, -0.721]


def tree(d):
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in d.items()}


def run(params, grads, **kw):
    state = sgd_init(params)
    for g in grads:
        params, state = sgd_update(params, tree(g), state, **kw)
    return params


def test_worked_example():
    kw = {"lr": 0.1, "momentum": 0.5, "weight_decay": 0.5}
    p = {"t": jnp.array([1.0])}
    grads = [{"t": [2.0]}, {"t": [-1.0]}]
    assert float(run(p, grads, lr=0.1, weight_decay=0.5)["t"][0]) == pytest.approx(0.8125, abs=1e-15)
    assert float(run(p, grads, lr=0.1, weight_decay=0.5, decoupled=True)["t"][0]) == pytest.approx(0.8125, abs=1e-15)
    assert float(run(p, grads, **kw)["t"][0]) == pytest.approx(0.6875, abs=1e-15)
    assert float(run(p, grads, decoupled=True, **kw)["t"][0]) == pytest.approx(0.7125, abs=1e-15)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(name):
    out = run(tree(P), G, **CONFIGS[name])
    for k in P:
        np.testing.assert_allclose(out[k], AFTER3[name][k], rtol=1e-12, atol=1e-15)


def test_fixture_masked_and_jit():
    params = tree(P)
    mask = decay_mask(params)
    assert mask == {"w": True, "b": False}
    step = jax.jit(lambda p, g, s: sgd_update(p, g, s, lr=0.1, momentum=0.9, weight_decay=0.1, mask=mask))
    state = sgd_init(params)
    for g in G:
        params, state = step(params, tree(g), state)
    np.testing.assert_allclose(params["w"], AFTER3["momentum"]["w"], rtol=1e-12)
    np.testing.assert_allclose(params["b"], AFTER3_MASKED_B, rtol=1e-12)


def test_grad_of_penalty():
    params = tree(P)
    g = jax.grad(l2_penalty)(params, 0.3)
    np.testing.assert_allclose(g["w"], 0.3 * params["w"], rtol=1e-14)
    np.testing.assert_allclose(g["b"], 0.3 * params["b"], rtol=1e-14)
    gm = jax.grad(l2_penalty)(params, 0.3, decay_mask(params))
    np.testing.assert_allclose(gm["b"], 0.0)


def test_l2_through_autodiff_equals_weight_decay():
    # SGD with momentum on loss + penalty, differentiated by jax.grad, is SGD with weight_decay as L2.
    params = tree(P)
    target = {"w": jnp.ones((2, 2)), "b": jnp.zeros(2)}

    def loss(p):
        return sum(jnp.sum((p[k] - target[k]) ** 2) for k in p)

    a, sa = params, sgd_init(params)
    b, sb = params, sgd_init(params)
    for _ in range(10):
        a, sa = sgd_update(a, jax.grad(lambda p: loss(p) + l2_penalty(p, 0.2))(a), sa, lr=0.05, momentum=0.9)
        b, sb = sgd_update(b, jax.grad(loss)(b), sb, lr=0.05, momentum=0.9, weight_decay=0.2)
    for k in P:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12)


def test_map_gradient_is_zero():
    key = jax.random.PRNGKey(0)
    kx, ky = jax.random.split(key)
    X = jax.random.normal(kx, (25, 4))
    y = jax.random.normal(ky, (25,))
    theta = map_ridge(X, y, 0.3, 2.0)
    g = jax.grad(neg_log_posterior)(theta, X, y, 0.3, 2.0)
    np.testing.assert_allclose(g, 0.0, atol=1e-10)


def scale_invariant_loss(w, H, b):
    u = w / jnp.linalg.norm(w)
    return 0.5 * u @ H @ u - b @ u


def test_scale_invariance_and_effective_lr():
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (5, 5))
    H, b = A @ A.T / 5, jax.random.normal(k2, (5,))
    w = jax.random.normal(k3, (5,))
    grad = jax.grad(scale_invariant_loss)
    g = grad(w, H, b)
    assert abs(float(w @ g)) < 1e-12
    np.testing.assert_allclose(grad(4.0 * w, H, b), g / 4.0, rtol=1e-10)
    u = w / jnp.linalg.norm(w)
    one = w - 0.3 * g
    other = u - effective_lr(0.3, w) * grad(u, H, b)
    np.testing.assert_allclose(one / jnp.linalg.norm(one), other / jnp.linalg.norm(other), rtol=1e-12)


def test_li_arora():
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (5, 5))
    H, b = A @ A.T / 5, jax.random.normal(k2, (5,))
    w0 = jax.random.normal(k3, (5,))
    grad = jax.grad(scale_invariant_loss)
    lr, wd = 0.3, 0.04
    q = 1 - lr * wd
    x = {"w": w0}
    s = sgd_init(x)
    y = w0
    for t in range(50):
        x, s = sgd_update(x, {"w": grad(x["w"], H, b)}, s, lr=lr, weight_decay=wd)
        y = y - exp_lr_schedule(lr, wd, t) * grad(y, H, b)
        np.testing.assert_allclose(y * q ** (t + 1), x["w"], rtol=1e-9)


def test_optax_add_decayed_weights():
    optax = pytest.importorskip("optax")
    params = tree(P)
    opt = optax.chain(optax.add_decayed_weights(0.1), optax.sgd(0.1, momentum=0.9))
    state = opt.init(params)
    for g in G:
        updates, state = opt.update(tree(g), state, params)
        params = optax.apply_updates(params, updates)
    for k in P:
        np.testing.assert_allclose(params[k], AFTER3["momentum"][k], rtol=1e-12)
