# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
"""Tests of the NumPy reference: the worked example, the fixture shared with the PyTorch and JAX tests, L2 against
decoupled decay under SGD, momentum and Adam, the mask, the MAP estimate, and the scale-invariance formulas."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weight_decay import (adam_config, adam_init, adam_step, decay_mask, decay_per_step, effective_lr,
                          equilibrium_angle, equilibrium_norm_sq, exp_lr_schedule, l2_from_decay, l2_grad, l2_penalty,
                          map_l2_coefficient, map_ridge, matched_decay, sgd_config, sgd_init, sgd_step)

# The shared fixture. The PyTorch tests check these numbers against torch.optim.SGD, and the JAX tests hold them too.
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
# momentum with the mask of decay_mask: the bias b is not decayed.
AFTER3_MASKED_B = [0.10450000000000001, -0.721]


def run(cfg, params, grads, mask=None):
    state = sgd_init(params)
    for g in grads:
        params, state = sgd_step(cfg, params, g, state, mask=mask)
    return params


def one(theta, gs, **kw):
    out = run(sgd_config(**kw), {"t": [theta]}, [{"t": [g]} for g in gs])
    return float(out["t"][0])


def test_worked_example():
    # theta_0 = 1, lr = 0.1, weight decay 0.5, gradients 2 then -1. Plain SGD: both forms give 0.75 then 0.8125.
    for decoupled in (False, True):
        assert one(1.0, [2.0], lr=0.1, weight_decay=0.5, decoupled=decoupled) == pytest.approx(0.75, abs=1e-15)
        assert one(1.0, [2.0, -1.0], lr=0.1, weight_decay=0.5, decoupled=decoupled) == pytest.approx(0.8125, abs=1e-15)


def test_worked_example_momentum():
    # With momentum 0.5 the first steps agree (0.75), and the second differ: 0.6875 with L2, 0.7125 decoupled.
    kw = {"lr": 0.1, "momentum": 0.5, "weight_decay": 0.5}
    assert one(1.0, [2.0], **kw) == pytest.approx(0.75, abs=1e-15)
    assert one(1.0, [2.0], decoupled=True, **kw) == pytest.approx(0.75, abs=1e-15)
    assert one(1.0, [2.0, -1.0], **kw) == pytest.approx(0.6875, abs=1e-15)
    assert one(1.0, [2.0, -1.0], decoupled=True, **kw) == pytest.approx(0.7125, abs=1e-15)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(name):
    out = run(sgd_config(**CONFIGS[name]), P, G)
    for k in P:
        np.testing.assert_allclose(out[k], AFTER3[name][k], rtol=1e-12, atol=1e-15)


def test_fixture_masked():
    out = run(sgd_config(**CONFIGS["momentum"]), P, G, mask=decay_mask(P))
    np.testing.assert_allclose(out["w"], AFTER3["momentum"]["w"], rtol=1e-12)
    np.testing.assert_allclose(out["b"], AFTER3_MASKED_B, rtol=1e-12)
    np.testing.assert_allclose(out["b"], run(sgd_config(lr=0.1, momentum=0.9), {"b": P["b"]},
                                             [{"b": g["b"]} for g in G])["b"], rtol=1e-12)


@settings(max_examples=60, deadline=None)
@given(st.floats(1e-3, 1.0), st.floats(0.0, 0.9), st.integers(0, 2**31 - 1))
def test_sgd_l2_is_decay(lr, wd, seed):
    # Plain SGD: L2 with weight_decay is decoupled decay with delta = lr * weight_decay, for any gradients.
    rng = np.random.default_rng(seed)
    params = {"w": rng.normal(size=(3, 2))}
    grads = [{"w": rng.normal(size=(3, 2))} for _ in range(8)]
    a = run(sgd_config(lr, weight_decay=wd), params, grads)
    b = run(sgd_config(lr, weight_decay=wd, decoupled=True), params, grads)
    np.testing.assert_allclose(a["w"], b["w"], rtol=1e-11, atol=1e-13)
    assert decay_per_step(lr, wd) == pytest.approx(lr * wd)
    assert l2_from_decay(lr, decay_per_step(lr, wd)) == pytest.approx(wd)


def test_l2_penalty_gradient():
    rng = np.random.default_rng(0)
    params = {"w": rng.normal(size=(2, 3)), "b": rng.normal(size=3)}
    mask = decay_mask(params)
    assert l2_penalty(params, 0.2) == pytest.approx(0.1 * (np.sum(params["w"] ** 2) + np.sum(params["b"] ** 2)))
    assert l2_penalty(params, 0.2, mask) == pytest.approx(0.1 * np.sum(params["w"] ** 2))
    grad = l2_grad(params, 0.2, mask)
    h = 1e-6
    for k in params:
        fd = np.zeros_like(params[k])
        for idx in np.ndindex(params[k].shape):
            up = {kk: vv.copy() for kk, vv in params.items()}
            dn = {kk: vv.copy() for kk, vv in params.items()}
            up[k][idx] += h
            dn[k][idx] -= h
            fd[idx] = (l2_penalty(up, 0.2, mask) - l2_penalty(dn, 0.2, mask)) / (2 * h)
        np.testing.assert_allclose(grad[k], fd, atol=1e-8)
    np.testing.assert_array_equal(grad["b"], 0.0)


def test_exponential_decay():
    # With no loss gradient, the weights shrink by (1 - lr wd) per step, with and without momentum 0.
    theta = {"w": np.array([[1.0, -2.0], [3.0, 0.5]])}
    zeros = [{"w": np.zeros((2, 2))}] * 20
    for decoupled in (False, True):
        out = run(sgd_config(0.1, weight_decay=0.3, decoupled=decoupled), theta, zeros)
        np.testing.assert_allclose(out["w"], 0.97**20 * theta["w"], rtol=1e-12)


def test_momentum_second_step_gap():
    rng = np.random.default_rng(1)
    params = {"w": rng.normal(size=(4, 4))}
    grads = [{"w": rng.normal(size=(4, 4))} for _ in range(2)]
    lr, mu, wd = 0.05, 0.9, 0.2
    a1 = run(sgd_config(lr, momentum=mu, weight_decay=wd), params, grads[:1])
    b1 = run(sgd_config(lr, momentum=mu, weight_decay=wd, decoupled=True), params, grads[:1])
    np.testing.assert_allclose(a1["w"], b1["w"], rtol=1e-12)
    a2 = run(sgd_config(lr, momentum=mu, weight_decay=wd), params, grads)
    b2 = run(sgd_config(lr, momentum=mu, weight_decay=wd, decoupled=True), params, grads)
    np.testing.assert_allclose(b2["w"] - a2["w"], lr * mu * wd * params["w"], rtol=1e-9, atol=1e-14)


def test_matched_decay_fixed_point():
    # On a quadratic, L2 under momentum and decoupled decay of lr wd / (1 - mu) per step reach the same point.
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4))
    H = A @ A.T / 4 + 0.2 * np.eye(4)
    c = rng.normal(size=4)
    lr, mu, wd = 0.05, 0.9, 0.3
    delta = matched_decay(lr, wd, mu)
    a = b = np.zeros(4)
    ba = bb = np.zeros(4)
    for _ in range(4000):
        ba = mu * ba + H @ a - c + wd * a
        a = a - lr * ba
        bb = mu * bb + H @ b - c
        b = (1 - delta) * b - lr * bb
    star = np.linalg.solve(H + wd * np.eye(4), c)
    np.testing.assert_allclose(a, star, rtol=1e-9)
    np.testing.assert_allclose(b, star, rtol=1e-9)
    # The decoupled form of the page, sgd_step with decoupled=True, decays by lr * weight_decay, so the matching
    # weight_decay to pass is wd / (1 - mu).
    p = {"x": np.zeros(4)}
    s = sgd_init(p)
    cfg = sgd_config(lr, momentum=mu, weight_decay=wd / (1 - mu), decoupled=True)
    for _ in range(4000):
        p, s = sgd_step(cfg, p, {"x": H @ p["x"] - c}, s)
    np.testing.assert_allclose(p["x"], star, rtol=1e-9)


def test_adam_l2_is_not_decoupled():
    # Two coordinates whose gradients differ by a factor of 100: under Adam with L2 the decay of the coordinate with
    # the large gradient is about 100 times smaller, while AdamW decays both by lr * wd.
    params = {"w": np.array([1.0, 1.0])}
    grads = [{"w": np.array([100.0 * (-1) ** t, 1.0 * (-1) ** t])} for t in range(10)]
    zero = {"w": np.zeros(2)}

    def go(cfg):
        p, s = params, adam_init(params)
        for g in grads:
            p, s = adam_step(cfg, p, g, s)
        q, s = params, adam_init(params)
        for g in grads:
            q, s = adam_step(adam_config(lr=cfg["lr"]), q, g, s)
        return p["w"] - q["w"]

    shift_l2 = go(adam_config(lr=1e-3, weight_decay=1.0))
    shift_w = go(adam_config(lr=1e-3, weight_decay=1.0, decoupled=True))
    assert abs(shift_l2[0]) < abs(shift_l2[1]) / 30
    np.testing.assert_allclose(shift_w[0], shift_w[1], rtol=0.05)
    # With a zero gradient, AdamW only decays: one step multiplies by 1 - lr * wd.
    p, _ = adam_step(adam_config(lr=0.01, weight_decay=0.1, decoupled=True), params, zero, adam_init(params))
    np.testing.assert_allclose(p["w"], params["w"] * (1 - 0.001), rtol=1e-12)


def test_decay_mask():
    params = {"conv": np.zeros((4, 3, 3, 3)), "fc": np.zeros((10, 4)), "bias": np.zeros(10), "bn_gamma": np.ones(4)}
    assert decay_mask(params) == {"conv": True, "fc": True, "bias": False, "bn_gamma": False}


def test_map_ridge():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    noise_var, prior_var = 0.25, 2.0
    theta = map_ridge(X, y, noise_var, prior_var)
    # The gradient of the negative log posterior is zero at the MAP estimate.
    g = -X.T @ (y - X @ theta) / noise_var + theta / prior_var
    np.testing.assert_allclose(g, 0.0, atol=1e-10)
    assert map_l2_coefficient(prior_var) == pytest.approx(0.5)
    assert map_l2_coefficient(prior_var, n=20) == pytest.approx(0.025)
    # The same point minimizes the mean squared error plus the mean-scaled penalty (L2 coefficient on the mean loss).
    lam_mean = noise_var * map_l2_coefficient(prior_var, n=20)
    g2 = -X.T @ (y - X @ theta) / 20 + lam_mean * theta
    np.testing.assert_allclose(g2, 0.0, atol=1e-10)


def test_map_lambda_is_optimal():
    # The exact expected squared error of the ridge weights is smallest at lambda = noise_var / prior_var.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 6))
    s2, sp2 = 0.4, 1.5
    XtX = X.T @ X

    def err(lam):
        A = np.linalg.inv(XtX + lam * np.eye(6))
        return lam**2 * sp2 * np.trace(A @ A) + s2 * np.trace(A @ XtX @ A)

    best = s2 / sp2
    for lam in (best / 3, best / 1.1, best * 1.1, best * 3, 1e-6):
        assert err(lam) > err(best)


def test_min_norm_limit():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, 7))
    y = rng.normal(size=3)
    theta = map_ridge(X, y, 1e-9, 1.0)
    np.testing.assert_allclose(theta, np.linalg.pinv(X) @ y, rtol=1e-5, atol=1e-7)


def scale_invariant(seed, d=6):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    H = A @ A.T / d
    b = rng.normal(size=d)

    def loss(w):
        u = w / np.linalg.norm(w)
        return 0.5 * u @ H @ u - b @ u

    def grad(w):
        r = np.linalg.norm(w)
        u = w / r
        gu = H @ u - b
        return (gu - (u @ gu) * u) / r

    return loss, grad, rng.normal(size=d)


def test_effective_lr():
    loss, grad, w = scale_invariant(6)
    lr = 0.3
    u = w / np.linalg.norm(w)
    a = w - lr * grad(w)
    b = u - effective_lr(lr, w) * grad(u)
    np.testing.assert_allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), rtol=1e-12)
    # Without decay, the norm only grows.
    v = w.copy()
    for _ in range(30):
        new = v - lr * grad(v)
        assert np.linalg.norm(new) >= np.linalg.norm(v)
        v = new


def test_exp_lr_schedule():
    # Li and Arora, momentum 0: decay with wd, or no decay and the exponential schedule, give the same directions.
    loss, grad, w = scale_invariant(7)
    lr, wd = 0.4, 0.05
    p = {"w": w}
    s = sgd_init(p)
    cfg = sgd_config(lr, weight_decay=wd)
    y = w.copy()
    for t in range(80):
        p, s = sgd_step(cfg, p, {"w": grad(p["w"])}, s)
        y = y - exp_lr_schedule(lr, wd, t) * grad(y)
        np.testing.assert_allclose(y * (1 - lr * wd) ** (t + 1), p["w"], rtol=1e-8)
    assert loss(y) == pytest.approx(loss(p["w"]), rel=1e-10)
    assert exp_lr_schedule(lr, wd, 79) > 20 * lr  # the schedule has grown by 0.98^-159, about 25 times


def test_equilibrium():
    # Iterate the norm recursion of a scale-invariant loss with |g| = G / |w|.
    lr, wd, G = 0.1, 5e-4, 2.0
    q2 = (1 - lr * wd) ** 2
    r = 50.0
    for _ in range(100000):
        r = q2 * r + lr**2 * G**2 / r
    assert r == pytest.approx(equilibrium_norm_sq(lr, wd, G), rel=1e-8)
    assert equilibrium_angle(lr, wd) == pytest.approx(np.sqrt(2 * lr * wd), rel=1e-3)
    assert effective_lr(lr, np.array([np.sqrt(r)])) == pytest.approx(np.sqrt(lr * wd * (2 - lr * wd)) / G, rel=1e-7)


def test_config_validation():
    with pytest.raises(ValueError):
        sgd_config(-0.1)
    with pytest.raises(ValueError):
        sgd_config(0.1, weight_decay=-1.0)
    with pytest.raises(ValueError):
        sgd_config(0.1, momentum=1.0)
    with pytest.raises(ValueError):
        sgd_config(0.1, nesterov=True)
