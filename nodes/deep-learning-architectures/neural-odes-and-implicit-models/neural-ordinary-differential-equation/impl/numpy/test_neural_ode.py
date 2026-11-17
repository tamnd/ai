# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""Tests of the NumPy reference: the solvers, the residual step, the two gradients, the change of variables,
the one-dimensional limit and augmentation, the shapes and the shared fixture."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neural_ode import (
    NAMES,
    adjoint_backward,
    augment,
    backprop,
    config,
    dopri5,
    euler,
    field,
    field_vjp,
    flops,
    forward,
    init,
    log_det_flow,
    mse_loss,
    nfe,
    param_count,
    rk4,
    trace_jacobian,
    train,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d"], k["d_hidden"], k["d_state"], k["n_steps"], **kw)


def random_params(rng, cfg, scale=1.0):
    params = init(rng, cfg, scale)
    params["b1"] = rng.normal(0.0, 0.3, params["b1"].shape)
    params["b2"] = rng.normal(0.0, 0.3, params["b2"].shape)
    return params


def expm(A, terms=40):
    """exp(A) by scaling and squaring of the Taylor series, for the small matrices of the tests."""
    k = max(0, int(np.ceil(np.log2(max(np.abs(A).sum(1).max(), 1e-300)))) + 1)
    M = A / 2.0**k
    E, term = np.eye(len(A)), np.eye(len(A))
    for i in range(1, terms):
        term = term @ M / i
        E = E + term
    for _ in range(k):
        E = E @ E
    return E


def linear(A):
    return lambda h, t: h @ A.T


def test_euler_step_is_a_residual_block():
    # One Euler step of size 1 is h + f(h, 0): the update of a residual block.
    rng = np.random.default_rng(0)
    cfg = config(3, 5, n_steps=1, T=1.0, method="euler")
    params, x = random_params(rng, cfg), rng.normal(size=(4, 3))
    np.testing.assert_allclose(forward(params, x, cfg), x + field(params, x, 0.0), rtol=0, atol=1e-15)


@pytest.mark.parametrize("method,order", [("euler", 1), ("rk4", 4)])
def test_order_of_convergence(method, order):
    # On dh/dt = A h the error at T falls as dt^order: halving the step divides it by about 2^order.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3)) * 0.7
    h0 = rng.normal(size=(2, 3))
    exact = h0 @ expm(A).T
    solver = {"euler": euler, "rk4": rk4}[method]
    errs = [np.abs(solver(linear(A), h0, 0.0, 1.0, n) - exact).max() for n in (16, 32, 64)]
    for e0, e1 in zip(errs, errs[1:]):
        assert e0 / e1 == pytest.approx(2.0**order, rel=0.15)


def test_dopri5_error_control():
    # A tighter tolerance gives a smaller error and costs more evaluations of f.
    A = np.array([[0.0, -4.0], [4.0, -0.1]])
    h0 = np.array([[1.0, 0.0]])
    exact = h0 @ expm(2.0 * A).T
    runs = [dopri5(linear(A), h0, 0.0, 2.0, rtol=r, atol=r * 1e-2) for r in (1e-3, 1e-6, 1e-9)]
    errs = [np.abs(h - exact).max() for h, _ in runs]
    evals = [n for _, n in runs]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-8
    assert evals[0] < evals[1] < evals[2]


def test_reverse_solve_recovers_the_input():
    # Solving the same ODE from T back to 0 returns the input, up to the error of the solver.
    rng = np.random.default_rng(2)
    cfg = config(3, 6, n_steps=40)
    params, x = random_params(rng, cfg), rng.normal(size=(5, 3))
    y = forward(params, x, cfg)
    back = rk4(lambda h, t: field(params, h, t), y, cfg["T"], 0.0, cfg["n_steps"])
    np.testing.assert_allclose(back, x, atol=1e-8)


def test_field_vjp_finite_differences():
    rng = np.random.default_rng(3)
    cfg = config(3, 4)
    params, h, a = random_params(rng, cfg), rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    t, eps = 0.4, 1e-6
    vh, grads = field_vjp(params, h, t, a)
    for i in np.ndindex(h.shape):
        hp, hm = h.copy(), h.copy()
        hp[i] += eps
        hm[i] -= eps
        fd = ((field(params, hp, t) - field(params, hm, t)) * a).sum() / (2 * eps)
        assert vh[i] == pytest.approx(fd, abs=1e-7)
    for k in NAMES:
        for i in np.ndindex(params[k].shape):
            pp = {n: v.copy() for n, v in params.items()}
            pm = {n: v.copy() for n, v in params.items()}
            pp[k][i] += eps
            pm[k][i] -= eps
            fd = ((field(pp, h, t) - field(pm, h, t)) * a).sum() / (2 * eps)
            assert grads[k][i] == pytest.approx(fd, abs=1e-7)


def loss_of(params, x, c, cfg):
    return float((forward(params, x, cfg) * c).sum())


def numeric_grads(params, x, c, cfg, eps=1e-6):
    gx = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        gx[i] = (loss_of(params, xp, c, cfg) - loss_of(params, xm, c, cfg)) / (2 * eps)
    gp = {}
    for k in NAMES:
        gp[k] = np.zeros_like(params[k])
        for i in np.ndindex(params[k].shape):
            pp = {n: v.copy() for n, v in params.items()}
            pm = {n: v.copy() for n, v in params.items()}
            pp[k][i] += eps
            pm[k][i] -= eps
            gp[k][i] = (loss_of(pp, x, c, cfg) - loss_of(pm, x, c, cfg)) / (2 * eps)
    return gx, gp


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_backprop_is_the_exact_gradient_of_the_solver(method):
    # Discretize, then optimize: the chain rule through the steps gives the gradient of the discrete map exactly.
    rng = np.random.default_rng(4)
    cfg = config(2, 3, d_state=3, n_steps=3, method=method)
    params, x = random_params(rng, cfg), rng.normal(size=(2, 2))
    c = rng.normal(size=(2, 3))
    gx, gp = backprop(params, x, c, cfg)
    nx, npar = numeric_grads(params, x, c, cfg)
    np.testing.assert_allclose(gx, nx, atol=1e-8)
    for k in NAMES:
        np.testing.assert_allclose(gp[k], npar[k], atol=1e-8)


def test_adjoint_matches_backprop_as_the_steps_shrink():
    # Optimize, then discretize: the adjoint gradient differs from the gradient of the discrete map by the error of
    # the solver, so the gap shrinks as the steps shrink (about 16 times per halving with RK4) and vanishes in the limit.
    rng = np.random.default_rng(5)
    base = config(2, 4, d_state=3, n_steps=2)
    params, x = random_params(rng, base, scale=1.5), rng.normal(size=(3, 2))
    c = rng.normal(size=(3, 3))
    gaps = []
    for n in (2, 4, 8, 16):
        cfg = config(2, 4, d_state=3, n_steps=n)
        y = forward(params, x, cfg)
        ax, ag = adjoint_backward(params, y, c, cfg)
        bx, bg = backprop(params, x, c, cfg)
        gaps.append(max([np.abs(ax - bx).max()] + [np.abs(ag[k] - bg[k]).max() for k in NAMES]))
    assert gaps[0] > 1e-6
    assert all(g0 > 8.0 * g1 for g0, g1 in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1e-5


def test_adjoint_on_a_linear_ode():
    # For dh/dt = A h and L = <c, h(T)>, dL/dh(0) = exp(A T)^T c: the adjoint ODE da/dt = -A^T a run backwards.
    rng = np.random.default_rng(6)
    A = rng.normal(size=(3, 3)) * 0.5
    c = rng.normal(size=(1, 3))
    a = rk4(lambda v, t: -v @ A, c, 1.0, 0.0, 200)
    np.testing.assert_allclose(a, c @ expm(A), atol=1e-10)


def test_adjoint_memory_does_not_grow_with_the_steps(monkeypatch):
    # The backward solve carries [h, a, g], 2 B D + P numbers, for any number of steps; backprop keeps 4 N B D.
    import neural_ode

    seen = []
    solve = neural_ode.SOLVERS["rk4"]

    def recording(f, z, t0, t1, n):
        seen.append(z.size)
        return solve(f, z, t0, t1, n)

    monkeypatch.setitem(neural_ode.SOLVERS, "rk4", recording)
    rng = np.random.default_rng(7)
    B, D = 3, 2
    for n in (2, 8, 32):
        cfg = config(2, 4, n_steps=n)
        params, x = random_params(rng, cfg), rng.normal(size=(B, 2))
        y = forward(params, x, cfg)
        seen.clear()
        adjoint_backward(params, y, np.ones_like(y), cfg)
        assert seen == [2 * B * D + param_count(cfg)]


def test_trace_jacobian():
    rng = np.random.default_rng(8)
    cfg = config(3, 5)
    params, h = random_params(rng, cfg), rng.normal(size=(4, 3))
    eps, t = 1e-6, 0.3
    for b in range(4):
        J = np.zeros((3, 3))
        for j in range(3):
            hp, hm = h[b:b + 1].copy(), h[b:b + 1].copy()
            hp[0, j] += eps
            hm[0, j] -= eps
            J[:, j] = (field(params, hp, t) - field(params, hm, t))[0] / (2 * eps)
        assert trace_jacobian(params, h[b:b + 1], t)[0] == pytest.approx(np.trace(J), abs=1e-7)


def test_instantaneous_change_of_variables():
    # The integral of -tr(df/dh) along a path equals -log |det| of the Jacobian of the flow map.
    rng = np.random.default_rng(9)
    cfg = config(2, 6, n_steps=50)
    params, x = random_params(rng, cfg), rng.normal(size=(3, 2))
    y, dlogp = log_det_flow(params, x, cfg)
    np.testing.assert_allclose(y, forward(params, x, cfg), atol=1e-12)
    eps = 1e-6
    for b in range(3):
        J = np.zeros((2, 2))
        for j in range(2):
            xp, xm = x[b:b + 1].copy(), x[b:b + 1].copy()
            xp[0, j] += eps
            xm[0, j] -= eps
            J[:, j] = (forward(params, xp, cfg) - forward(params, xm, cfg))[0] / (2 * eps)
        assert dlogp[b] == pytest.approx(-np.log(abs(np.linalg.det(J))), abs=1e-6)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10**6))
def test_one_dimensional_flow_preserves_order(seed):
    # In one dimension trajectories cannot cross, so the flow map is increasing: x1 < x2 gives h1(T) < h2(T).
    rng = np.random.default_rng(seed)
    cfg = config(1, 6, n_steps=20, T=2.0)
    params = random_params(rng, cfg, scale=2.0)
    x = np.sort(rng.uniform(-3, 3, size=(12, 1)), axis=0)
    y = forward(params, x, cfg)
    assert np.all(np.diff(y[:, 0]) > 0)


def test_one_dimensional_flow_cannot_fit_g1d():
    # A flow keeps h(-1) <= h(1), so on g(-1) = 1, g(1) = -1 its squared errors sum to at least 2 and its loss, half
    # their mean, is at least 1/2; training the plain model stays above that bound, while one residual step h - 2h
    # fits g exactly.
    rng = np.random.default_rng(10)
    x, t = np.array([[-1.0], [1.0]]), np.array([[1.0], [-1.0]])
    cfg = config(1, 8, n_steps=10)
    _, losses = train(random_params(rng, cfg), x, t, cfg, lr=0.2, steps=150)
    assert min(losses) >= 0.5 - 1e-9
    assert np.array_equal(x + (-2.0 * x), t)


def test_augmentation_can_fit_g1d():
    # With one extra dimension, the rotation by pi, dh/dt = [[0, -pi], [pi, 0]] h, maps (x, 0) to (-x, 0).
    R = np.array([[0.0, -np.pi], [np.pi, 0.0]])
    x = augment(np.array([[-1.0], [1.0]]), config(1, 1, d_state=2))
    y = rk4(linear(R), x, 0.0, 1.0, 200)
    np.testing.assert_allclose(y, [[1.0, 0.0], [-1.0, 0.0]], atol=1e-8)


def test_training_reduces_loss():
    rng = np.random.default_rng(11)
    cfg = config(2, 8, d_state=3, n_steps=4)
    x = rng.normal(size=(16, 2))
    t = np.stack([x[:, 0] * x[:, 1], np.sin(x[:, 0])], 1)
    _, losses = train(random_params(rng, cfg, scale=0.5), x, t, cfg, lr=0.1, steps=200)
    assert losses[-1] < 0.5 * losses[0]


def test_batch_independence():
    rng = np.random.default_rng(12)
    cfg = config(2, 5, d_state=4, n_steps=3)
    params, x = random_params(rng, cfg), rng.normal(size=(6, 2))
    y = forward(params, x, cfg)
    for b in range(6):
        np.testing.assert_allclose(forward(params, x[b:b + 1], cfg), y[b:b + 1], rtol=0, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(np.random.default_rng(0), cfg)
    x = np.zeros((k["B"], k["d"]))
    assert list(augment(x, cfg).shape) == c["blocks"]["augment"]["outputs"]["h0"]
    assert list(forward(params, x, cfg).shape) == c["blocks"]["solve"]["outputs"]["y"]
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    assert flops(cfg, k["B"]) == c["model"]["flops_forward"]
    assert nfe(cfg) == 4 * k["n_steps"]


def test_shared_fixture():
    raw = dict(np.load(FIXTURES / "params-small.npz"))
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    params = {k: raw[k] for k in NAMES}
    y = forward(params, inp["x"], cfg)
    np.testing.assert_allclose(y, want["y"], rtol=1e-12, atol=1e-14)
    loss, dyd = mse_loss(y[:, :cfg["d"]], inp["t"])
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    dy = np.zeros_like(y)
    dy[:, :cfg["d"]] = dyd
    dx, grads = backprop(params, inp["x"], dy, cfg)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(grads[k], want[f"d{k}"], rtol=1e-10, atol=1e-12)


def test_config_errors():
    for bad in ({"d": 0}, {"d_hidden": 0}, {"n_steps": 0}, {"T": 0.0}, {"method": "rk45"}, {"d": 3, "d_state": 2}):
        with pytest.raises(ValueError):
            config(**bad)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), rng.normal(size=(k["B"], k["d"])), cfg)
