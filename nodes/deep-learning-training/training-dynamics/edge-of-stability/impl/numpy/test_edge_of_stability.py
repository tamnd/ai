# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://NEW-edge-of-stability).
"""Tests of the NumPy reference: gradient descent on a quadratic converges just below 2 / lambda_max and diverges
just above, with each eigen-coordinate multiplied by 1 - eta lambda_i; the momentum thresholds; the central
difference HVP and power iteration against eigvalsh; the hand-written gradient of the tanh MLP against finite
differences; the exact catapult updates; the two-variable self-stabilization model; and the small experiment of the
page, in which the sharpness rises to 2 / eta and hovers there while the loss goes down non-monotonically."""

import numpy as np
import pytest

from edge_of_stability import (
    catapult_closed_form,
    catapult_f_lambda,
    catapult_step,
    conserved_g,
    demo,
    eos_map,
    forward,
    hvp_central,
    init_mlp,
    is_stable,
    loss_and_grad,
    make_regression,
    mode_factors,
    momentum_gd_1d,
    mss_nesterov,
    mss_polyak,
    power_iteration,
    quadratic_gd,
    sharpness,
    train_gd,
    unflatten,
)


def random_psd(d, rng):
    A = rng.standard_normal((d, d))
    return A @ A.T / d


# Quadratics


def test_mode_factors_values():
    assert np.allclose(mode_factors([20.0, 1.0], 0.09), [-0.8, 0.91])
    assert np.allclose(mode_factors([20.0, 1.0], 0.11), [-1.2, 0.89])
    assert mode_factors([0.0], 0.5)[0] == 1.0


def test_is_stable_at_the_threshold():
    for lam in (0.5, 3.0, 20.0, 1000.0):
        assert is_stable([lam, lam / 7], 0.99 * 2 / lam)
        assert not is_stable([lam, lam / 7], 1.01 * 2 / lam)
        assert not is_stable([lam], 2 / lam)  # the boundary oscillates without decay
    assert not is_stable([1.0, 0.0], 0.5)  # a flat direction never contracts
    assert is_stable([1.0], 1.0)


def test_cohen_figure_two():
    H = np.diag([20.0, 1.0])
    x0 = np.array([1.0, 1.0])
    conv = quadratic_gd(H, x0, 0.09, 100)
    div = quadratic_gd(H, x0, 0.11, 100)
    assert conv.shape == (101, 2)
    assert np.allclose(conv[0], x0) and np.allclose(div[0], x0)
    assert abs(conv[-1, 0] - 0.8**100) < 1e-14 and abs(conv[-1, 0] - 2.037e-10) < 1e-12
    assert abs(div[-1, 0] / 1.2**100 - 1) < 1e-10 and abs(div[-1, 0] / 8.282e7 - 1) < 1e-3
    assert np.allclose(div[-1, 1], 0.89**100) and np.allclose(conv[-1, 1], 0.91**100)
    signs = np.sign(div[1:, 0])
    assert np.all(signs[1:] == -signs[:-1])  # it oscillates along q1


def test_quadratic_matches_modes_in_eigenbasis():
    rng = np.random.default_rng(3)
    H = random_psd(6, rng)
    lam, Q = np.linalg.eigh(H)
    x0 = rng.standard_normal(6)
    for scale in (0.99, 1.01):
        eta = scale * 2 / lam[-1]
        xs = quadratic_gd(H, x0, eta, 400)
        coords = xs @ Q
        expect = (Q.T @ x0)[None, :] * mode_factors(lam, eta)[None, :] ** np.arange(401)[:, None]
        assert np.allclose(coords, expect, atol=1e-8, rtol=1e-8)
        top = abs(coords[-1, -1])
        if scale < 1:
            assert top < 1e-1 * abs(coords[0, -1])
        else:
            assert top > 10 * abs(coords[0, -1])


def test_mss_formulas():
    assert mss_polyak(0.01, 0.9) == pytest.approx(380.0)
    assert mss_nesterov(0.01, 0.9) == pytest.approx(380.0 / 2.8)
    assert mss_polyak(0.1, 0.0) == pytest.approx(20.0) == mss_nesterov(0.1, 0.0)
    assert mss_nesterov(0.01, 0.5) == pytest.approx(150.0)


@pytest.mark.parametrize("beta", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("nesterov", [False, True])
def test_momentum_diverges_above_mss(beta, nesterov):
    eta = 0.01
    mss = mss_nesterov(eta, beta) if nesterov else mss_polyak(eta, beta)
    below = momentum_gd_1d(0.98 * mss, eta, beta, 4000, nesterov)
    above = momentum_gd_1d(1.02 * mss, eta, beta, 400, nesterov)
    assert len(below) == 4001 and below[0] == 1.0
    assert abs(below[-1]) < 1e-3
    assert abs(above[-1]) > 1e3


def test_momentum_first_steps():
    p = momentum_gd_1d(2.0, 0.1, 0.5, 2)
    assert np.allclose(p, [1.0, 0.8, 0.8 - 0.16 - 0.1])
    n = momentum_gd_1d(2.0, 0.1, 0.5, 2, nesterov=True)
    y = 0.8 + 0.5 * (0.8 - 1.0)
    assert np.allclose(n, [1.0, 0.8, 0.8 * y])


# HVPs and power iteration


def test_hvp_is_exact_on_a_quadratic():
    rng = np.random.default_rng(1)
    H = random_psd(8, rng)
    theta, v = rng.standard_normal(8), rng.standard_normal(8)
    for eps in (1e-1, 1e-4):
        assert np.allclose(hvp_central(lambda t: H @ t, theta, v, eps), H @ v, atol=1e-9)


def test_hvp_error_is_second_order():
    # grad of sum(t^4)/4 is t^3; the central difference error is eps^2 v^3
    theta, v = np.array([0.5, -1.0]), np.array([1.0, 2.0])
    exact = 3 * theta**2 * v
    for eps in (1e-1, 1e-2):
        err = hvp_central(lambda t: t**3, theta, v, eps) - exact
        assert np.allclose(err, eps**2 * v**3, rtol=1e-6)


def test_power_iteration_matches_eigvalsh():
    rng = np.random.default_rng(2)
    for d in (3, 10, 25):
        H = random_psd(d, rng)
        lam, q = power_iteration(lambda v, H=H: H @ v, d, iters=2000, rng=np.random.default_rng(d))
        top = np.linalg.eigvalsh(H)[-1]
        assert lam == pytest.approx(top, rel=1e-6)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert np.linalg.norm(H @ q - top * q) < 1e-3 * top


def test_power_iteration_finds_largest_modulus():
    H = np.diag([3.0, 1.0, -5.0])
    lam, _ = power_iteration(lambda v, H=H: H @ v, 3, 300, np.random.default_rng(0))
    assert lam == pytest.approx(-5.0, rel=1e-8)
    lam, q = power_iteration(lambda v: 0 * v, 3, 5, np.random.default_rng(0))
    assert lam == 0.0 and np.linalg.norm(q) == pytest.approx(1.0)
    lam, q = power_iteration(lambda v, H=H: H @ v, 3, 0, np.random.default_rng(0))
    assert lam == 0.0 and np.linalg.norm(q) == pytest.approx(1.0)


def test_power_iteration_needs_iterations():
    H = np.diag([1.0, 0.99, 0.5])
    one, _ = power_iteration(lambda v, H=H: H @ v, 3, iters=1, rng=np.random.default_rng(0))
    many, _ = power_iteration(lambda v, H=H: H @ v, 3, iters=3000, rng=np.random.default_rng(0))
    assert many == pytest.approx(1.0, abs=1e-6)
    assert abs(one - 1.0) > 1e-3


# The MLP


def test_init_and_unflatten():
    rng = np.random.default_rng(0)
    sizes = [5, 400, 3]
    theta = init_mlp(sizes, rng)
    assert theta.size == 5 * 400 + 400 + 400 * 3 + 3
    (W1, b1), (W2, b2) = unflatten(theta, sizes)
    assert W1.shape == (5, 400) and b1.shape == (400,) and W2.shape == (400, 3) and b2.shape == (3,)
    assert np.all(b1 == 0) and np.all(b2 == 0)
    assert np.std(W1) == pytest.approx(5**-0.5, rel=0.05)
    assert np.std(W2) == pytest.approx(400**-0.5, rel=0.05)
    with pytest.raises(ValueError):
        unflatten(theta[:-1], sizes)


def test_forward_and_loss():
    sizes = [2, 3, 1]
    theta = np.zeros(2 * 3 + 3 + 3 * 1 + 1)
    W1, b1 = np.arange(6.0).reshape(2, 3) / 10, np.array([0.1, -0.2, 0.3])
    W2, b2 = np.array([[1.0], [-2.0], [0.5]]), np.array([0.25])
    theta[:] = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
    X = np.array([[1.0, -1.0], [0.5, 2.0]])
    Y = np.array([[0.0], [1.0]])
    out = np.tanh(X @ W1 + b1) @ W2 + b2
    assert np.allclose(forward(theta, sizes, X), out)
    loss, _ = loss_and_grad(theta, sizes, X, Y)
    assert loss == pytest.approx(0.5 * np.sum((out - Y) ** 2) / 2)


def test_gradient_against_finite_differences():
    rng = np.random.default_rng(4)
    sizes = [3, 4, 5, 2]
    theta = init_mlp(sizes, rng) + 0.1 * rng.standard_normal(3 * 4 + 4 + 4 * 5 + 5 + 5 * 2 + 2)
    X, Y = rng.standard_normal((7, 3)), rng.standard_normal((7, 2))
    _, g = loss_and_grad(theta, sizes, X, Y)
    num = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = 1e-6
        num[i] = (loss_and_grad(theta + e, sizes, X, Y)[0] - loss_and_grad(theta - e, sizes, X, Y)[0]) / 2e-6
    assert np.allclose(g, num, atol=1e-8, rtol=1e-6)


def test_sharpness_matches_full_hessian():
    rng = np.random.default_rng(5)
    sizes = [3, 6, 1]
    X, Y = make_regression(20, 3, rng)
    theta = init_mlp(sizes, rng)
    for _ in range(200):  # move toward a minimum, where the Hessian is close to PSD
        theta -= 0.2 * loss_and_grad(theta, sizes, X, Y)[1]
    d = theta.size

    def grad(t):
        return loss_and_grad(t, sizes, X, Y)[1]

    Hm = np.stack([hvp_central(grad, theta, e, 1e-5) for e in np.eye(d)])
    Hm = (Hm + Hm.T) / 2
    ev = np.linalg.eigvalsh(Hm)
    assert ev[-1] > abs(ev[0])
    assert sharpness(theta, sizes, X, Y, 3000, 1e-4, 0) == pytest.approx(ev[-1], rel=1e-5)


def test_make_regression():
    X, Y = make_regression(50, 4, np.random.default_rng(0))
    assert X.shape == (50, 4) and Y.shape == (50, 1)
    assert np.all(np.abs(Y) <= 1)
    X2, Y2 = make_regression(50, 4, np.random.default_rng(0))
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2)
    w = np.linalg.lstsq(X, np.arcsin(np.clip(Y, -1, 1)), rcond=None)[0]
    assert np.std(Y) > 0.3 and w.shape == (4, 1)


def test_train_gd_records_and_descends_at_small_step():
    rng = np.random.default_rng(6)
    sizes = [3, 8, 1]
    X, Y = make_regression(16, 3, rng)
    theta0 = init_mlp(sizes, rng)
    keep = theta0.copy()
    theta, losses, at, sharp = train_gd(theta0, sizes, X, Y, 0.05, 50, 10, 50)
    assert np.array_equal(theta0, keep)  # the start is not modified
    assert len(losses) == 51 and list(at) == [0, 10, 20, 30, 40, 50] and len(sharp) == 6
    assert np.all(np.diff(losses) < 0)
    assert losses[0] == pytest.approx(loss_and_grad(theta0, sizes, X, Y)[0])
    assert losses[-1] == pytest.approx(loss_and_grad(theta, sizes, X, Y)[0])
    assert sharp[0] == sharpness(theta0, sizes, X, Y, 50, 1e-4, 0)
    t1, _ = loss_and_grad(theta0, sizes, X, Y)[1], None
    one = train_gd(theta0, sizes, X, Y, 0.05, 1, 1, 5)[0]
    assert np.allclose(one, theta0 - 0.05 * t1)


# The catapult model and the self-stabilization model


def test_catapult_closed_form_is_exact():
    rng = np.random.default_rng(7)
    for n, eta in [(10, 0.5), (100, 1.5), (1000, 3.0), (50, 5.0)]:
        u, v = rng.standard_normal(n), rng.standard_normal(n)
        for _ in range(5):
            f, lam = catapult_f_lambda(u, v)
            u, v = catapult_step(u, v, eta)
            f2, lam2 = catapult_f_lambda(u, v)
            pf, pl = catapult_closed_form(f, lam, eta, n)
            assert pf == pytest.approx(f2, rel=1e-9, abs=1e-12)
            assert pl == pytest.approx(lam2, rel=1e-12)
            if f != 0 and eta * lam < 4:
                assert lam2 < lam


def test_catapult_f_lambda_values():
    u, v = np.array([1.0, 2.0, 2.0, 0.0]), np.array([0.0, 1.0, -1.0, 3.0])
    f, lam = catapult_f_lambda(u, v)
    assert f == pytest.approx(0.0) and lam == pytest.approx((9 + 11) / 4)
    u2, v2 = catapult_step(np.array([1.0, 1.0]), np.array([1.0, 0.0]), 0.5)
    f = 1 / np.sqrt(2)
    assert np.allclose(u2, [1 - 0.5 * f / np.sqrt(2), 1.0]) and np.allclose(
        v2, [1 - 0.5 * f / np.sqrt(2), -0.5 * f / np.sqrt(2)]
    )


def test_catapult_phases():
    # lazy below 2 / lambda0: lambda barely moves; catapult between 2 / lambda0 and 4 / lambda0: lambda ends below 2 / eta
    rng = np.random.default_rng(8)
    n = 2000
    u0, v0 = rng.standard_normal(n), rng.standard_normal(n)
    _, lam0 = catapult_f_lambda(u0, v0)
    for c, lazy in [(1.5, True), (3.0, False)]:
        eta = c / lam0
        u, v = u0.copy(), v0.copy()
        fmax = 0.0
        for _ in range(300):
            u, v = catapult_step(u, v, eta)
            fmax = max(fmax, abs(catapult_f_lambda(u, v)[0]))
        f, lam = catapult_f_lambda(u, v)
        assert abs(f) < 1e-6
        if lazy:
            assert lam == pytest.approx(lam0, rel=0.01)
        else:
            assert lam < 2 / eta and fmax > 3
    u, v = u0.copy(), v0.copy()
    for _ in range(300):
        u, v = catapult_step(u, v, 4.5 / lam0)
    assert not np.isfinite(catapult_f_lambda(u, v)[0]) or abs(catapult_f_lambda(u, v)[0]) > 1e6


def test_eos_map_step_and_fixed_point():
    x, y = eos_map(0.5, 0.2, 0.1, 1.0, 2.0)
    assert x == pytest.approx(-(1.02) * 0.5) and y == pytest.approx(0.2 + 0.1 * (1.0 - 0.25))
    delta = np.sqrt(2 * 3.0 / 1.5)
    x, y = eos_map(delta, 0.0, 0.01, 3.0, 1.5)
    assert x == pytest.approx(-delta) and y == pytest.approx(0.0, abs=1e-15)


def test_eos_map_cycles_and_stays_bounded():
    x, y = 0.5, 0.0
    xs, ys = [], []
    for _ in range(5000):
        x, y = eos_map(x, y, 0.01, 1.0, 1.0)
        xs.append(abs(x))
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    assert xs.max() < 5 and np.abs(ys).max() < 3
    assert ys.max() > 0.5 and ys.min() < -0.5  # the sharpness goes above and below 2 / eta


def test_conserved_quantity_under_rk4():
    alpha, beta = 1.3, 0.7

    def f(s):
        return np.array([s[0] * s[1], alpha - beta * s[0] ** 2 / 2])

    s = np.array([0.2, 0.0])
    g0 = conserved_g(s[0], s[1], alpha, beta)
    h = 1e-3
    for _ in range(20000):
        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert conserved_g(s[0], s[1], alpha, beta) == pytest.approx(g0, rel=1e-8)
    assert conserved_g(np.sqrt(2 * alpha / beta), 0.0, alpha, beta) == pytest.approx(0.0, abs=1e-12)
    assert conserved_g(1.0, 2.0, 2.0, 4.0) == pytest.approx(1 - 0.0 - 1 + 2.0)


# The experiment


def late(r):
    """The sharpness measured at steps 1000 to 1600 of a demo run."""
    return [s for t, s in zip(r["measured_at"], r["sharpness"], strict=True) if t >= 1000]


def test_sharpness_rises_to_two_over_eta_and_hovers():
    r = demo(0, 0.25, 1600)
    assert r["sizes"] == [5, 16, 16, 1] and r["examples"] == 64
    assert r["measured_at"] == list(range(0, 1601, 100)) and len(r["losses"]) == 1601
    thr = r["threshold"]
    assert thr == 8.0
    assert r["sharpness"][0] < 0.5 * thr
    assert len(late(r)) == 7
    assert all(abs(s - thr) < 0.05 * thr for s in late(r))
    assert np.sum(np.diff(r["losses"]) > 0) > 100  # the loss goes up on many steps
    assert r["losses"][-1] < r["losses"][0] / 100


def test_small_step_stays_below_threshold_and_descends():
    r = demo(0, 0.05, 1600)
    assert r["threshold"] == pytest.approx(40.0)
    assert max(late(r)) < 0.25 * r["threshold"]
    assert late(r)[-1] > late(r)[0] > r["sharpness"][0]  # progressive sharpening
    assert np.all(np.diff(r["losses"]) <= 0)
