# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://NEW-mode-connectivity).
"""Tests of the NumPy reference: the endpoints and midpoints of the linear, Bezier and polychain paths; the bend
weight against a finite difference; the loss barrier in both conventions, zero for a convex loss and positive between
a network and a permuted copy; the hand-written gradient of the MLP against finite differences; the invariance of
the network under a hidden-unit permutation; the Hungarian solver against brute force and SciPy; weight matching
that recovers a known permutation; curve fitting on a ring-shaped loss; and the small experiment of the page."""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

from mode_connectivity import (
    bend_weight,
    bezier_path,
    demo,
    error,
    fit_bend,
    flatten,
    forward,
    grad,
    init_mlp,
    linear_assignment,
    linear_path,
    loss,
    loss_barrier,
    make_data,
    permute_hidden,
    polychain_path,
    train_gd,
    unflatten,
    weight_matching,
)

W1 = np.array([1.0, -2.0, 0.5])
TH = np.array([3.0, 4.0, -1.0])
W2 = np.array([-1.0, 0.0, 2.5])


def ring(w):
    """(|w|^2 - 1)^2, minimal on the unit circle."""
    w = np.asarray(w)
    return float((w @ w - 1.0) ** 2)


def ring_grad(w):
    return 4.0 * (w @ w - 1.0) * w


def small_net(rng, d=3, h=5, k=4):
    p = init_mlp(d, h, k, rng)
    p["b1"] = rng.standard_normal(h)
    p["b2"] = rng.standard_normal(k)
    return p


# Paths


def test_linear_path_points():
    assert np.allclose(linear_path(W1, W2, 0.0), W1)
    assert np.allclose(linear_path(W1, W2, 1.0), W2)
    assert np.allclose(linear_path(W1, W2, 0.25), 0.75 * W1 + 0.25 * W2)


def test_bezier_endpoints_and_midpoint():
    assert np.allclose(bezier_path(W1, TH, W2, 0.0), W1)
    assert np.allclose(bezier_path(W1, TH, W2, 1.0), W2)
    assert np.allclose(bezier_path(W1, TH, W2, 0.5), (W1 + 2.0 * TH + W2) / 4.0)
    assert np.allclose(bezier_path(W1, TH, W2, 0.25), (9.0 * W1 + 6.0 * TH + W2) / 16.0)


def test_bezier_with_bend_on_segment_is_the_segment_reparametrized():
    mid = 0.5 * (W1 + W2)
    for t in (0.1, 0.3, 0.8):
        assert np.allclose(bezier_path(W1, mid, W2, t), linear_path(W1, W2, t))


def test_polychain_points():
    assert np.allclose(polychain_path(W1, TH, W2, 0.0), W1)
    assert np.allclose(polychain_path(W1, TH, W2, 0.25), 0.5 * (W1 + TH))
    assert np.allclose(polychain_path(W1, TH, W2, 0.5), TH)
    assert np.allclose(polychain_path(W1, TH, W2, 0.75), 0.5 * (TH + W2))
    assert np.allclose(polychain_path(W1, TH, W2, 1.0), W2)


@pytest.mark.parametrize("kind,path", [("bezier", bezier_path), ("polychain", polychain_path)])
@pytest.mark.parametrize("t", [0.0, 0.2, 0.5, 0.7, 1.0])
def test_bend_weight_is_derivative_in_theta(kind, path, t):
    e = np.array([1.0, 0.0, 0.0]) * 1e-6
    fd = (path(W1, TH + e, W2, t) - path(W1, TH - e, W2, t))[0] / 2e-6
    assert bend_weight(kind, t) == pytest.approx(fd, abs=1e-8)


def test_bend_weight_values_and_unknown_kind():
    assert bend_weight("bezier", 0.5) == 0.5
    assert bend_weight("polychain", 0.5) == 1.0
    assert bend_weight("polychain", 0.25) == 0.5
    assert bend_weight("polychain", 0.75) == 0.5
    with pytest.raises(ValueError):
        bend_weight("spline", 0.5)


# The loss barrier


def test_barrier_is_zero_for_a_convex_loss():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    H = A @ A.T + np.eye(3)

    def f(w):
        return float(0.5 * w @ H @ w + w.sum())

    ts = np.linspace(0.0, 1.0, 51)
    b_lin, t_lin, losses = loss_barrier(f, lambda t: linear_path(W1, W2, t), ts, reference="linear")
    assert b_lin == pytest.approx(0.0, abs=1e-12)
    assert t_lin in (0.0, 1.0)
    assert losses.shape == (51,)
    assert losses[0] == pytest.approx(f(W1))
    assert losses[-1] == pytest.approx(f(W2))
    b_mean, t_mean, _ = loss_barrier(f, lambda t: linear_path(W1, W2, t), ts, reference="mean")
    # With unequal endpoint losses, the mean reference gives max(L0, L1) - mean, reached at an end.
    assert b_mean == pytest.approx(0.5 * abs(f(W1) - f(W2)), rel=1e-9)
    assert t_mean == (0.0 if f(W1) > f(W2) else 1.0)


def test_barrier_of_the_ring_on_the_segment_and_on_a_bezier_curve():
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    ts = np.linspace(0.0, 1.0, 401)
    b_lin, t_lin, _ = loss_barrier(ring, lambda t: linear_path(a, b, t), ts)
    assert b_lin == pytest.approx(1.0)
    assert t_lin == pytest.approx(0.5)
    b_bez, t_bez, _ = loss_barrier(ring, lambda t: bezier_path(a, [0.0, 2.0], b, t), ts)
    assert b_bez == pytest.approx(1.0 / 16.0, abs=1e-4)
    assert min(abs(t_bez - (1 - np.sqrt(0.5)) / 2), abs(t_bez - (1 + np.sqrt(0.5)) / 2)) < 3e-3


def test_barrier_conventions_differ_on_a_tilted_loss():
    # L(t) = t + 4 t (1 - t): the linear reference removes the tilt and leaves max 4 t (1 - t) = 1 at t = 1/2.
    ts = np.linspace(0.0, 1.0, 101)

    def f(w):
        return float(w[0] + 4.0 * w[0] * (1.0 - w[0]))

    b_lin, t_lin, _ = loss_barrier(f, lambda t: np.array([t]), ts, reference="linear")
    b_mean, t_mean, _ = loss_barrier(f, lambda t: np.array([t]), ts, reference="mean")
    assert b_lin == pytest.approx(1.0)
    assert t_lin == pytest.approx(0.5)
    # max of t + 4 t (1 - t) is at t = 5/8, value 25/16; minus the mean 1/2.
    assert b_mean == pytest.approx(25.0 / 16.0 - 0.5, abs=1e-3)
    assert t_mean == pytest.approx(0.62, abs=0.011)


def test_barrier_rejects_bad_grids_and_references():
    with pytest.raises(ValueError):
        loss_barrier(ring, lambda t: np.array([t, 0.0]), [0.1, 1.0])
    with pytest.raises(ValueError):
        loss_barrier(ring, lambda t: np.array([t, 0.0]), [0.0, 0.9])
    with pytest.raises(ValueError):
        loss_barrier(ring, lambda t: np.array([t, 0.0]), [0.0, 1.0], reference="max")


# The MLP


def test_init_shapes_and_scale():
    p = init_mlp(50, 400, 3, np.random.default_rng(0))
    assert p["W1"].shape == (400, 50) and p["W2"].shape == (3, 400)
    assert p["b1"].shape == (400,) and p["b2"].shape == (3,)
    assert not p["b1"].any() and not p["b2"].any()
    assert p["W1"].std() == pytest.approx(np.sqrt(2.0 / 50), rel=0.05)
    assert p["W2"].std() == pytest.approx(np.sqrt(2.0 / 400), rel=0.1)


def test_flatten_roundtrip_and_order():
    p = small_net(np.random.default_rng(1))
    v = flatten(p)
    assert v.size == 5 * 3 + 5 + 4 * 5 + 4
    assert np.array_equal(v[:15], p["W1"].ravel())
    assert np.array_equal(v[15:20], p["b1"])
    assert np.array_equal(v[-4:], p["b2"])
    q = unflatten(v, 3, 5, 4)
    for n in p:
        assert np.array_equal(p[n], q[n])
    with pytest.raises(ValueError):
        unflatten(np.zeros(v.size + 1), 3, 5, 4)


def test_forward_loss_and_error_against_loops():
    rng = np.random.default_rng(2)
    p = small_net(rng)
    X = rng.standard_normal((6, 3))
    y = np.array([0, 1, 2, 3, 0, 1])
    z = forward(p, X)
    for i in range(6):
        hidden = np.maximum(p["W1"] @ X[i] + p["b1"], 0.0)
        assert np.allclose(z[i], p["W2"] @ hidden + p["b2"])
    ce = np.mean([np.log(np.exp(z[i]).sum()) - z[i, y[i]] for i in range(6)])
    assert loss(p, X, y) == pytest.approx(ce)
    assert error(p, X, y) == pytest.approx(np.mean(z.argmax(axis=1) != y))
    big = {n: v * 100.0 for n, v in p.items()}
    assert np.isfinite(loss(big, X, y))


def test_grad_against_finite_differences():
    rng = np.random.default_rng(3)
    p = small_net(rng)
    X = rng.standard_normal((7, 3))
    y = rng.integers(0, 4, size=7)
    g = flatten(grad(p, X, y))
    v = flatten(p)
    eps = 1e-6
    for i in range(v.size):
        e = np.zeros_like(v)
        e[i] = eps
        fd = (loss(unflatten(v + e, 3, 5, 4), X, y) - loss(unflatten(v - e, 3, 5, 4), X, y)) / (2 * eps)
        assert g[i] == pytest.approx(fd, abs=1e-6)


def test_train_gd_is_gradient_descent():
    rng = np.random.default_rng(4)
    p = small_net(rng)
    X = rng.standard_normal((9, 3))
    y = rng.integers(0, 4, size=9)
    before = {n: v.copy() for n, v in p.items()}
    q = train_gd(p, X, y, 0.1, 1)
    g = grad(p, X, y)
    for n in p:
        assert np.allclose(q[n], p[n] - 0.1 * g[n])
        assert np.array_equal(p[n], before[n])
    assert loss(train_gd(p, X, y, 0.1, 50), X, y) < loss(p, X, y)


# Permutations


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_permutation_leaves_the_function_unchanged(seed):
    rng = np.random.default_rng(seed)
    p = small_net(rng)
    perm = rng.permutation(5)
    q = permute_hidden(p, perm)
    X = rng.standard_normal((4, 3))
    assert np.allclose(forward(p, X), forward(q, X))


def test_permute_hidden_moves_units():
    p = small_net(np.random.default_rng(5))
    perm = np.array([2, 0, 4, 1, 3])
    q = permute_hidden(p, perm)
    for i, j in enumerate(perm):
        assert np.array_equal(q["W1"][i], p["W1"][j])
        assert q["b1"][i] == p["b1"][j]
        assert np.array_equal(q["W2"][:, i], p["W2"][:, j])
    assert np.array_equal(q["b2"], p["b2"])
    q["b2"][0] += 1.0
    assert q["b2"][0] != p["b2"][0]


def test_a_permuted_copy_has_a_barrier():
    X, y = make_data(128, np.random.default_rng(0))
    a = train_gd(init_mlp(2, 16, 2, np.random.default_rng(1)), X, y, 0.5, 600)
    perm = np.roll(np.arange(16), 1)
    b = permute_hidden(a, perm)
    assert loss(b, X, y) == pytest.approx(loss(a, X, y))

    def f(w):
        return loss(unflatten(w, 2, 16, 2), X, y)

    ts = np.linspace(0.0, 1.0, 21)
    barrier, t, _ = loss_barrier(f, lambda t: linear_path(flatten(a), flatten(b), t), ts)
    assert barrier > 0.05
    assert 0.0 < t < 1.0


# Linear assignment and weight matching


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_linear_assignment_against_brute_force(n):
    rng = np.random.default_rng(n)
    for _ in range(10):
        S = rng.standard_normal((n, n))
        perm = linear_assignment(S)
        best = max(sum(S[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        assert sorted(perm.tolist()) == list(range(n))
        assert S[np.arange(n), perm].sum() == pytest.approx(best)


@pytest.mark.parametrize("n", [10, 31, 64])
def test_linear_assignment_against_scipy(n):
    rng = np.random.default_rng(100 + n)
    S = rng.standard_normal((n, n))
    r, c = linear_sum_assignment(S, maximize=True)
    perm = linear_assignment(S)
    assert S[np.arange(n), perm].sum() == pytest.approx(S[r, c].sum())


def test_linear_assignment_finds_a_planted_permutation():
    rng = np.random.default_rng(7)
    true = rng.permutation(12)
    S = 0.1 * rng.standard_normal((12, 12))
    S[np.arange(12), true] += 5.0
    assert np.array_equal(linear_assignment(S), true)
    with pytest.raises(ValueError):
        linear_assignment(np.zeros((2, 3)))


def test_weight_matching_recovers_a_known_permutation():
    rng = np.random.default_rng(8)
    a = small_net(rng, d=4, h=9, k=3)
    true = rng.permutation(9)
    b = permute_hidden(a, np.argsort(true))
    perm, b_matched = weight_matching(a, b)
    assert np.array_equal(perm, true)
    for n in a:
        assert np.allclose(b_matched[n], a[n])


def test_weight_matching_uses_both_layers_and_biases():
    rng = np.random.default_rng(9)
    a = small_net(rng, d=2, h=6, k=2)
    for zeroed in (("W1", "b1"), ("W1", "W2"), ("b1", "W2")):
        c = {n: (np.zeros_like(v) if n in zeroed else v) for n, v in a.items()}
        true = rng.permutation(6)
        perm, _ = weight_matching(c, permute_hidden(c, np.argsort(true)))
        assert np.array_equal(perm, true)


# Curve finding


def test_fit_bend_starts_at_the_midpoint_or_theta0():
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    rng = np.random.default_rng(0)
    assert np.allclose(fit_bend(ring_grad, a, b, 0, 0.1, rng), [0.0, 0.0])
    assert np.allclose(fit_bend(ring_grad, a, b, 0, 0.1, rng, theta0=[0.0, 0.3]), [0.0, 0.3])


def test_fit_bend_finds_a_low_curve_on_the_ring():
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    theta = fit_bend(ring_grad, a, b, 3000, 0.05, np.random.default_rng(0), theta0=[0.0, 0.3])
    ts = np.linspace(0.0, 1.0, 201)
    barrier, _, _ = loss_barrier(ring, lambda t: bezier_path(a, theta, b, t), ts)
    assert barrier < 0.07
    assert 1.8 < theta[1] < 2.5
    assert abs(theta[0]) < 0.1


def test_fit_bend_takes_one_gradient_step_per_sample():
    a, b = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    t = np.random.default_rng(3).uniform()
    theta = fit_bend(ring_grad, a, b, 1, 0.1, np.random.default_rng(3), theta0=[0.0, 0.3])
    th0 = np.array([0.0, 0.3])
    expect = th0 - 0.1 * 2.0 * t * (1.0 - t) * ring_grad(bezier_path(a, th0, b, t))
    assert np.allclose(theta, expect)


# The run of the page


def test_make_data_is_two_moons():
    X, y = make_data(400, np.random.default_rng(0))
    assert X.shape == (400, 2) and y.shape == (400,)
    assert (y == 0).sum() == 200 and (y == 1).sum() == 200
    r0 = np.linalg.norm(X[y == 0], axis=1)
    r1 = np.linalg.norm(X[y == 1] - np.array([1.0, 0.5]), axis=1)
    assert abs(r0.mean() - 1.0) < 0.05 and abs(r1.mean() - 1.0) < 0.05
    assert X[y == 0, 1].mean() > 0.4 and X[y == 1, 1].mean() < 0.1


def test_demo_matching_and_curves_lower_the_barrier():
    r = demo(1, 2)
    assert max(r["loss_a"], r["loss_b"]) < 0.01
    assert max(r["error_a"], r["error_b"]) < 0.01
    assert r["barrier_naive"] > 0.04
    assert r["barrier_bezier"] < 0.3 * r["barrier_naive"]
    assert r["barrier_matched"] < 0.05 * r["barrier_naive"]
    assert 0.0 < r["t_naive"] < 1.0
    assert sorted(r["perm"].tolist()) == list(range(64))
