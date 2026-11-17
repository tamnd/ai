# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://3SJCNH).
"""Tests of the NumPy reference: the worked example, the Jacobian, the limit kernel, the flow and the lazy regime."""

import numpy as np
import pytest

from neural_tangent_kernel import (
    empirical_ntk,
    empirical_ntk_terms,
    flatten,
    forward,
    grad,
    init,
    jacobian,
    relative_change,
    relu_grad,
    relu_ntk,
    relu_ntk_terms,
    residual_flow,
    train_linearized,
    train_network,
    unflatten,
)

# The worked example of the page: x = (1, 0) and x' at 60 degrees, both of norm 1.
X_A = np.array([[1.0, 0.0]])
X_B = np.array([[0.5, np.sqrt(3) / 2]])
FIRST_AB = 1 / 6
SECOND_AB = (np.sqrt(3) / 2 + np.pi / 3) / (2 * np.pi)

# A small shared data set on the unit sphere of R^3.
_r = np.random.default_rng(7)
XS = _r.standard_normal((5, 3))
XS /= np.linalg.norm(XS, axis=1, keepdims=True)
YS = np.sin(3.0 * XS[:, 0])


def test_worked_example():
    first, second = relu_ntk_terms(X_A, X_B)
    assert first[0, 0] == pytest.approx(FIRST_AB, abs=1e-12)
    assert second[0, 0] == pytest.approx(SECOND_AB, abs=1e-12)
    assert relu_ntk(X_A, X_B)[0, 0] == pytest.approx(0.4711655572, abs=1e-9)


def test_special_angles():
    e1, e2 = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert relu_ntk(e1, e1)[0, 0] == pytest.approx(1.0)
    assert relu_ntk(e1, e2)[0, 0] == pytest.approx(1 / (2 * np.pi))
    assert relu_ntk(e1, -e1)[0, 0] == pytest.approx(0.0, abs=1e-15)
    first, second = relu_ntk_terms(e1, e1)
    assert first[0, 0] == pytest.approx(0.5) and second[0, 0] == pytest.approx(0.5)


def test_diagonal_and_homogeneity():
    X = np.array([[3.0, -1.0, 2.0], [0.5, 0.5, -4.0]])
    assert np.allclose(np.diag(relu_ntk(X, X)), np.sum(X**2, axis=1))
    for c in (0.5, 2.0, 7.0):
        assert np.allclose(relu_ntk(c * X, X), c * relu_ntk(X, X))
        assert np.allclose(relu_ntk(X, c * X), c * relu_ntk(X, X))


def test_zero_input_gives_zero_kernel():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    K = relu_ntk(X, X)
    assert np.all(np.isfinite(K))
    assert np.allclose(K[0], 0.0) and np.allclose(K[:, 0], 0.0)


def test_rounding_past_one_is_clipped():
    x = np.array([[1e8, 1.0]])
    K = relu_ntk(x, x)
    assert np.isfinite(K[0, 0]) and K[0, 0] == pytest.approx(np.sum(x**2), rel=1e-12)


def test_analytic_kernel_is_symmetric_and_psd():
    K = relu_ntk(XS, XS)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > 0


def test_init_shapes_and_errors():
    p = init(np.random.default_rng(0), 3, 7)
    assert p["W"].shape == (7, 3) and p["a"].shape == (7,)
    for d, m in ((0, 4), (3, 0)):
        with pytest.raises(ValueError):
            init(np.random.default_rng(0), d, m)
    q = unflatten(flatten(p), 3, 7)
    assert np.array_equal(q["W"], p["W"]) and np.array_equal(q["a"], p["a"])


def test_relu_grad_at_zero():
    assert np.array_equal(relu_grad(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 1.0])


def test_forward_by_hand():
    p = {
        "W": np.array([[1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-2.0, 0.5]]),
        "a": np.array([1.0, -2.0, 0.5, 3.0]),
    }
    x = np.array([[2.0, -1.0]])
    # Preactivations 2, 1, 1, -4.5; relu 2, 1, 1, 0; sum a_j relu = 2 - 2 + 0.5 = 0.5; divided by sqrt(4).
    assert forward(p, x)[0] == pytest.approx(0.25)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    p = init(rng, 3, 6)
    th = flatten(p)
    J = jacobian(p, XS)
    h = 1e-6
    for k in range(th.size):
        e = np.zeros_like(th)
        e[k] = h
        num = (forward(unflatten(th + e, 3, 6), XS) - forward(unflatten(th - e, 3, 6), XS)) / (2 * h)
        assert np.allclose(J[:, k], num, atol=1e-6)


def test_empirical_ntk_is_sum_of_layer_terms():
    p = init(np.random.default_rng(2), 3, 50)
    K = empirical_ntk(p, XS, XS)
    first, second = empirical_ntk_terms(p, XS, XS)
    assert np.allclose(K, first + second)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-12
    assert np.allclose(empirical_ntk(p, XS[:2], XS), K[:2])


def test_empirical_ntk_converges_to_limit():
    """The two layer terms each tend to their limit, and the error falls like 1/sqrt(m)."""
    A1, A2 = relu_ntk_terms(XS, XS)
    widths = [256, 4096, 65536]
    errs = []
    for m in widths:
        sq = []
        for s in range(8):
            F, S = empirical_ntk_terms(init(np.random.default_rng(100 + s), 3, m), XS, XS)
            sq.append(np.mean((F + S - A1 - A2) ** 2))
            if m == widths[-1]:
                assert np.abs(F - A1).max() < 0.05 and np.abs(S - A2).max() < 0.05
        errs.append(np.sqrt(np.mean(sq)))
    slope = np.polyfit(np.log(widths), np.log(errs), 1)[0]
    assert -0.62 < slope < -0.38


def test_monte_carlo_mean_is_the_limit():
    """At small width, the average of the empirical NTK over many draws is the limit kernel (it is unbiased)."""
    A = relu_ntk(XS, XS)
    draws = np.array([empirical_ntk(init(np.random.default_rng(s), 3, 4), XS, XS) for s in range(800)])
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - A) < 5 * se)


def test_residual_flow_by_hand():
    K = np.diag([2.0, 0.5])
    r0 = np.array([1.0, -3.0])
    r = residual_flow(K, r0, eta=0.5, t=4.0, n=2)
    assert np.allclose(r, [np.exp(-2.0), -3.0 * np.exp(-0.5)])
    assert np.allclose(residual_flow(K, r0, 0.5, 0.0, 2), r0)


def test_linearized_gd_is_the_discrete_kernel_map():
    p = init(np.random.default_rng(3), 3, 200)
    n = XS.shape[0]
    K = empirical_ntk(p, XS, XS)
    lam, V = np.linalg.eigh(K)
    rs, _ = train_linearized(p, XS, YS, eta=0.7, steps=40)
    for k in (0, 1, 13, 40):
        exact = V @ ((1 - 0.7 * lam / n) ** k * (V.T @ rs[0]))
        assert np.allclose(rs[k], exact, atol=1e-12)


def test_linearized_gd_tends_to_the_flow():
    p = init(np.random.default_rng(4), 3, 200)
    n = XS.shape[0]
    K = empirical_ntk(p, XS, XS)
    T = 30.0
    flow = residual_flow(K, forward(p, XS) - YS, eta=1.0, t=T, n=n)
    gaps = [np.abs(train_linearized(p, XS, YS, eta, int(T / eta))[0][-1] - flow).max() for eta in (0.5, 0.25)]
    assert gaps[1] < gaps[0] < 0.01
    assert gaps[0] / gaps[1] == pytest.approx(2.0, rel=0.1)


def test_eigencomponents_decay_at_their_rates():
    p = init(np.random.default_rng(5), 3, 300)
    n = XS.shape[0]
    K = empirical_ntk(p, XS, XS)
    lam, V = np.linalg.eigh(K)
    r0 = forward(p, XS) - YS
    for t in (1.0, 10.0, 50.0):
        c = V.T @ residual_flow(K, r0, eta=0.8, t=t, n=n)
        assert np.allclose(c, np.exp(-0.8 * lam * t / n) * (V.T @ r0))
    assert np.abs(residual_flow(K, r0, 0.8, 2000.0, n)).max() < 1e-6


def test_grad_matches_finite_differences():
    p = init(np.random.default_rng(6), 3, 5)
    g = flatten(grad(p, XS, YS))
    th = flatten(p)

    def loss(t):
        return 0.5 * np.mean((forward(unflatten(t, 3, 5), XS) - YS) ** 2)

    h = 1e-6
    num = np.array([(loss(th + h * e) - loss(th - h * e)) / (2 * h) for e in np.eye(th.size)])
    assert np.allclose(g, num, atol=1e-7)


def test_train_network_reduces_loss_and_leaves_input():
    p = init(np.random.default_rng(8), 3, 256)
    keep = flatten(p).copy()
    q = train_network(p, XS, YS, eta=1.0, steps=200)
    assert np.array_equal(flatten(p), keep)
    assert np.abs(forward(q, XS) - YS).max() < 0.05 * np.abs(forward(p, XS) - YS).max()


def test_network_follows_linearized_model_more_closely_when_wider():
    gaps = []
    for m in (64, 4096):
        g = []
        for s in range(3):
            p = init(np.random.default_rng(20 + s), 3, m)
            q = train_network(p, XS, YS, eta=1.0, steps=20)
            lin, _ = train_linearized(p, XS, YS, eta=1.0, steps=20)
            g.append(np.abs(forward(q, XS) - YS - lin[-1]).max())
        gaps.append(np.mean(g))
    assert gaps[1] < gaps[0] / 4


def test_relative_change_falls_like_inverse_sqrt_width():
    widths = [32, 256, 2048]
    rel = []
    for m in widths:
        rel.append(
            np.mean(
                [
                    relative_change(p, train_network(p, XS, YS, 1.0, 100))
                    for p in (init(np.random.default_rng(30 + s), 3, m) for s in range(3))
                ]
            )
        )
    slope = np.polyfit(np.log(widths), np.log(rel), 1)[0]
    assert -0.62 < slope < -0.38


def test_relative_change_by_hand():
    p0 = {"W": np.array([[3.0]]), "a": np.array([4.0])}
    p1 = {"W": np.array([[3.0]]), "a": np.array([4.5])}
    assert relative_change(p0, p1) == pytest.approx(0.1)


def test_linearized_weights_stay_in_the_kernel_bound():
    p = init(np.random.default_rng(9), 3, 400)
    n = XS.shape[0]
    K = empirical_ntk(p, XS, XS)
    r0 = forward(p, XS) - YS
    eta = n / np.linalg.eigvalsh(K).max()
    _, omega = train_linearized(p, XS, YS, eta=eta, steps=3000)
    assert np.linalg.norm(omega) <= np.sqrt(r0 @ np.linalg.solve(K, r0)) + 1e-9
    assert np.linalg.norm(omega) == pytest.approx(np.sqrt(r0 @ np.linalg.solve(K, r0)), rel=1e-3)
    assert np.allclose(jacobian(p, XS) @ omega, -r0, atol=1e-6)
