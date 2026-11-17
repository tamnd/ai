# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://NEW-sharpness-aware-minimization).
"""Tests of the NumPy reference: the perturbation against brute-force maximization of the linearized loss on the p-ball,
its closed forms at p = 2 and p = inf and over a list of arrays; the SAM gradient and one SAM step on a quadratic,
against the closed form H(w + rho Hw / ||Hw||); momentum and weight decay; m-sharpness; the filter-normalized
direction; and the worked example of the page."""

import numpy as np
import pytest

from sharpness_aware_minimization import (
    filter_normalized_direction,
    loss_along,
    m_sharpness_gradient,
    sam_gradient,
    sam_perturbation,
    sam_step,
)

rng = np.random.default_rng(7)


def pnorm(x, p):
    x = np.ravel(x)
    return np.max(np.abs(x)) if np.isinf(p) else np.sum(np.abs(x) ** p) ** (1.0 / p)


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0, np.inf])
def test_perturbation_maximizes_linear_term(p):
    q = 1.0 if np.isinf(p) else p / (p - 1.0)
    for _ in range(20):
        g = rng.normal(size=4)
        rho = float(rng.uniform(0.05, 2.0))
        e = sam_perturbation(g, rho, p)
        assert e.shape == g.shape
        assert pnorm(e, p) == pytest.approx(rho, rel=1e-9)
        assert e @ g == pytest.approx(rho * pnorm(g, q), rel=1e-9)
        z = rng.normal(size=(5000, 4))
        zn = np.max(np.abs(z), axis=1) if np.isinf(p) else np.sum(np.abs(z) ** p, axis=1) ** (1.0 / p)
        z = rho * z / zn[:, None]
        assert (z @ g).max() <= e @ g + 1e-9


def test_closed_forms():
    g = np.array([3.0, -4.0, 0.0])
    assert np.allclose(sam_perturbation(g, 0.5), [0.3, -0.4, 0.0])
    assert np.allclose(sam_perturbation(g, 0.5, p=np.inf), [0.5, -0.5, 0.0])
    # the default is p = 2
    assert np.allclose(sam_perturbation(g, 0.5, 2.0), sam_perturbation(g, 0.5))
    # p = 4, q = 4/3: rho sign(g) |g|^(1/3) / (sum |g|^(4/3))^(1/4)
    e = sam_perturbation(np.array([1.0, 8.0]), 1.0, 4.0)
    s = (1.0 + 8.0 ** (4 / 3)) ** 0.25
    assert np.allclose(e, [1.0 / s, 2.0 / s])


def test_scale_invariance_and_large_q():
    g = rng.normal(size=5)
    for c in [1e-6, 0.3, 7.0, 1e6]:
        assert np.allclose(sam_perturbation(c * g, 0.2, 3.0), sam_perturbation(g, 0.2, 3.0))
    # p close to 1 puts the whole budget on the largest entry without overflow
    e = sam_perturbation(np.array([0.5, -200.0, 100.0]), 1.0, 1.001)
    assert np.all(np.isfinite(e))
    assert e[1] == pytest.approx(-1.0, abs=1e-6)


def test_zero_gradient_and_bad_arguments():
    assert np.all(sam_perturbation(np.zeros(3), 0.5) == 0)
    z = sam_perturbation([np.zeros(2), np.zeros((2, 2))], 0.5, 3.0)
    assert all(np.all(v == 0) for v in z)
    assert np.all(sam_perturbation(np.array([1.0, -2.0]), 0.0) == 0)
    with pytest.raises(ValueError):
        sam_perturbation(np.ones(2), -0.1)
    with pytest.raises(ValueError):
        sam_perturbation(np.ones(2), 0.1, p=1.0)


def test_list_of_arrays_is_one_vector():
    a, b = rng.normal(size=(2, 3)), rng.normal(size=4)
    flat = np.concatenate([a.ravel(), b])
    for p in [2.0, 3.0, np.inf]:
        ea, eb = sam_perturbation([a, b], 0.3, p)
        assert ea.shape == a.shape and eb.shape == b.shape
        assert np.allclose(np.concatenate([ea.ravel(), eb]), sam_perturbation(flat, 0.3, p))


def quad(H):
    return lambda w: H @ w


def test_sam_gradient_on_quadratic():
    for _ in range(20):
        A = rng.normal(size=(4, 4))
        H = A @ A.T + 0.1 * np.eye(4)
        w = rng.normal(size=4)
        rho = float(rng.uniform(0.01, 1.0))
        g_adv, e = sam_gradient(w, quad(H), rho)
        g = H @ w
        assert np.allclose(e, rho * g / np.linalg.norm(g))
        assert np.allclose(g_adv, H @ (w + rho * g / np.linalg.norm(g)))
        assert np.allclose(g_adv, g + rho * H @ g / np.linalg.norm(g))


def test_sam_gradient_list_params():
    H1, H2 = np.diag([2.0, 3.0]), np.diag([5.0])

    def gf(ws):
        return [H1 @ ws[0], H2 @ ws[1]]

    w = [np.array([1.0, -1.0]), np.array([0.5])]
    g_adv, e = sam_gradient(w, gf, 0.4)
    g = np.array([2.0, -3.0, 2.5])
    eh = 0.4 * g / np.linalg.norm(g)
    assert np.allclose(np.concatenate(e), eh)
    assert np.allclose(g_adv[0], H1 @ (w[0] + eh[:2]))
    assert np.allclose(g_adv[1], H2 @ (w[1] + eh[2:]))


def test_sam_step_plain():
    H = np.diag([4.0, 1.0])
    w = np.array([1.0, 2.0])
    w_new, buf = sam_step(w, quad(H), lr=0.1, rho=0.5)
    assert buf is None
    assert np.allclose(w_new, [0.42111456, 1.77763932])
    # the step starts from w, not from the perturbed point
    g_adv, e = sam_gradient(w, quad(H), 0.5)
    assert np.allclose(w_new, w - 0.1 * g_adv)
    # rho = 0 is plain gradient descent
    assert np.allclose(sam_step(w, quad(H), lr=0.1, rho=0.0)[0], w - 0.1 * H @ w)
    # the input is not modified
    assert np.allclose(w, [1.0, 2.0])


def test_sam_step_momentum_and_weight_decay():
    H = np.diag([4.0, 1.0])
    w = np.array([1.0, 2.0])
    g1, _ = sam_gradient(w, quad(H), 0.5)
    w1, b1 = sam_step(w, quad(H), lr=0.1, rho=0.5, momentum=0.9)
    assert np.allclose(b1, g1) and np.allclose(w1, w - 0.1 * g1)
    g2, _ = sam_gradient(w1, quad(H), 0.5)
    w2, b2 = sam_step(w1, quad(H), lr=0.1, rho=0.5, momentum=0.9, buf=b1)
    assert np.allclose(b2, 0.9 * g1 + g2)
    assert np.allclose(w2, w1 - 0.1 * (0.9 * g1 + g2))
    w3, _ = sam_step(w, quad(H), lr=0.1, rho=0.5, weight_decay=0.01)
    assert np.allclose(w3, w - 0.1 * (g1 + 0.01 * w))
    # list parameters with momentum
    wl, bl = sam_step([w], lambda ws: [H @ ws[0]], lr=0.1, rho=0.5, momentum=0.9)
    assert np.allclose(wl[0], w1) and np.allclose(bl[0], b1)
    wl2, bl2 = sam_step(wl, lambda ws: [H @ ws[0]], lr=0.1, rho=0.5, momentum=0.9, buf=bl)
    assert np.allclose(wl2[0], w2) and np.allclose(bl2[0], b2)


def test_sam_shrinks_sharp_direction_faster():
    H = np.diag([10.0, 0.1])
    w_gd = w_sam = np.array([1.0, 1.0])
    for _ in range(100):
        w_gd = w_gd - 0.01 * H @ w_gd
        w_sam, _ = sam_step(w_sam, quad(H), lr=0.01, rho=0.05)
        assert abs(w_sam[0] / w_sam[1]) < abs(w_gd[0] / w_gd[1])


def test_valley_drifts_to_flat_minimum():
    def vgrad(x):
        r = x[0] * x[1] - 1.0
        return r * np.array([x[1], x[0]])

    x = x_gd = np.array([3.0, 0.4])
    for _ in range(4000):
        x, _ = sam_step(x, vgrad, lr=0.02, rho=0.05)
        x_gd = x_gd - 0.02 * vgrad(x_gd)
    # GD stops at the first minimum it reaches, where the top Hessian eigenvalue a^2 + b^2 is about 9.06
    assert abs(x_gd[0] * x_gd[1] - 1.0) < 1e-6 and x_gd @ x_gd > 9.0
    # SAM stays near the valley and slides along it toward a = b, where a^2 + b^2 = 2
    assert abs(x[0] * x[1] - 1.0) < 0.05
    assert x @ x < 6.5


def test_m_sharpness():
    X = rng.normal(size=(8, 3))
    y = rng.normal(size=8)

    def gf(w, idx):
        r = X[idx] @ w - y[idx]
        return X[idx].T @ r / len(idx)

    w = rng.normal(size=3)
    # m = n is SAM on the whole batch
    full, _ = sam_gradient(w, lambda v: gf(v, np.arange(8)), 0.1)
    assert np.allclose(m_sharpness_gradient(w, gf, 8, 8, 0.1), full)
    # m = 2: four separate perturbations, averaged
    parts = [sam_gradient(w, lambda v, i=i: gf(v, np.arange(i, i + 2)), 0.1)[0] for i in range(0, 8, 2)]
    m2 = m_sharpness_gradient(w, gf, 8, 2, 0.1)
    assert np.allclose(m2, np.mean(parts, axis=0))
    assert not np.allclose(m2, full)
    # rho = 0 gives the ordinary gradient for any m
    assert np.allclose(m_sharpness_gradient(w, gf, 8, 2, 0.0), gf(w, np.arange(8)))
    lst = m_sharpness_gradient([w], lambda v, idx: [gf(v[0], idx)], 8, 2, 0.1)
    assert isinstance(lst, list) and np.allclose(lst[0], m2)
    with pytest.raises(ValueError):
        m_sharpness_gradient(w, gf, 8, 3, 0.1)
    with pytest.raises(ValueError):
        m_sharpness_gradient(w, gf, 8, 0, 0.1)


def test_filter_normalized_direction():
    conv = rng.normal(size=(4, 3, 3, 3))
    fc = rng.normal(size=(5, 6)) * 10.0
    bias = rng.normal(size=5)
    d = filter_normalized_direction([conv, fc, bias], np.random.default_rng(0))
    assert [x.shape for x in d] == [conv.shape, fc.shape, bias.shape]
    for theta, dd in [(conv, d[0]), (fc, d[1])]:
        tn = np.linalg.norm(theta.reshape(theta.shape[0], -1), axis=1)
        dn = np.linalg.norm(dd.reshape(dd.shape[0], -1), axis=1)
        assert np.allclose(tn, dn)
    assert np.all(d[2] == 0)
    # the direction of each filter is that of the Gaussian draw
    g = np.random.default_rng(0).standard_normal(conv.shape)
    for j in range(4):
        a, b = g[j].ravel(), d[0][j].ravel()
        assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) == pytest.approx(1.0)
    # with ignore_1d false, each entry of a vector keeps its magnitude
    d2 = filter_normalized_direction([bias], np.random.default_rng(1), ignore_1d=False)
    assert np.allclose(np.abs(d2[0]), np.abs(bias))
    # a zero filter gets a zero direction
    z = np.ones((2, 3))
    z[1] = 0
    dz = filter_normalized_direction([z], np.random.default_rng(2))
    assert np.all(dz[0][1] == 0) and np.linalg.norm(dz[0][0]) == pytest.approx(np.sqrt(3))


def test_filter_normalization_is_scale_invariant():
    # rescaling one filter of the weights rescales the same filter of the direction
    theta = rng.normal(size=(3, 4))
    scaled = theta.copy()
    scaled[1] *= 10.0
    d1 = filter_normalized_direction([theta], np.random.default_rng(5))[0]
    d2 = filter_normalized_direction([scaled], np.random.default_rng(5))[0]
    assert np.allclose(d2[1], 10.0 * d1[1])
    assert np.allclose(d2[[0, 2]], d1[[0, 2]])


def test_loss_along():
    H = np.diag([2.0, 1.0])

    def loss(ps):
        return 0.5 * ps[0] @ H @ ps[0]

    vals = loss_along(loss, [np.array([1.0, 0.0])], [np.array([0.0, 1.0])], [-1.0, 0.0, 2.0])
    assert np.allclose(vals, [1.5, 1.0, 3.0])


def test_worked_example():
    H = np.diag([4.0, 1.0])
    w = np.array([1.0, 2.0])
    g_adv, e = sam_gradient(w, quad(H), 0.5)
    assert np.allclose(e, [0.4472136, 0.2236068])
    assert np.allclose(g_adv, [5.78885438, 2.2236068])
    assert np.allclose(g_adv / (H @ w), [1.4472136, 1.1118034])
    w_sam, _ = sam_step(w, quad(H), lr=0.1, rho=0.5)
    w_gd = w - 0.1 * H @ w

    def L(v):
        return 0.5 * v @ H @ v

    assert L(w_sam) == pytest.approx(1.93468, abs=1e-5)
    assert L(w_gd) == pytest.approx(2.34)
