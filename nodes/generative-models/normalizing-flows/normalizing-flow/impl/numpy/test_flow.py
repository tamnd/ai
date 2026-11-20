# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the inverse, the log-determinant
against the Jacobian, the hand-written gradients against finite differences, a density that integrates to 1, the zero
init as the identity, the worked example of the page, the bits-per-dimension conversion, the planar flow and the fix
that keeps it invertible, the one-pass density and d-pass sampling of the autoregressive layer, and a flow that learns
a banana-shaped density down to its entropy."""

import json
import math
import pathlib

import numpy as np
import pytest

import flow

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def random_params(rng, cfg, scale=0.5):
    return {k: rng.normal(0.0, scale, s) for k, s in flow.param_shapes(cfg).items()}


def jacobian(f, x, eps=1e-6):
    """The Jacobian of f: R^D -> R^D at one point x by central differences."""
    cols = [(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(len(x))]
    return np.stack(cols, 1)


def banana(n, rng):
    """x2 ~ N(0, 4), x1 | x2 ~ N(x2^2 / 4, 1): the density of Fig. 1 of Papamakarios et al. (2017)."""
    x2 = 2.0 * rng.standard_normal(n)
    return np.stack([x2**2 / 4 + rng.standard_normal(n), x2], 1)


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = flow.config()
    z, ld, _ = flow.forward(p, cfg, inp["x"])
    np.testing.assert_allclose(z, want["z"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ld, want["logdet"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(flow.log_prob(p, cfg, inp["x"]), want["log_prob"], rtol=1e-12)
    np.testing.assert_allclose(flow.inverse(p, cfg, inp["u"]), want["x_inv"], rtol=1e-12, atol=1e-14)
    loss, g, gx = flow.nll_and_grads(p, cfg, inp["x"])
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    np.testing.assert_allclose(gx, want["dx"], rtol=1e-10, atol=1e-13)
    assert set(g) == set(flow.param_shapes(cfg))
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    w, u, b = p["planar_w"], p["planar_u"], p["planar_b"][0]
    f, pld = flow.planar(inp["x"], w, flow.u_hat(w, u), b)
    np.testing.assert_allclose(f, want["planar_f"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pld, want["planar_logdet"], rtol=1e-12, atol=1e-14)
    q = {k[3:]: v for k, v in p.items() if k.startswith("ar_")}
    au, ald = flow.ar_density(q, inp["x"])
    np.testing.assert_allclose(au, want["ar_u"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ald, want["ar_logdet"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(flow.ar_sample(q, inp["u"]), want["ar_x"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = flow.config(k["d_data"], k["d_hidden"], k["n_layers"])
    assert flow.param_count(cfg) == mod["params"]
    assert flow.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    assert sum(v.size for v in flow.init(np.random.default_rng(0), cfg).values()) == mod["params"]


def test_masks_alternate():
    m = flow.masks(flow.config(5, 4, 4))
    np.testing.assert_array_equal(m[0], [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(m[0] + m[1], np.ones(5))
    np.testing.assert_array_equal(m[2], m[0])


@pytest.mark.parametrize("dims", [(2, 3, 1), (3, 5, 3), (6, 8, 5)])
def test_inverse_round_trip(dims):
    rng = np.random.default_rng(1)
    cfg = flow.config(*dims)
    p = random_params(rng, cfg)
    x = 2.0 * rng.standard_normal((50, cfg["d_data"]))
    z, _, _ = flow.forward(p, cfg, x)
    np.testing.assert_allclose(flow.inverse(p, cfg, z), x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(flow.forward(p, cfg, flow.inverse(p, cfg, x))[0], x, rtol=1e-12, atol=1e-12)


def test_logdet_is_the_log_abs_det_of_the_jacobian():
    """eq-logdet: the sum of the scales equals log |det df/dx| of the whole stack."""
    rng = np.random.default_rng(2)
    cfg = flow.config(4, 6, 3)
    p = random_params(rng, cfg)
    for x in rng.standard_normal((5, 4)):
        J = jacobian(lambda v: flow.forward(p, cfg, v[None])[0][0], x)
        _, ld, _ = flow.forward(p, cfg, x[None])
        assert ld[0] == pytest.approx(np.linalg.slogdet(J)[1], abs=1e-7)


def test_one_coupling_layer_has_a_triangular_jacobian():
    """eq-coupling-jacobian: the kept coordinates pass through, so dy_kept/dx is a row of the identity and the
    transformed block is diagonal with entries exp(s)."""
    rng = np.random.default_rng(3)
    cfg = flow.config(4, 6, 1)
    p = random_params(rng, cfg)
    x = rng.standard_normal(4)
    J = jacobian(lambda v: flow.forward(p, cfg, v[None])[0][0], x)
    keep, move = [0, 2], [1, 3]
    np.testing.assert_allclose(J[np.ix_(keep, keep)], np.eye(2), atol=1e-9)
    np.testing.assert_allclose(J[np.ix_(keep, move)], 0.0, atol=1e-9)
    np.testing.assert_allclose(J[np.ix_(move, move)], np.diag(np.diag(J[np.ix_(move, move)])), atol=1e-9)
    _, _, _, s, _ = flow._layer(p, 0, flow.masks(cfg)[0], x[None])
    np.testing.assert_allclose(np.diag(J)[move], np.exp(s[0, move]), rtol=1e-8)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    cfg = flow.config(3, 4, 3)
    p = random_params(rng, cfg)
    x = rng.standard_normal((6, 3))
    _, g, gx = flow.nll_and_grads(p, cfg, x)
    eps = 1e-6
    for k in p:
        num = np.zeros_like(p[k])
        for i in np.ndindex(p[k].shape):
            hi, lo = dict(p), dict(p)
            hi[k], lo[k] = p[k].copy(), p[k].copy()
            hi[k][i] += eps
            lo[k][i] -= eps
            num[i] = (flow.nll_and_grads(hi, cfg, x)[0] - flow.nll_and_grads(lo, cfg, x)[0]) / (2 * eps)
        np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-8, err_msg=k)
    num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        hi, lo = x.copy(), x.copy()
        hi[i] += eps
        lo[i] -= eps
        num[i] = (flow.nll_and_grads(p, cfg, hi)[0] - flow.nll_and_grads(p, cfg, lo)[0]) / (2 * eps)
    np.testing.assert_allclose(gx, num, rtol=1e-6, atol=1e-8)


def test_density_integrates_to_one():
    """eq-change-of-variables gives a normalized density with no normalizing constant to estimate."""
    rng = np.random.default_rng(3)
    cfg = flow.config(2, 6, 4)
    p = random_params(rng, cfg, 0.3)
    h = 0.05
    g = np.arange(-20, 20, h) + h / 2
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    assert np.exp(flow.log_prob(p, cfg, pts)).sum() * h * h == pytest.approx(1.0, abs=1e-8)


def test_zero_init_is_the_identity():
    rng = np.random.default_rng(5)
    cfg = flow.config(5, 7, 4)
    p = flow.init(rng, cfg)
    x = rng.standard_normal((10, 5))
    z, ld, _ = flow.forward(p, cfg, x)
    np.testing.assert_array_equal(z, x)
    np.testing.assert_array_equal(ld, 0.0)
    np.testing.assert_allclose(flow.log_prob(p, cfg, x), flow.log_normal(x))


def test_worked_example():
    """One coupling layer with s = log 2 and t = 1 on the second coordinate maps (1, 2) to (1, 5)."""
    cfg = flow.config(2, 1, 1)
    p = {k: np.zeros(s) for k, s in flow.param_shapes(cfg).items()}
    p["c0_bs"][:] = math.log(2)
    p["c0_bt"][:] = 1.0
    x = np.array([[1.0, 2.0]])
    z, ld, _ = flow.forward(p, cfg, x)
    np.testing.assert_allclose(z, [[1.0, 5.0]])
    assert ld[0] == pytest.approx(math.log(2))
    assert flow.log_prob(p, cfg, x)[0] == pytest.approx(-13 - math.log(2 * math.pi) + math.log(2))
    assert flow.log_prob(p, cfg, x)[0] == pytest.approx(-14.1447, abs=1e-4)
    np.testing.assert_allclose(flow.inverse(p, cfg, z), x)


def test_bits_per_dim():
    """A uniform density on [0, 1)^D is 8 bits per dimension of 8-bit data, and 3.49 bits per dimension on CIFAR-10
    (Real NVP) is a log-likelihood of about 9603 nats in the scaled space."""
    D = 32 * 32 * 3
    assert flow.bits_per_dim(0.0, D) == pytest.approx(8.0)
    nll = 3.49 * D * math.log(2) - D * math.log(256)
    assert nll == pytest.approx(-9603.4, abs=0.1)
    assert flow.bits_per_dim(nll, D) == pytest.approx(3.49)


def test_dequantized_likelihood_bounds_the_discrete_one():
    """eq-dequantization: on 1-dimensional pixels with 4 levels, the expected log-density of the dequantized data under
    any density q on [0, 1) is at most the log-probability of the discrete model P(k) = integral of q over its bin,
    after the change of scale by 1/levels."""
    rng = np.random.default_rng(6)
    levels = 4
    pk = np.array([0.1, 0.2, 0.3, 0.4])
    k = rng.choice(levels, 200000, p=pk)
    y = flow.dequantize(k, rng, levels)
    assert np.all((y >= k / levels) & (y < (k + 1) / levels))

    def q(v):  # a density on [0, 1) that is not constant on the bins
        return 0.5 + v

    bins = np.linspace(0, 1, levels + 1)
    big_p = [(b**2 / 2 + b / 2) - (a**2 / 2 + a / 2) for a, b in zip(bins[:-1], bins[1:], strict=True)]
    cont = np.mean(np.log(q(y))) - math.log(levels)
    disc = np.mean(np.log(np.array(big_p)[k]))
    assert cont < disc
    assert cont == pytest.approx(disc, abs=0.05)


def test_planar_logdet_and_the_invertibility_fix():
    rng = np.random.default_rng(7)
    w, u, b = rng.standard_normal(3), rng.standard_normal(3), 0.3
    u = u - (w @ u + 3.0) * w / (w @ w)  # w.u = -3: the map folds
    assert w @ u == pytest.approx(-3.0)
    uh = flow.u_hat(w, u)
    assert w @ uh > -1.0
    for z in rng.standard_normal((5, 3)):
        J = jacobian(lambda v: flow.planar(v[None], w, uh, b)[0][0], z)
        assert flow.planar(z[None], w, uh, b)[1][0] == pytest.approx(np.linalg.slogdet(J)[1], abs=1e-7)
    t = np.linspace(-6, 6, 2001)
    line = t[:, None] * w / (w @ w)  # w.z = t along this line
    dets = 1 + (1 - np.tanh(t + b) ** 2) * (w @ u)
    dets_hat = 1 + (1 - np.tanh(t + b) ** 2) * (w @ uh)
    assert dets.min() < 0 < dets.max()
    assert dets_hat.min() > 0
    proj = flow.planar(line, w, uh, b)[0] @ w
    assert np.all(np.diff(proj) > 0)
    proj_bad = flow.planar(line, w, u, b)[0] @ w
    assert not np.all(np.diff(proj_bad) > 0)


def test_u_hat_sets_w_dot_u_to_m():
    """eq-planar: w.u_hat = m(w.u) = -1 + log(1 + exp(w.u)) > -1 for every u, and the part of u orthogonal to w is
    unchanged."""
    rng = np.random.default_rng(8)
    w = rng.standard_normal(4)
    for wu in (-30.0, -3.0, -1.0, 0.0, 2.0, 25.0):
        u = rng.standard_normal(4)
        u = u + (wu - w @ u) * w / (w @ w)
        uh = flow.u_hat(w, u)
        assert w @ uh == pytest.approx(-1 + math.log1p(math.exp(wu)), abs=1e-10)
        assert w @ uh > -1
        np.testing.assert_allclose(uh - (uh @ w) * w / (w @ w), u - (u @ w) * w / (w @ w), atol=1e-12)


def test_autoregressive_density_in_one_pass_sampling_in_d():
    rng = np.random.default_rng(8)
    D = 5
    q = {"Wm": rng.normal(0, 0.7, (D, D)), "bm": rng.normal(0, 0.5, D),
         "Wa": rng.normal(0, 0.4, (D, D)), "ba": rng.normal(0, 0.3, D)}  # fmt: skip
    u = rng.standard_normal((20, D))
    x = flow.ar_sample(q, u)
    back, ld = flow.ar_density(q, x)
    np.testing.assert_allclose(back, u, rtol=1e-12, atol=1e-12)
    assert np.abs(flow.ar_sample(q, u, passes=D - 1) - x).max() > 1e-3
    np.testing.assert_allclose(flow.ar_sample(q, u, passes=D + 3), x, rtol=1e-12, atol=1e-12)
    J = jacobian(lambda v: flow.ar_density(q, v[None])[0][0], x[0])
    np.testing.assert_allclose(np.triu(J, 1), 0.0, atol=1e-9)  # u_i depends on x_1..x_i only
    assert ld[0] == pytest.approx(np.linalg.slogdet(J)[1], abs=1e-7)


def test_learns_a_banana():
    """Maximum likelihood fits the banana of Papamakarios et al. (2017) to within 0.05 nats of its entropy,
    log(2 pi e) + log(4) / 2 = 3.531, where the best Gaussian stays above 4."""
    cfg = flow.config(2, 32, 4)
    p = flow.init(np.random.default_rng(1), cfg)
    rng = np.random.default_rng(2)
    flow.train(p, cfg, banana, 2000, rng)
    held = banana(20000, np.random.default_rng(9))
    nll = -np.mean(flow.log_prob(p, cfg, held))
    entropy = math.log(2 * math.pi * math.e) + 0.5 * math.log(4)
    assert entropy == pytest.approx(3.5310, abs=1e-4)
    assert nll == pytest.approx(entropy, abs=0.05)
    mu, cov = held.mean(0), np.cov(held.T)
    d = held - mu
    gauss = 0.5 * np.mean(np.sum(d @ np.linalg.inv(cov) * d, 1)) + 0.5 * np.linalg.slogdet(2 * math.pi * cov)[1]
    assert gauss > nll + 0.4
    s = flow.sample(p, cfg, 20000, np.random.default_rng(10))
    assert np.mean(s[:, 0]) == pytest.approx(1.0, abs=0.1)  # E[x1] = E[x2^2] / 4 = 1
    cold = flow.sample(p, cfg, 20000, np.random.default_rng(10), temperature=0.5)
    assert np.std(cold[:, 1]) < 0.7 * np.std(s[:, 1])
