# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the path and its two velocities, the
hand-written gradients and divergence against finite differences, the continuity equation of the exact marginal field
(Theorem 1 of Lipman et al.), the equal gradients of the conditional and the marginal loss (their Theorem 2), the
samplers driven by the exact field, the collapse of one Euler step to the data mean, the worked example of the page,
the log-likelihood by the change of variables, the Hungarian algorithm against brute force, minibatch optimal
transport, reflow lowering the transport cost, logit-normal times, and a model that learns a two-mode mixture."""

import itertools
import json
import math
import pathlib

import numpy as np
import pytest

import flow_matching as fm

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
MODES = np.array([[-2.0, 0.0], [2.0, 0.0]])
HALF = np.array([0.5, 0.5])


def two_modes(n, rng):
    """Half the points around (-2, 0), half around (2, 0), each N(mode, 0.3^2 I)."""
    return MODES[rng.integers(0, 2, n)] + 0.3 * rng.standard_normal((n, 2))


def exact(means, s, weights, sigma_min=0.0):
    return lambda x, t: fm.mixture_field(x, t, means, s, weights, sigma_min)


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = fm.config(sigma_min=float(inp["sigma_min"]))
    x0, x1, t = inp["x0"], inp["x1"], inp["t"]
    np.testing.assert_allclose(fm.time_features(t, cfg["d_time"]), want["feat"], rtol=1e-12, atol=1e-14)
    xt = fm.path(cfg, x0, x1, t)
    np.testing.assert_allclose(xt, want["xt"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fm.target(cfg, x0, x1), want["u"], rtol=1e-12, atol=1e-14)
    out, _ = fm.velocity(p, cfg, xt, t)
    np.testing.assert_allclose(out, want["v"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fm.divergence(p, cfg, xt, t), want["div"], rtol=1e-12, atol=1e-14)
    loss, g = fm.loss_and_grads(p, cfg, x0, x1, t)
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    assert set(g) == set(fm.param_shapes(cfg))
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)

    def v_fn(x, s):
        return fm.velocity(p, cfg, x, s)[0]

    np.testing.assert_allclose(fm.euler_sample(v_fn, x0, 4), want["x_euler"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fm.midpoint_sample(v_fn, x0, 2), want["x_midpoint"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = fm.config(k["d_data"], k["d_hidden"], k["d_time"])
    assert fm.param_count(cfg) == mod["params"]
    assert fm.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = fm.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out, _ = fm.velocity(p, cfg, np.zeros((k["B"], k["d_data"])), np.zeros(k["B"]))
    assert out.shape == (k["B"], k["d_data"])


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        fm.config(d_hidden=0)
    with pytest.raises(ValueError):
        fm.config(sigma_min=1.0)


@pytest.mark.parametrize("sigma_min", [0.0, 0.1])
def test_path_ends_and_velocities(sigma_min):
    cfg = fm.config(sigma_min=sigma_min)
    rng = np.random.default_rng(1)
    x0, x1 = rng.standard_normal((5, 3)), rng.standard_normal((5, 3))
    np.testing.assert_allclose(fm.path(cfg, x0, x1, np.zeros(5)), x0)
    np.testing.assert_allclose(fm.path(cfg, x0, x1, np.ones(5)), x1 + sigma_min * x0)
    t, h = rng.uniform(0.1, 0.9, 5), 1e-6
    dpath = (fm.path(cfg, x0, x1, t + h) - fm.path(cfg, x0, x1, t - h)) / (2 * h)
    np.testing.assert_allclose(dpath, fm.target(cfg, x0, x1), rtol=1e-8)
    np.testing.assert_allclose(fm.cond_field(cfg, fm.path(cfg, x0, x1, t), x1, t), fm.target(cfg, x0, x1), rtol=1e-12)


def test_gradients_match_finite_differences():
    cfg = fm.config()
    rng = np.random.default_rng(2)
    p = {k: rng.normal(0, 0.5, s) for k, s in fm.param_shapes(cfg).items()}
    x0, x1, t = rng.standard_normal((6, 3)), rng.standard_normal((6, 3)), rng.uniform(0, 1, 6)
    _, g = fm.loss_and_grads(p, cfg, x0, x1, t)
    for k, arr in p.items():
        for idx in itertools.islice(np.ndindex(arr.shape), 6):
            q = {j: v.copy() for j, v in p.items()}
            q[k][idx] += 1e-6
            up = fm.loss_and_grads(q, cfg, x0, x1, t)[0]
            q[k][idx] -= 2e-6
            down = fm.loss_and_grads(q, cfg, x0, x1, t)[0]
            assert g[k][idx] == pytest.approx((up - down) / 2e-6, rel=1e-5, abs=1e-8), (k, idx)


def test_divergence_matches_finite_differences():
    cfg = fm.config()
    rng = np.random.default_rng(3)
    p = {k: rng.normal(0, 0.5, s) for k, s in fm.param_shapes(cfg).items()}
    x, t, h = rng.standard_normal((4, 3)), rng.uniform(0, 1, 4), 1e-6
    fd = sum(
        (fm.velocity(p, cfg, x + h * e, t)[0][:, i] - fm.velocity(p, cfg, x - h * e, t)[0][:, i]) / (2 * h)
        for i, e in enumerate(np.eye(3))
    )
    np.testing.assert_allclose(fm.divergence(p, cfg, x, t), fd, rtol=1e-7)


@pytest.mark.parametrize("sigma_min", [0.0, 0.1])
def test_marginal_field_satisfies_the_continuity_equation(sigma_min):
    """Theorem 1 of Lipman et al.: d/dt p_t + d/dx (p_t u_t) = 0 for the marginal field of eq-marginal-field."""
    means, w, s = np.array([[-2.0], [1.5]]), np.array([0.4, 0.6]), 0.3
    xs, t, h = np.linspace(-4, 4, 17)[:, None], 0.37, 1e-5

    def dens(x, u):
        return np.exp(fm.mixture_log_density(x, u, means, s, w, sigma_min))

    def flux(x):
        return dens(x, t) * fm.mixture_field(x, t, means, s, w, sigma_min)[:, 0]

    dp = (dens(xs, t + h) - dens(xs, t - h)) / (2 * h)
    dj = (flux(xs + h) - flux(xs - h)) / (2 * h)
    np.testing.assert_allclose(dp + dj, 0.0, atol=1e-8)


def test_marginal_field_is_the_conditional_mean_of_the_target():
    """eq-marginal-field: averaging the target over pairs whose path passes near x gives u_t(x)."""
    rng = np.random.default_rng(4)
    cfg = fm.config(d_data=1)
    means, w, s = np.array([[-2.0], [1.5]]), np.array([0.4, 0.6]), 0.3
    n, t = 2_000_000, 0.6
    k = rng.choice(2, n, p=w)
    x1 = means[k] + s * rng.standard_normal((n, 1))
    x0 = rng.standard_normal((n, 1))
    xt, u = fm.path(cfg, x0, x1, np.full(n, t)), fm.target(cfg, x0, x1)
    for x in (-1.2, 0.0, 0.9):
        near = np.abs(xt[:, 0] - x) < 0.01
        assert u[near, 0].mean() == pytest.approx(fm.mixture_field(np.array([[x]]), t, means, s, w)[0, 0], abs=0.03)


def test_conditional_and_marginal_losses_have_the_same_gradient():
    """Theorem 2 of Lipman et al.: on the same draws of x_t, regressing on the target x_1 - x_0 and on the marginal
    field u_t(x_t) gives the same gradient up to Monte Carlo error, and the losses differ by a constant."""
    rng = np.random.default_rng(5)
    cfg = fm.config(d_data=2, d_hidden=8)
    n = 200_000
    x1, x0, t = two_modes(n, rng), rng.standard_normal((n, 2)), rng.uniform(0, 1, n)
    xt, u = fm.path(cfg, x0, x1, t), fm.target(cfg, x0, x1)
    um = fm.mixture_field(xt, t, MODES, 0.3, HALF)
    gaps = []
    for seed in (6, 7):
        p = {k: np.random.default_rng(seed).normal(0, 0.5, s) for k, s in fm.param_shapes(cfg).items()}
        lc, gc = fm.regress(p, cfg, xt, t, u)
        lm, gm = fm.regress(p, cfg, xt, t, um)
        gaps.append(lc - lm)
        for k in p:
            np.testing.assert_allclose(gc[k], gm[k], atol=0.02 * (1 + np.abs(gm[k]).max()), err_msg=k)
    assert gaps[0] == pytest.approx(gaps[1], abs=0.02)


def test_exact_field_carries_noise_to_the_mixture():
    rng = np.random.default_rng(8)
    means, w, s = np.array([[-2.0], [1.5]]), np.array([0.4, 0.6]), 0.3
    x = fm.midpoint_sample(exact(means, s, w), rng.standard_normal((50_000, 1)), 30)
    mean = float(w @ means[:, 0])
    assert x.mean() == pytest.approx(mean, abs=0.03)
    assert x.var() == pytest.approx(float(w @ (means[:, 0] ** 2 + s**2)) - mean**2, rel=0.03)
    assert (x[:, 0] > 0).mean() == pytest.approx(0.6, abs=0.01)


def test_one_euler_step_lands_on_the_data_mean():
    """Under the independent coupling u_0(x) = E[x_1] - x, so a single Euler step sends every point to E[x_1]."""
    rng = np.random.default_rng(9)
    means, w = np.array([[-2.0, 0.0], [2.0, 1.0]]), np.array([0.3, 0.7])
    x = fm.euler_sample(exact(means, 0.3, w), rng.standard_normal((7, 2)), 1)
    np.testing.assert_allclose(x, np.broadcast_to(w @ means, x.shape), atol=1e-12)


def test_worked_example():
    """N(0, 1) to N(2, 1): the 1-rectified flow is x_t = 2t + x_0 sqrt((1 - t)^2 + t^2), which ends at x_0 + 2; Euler
    with n steps shrinks the spread to 0, 0.5, 0.72, while one midpoint step is exact."""
    v = exact(np.array([[2.0]]), 1.0, np.ones(1))
    x0 = np.array([[0.0], [1.0]])
    for n, spread in [(1, 0.0), (2, 0.5), (4, 0.72)]:
        x = fm.euler_sample(v, x0, n)
        assert x[0, 0] == pytest.approx(2.0, abs=1e-12)
        assert x[1, 0] - x[0, 0] == pytest.approx(spread, abs=1e-12)
    np.testing.assert_allclose(fm.midpoint_sample(v, x0, 1), [[2.0], [3.0]], atol=1e-12)
    xs = fm.midpoint_sample(v, x0, 50, 0.0, 0.5)
    np.testing.assert_allclose(xs[:, 0], [1.0, 1.0 + math.sqrt(0.5)], atol=1e-6)


def test_log_likelihood_by_the_change_of_variables():
    means, w, s = np.array([[-2.0], [1.5]]), np.array([0.4, 0.6]), 0.5
    v, h = exact(means, s, w), 1e-5

    def div(x, t):
        return (v(x + h, t) - v(x - h, t))[:, 0] / (2 * h)

    x1 = np.array([[-2.5], [-0.3], [0.8], [2.6]])
    got = fm.log_likelihood(v, div, x1, 200, lambda x: fm.log_normal(x, 0.0, 1.0))
    np.testing.assert_allclose(got, fm.mixture_log_density(x1, 1.0, means, s, w), atol=1e-4)


def test_network_log_likelihood_integrates_to_one():
    cfg = fm.config(d_data=1, d_hidden=6)
    rng = np.random.default_rng(10)
    p = {k: rng.normal(0, 0.7, s) for k, s in fm.param_shapes(cfg).items()}
    xs = np.linspace(-9, 9, 1201)[:, None]
    ll = fm.log_likelihood(
        lambda x, t: fm.velocity(p, cfg, x, t)[0],
        lambda x, t: fm.divergence(p, cfg, x, t),
        xs,
        100,
        lambda x: fm.log_normal(x, 0.0, 1.0),
    )
    assert np.exp(ll).sum() * (xs[1, 0] - xs[0, 0]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("n", [1, 2, 5, 7])
def test_assignment_matches_brute_force(n):
    rng = np.random.default_rng(n)
    for _ in range(10):
        c = rng.standard_normal((n, n))
        col = fm.assignment(c)
        assert sorted(col) == list(range(n))
        best = min(sum(c[i, q[i]] for i in range(n)) for q in itertools.permutations(range(n)))
        assert c[np.arange(n), col].sum() == pytest.approx(best, abs=1e-12)


def test_ot_pairs_in_one_dimension_sort_both_sides():
    rng = np.random.default_rng(11)
    x0, x1 = rng.standard_normal((50, 1)), rng.standard_normal((50, 1)) + 3
    a, b = fm.ot_pairs(x0, x1)
    np.testing.assert_array_equal(np.argsort(a[:, 0]), np.argsort(b[:, 0]))


def test_ot_pairs_shorten_and_uncross_the_paths():
    rng = np.random.default_rng(12)
    x0, x1 = rng.standard_normal((128, 2)), two_modes(128, rng)
    a, b = fm.ot_pairs(x0, x1)
    assert np.sum((b - a) ** 2) < 0.7 * np.sum((x1 - x0) ** 2)
    i, j = np.triu_indices(128, 1)
    swapped = np.sum((b[j] - a[i]) ** 2, 1) + np.sum((b[i] - a[j]) ** 2, 1)
    assert np.all(swapped >= np.sum((b[i] - a[i]) ** 2, 1) + np.sum((b[j] - a[j]) ** 2, 1) - 1e-9)


def test_reflow_lowers_the_transport_cost():
    """Theorem 3.5 of Liu et al.: the coupling of the rectified flow costs no more than the independent one under
    every convex cost; the paths of the 1-rectified flow still bend between their two ends."""
    rng = np.random.default_rng(13)
    v = exact(MODES, 0.3, HALF)
    x0 = rng.standard_normal((5_000, 2))
    z0, z1 = fm.reflow_pairs(v, x0, 30)
    x1 = two_modes(5_000, rng)
    assert np.mean(np.sum((z1 - z0) ** 2, 1)) < np.mean(np.sum((x1 - x0) ** 2, 1))
    assert np.mean(np.abs(z1 - z0)) < np.mean(np.abs(x1 - x0))
    mid = fm.midpoint_sample(v, x0, 15, 0.0, 0.5)
    assert np.mean(np.sum((mid - 0.5 * (z0 + z1)) ** 2, 1)) > 0.05


def test_logit_normal_times():
    rng = np.random.default_rng(14)
    t = fm.logit_normal(rng, 200_000, 0.5, 0.6)
    assert np.all((t > 0) & (t < 1))
    z = np.log(t / (1 - t))
    assert z.mean() == pytest.approx(0.5, abs=0.01)
    assert z.std() == pytest.approx(0.6, rel=0.01)
    assert np.median(fm.logit_normal(rng, 200_000)) == pytest.approx(0.5, abs=0.01)


def test_ot_coupling_lowers_the_loss():
    """Minibatch optimal transport pairs noise with nearby data, so the regression target varies less and the
    conditional loss settles lower than under the independent coupling."""
    cfg = fm.config(2, 32, 8)
    final = {}
    for coupling in ("independent", "ot"):
        rng = np.random.default_rng(16)
        p = fm.init(rng, cfg)
        final[coupling] = np.mean(fm.train(p, cfg, two_modes, 200, rng, batch=32, coupling=coupling)[-100:])
    assert final["ot"] < 0.5 * final["independent"]


def test_learns_two_modes():
    rng = np.random.default_rng(15)
    cfg = fm.config(2, 64, 16)
    p = fm.init(rng, cfg)
    trace = fm.train(p, cfg, two_modes, 1500, rng)
    assert np.mean(trace[-100:]) < 0.8 * np.mean(trace[:10])

    def v(x, t):
        return fm.velocity(p, cfg, x, t)[0]

    xs = fm.midpoint_sample(v, rng.standard_normal((2000, 2)), 20)
    dist = np.min(np.linalg.norm(xs[:, None, :] - MODES[None], axis=-1), 1)
    assert 0.4 < (xs[:, 0] > 0).mean() < 0.6
    assert (dist < 1.0).mean() > 0.9
