# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the schedule, the forward marginal
against the chained forward steps, the posterior against Bayes' rule, the hand-written gradients against finite
differences, the score of the exact noise prediction, the samplers driven by the exact noise prediction for Gaussian
data, DDIM with eta = 1 as the DDPM step and with eta = 0 as a deterministic map, the DDIM marginals, the KL term as a
weighted noise error, the variational bound above the model's negative log-likelihood, classifier-free guidance as an
implicit classifier's gradient, the worked example of the page, and a model that learns a two-mode mixture."""

import json
import math
import pathlib

import numpy as np
import pytest

import diffusion as dm

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
MODES = np.array([[-2.0, 0.0], [2.0, 0.0]])


def two_modes(n, rng):
    """Half the points around (-2, 0), half around (2, 0), each N(mode, 0.3^2 I)."""
    return MODES[rng.integers(0, 2, n)] + 0.3 * rng.standard_normal((n, 2))


def exact_eps(cfg, means, s, weights):
    """E[eps | x_t] for data drawn from the Gaussian mixture, as an eps_fn(x, t) of the samplers."""
    abar = dm.schedule(cfg)["abar"]
    return lambda x, t: dm.mixture_eps(x, abar[t - 1][:, None], means, s, weights)


def affine_moments(step, m, v, noisy):
    """Push N(m, v) in one dimension through a step x -> a x + c + g z that is affine in x; returns the new moments."""
    c = step(np.zeros((1, 1)), np.zeros((1, 1)))[0, 0]
    a = step(np.ones((1, 1)), np.zeros((1, 1)))[0, 0] - c
    g = step(np.zeros((1, 1)), np.ones((1, 1)))[0, 0] - c if noisy else 0.0
    return a * m + c, a * a * v + g * g


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = dm.config()
    x0, t, eps, z = inp["x0"], inp["t"], inp["eps"], inp["z"]
    np.testing.assert_allclose(dm.time_embedding(t, cfg["d_time"]), want["emb"], rtol=1e-12, atol=1e-14)
    xt = dm.q_sample(cfg, x0, t, eps)
    np.testing.assert_allclose(xt, want["xt"], rtol=1e-12, atol=1e-14)
    out, _ = dm.eps_model(p, cfg, xt, t)
    np.testing.assert_allclose(out, want["eps_pred"], rtol=1e-12, atol=1e-14)
    loss, g = dm.loss_and_grads(p, cfg, x0, t, eps)
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    assert set(g) == set(dm.param_shapes(cfg))
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    pm, pv = dm.posterior(cfg, xt, x0, t)
    np.testing.assert_allclose(pm, want["post_mean"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pv, want["post_var"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dm.reverse_mean(cfg, xt, t, out), want["mu"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dm.ddpm_step(cfg, xt, t, out, z), want["x_prev"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        dm.ddim_step(cfg, xt, t, t // 2, out, z, 0.5),
        want["x_ddim"],
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = dm.config(k["d_data"], k["d_hidden"], k["d_time"], k["n_steps"])
    assert dm.param_count(cfg) == mod["params"]
    assert dm.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    assert sum(v.size for v in dm.init(np.random.default_rng(0), cfg).values()) == mod["params"]
    out, _ = dm.eps_model(dm.init(np.random.default_rng(0), cfg), cfg, np.zeros((k["B"], k["d_data"])),
                          np.ones(k["B"], int))  # fmt: skip
    assert out.shape == (k["B"], k["d_data"])


def test_schedule_of_ho_et_al():
    s = dm.schedule(dm.config(n_steps=1000, beta_1=1e-4, beta_T=0.02))
    np.testing.assert_allclose(s["abar"], np.cumprod(1 - s["beta"]))
    assert s["beta_tilde"][0] == 0.0
    assert np.all(s["beta_tilde"][1:] < s["beta"][1:])
    assert 3e-5 < s["abar"][-1] < 5e-5
    # L_T for data uniform on [-1, 1]: E[KL(N(sqrt(abar_T) x0, 1 - abar_T) || N(0, 1))] per dimension, in bits.
    a = s["abar"][-1]
    lt = 0.5 * (a / 3 + (1 - a) - 1 - math.log(1 - a)) / math.log(2)
    assert 1e-6 < lt < 2e-5


def test_forward_marginal_matches_the_chain():
    cfg = dm.config(n_steps=10)
    rng = np.random.default_rng(0)
    x0 = np.array([[1.0, -2.0, 0.5]])
    n, t = 200_000, 7
    x = np.repeat(x0, n, 0)
    for k in range(1, t + 1):
        x = dm.forward_step(cfg, x, np.full(n, k), rng.standard_normal(x.shape))
    ab = dm.schedule(cfg)["abar"][t - 1]
    np.testing.assert_allclose(x.mean(0), math.sqrt(ab) * x0[0], atol=0.01)
    np.testing.assert_allclose(x.var(0), np.full(3, 1 - ab), rtol=0.01)
    one = dm.q_sample(cfg, np.repeat(x0, n, 0), np.full(n, t), rng.standard_normal((n, 3)))
    np.testing.assert_allclose(one.mean(0), x.mean(0), atol=0.01)
    np.testing.assert_allclose(one.var(0), x.var(0), rtol=0.02)


@pytest.mark.parametrize("t", [2, 5, 10])
def test_posterior_is_bayes_rule(t):
    """q(x_{t-1} | x_t, x_0) is proportional to q(x_t | x_{t-1}) q(x_{t-1} | x_0) as a function of x_{t-1}."""
    cfg = dm.config(d_data=1)
    s = dm.schedule(cfg)
    x0, xt = np.array([[0.7]]), np.array([[-0.4]])
    m, v = dm.posterior(cfg, xt, x0, np.array([t]))
    grid = np.linspace(-3, 3, 61)[:, None]
    b, abp = s["beta"][t - 1], s["abar_prev"][t - 1]
    joint = dm.log_normal(xt.repeat(61, 0), math.sqrt(1 - b) * grid, b) + dm.log_normal(grid, math.sqrt(abp) * 0.7,
                                                                                         1 - abp)  # fmt: skip
    diff = joint - dm.log_normal(grid, m[0, 0], v[0, 0])
    np.testing.assert_allclose(diff, diff[0], atol=1e-9)


def test_gradients_against_finite_differences():
    cfg = dm.config()
    rng = np.random.default_rng(1)
    p = {k: rng.normal(0, 0.5, s) for k, s in dm.param_shapes(cfg).items()}
    x0, t, eps = (
        rng.standard_normal((5, 3)),
        np.array([1, 3, 5, 8, 10]),
        rng.standard_normal((5, 3)),
    )
    _, g = dm.loss_and_grads(p, cfg, x0, t, eps)
    for k in p:
        num = np.zeros_like(p[k])
        for i in np.ndindex(p[k].shape):
            old = p[k][i]
            p[k][i] = old + 1e-6
            up, _ = dm.loss_and_grads(p, cfg, x0, t, eps)
            p[k][i] = old - 1e-6
            dn, _ = dm.loss_and_grads(p, cfg, x0, t, eps)
            p[k][i] = old
            num[i] = (up - dn) / 2e-6
        np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-8, err_msg=k)


def test_exact_eps_is_the_scaled_score():
    """eq-tweedie: E[eps | x_t] = -sqrt(1 - abar_t) grad log q_t(x_t), checked by finite differences on a mixture."""
    rng = np.random.default_rng(2)
    x, means, w = rng.standard_normal((6, 2)), MODES, np.array([0.3, 0.7])
    for abar in (0.9, 0.3, 0.01):
        num = np.stack([(dm.mixture_log_density(x + 1e-5 * e, abar, means, 0.3, w)
                         - dm.mixture_log_density(x - 1e-5 * e, abar, means, 0.3, w)) / 2e-5 for e in np.eye(2)], 1)  # fmt: skip
        eps = dm.mixture_eps(x, abar, means, 0.3, w)
        np.testing.assert_allclose(eps, -math.sqrt(1 - abar) * num, rtol=1e-6, atol=1e-8)
    cfg = dm.config(d_data=2)
    t = np.full(6, 4)
    np.testing.assert_allclose(
        dm.score_from_eps(cfg, eps, t),
        -eps / math.sqrt(1 - dm.schedule(cfg)["abar"][3]),
    )


@pytest.mark.parametrize("var", ["beta", "beta_tilde"])
def test_ddpm_with_the_exact_eps_samples_the_data(var):
    """Gaussian data N(1.5, 0.5^2) and the schedule of Ho et al.: the exact eps makes each step affine, so the moments
    of x_0 after T = 1000 steps follow exactly; both choices of sigma_t^2 land within 1% of the data's."""
    cfg = dm.config(d_data=1, n_steps=1000, beta_1=1e-4, beta_T=0.02)
    eps_fn = exact_eps(cfg, [[1.5]], 0.5, [1.0])
    m, v = 0.0, 1.0
    for t in range(1000, 0, -1):
        tt = np.array([t])
        m, v = affine_moments(
            lambda x, z, tt=tt: dm.ddpm_step(cfg, x, tt, eps_fn(x, tt), z, var),
            m,
            v,
            t > 1,
        )
    assert m == pytest.approx(1.5, abs=1e-3)
    assert math.sqrt(v) == pytest.approx(0.5, rel=0.01)


def test_ddim_with_the_exact_eps_and_few_steps_shrinks_the_spread():
    """The deterministic sampler with all 1000 steps lands within 1% of the data's spread; with 10 of them, each step
    is a shrinking affine map and the spread falls below 0.8 of the data's."""
    cfg = dm.config(d_data=1, n_steps=1000, beta_1=1e-4, beta_T=0.02)
    eps_fn = exact_eps(cfg, [[1.5]], 0.5, [1.0])
    sd = {}
    for n in (1000, 10):
        tau = dm.ddim_steps(cfg, n)
        m, v = 0.0, 1.0
        for t, s in zip(tau[::-1], np.concatenate([[0], tau[:-1]])[::-1], strict=True):
            tt, ss = np.array([t]), np.array([s])
            m, v = affine_moments(lambda x, z, tt=tt, ss=ss: dm.ddim_step(cfg, x, tt, ss, eps_fn(x, tt), z), m, v,
                                  False)  # fmt: skip
        sd[n] = math.sqrt(v)
        assert m == pytest.approx(1.5, abs=0.01)
    assert sd[1000] == pytest.approx(0.5, rel=0.01)
    assert sd[10] < 0.8 * 0.5


def test_ddim_with_eta_one_is_the_ddpm_step():
    cfg = dm.config()
    rng = np.random.default_rng(3)
    xt, eps, z = (rng.standard_normal((4, 3)) for _ in range(3))
    t = np.array([2, 4, 7, 10])
    s = dm.schedule(cfg)
    np.testing.assert_allclose(dm.ddim_sigma(cfg, t, t - 1, 1.0)[:, 0] ** 2, s["beta_tilde"][t - 1])
    np.testing.assert_allclose(dm.ddim_step(cfg, xt, t, t - 1, eps, z, 1.0),
                               dm.ddpm_step(cfg, xt, t, eps, z, "beta_tilde"), rtol=1e-12)  # fmt: skip


def test_ddim_with_eta_zero_is_deterministic():
    cfg = dm.config()
    p = {k: np.random.default_rng(4).normal(0, 0.5, s) for k, s in dm.param_shapes(cfg).items()}
    eps_fn = lambda x, t: dm.eps_model(p, cfg, x, t)[0]  # noqa: E731
    xT = np.random.default_rng(5).standard_normal((3, 3))
    a = dm.ddim_sample(eps_fn, cfg, 3, np.random.default_rng(6), 5, 0.0, xT)
    b = dm.ddim_sample(eps_fn, cfg, 3, np.random.default_rng(7), 5, 0.0, xT)
    np.testing.assert_array_equal(a, b)
    c = dm.ddim_sample(eps_fn, cfg, 3, np.random.default_rng(6), 5, 1.0, xT)
    assert not np.allclose(a, c)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_ddim_step_keeps_the_forward_marginals(eta):
    """eq-ddim-step with the true eps of x_t: if x_t ~ q(x_t | x_0) then x_s ~ q(x_s | x_0), for every eta."""
    cfg = dm.config(d_data=1)
    rng = np.random.default_rng(8)
    n, x0 = 400_000, np.full((400_000, 1), 0.8)
    t, s = np.full(n, 9), np.full(n, 3)
    eps = rng.standard_normal((n, 1))
    xs = dm.ddim_step(cfg, dm.q_sample(cfg, x0, t, eps), t, s, eps, rng.standard_normal((n, 1)), eta)
    ab = dm.schedule(cfg)["abar"][2]
    assert xs.mean() == pytest.approx(math.sqrt(ab) * 0.8, abs=0.005)
    assert xs.var() == pytest.approx(1 - ab, rel=0.01)


@pytest.mark.parametrize("var", ["beta", "beta_tilde"])
def test_kl_term_is_a_weighted_eps_error(var):
    """eq-eps-weight: KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t)) = beta_t^2 / (2 sigma_t^2 alpha_t (1 - abar_t))
    ||eps - eps_theta||^2 plus a term that does not depend on eps_theta."""
    cfg = dm.config()
    s = dm.schedule(cfg)
    rng = np.random.default_rng(9)
    x0, eps, guess = (rng.standard_normal((4, 3)) for _ in range(3))
    t = np.array([2, 3, 6, 10])
    xt = dm.q_sample(cfg, x0, t, eps)
    mq, vq = dm.posterior(cfg, xt, x0, t)
    sig = dm.sigma2(cfg, t, var)
    kl = dm.gauss_kl(mq, vq, dm.reverse_mean(cfg, xt, t, guess), sig)
    b, a, ab = s["beta"][t - 1], s["alpha"][t - 1], s["abar"][t - 1]
    r = vq[:, 0] / sig[:, 0]
    want = b**2 / (2 * sig[:, 0] * a * (1 - ab)) * np.sum((eps - guess) ** 2, 1) + 1.5 * (r - 1 - np.log(r))
    np.testing.assert_allclose(kl, want, rtol=1e-10)


@pytest.mark.parametrize("var", ["beta", "beta_tilde"])
def test_bound_is_above_the_models_negative_log_likelihood(var):
    """eq-elbo: with the exact eps for N(1.5, 0.5^2) and T = 10 the model's own density of x_0 is Gaussian (each step is
    affine and the decoder adds beta_1); the average bound lies above its negative log-likelihood, which lies above the
    data's entropy."""
    cfg = dm.config(d_data=1)
    eps_fn = exact_eps(cfg, [[1.5]], 0.5, [1.0])
    m, v = 0.0, 1.0
    for t in range(10, 1, -1):
        tt = np.array([t])
        m, v = affine_moments(
            lambda x, z, tt=tt: dm.ddpm_step(cfg, x, tt, eps_fn(x, tt), z, var),
            m,
            v,
            True,
        )
    one = np.array([1])
    m, v = affine_moments(lambda x, z: dm.reverse_mean(cfg, x, one, eps_fn(x, one)), m, v, False)
    v += dm.schedule(cfg)["beta"][0]
    rng = np.random.default_rng(10)
    x0 = 1.5 + 0.5 * rng.standard_normal((20_000, 1))
    bound = dm.neg_elbo(eps_fn, cfg, x0, rng, var).mean()
    nll = -dm.log_normal(x0, m, v).mean()
    entropy = 0.5 * math.log(2 * math.pi * math.e * 0.25)
    assert bound > nll + 0.03
    assert nll > entropy - 0.02


def test_guidance_follows_an_implicit_classifier():
    """eq-guidance with exact predictions: (1 + w) eps(x, c) - w eps(x) = -sqrt(1 - abar) grad[log q_t(x | c)
    + w log q_t(c | x)], with q_t(c | x) from Bayes' rule over two classes; w = 0 is the conditional prediction."""
    rng = np.random.default_rng(11)
    x, w, abar = rng.standard_normal((5, 2)), np.array([0.5, 0.5]), 0.4
    ec = dm.mixture_eps(x, abar, MODES[1:], 0.3, [1.0])
    eu = dm.mixture_eps(x, abar, MODES, 0.3, w)
    np.testing.assert_allclose(dm.guided_eps(ec, eu, 0.0), ec)

    def log_class(y):
        return (
            math.log(0.5)
            + dm.mixture_log_density(y, abar, MODES[1:], 0.3, [1.0])
            - dm.mixture_log_density(y, abar, MODES, 0.3, w)
        )

    grad = np.stack(
        [(log_class(x + 1e-5 * e) - log_class(x - 1e-5 * e)) / 2e-5 for e in np.eye(2)],
        1,
    )
    for gw in (1.0, 3.0):
        want = ec - gw * math.sqrt(1 - abar) * grad
        np.testing.assert_allclose(dm.guided_eps(ec, eu, gw), want, rtol=1e-6, atol=1e-8)


def test_worked_example():
    """The page's example: T = 2, beta = (0.1, 0.5), x_0 = 1, eps = 0.5."""
    cfg = dm.config(d_data=1, n_steps=2, beta_1=0.1, beta_T=0.5)
    s = dm.schedule(cfg)
    np.testing.assert_allclose(s["abar"], [0.9, 0.45])
    t, x0, eps = np.array([2]), np.array([[1.0]]), np.array([[0.5]])
    xt = dm.q_sample(cfg, x0, t, eps)
    assert xt[0, 0] == pytest.approx(1.0416, abs=1e-4)
    m, v = dm.posterior(cfg, xt, x0, t)
    assert v[0, 0] == pytest.approx(0.0909, abs=1e-4)
    assert m[0, 0] == pytest.approx(0.9964, abs=1e-4)
    assert dm.reverse_mean(cfg, xt, t, eps)[0, 0] == pytest.approx(m[0, 0], rel=1e-12)
    assert dm.predict_x0(cfg, xt, t, eps)[0, 0] == pytest.approx(1.0, rel=1e-12)


def test_learns_two_modes():
    """Algorithm 1 for 3000 Adam steps on two_modes with T = 100: the loss ends within 0.05 of the floor set by the
    exact eps, and ancestral samples split between the modes and sit near them."""
    rng = np.random.default_rng(1)
    cfg = dm.config(2, 64, 16, 100, 1e-3, 0.1)
    p = dm.init(rng, cfg)
    trace = dm.train(p, cfg, two_modes, 3000, rng)
    x0, t = two_modes(8000, rng), rng.integers(1, 101, 8000)
    eps = rng.standard_normal(x0.shape)
    floor = np.mean(
        np.sum(
            (exact_eps(cfg, MODES, 0.3, [0.5, 0.5])(dm.q_sample(cfg, x0, t, eps), t) - eps) ** 2,
            1,
        )
    )
    assert np.mean(trace[-200:]) < floor + 0.05
    assert np.mean(trace[:20]) > floor + 0.3
    xs = dm.ddpm_sample(lambda x, tt: dm.eps_model(p, cfg, x, tt)[0], cfg, 2000, rng)
    dist = np.min(np.linalg.norm(xs[:, None] - MODES[None], axis=-1), 1)
    assert 0.4 < np.mean(xs[:, 0] > 0) < 0.6
    assert np.mean(dist < 1.0) > 0.95
    assert 0.25 < np.median(dist) < 0.5


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: one evaluation of the noise predictor."""
    k = c["config"]
    cfg = dm.config(k["d_data"], k["d_hidden"], k["d_time"], k["n_steps"])
    rng = np.random.default_rng(0)
    t = rng.integers(1, k["n_steps"] + 1, k["B"])
    return dm.eps_model, (dm.init(rng, cfg), cfg, rng.standard_normal((k["B"], k["d_data"])), t)
