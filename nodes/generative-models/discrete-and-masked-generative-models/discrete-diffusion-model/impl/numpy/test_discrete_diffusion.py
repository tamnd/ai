# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the worked example, the transition
matrices and their closed-form products, the D3PM posterior against Bayes' rule and against the multinomial diffusion
posterior, the exact D3PM bound against the entropy, the masked posterior and its KL term, the weighted cross-entropy
bound against the any-order chain rule and across schedules, its Monte Carlo estimate, the ancestral sampler's exact
distribution, MaskGIT's decoding, score entropy and its denoising form, the concrete score of the absorbing process,
the hand-written gradient against finite differences, and a denoiser trained by the exact bound."""

import itertools
import json
import math
import pathlib

import numpy as np
import pytest

import discrete_diffusion as dd

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return dd.config(k["seq_len"], k["vocab"], k["hidden"])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: logits for B partly masked sequences."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    z = rng.integers(0, k["vocab"] + 1, (k["B"], k["seq_len"]))
    return dd.forward, (dd.init(rng, cfg), z, cfg)


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    L, K, H = (int(v) for v in inp["cfg"])
    cfg = dd.config(L, K, H)
    z, x, t, u = inp["z"], inp["x"], inp["t"], inp["u"]
    np.testing.assert_allclose(dd.forward(P, z, cfg), want["logits"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dd.subs_probs(want["logits"], z, K), want["probs"], rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(dd.mask_tokens(x, t, u, K, "cosine"), want["z_masked"])
    Z, X, Wt = dd.mc_batch(x, t, u, K, "cosine")
    assert dd.weighted_ce(P, cfg, Z, X, Wt) == pytest.approx(float(want["mc_loss"]), rel=1e-12)
    for k, g in dd.weighted_ce_grad(P, cfg, Z, X, Wt).items():
        np.testing.assert_allclose(g, want[f"d{k}"], err_msg=k, rtol=1e-10, atol=1e-13)
    lp = dd.log_softmax(dd.forward(P, Z, cfg))
    np.testing.assert_allclose(dd.mdlm_estimate(lp, Z, X, t, K, "cosine"), want["per_seq"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dd.posterior(dd.one_hot(inp["xt"], 4), inp["p0"], dd.uniform_q(0.3, 4),
                                            dd.uniform_qbar(0.6, 4)), want["reverse_uniform"], rtol=1e-12, atol=1e-13)  # fmt: skip
    assert dd.exact_nelbo(dd.denoise_fn(P, cfg), x[0], K) == pytest.approx(float(want["nelbo_x0"]), rel=1e-12)
    s = inp["scores"]
    assert dd.score_entropy(s, inp["p_se"], np.ones_like(s)) == pytest.approx(float(want["score_entropy"]), rel=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert dd.param_count(cfg) == mod["params"]
    assert dd.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = dd.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    z = np.full((k["B"], k["seq_len"]), k["vocab"])
    assert dd.forward(p, z, cfg).shape == (k["B"], k["seq_len"], k["vocab"])


def test_worked_example():
    # Two tokens over {0, 1}, equal with probability 0.8: p(0,0) = p(1,1) = 0.4 and p(0,1) = p(1,0) = 0.1.
    p = np.array([[0.4, 0.1], [0.1, 0.4]])
    den = dd.conditional_denoiser(p, 2)
    x = np.array([0, 0])
    assert dd.mask_coefficient(1, 2) == 0.5 and dd.mask_coefficient(2, 2) == 0.5
    # 0.5 (-log 0.8) twice for one masked token, 0.5 (-log 0.5 - log 0.5) for both: -log 0.8 - log 0.5 = -log 0.4.
    assert dd.exact_nelbo(den, x, 2) == pytest.approx(-math.log(0.4), rel=1e-14)
    assert dd.exact_nelbo(den, x, 2) == pytest.approx(0.9163, abs=1e-4)
    one = dd.sampler_distribution(den, 2, 2, 1)
    np.testing.assert_allclose(one, np.full((2, 2), 0.25))  # one step draws both tokens from their marginals
    two = dd.sampler_distribution(den, 2, 2, 2)
    # linear schedule, two steps: each token is unmasked at t = 1/2 with probability 1/2.
    # p(0,0) = 1/4 * 1/4 (both in the first step) + 1/2 * 0.4 (one in each step: 0.5 for the first, 0.8 for the
    # second given it) + 1/4 * 1/4 (both in the last step)
    assert two[0, 0] == pytest.approx(0.25 * 0.25 + 0.5 * 0.5 * 0.8 + 0.25 * 0.25, abs=1e-12)
    assert two[0, 0] == pytest.approx(0.325, abs=1e-12)


@pytest.mark.parametrize("K", [3, 5])
def test_transition_matrices_and_their_products(K):
    betas = [0.1, 0.25, 0.4, 0.7]
    for make, closed in (
        (dd.uniform_q, lambda a: dd.uniform_qbar(a, K)),
        (dd.absorbing_q, lambda a: dd.absorbing_qbar(a, K)),
    ):
        Qs = [make(b, K) for b in betas]
        for Q in Qs:
            np.testing.assert_allclose(Q.sum(1), 1.0)
            assert np.all(Q >= 0)
        Qbar = dd.cumulative(Qs)
        abar = np.cumprod([1 - b for b in betas])
        for t in range(1, len(betas) + 1):
            np.testing.assert_allclose(Qbar[t], closed(abar[t - 1]), atol=1e-14)
    np.testing.assert_allclose(dd.uniform_q(1.0, K), np.full((K, K), 1.0 / K))
    np.testing.assert_allclose(dd.absorbing_q(1.0, K)[:, K], 1.0)  # beta = 1 absorbs every state
    np.testing.assert_allclose(dd.uniform_q(0.3, K).sum(0), 1.0)  # doubly stochastic, so the uniform is stationary


def test_posterior_is_bayes_rule_and_matches_multinomial_diffusion():
    rng = np.random.default_rng(1)
    K = 4
    betas = [0.2, 0.3, 0.35]
    Qs = [dd.uniform_q(b, K) for b in betas]
    Qbar = dd.cumulative(Qs)
    for x0, xt in itertools.product(range(K), repeat=2):
        joint = Qbar[1][x0, :] * Qs[1][:, xt]  # q(x_1 | x_0) q(x_2 | x_1)
        want = joint / joint.sum()
        got = dd.posterior(dd.one_hot(xt, K), dd.one_hot(x0, K), Qs[1], Qbar[1])
        np.testing.assert_allclose(got, want, atol=1e-15)
    abar = np.cumprod([1 - b for b in betas])
    for _ in range(5):
        xt, p0 = dd.one_hot(rng.integers(K), K), rng.dirichlet(np.ones(K))
        np.testing.assert_allclose(dd.posterior(xt, p0, Qs[2], Qbar[2]),
                                   dd.multinomial_posterior(xt, p0, 1 - betas[2], abar[1], K), atol=1e-15)  # fmt: skip


@pytest.mark.parametrize("kind", ["uniform", "absorbing"])
def test_d3pm_bound_equals_the_entropy_at_its_optimum(kind):
    rng = np.random.default_rng(2)
    K, betas = 4, [0.1, 0.2, 0.3, 0.5, 1.0]  # the last step reaches the stationary distribution exactly
    if kind == "uniform":
        Qs, n = [dd.uniform_q(b, K) for b in betas], K
        p0, prior = rng.dirichlet(np.ones(K)), np.full(K, 1.0 / K)
    else:
        Qs, n = [dd.absorbing_q(b, K) for b in betas], K + 1
        p0, prior = np.r_[rng.dirichlet(np.ones(K)), 0.0], np.eye(K + 1)[K]
    H = -np.sum(p0[p0 > 0] * np.log(p0[p0 > 0]))

    def mean_bound(den):
        out = 0.0
        for x0 in range(n):
            if p0[x0] > 0:
                LT, mids, L0 = dd.d3pm_bound(x0, Qs, den, prior)
                assert LT == pytest.approx(0.0, abs=1e-14) and min(mids) >= -1e-15
                out += p0[x0] * (LT + sum(mids) + L0)
        return out

    assert mean_bound(lambda t, xt: p0) == pytest.approx(H, rel=1e-12)
    post = mean_bound(dd.posterior_mean(p0, Qs))
    if kind == "absorbing":
        assert post == pytest.approx(H, rel=1e-12)
    else:
        assert post > H + 0.05  # the posterior mean counts q(x_t | x_0) twice under eq-x0-param
    other = np.r_[rng.dirichlet(np.ones(K)), np.zeros(n - K)]  # any other prediction does worse
    assert mean_bound(lambda t, xt: other) > H


def test_masked_posterior_and_kl_term():
    rng = np.random.default_rng(3)
    K = 5
    for _ in range(10):
        a_t = rng.uniform(0.05, 0.9)
        a_s = rng.uniform(a_t, 1.0)
        x = int(rng.integers(K))
        probs = rng.dirichlet(np.ones(K))
        # eq-mask-posterior is the general D3PM posterior of the absorbing matrices from s to t
        Q_st = dd.absorbing_qbar(a_t / a_s, K)
        got = dd.posterior(dd.one_hot(K, K + 1), dd.one_hot(x, K + 1), Q_st, dd.absorbing_qbar(a_s, K))
        np.testing.assert_allclose(dd.mask_posterior(K, x, a_s, a_t, K), got, atol=1e-15)
        q = dd.mask_posterior(K, x, a_s, a_t, K)
        pr = dd.subs_reverse(K, probs, a_s, a_t, K)
        assert float(dd.kl(q, pr)) == pytest.approx(dd.masked_kl_term(probs, x, a_s, a_t), rel=1e-12)
        y = int(rng.integers(K))
        assert float(dd.kl(dd.mask_posterior(y, x, a_s, a_t, K), dd.subs_reverse(y, probs, a_s, a_t, K))) == 0.0


@pytest.mark.parametrize("L", [2, 3, 4])
def test_nelbo_is_the_any_order_chain_rule_and_ignores_the_schedule(L):
    K = 3
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)
    for x in itertools.product(range(K), repeat=L):
        x = np.array(x)
        e = dd.exact_nelbo(den, x, K)
        assert e == pytest.approx(-math.log(p[tuple(x)]), rel=1e-12)
    rng = np.random.default_rng(L)
    logits = rng.normal(size=(K + 1,) * L + (L, K))  # an arbitrary denoiser that ignores t

    def den2(z):
        return dd.subs_probs(logits[tuple(z)][None], np.asarray(z)[None], K)[0]

    x = rng.integers(0, K, L)
    e = dd.exact_nelbo(den2, x, K)
    for kind in ("linear", "cosine"):
        assert dd.nelbo_by_quadrature(den2, x, K, kind) == pytest.approx(e, rel=1e-10)
    for m in range(1, L + 1):
        a = np.linspace(0, 1, 200001)
        num = np.trapezoid(a ** (L - m) * (1 - a) ** (m - 1), a)
        assert dd.mask_coefficient(m, L) == pytest.approx(num, rel=1e-6)


def test_monte_carlo_estimate_is_unbiased():
    K, L = 3, 3
    p = dd.markov_data(L, K)
    cfg = dd.config(L, K, 8)
    prm = dd.init(np.random.default_rng(4), cfg)
    x = np.array([[0, 1, 2]])
    exact = dd.exact_nelbo(dd.denoise_fn(prm, cfg), x[0], K)
    rng = np.random.default_rng(5)
    n = 40000
    xs = np.repeat(x, n, 0)
    for kind in ("linear", "cosine"):
        t = rng.uniform(1e-9, 1.0, n)
        Z, X, Wt = dd.mc_batch(xs, t, rng.random((n, L)), K, kind)
        lp = dd.log_softmax(dd.forward(prm, Z, cfg))
        est = dd.mdlm_estimate(lp, Z, X, t, K, kind)
        assert dd.weighted_ce(prm, cfg, Z, X, Wt) == pytest.approx(est.mean(), rel=1e-12)
        assert abs(est.mean() - exact) < 4 * est.std() / math.sqrt(n)
    assert p.sum() == pytest.approx(1.0)


def test_ancestral_sampler_distribution():
    K, L = 3, 3
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)
    marg = [p.sum(axis=tuple(j for j in range(L) if j != i)) for i in range(L)]
    prod = np.einsum("a,b,c->abc", *marg)
    np.testing.assert_allclose(dd.sampler_distribution(den, L, K, 1), prod, atol=1e-15)
    tv = [0.5 * np.abs(dd.sampler_distribution(den, L, K, T) - p).sum() for T in (1, 2, 4, 8, 32)]
    assert all(a > b for a, b in zip(tv, tv[1:], strict=False)) and tv[-1] < 0.025
    rng = np.random.default_rng(6)
    table = {z: den(np.array(z)) for z in itertools.product(range(K + 1), repeat=L)}

    def den_b(z):
        return np.stack([table[tuple(r)] for r in z])

    S = dd.sample(den_b, 20000, L, K, 4, rng)
    emp = np.zeros((K,) * L)
    np.add.at(emp, tuple(S.T), 1.0 / len(S))
    assert 0.5 * np.abs(emp - dd.sampler_distribution(den, L, K, 4)).sum() < 0.03


def test_maskgit_decoding():
    assert dd.maskgit_counts(16, 8)[-1] == 0
    for N, T in ((16, 8), (5, 12), (256, 8)):
        c = dd.maskgit_counts(N, T)
        assert len(c) == T and all(a > b or a == 0 for a, b in zip([N] + c, c, strict=False))
    c = dd.maskgit_counts(256, 8)
    assert c[:3] == [math.ceil(math.cos(math.pi / 16 * k) * 256) for k in (1, 2, 3)]
    K, L, B, T = 3, 6, 50, 4
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)

    def den_b(z):
        return np.stack([den(r) for r in z])

    u = np.random.default_rng(7).random((T, B, L))
    z, hist = dd.maskgit_decode(den_b, B, L, K, T, u)
    assert np.all(z < K)
    for n, (a, b) in zip(dd.maskgit_counts(L, T), itertools.pairwise(hist), strict=True):
        assert np.all(np.sum(b == K, axis=1) == n)
        fixed = a != K
        assert np.all(b[fixed] == a[fixed])  # a token once fixed is never masked again or changed
    # One token per step, the most confident first: the samples are likelier than the data on average, a mode-seeking
    # bias that ancestral sampling, which draws the order at random, does not have.
    z, _ = dd.maskgit_decode(den_b, B, L, K, L, np.random.default_rng(8).random((L, B, L)))
    assert np.mean([math.log(p[tuple(r)]) for r in z]) > np.sum(p * np.log(p)) + 1.0


def test_score_entropy():
    rng = np.random.default_rng(8)
    N = 5
    p = rng.dirichlet(np.ones(N))
    w = rng.uniform(0.5, 2.0, (N, N))
    r = dd.concrete_score(p)
    assert dd.score_entropy(r, p, w) == pytest.approx(0.0, abs=1e-14)
    for _ in range(20):
        s = np.exp(rng.normal(size=(N, N)))
        assert dd.score_entropy(s, p, w) > 0
    # Denoising score entropy differs from score entropy by a constant that does not depend on s.
    p0 = rng.dirichlet(np.ones(N))
    kern = rng.dirichlet(np.ones(N), size=N)
    pk = p0 @ kern
    diffs = [dd.score_entropy(s, pk, w) - dd.denoising_score_entropy(s, p0, kern, w)
             for s in (np.exp(rng.normal(size=(N, N))) for _ in range(6))]  # fmt: skip
    np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)
    g = np.zeros((N, N))
    s0 = dd.concrete_score(pk)
    for i, j in itertools.product(range(N), repeat=2):
        if i != j:
            e = np.zeros((N, N))
            e[i, j] = 1e-6
            g[i, j] = (
                dd.denoising_score_entropy(s0 + e, p0, kern, w) - dd.denoising_score_entropy(s0 - e, p0, kern, w)
            ) / 2e-6
    np.testing.assert_allclose(g, 0.0, atol=1e-7)  # the denoising form is minimized at the true ratios too


def test_absorbing_concrete_score_is_the_denoiser_times_alpha_ratio():
    K, L = 3, 3
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)
    for a in (0.2, 0.5, 0.9):
        for z in itertools.product(range(K + 1), repeat=L):
            z = np.array(z)
            for i in np.flatnonzero(z == K):
                pz = dd.masked_marginal(p, z, a, K)
                for y in range(K):
                    z2 = z.copy()
                    z2[i] = y
                    ratio = dd.masked_marginal(p, z2, a, K) / pz
                    assert ratio == pytest.approx(a / (1 - a) * den(z)[i, y], rel=1e-12, abs=1e-15)


def test_gradient_matches_finite_differences():
    K, L = 3, 3
    cfg = dd.config(L, K, 5)
    rng = np.random.default_rng(9)
    prm = dd.init(rng, cfg, scale=2.0)
    prm = {k: v + 0.1 * rng.standard_normal(v.shape) for k, v in prm.items()}
    Z, X, Wt = dd.exact_batch(dd.markov_data(L, K), K)
    g = dd.weighted_ce_grad(prm, cfg, Z, X, Wt)
    for k in prm:
        num = np.zeros_like(prm[k])
        for idx in np.ndindex(prm[k].shape):
            d = np.zeros_like(prm[k])
            d[idx] = 1e-6
            hi = dd.weighted_ce({**prm, k: prm[k] + d}, cfg, Z, X, Wt)
            lo = dd.weighted_ce({**prm, k: prm[k] - d}, cfg, Z, X, Wt)
            num[idx] = (hi - lo) / 2e-6
        np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-8, err_msg=k)


def test_trained_denoiser_reaches_the_entropy_and_samples_the_data():
    K, L = 3, 4
    p = dd.markov_data(L, K)
    cfg = dd.config(L, K, 32)
    prm, trace = dd.train_exact(dd.init(np.random.default_rng(1), cfg), cfg, p, 600, lr=0.01)
    H = dd.entropy(p)
    assert trace[0] > H + 1.0
    assert trace[-1] >= H - 1e-9 and trace[-1] < H + 0.01  # the bound can only approach the entropy from above
    den = dd.denoise_fn(prm, cfg)
    tv1 = 0.5 * np.abs(dd.sampler_distribution(den, L, K, 1) - p).sum()
    tv16 = 0.5 * np.abs(dd.sampler_distribution(den, L, K, 16) - p).sum()
    assert tv1 > 0.6 and tv16 < 0.06
