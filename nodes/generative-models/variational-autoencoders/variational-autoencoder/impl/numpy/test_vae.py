# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the gradients of the three objectives
against finite differences, the closed-form KL against sampling, the two estimators of the ELBO, the importance
weighted bound, and three small experiments: the latent dimensions that the data does not need go inactive, the weight
of the KL term trades rate for distortion, and the bound tightens with the number of samples."""

import json
import math
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import vae

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, decoder="bernoulli"):
    k = c["config"]
    return vae.config(k["d_in"], k["d_hidden"], k["d_latent"], decoder)


def random_params(rng, cfg):
    p = vae.init(rng, cfg)
    return {k: v if v.ndim == 2 else rng.normal(0.0, 0.3, v.shape) for k, v in p.items()}


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    for dec in vae.DECODERS:
        cfg = vae.config(decoder=dec)
        p = {k: v for k, v in params.items() if k in vae.param_shapes(cfg)}
        out, _ = vae.forward(p, cfg, inp["x"], inp["eps"])
        for k in ("mu", "lv", "z", "a") + (("olv",) if dec == "gaussian" else ()):
            np.testing.assert_allclose(out[k], want[f"{dec}/{k}"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(vae.log_weights(cfg, inp["x"], out, inp["eps"]), want[f"{dec}/log_w"], rtol=1e-12)
        for kind in vae.OBJECTIVES:
            value, grads = vae.loss(p, cfg, inp["x"], inp["eps"], kind)
            assert value == pytest.approx(float(want[f"{dec}/{kind}/loss"]), rel=1e-12)
            for k, g in grads.items():
                np.testing.assert_allclose(g, want[f"{dec}/{kind}/d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    bern, gauss = cfg_of(c), cfg_of(c, "gaussian")
    assert vae.param_count(bern) == mod["params"]
    assert vae.param_count(gauss) == mod["params_gaussian"]
    assert vae.flops_forward(bern, k["B"], k["S"]) == mod["flops_forward"]
    assert vae.flops_forward(gauss, k["B"], k["S"]) == mod["flops_gaussian"]
    if k["d_in"] < 100:  # build the small ones and count their arrays
        assert sum(v.size for v in vae.init(np.random.default_rng(0), gauss).values()) == mod["params_gaussian"]


def numeric_grads(f, p, eps=1e-6):
    out = {}
    for k, v in p.items():
        g = np.zeros_like(v)
        for i in np.ndindex(v.shape):
            old = v[i]
            v[i] = old + eps
            up = f()
            v[i] = old - eps
            down = f()
            v[i] = old
            g[i] = (up - down) / (2 * eps)
        out[k] = g
    return out


@pytest.mark.parametrize("decoder", vae.DECODERS)
@pytest.mark.parametrize("kind,beta", [("elbo", 1.0), ("elbo", 0.25), ("elbo_mc", 1.0), ("iwae", 1.0)])
def test_gradients_match_finite_differences(decoder, kind, beta):
    rng = np.random.default_rng(7)
    cfg = vae.config(5, 4, 3, decoder)
    p = random_params(rng, cfg)
    x = rng.uniform(size=(3, 5)) if decoder == "gaussian" else (rng.uniform(size=(3, 5)) > 0.5).astype(float)
    eps = rng.standard_normal((4, 3, 3))
    _, grads = vae.loss(p, cfg, x, eps, kind, beta)
    num = numeric_grads(lambda: vae.objective(p, cfg, x, eps, kind, beta)[0], p)
    for k in p:
        np.testing.assert_allclose(grads[k], num[k], rtol=1e-6, atol=1e-8, err_msg=k)


def test_worked_example():
    """The worked example of the page: mu = (1, 0), sigma^2 = (1, 1/4), eps = (1/2, -2)."""
    mu, lv = np.array([[1.0, 0.0]]), np.log([[1.0, 0.25]])
    kl = vae.kl(mu, lv)[0]
    assert kl == pytest.approx(0.5 * (1.0 + 0.25 - 1.0 + math.log(4.0)))
    assert kl == pytest.approx(0.8181, abs=1e-4)
    z = mu + np.exp(0.5 * lv) * np.array([[0.5, -2.0]])
    np.testing.assert_allclose(z, [[1.5, -1.0]])
    # the single-draw estimate of the KL, log q(z) - log p(z) at that z
    one = vae.log_q(lv, np.array([[0.5, -2.0]]))[0] - vae.log_prior(z)[0]
    assert one == pytest.approx(-0.5 * (0.25 + 4.0) - 0.5 * math.log(0.25) + 0.5 * (2.25 + 1.0))
    assert one == pytest.approx(0.1931, abs=1e-4)


def test_closed_form_kl_is_the_mean_of_the_sampled_one():
    rng = np.random.default_rng(1)
    mu, lv = rng.normal(0, 1, (5, 3)), rng.normal(0, 0.7, (5, 3))
    eps = rng.standard_normal((200_000, 5, 3))
    z = mu + np.exp(0.5 * lv) * eps
    sampled = np.mean(vae.log_q(lv, eps) - vae.log_prior(z), 0)
    np.testing.assert_allclose(sampled, vae.kl(mu, lv), atol=0.02)
    assert np.all(vae.kl(mu, lv) >= 0) and vae.kl(np.zeros((1, 3)), np.zeros((1, 3)))[0] == 0.0


@pytest.mark.parametrize("decoder", vae.DECODERS)
def test_both_estimators_have_the_same_mean(decoder):
    """Estimator A (every term sampled) and estimator B (the KL in closed form) estimate the same ELBO; B has the
    smaller variance here, the claim of Kingma and Welling."""
    rng = np.random.default_rng(2)
    cfg = vae.config(6, 5, 2, decoder)
    p = random_params(rng, cfg)
    x = rng.uniform(size=(1, 6)) if decoder == "gaussian" else (rng.uniform(size=(1, 6)) > 0.5).astype(float)
    a = np.array([vae.objective(p, cfg, x, rng.standard_normal((1, 1, 2)), "elbo_mc")[0] for _ in range(20_000)])
    b = np.array([vae.objective(p, cfg, x, rng.standard_normal((1, 1, 2)), "elbo")[0] for _ in range(20_000)])
    assert abs(a.mean() - b.mean()) < 4 * math.sqrt(a.var() / len(a) + b.var() / len(b))
    assert b.var() < a.var()


def test_one_sample_iwae_is_the_sampled_elbo():
    rng = np.random.default_rng(3)
    cfg = vae.config(decoder="bernoulli")
    p, x = random_params(rng, cfg), (rng.uniform(size=(4, 6)) > 0.5).astype(float)
    eps = rng.standard_normal((1, 4, 2))
    assert vae.objective(p, cfg, x, eps, "iwae")[0] == pytest.approx(vae.objective(p, cfg, x, eps, "elbo_mc")[0])
    # with k draws, the log of the mean weight is at least the mean of the log weights, draw by draw (Jensen)
    eps = rng.standard_normal((7, 4, 2))
    assert vae.objective(p, cfg, x, eps, "iwae")[0] <= vae.objective(p, cfg, x, eps, "elbo_mc")[0]


def test_logmeanexp_is_stable():
    a = np.array([[1000.0, -1000.0], [1000.0, -1001.0]])
    np.testing.assert_allclose(vae.logmeanexp(a), [1000.0, -1000.0 + np.log((1 + np.exp(-1)) / 2)])


def manifold(rng, n, D=8):
    """Points near a two-dimensional surface in D dimensions: tanh of a random linear map of two normal coordinates,
    plus a little noise."""
    W = np.random.default_rng(99).normal(0.0, 1.0, (2, D))
    return 0.8 * np.tanh(rng.standard_normal((n, 2)) @ W) + 0.05 * rng.standard_normal((n, D))


def trained(beta=1.0, J=6, steps=3000):
    X = manifold(np.random.default_rng(0), 2000)
    cfg = vae.config(8, 16, J, "gaussian")
    p = vae.init(np.random.default_rng(1), cfg)
    trace = vae.train(p, cfg, X, np.random.default_rng(2), steps, lr=3e-3, batch=50, beta=beta)
    return p, cfg, X, trace


def test_training_raises_the_elbo_and_the_unneeded_latents_go_inactive():
    """Six latent dimensions for data with two: the KL term switches four of them off, so that their mean no longer
    depends on x and their posterior is the prior; the claim of Kingma and Welling that superfluous latents do not
    overfit, and the active units of Burda et al."""
    p, cfg, X, trace = trained()
    assert np.mean(trace[-200:]) < np.mean(trace[:50]) - 10.0
    activity, active = vae.active_units(p, X)
    assert len(active) == 2
    off = np.setdiff1d(np.arange(6), active)
    assert np.all(activity[off] < 1e-2)
    mu, lv, _ = vae.encode(p, X)
    np.testing.assert_allclose(np.mean(vae.kl(mu[:, off], lv[:, off])), 0.0, atol=0.02)  # q is the prior there


def test_the_weight_of_the_kl_trades_rate_for_distortion():
    """Minimizing D + beta R (Alemi et al.): a small beta buys a low distortion with a high rate, a large one drives
    the rate to zero and the code is ignored, the posterior collapse."""
    eps = np.random.default_rng(3).standard_normal((1, 2000, 6))
    rd = {}
    for beta in (0.1, 1.0, 10.0):
        p, cfg, X, _ = trained(beta)
        rd[beta] = vae.rate_distortion(p, cfg, X, eps), len(vae.active_units(p, X)[1])
    (r1, d1), (r2, d2), (r3, d3) = (rd[b][0] for b in (0.1, 1.0, 10.0))
    assert r1 > r2 > r3 and d1 < d2 < d3
    assert r3 < 0.01 and rd[10.0][1] == 0 and rd[0.1][1] == 6


def test_the_bound_tightens_with_more_samples():
    """Theorem 1 of Burda et al.: E L_1 <= E L_5 <= E L_50 <= log p(x), on a trained model."""
    p, cfg, X, _ = trained()
    rng = np.random.default_rng(5)
    bound = {k: np.mean([vae.iwae_bound(p, cfg, X[:200], rng, k).mean() for _ in range(20)]) for k in (1, 5, 50)}
    assert bound[1] < bound[5] < bound[50]
    assert bound[5] - bound[1] > 0.1


def test_kl_annealing_starts_from_zero():
    X = manifold(np.random.default_rng(0), 200)
    cfg = vae.config(8, 4, 2, "gaussian")
    p = vae.init(np.random.default_rng(1), cfg)
    first = vae.train({k: v.copy() for k, v in p.items()}, cfg, X, np.random.default_rng(2), 1, batch=20, anneal=10)[0]
    eps = np.random.default_rng(2)
    xb = X[eps.integers(0, 200, 20)]
    value, _ = vae.loss(p, cfg, xb, eps.standard_normal((1, 20, 2)), "elbo", beta=0.0)
    assert first == pytest.approx(value)


@settings(max_examples=25, deadline=None)
@given(D=st.integers(1, 6), H=st.integers(1, 6), J=st.integers(1, 4), B=st.integers(1, 5), S=st.integers(1, 4),
       decoder=st.sampled_from(vae.DECODERS))  # fmt: skip
def test_shapes(D, H, J, B, S, decoder):
    cfg = vae.config(D, H, J, decoder)
    rng = np.random.default_rng(0)
    p = vae.init(rng, cfg)
    out, _ = vae.forward(p, cfg, rng.uniform(size=(B, D)), rng.standard_normal((S, B, J)))
    assert out["mu"].shape == out["lv"].shape == (B, J) and out["z"].shape == (S, B, J) and out["a"].shape == (S, B, D)
    assert vae.sample(p, cfg, rng, 3).shape == (3, D)
    assert sum(v.size for v in p.values()) == vae.param_count(cfg)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: the Bernoulli decoder, S draws per example."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((k["S"], k["B"], k["d_latent"]))
    return vae.forward, (vae.init(rng, cfg), cfg, rng.uniform(size=(k["B"], k["d_in"])), eps)
