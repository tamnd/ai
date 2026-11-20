# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Tests of the JAX version: the shared fixture with the gradient from jax.grad, the counts of shapes.json, the
transition matrices, the posterior against Bayes' rule and multinomial diffusion, the schedule weights from jax.grad
against their closed forms, the one-sample bound as an unbiased estimate of the exact one, score entropy with its
gradient vanishing at the true ratios, and a denoiser trained by a scanned Adam whose sampler, a fori_loop, draws the
data."""

import itertools
import json
import math
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import discrete_diffusion as dd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return dd.config(k["seq_len"], k["vocab"], k["hidden"])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: logits for B partly masked sequences."""
    cfg, k = cfg_of(c), c["config"]
    z = jax.random.randint(jax.random.key(1), (k["B"], k["seq_len"]), 0, k["vocab"] + 1)
    return dd.forward, (dd.init(jax.random.key(0), cfg), z, cfg)


def load(name):
    return {k: jnp.asarray(v) for k, v in np.load(FIXTURES / name).items()}


def test_shared_fixture():
    P, inp, want = load("params-small.npz"), load("input-small.npz"), load("expected-small.npz")
    L, K, H = (int(v) for v in inp["cfg"])
    cfg = dd.config(L, K, H)
    z, x, t, u = inp["z"], inp["x"], inp["t"], inp["u"]
    np.testing.assert_allclose(dd.forward(P, z, cfg), want["logits"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dd.subs_probs(want["logits"], z, K), want["probs"], rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(dd.mask_tokens(x, t, u, K, "cosine"), want["z_masked"])
    Z, X, Wt = dd.mc_batch(x, t, u, K, "cosine")
    assert float(dd.weighted_ce(P, cfg, Z, X, Wt)) == pytest.approx(float(want["mc_loss"]), rel=1e-12)
    for k, g in dd.weighted_ce_grad(P, cfg, Z, X, Wt).items():
        np.testing.assert_allclose(g, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    lp = jax.nn.log_softmax(dd.forward(P, Z, cfg), -1)
    np.testing.assert_allclose(dd.mdlm_estimate(lp, Z, X, t, K, "cosine"), want["per_seq"], rtol=1e-12, atol=1e-13)
    rev = dd.posterior(dd.one_hot(inp["xt"], 4), inp["p0"], dd.uniform_q(0.3, 4), dd.uniform_qbar(0.6, 4))
    np.testing.assert_allclose(rev, want["reverse_uniform"], rtol=1e-12, atol=1e-13)
    assert float(dd.exact_nelbo(P, cfg, x[0])) == pytest.approx(float(want["nelbo_x0"]), rel=1e-12)
    s = inp["scores"]
    se = dd.score_entropy(s, inp["p_se"], jnp.ones_like(s))
    assert float(se) == pytest.approx(float(want["score_entropy"]), rel=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert dd.param_count(cfg) == mod["params"]
    assert dd.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = dd.init(jax.random.key(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out = jax.eval_shape(lambda q, z: dd.forward(q, z, cfg), p, jnp.zeros((k["B"], k["seq_len"]), int))
    assert out.shape == (k["B"], k["seq_len"], k["vocab"])


def test_transition_matrices_and_posterior():
    K, betas = 4, [0.2, 0.3, 0.35]
    abar = np.cumprod([1 - b for b in betas])
    for make, closed in ((dd.uniform_q, dd.uniform_qbar), (dd.absorbing_q, dd.absorbing_qbar)):
        Qbar = dd.cumulative([make(b, K) for b in betas])
        for k, a in enumerate(abar):
            np.testing.assert_allclose(Qbar[k + 1], closed(float(a), K), atol=1e-15)
    Qs = [dd.uniform_q(b, K) for b in betas]
    Qbar = dd.cumulative(Qs)
    for x0, xt in itertools.product(range(K), repeat=2):
        joint = Qbar[1][x0, :] * Qs[1][:, xt]
        got = dd.posterior(dd.one_hot(xt, K), dd.one_hot(x0, K), Qs[1], Qbar[1])
        np.testing.assert_allclose(got, joint / joint.sum(), atol=1e-15)
    p0 = jax.nn.softmax(jax.random.normal(jax.random.key(2), (K,)))
    for xt in range(K):
        np.testing.assert_allclose(dd.posterior(dd.one_hot(xt, K), p0, Qs[2], Qbar[2]),
                                   dd.multinomial_posterior(dd.one_hot(xt, K), p0, 1 - betas[2], abar[1], K),
                                   atol=1e-15)  # fmt: skip


def test_schedule_weights():
    t = jnp.array([0.1, 0.3, 0.6, 0.95])
    np.testing.assert_allclose(dd.weight(t, "linear"), 1.0 / t, rtol=1e-13)
    c = 0.5 * math.pi
    want = c * jnp.sin(c * (1 - t)) / jnp.cos(c * (1 - t))  # alpha = 1 - cos(c (1 - t)), so -alpha' / (1 - alpha)
    np.testing.assert_allclose(dd.weight(t, "cosine"), want, rtol=1e-13)


def test_monte_carlo_estimate_is_unbiased():
    K, L = 3, 3
    cfg = dd.config(L, K, 8)
    p = dd.init(jax.random.key(4), cfg)
    x = jnp.array([0, 1, 2])
    exact = float(dd.exact_nelbo(p, cfg, x))
    n = 40000
    xs = jnp.broadcast_to(x, (n, L))
    for kind, key in (("linear", 5), ("cosine", 6)):
        k1, k2 = jax.random.split(jax.random.key(key))
        t = jax.random.uniform(k1, (n,), minval=1e-9, maxval=1.0)
        Z, X, _ = dd.mc_batch(xs, t, jax.random.uniform(k2, (n, L)), K, kind)
        est = dd.mdlm_estimate(jax.nn.log_softmax(dd.forward(p, Z, cfg), -1), Z, X, t, K, kind)
        assert abs(float(est.mean()) - exact) < 4 * float(est.std()) / math.sqrt(n)


def test_score_entropy_gradient_vanishes_at_the_ratios():
    N = 5
    ks = jax.random.split(jax.random.key(7), 4)
    w = jax.random.uniform(ks[0], (N, N), minval=0.5, maxval=2.0)
    p0 = jax.nn.softmax(jax.random.normal(ks[1], (N,)))
    kern = jax.nn.softmax(jax.random.normal(ks[2], (N, N)), -1)
    pk = p0 @ kern
    assert float(dd.score_entropy(dd.concrete_score(pk), pk, w)) == pytest.approx(0.0, abs=1e-14)
    g = jax.grad(dd.denoising_score_entropy)(dd.concrete_score(pk), p0, kern, w)
    np.testing.assert_allclose(g * (1 - jnp.eye(N)), 0.0, atol=1e-12)
    g2 = jax.grad(dd.score_entropy)(dd.concrete_score(pk), pk, w)
    np.testing.assert_allclose(g2 * (1 - jnp.eye(N)), 0.0, atol=1e-12)
    s = jnp.exp(jax.random.normal(ks[3], (N, N)))
    assert float(dd.score_entropy(s, pk, w)) > 0


def test_trained_denoiser_reaches_the_entropy_and_samples_the_data():
    K, L = 3, 4
    data = dd.markov_data(L, K)
    cfg = dd.config(L, K, 32)
    p, trace = dd.train_exact(dd.init(jax.random.key(1), cfg), cfg, data, 600, lr=0.01)
    H = dd.entropy(data)
    assert float(trace[0]) > H + 1.0
    assert H - 1e-9 <= float(trace[-1]) < H + 0.01

    def tv(T, key):
        z = np.asarray(dd.sample(p, cfg, jax.random.key(key), 20000, T))
        emp = np.zeros((K,) * L)
        np.add.at(emp, tuple(z.T), 1.0 / len(z))
        return 0.5 * np.abs(emp - np.asarray(data)).sum()

    assert tv(1, 8) > 0.55  # one step draws every token independently
    assert tv(16, 9) < 0.1
