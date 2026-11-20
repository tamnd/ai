# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the hand-written gradients of the five
losses against finite differences, the optimal discriminator and the value -log 4 + 2 JSD of Theorem 1, and small
experiments: a trained discriminator approaches D*, the minimax generator loss saturates where the non-saturating one
does not, simultaneous steps on the Dirac-GAN spiral out and R1 brings them back, a gradient-penalised critic
estimates the Wasserstein distance, and the generator learns a Gaussian, a two-mode mixture and a shifted plane."""

import json
import math
import pathlib

import numpy as np
import pytest

import gan

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def random_params(rng, cfg):
    p = gan.init(rng, cfg)
    return {k: v if v.ndim == 2 or k == "dis_w2" else rng.normal(0.0, 0.3, v.shape) for k, v in p.items()}


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    fake = gan.generate(p, inp["z"])[0]
    a_real, h_real = gan.critic(p, inp["x"])
    np.testing.assert_allclose(fake, want["fake"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a_real, want["a_real"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gan.critic(p, fake)[0], want["a_fake"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gan.input_grad(p, h_real), want["grad_real"], rtol=1e-12, atol=1e-14)
    for kind in gan.LOSSES:
        for side, (value, grads) in (("d", gan.d_step(p, inp["x"], inp["z"], kind, inp["alpha"], 10.0, 1.0)),
                                     ("g", gan.g_step(p, inp["z"], kind))):  # fmt: skip
            assert value == pytest.approx(float(want[f"{kind}/{side}/loss"]), rel=1e-12)
            assert set(grads) == set(gan.DIS if side == "d" else gan.GEN)
            for k, g in grads.items():
                np.testing.assert_allclose(g, want[f"{kind}/{side}/d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = gan.config(k["d_noise"], k["d_hidden"], k["d_data"], k["d_critic"])
    assert gan.param_count(cfg) == (mod["params_gen"], mod["params_dis"])
    assert sum(gan.param_count(cfg)) == mod["params"]
    assert gan.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    assert sum(v.size for v in gan.init(np.random.default_rng(0), cfg).values()) == mod["params"]


def numeric_grads(f, p, keys, eps=1e-6):
    out = {}
    for k in keys:
        v, g = p[k], np.zeros_like(p[k])
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


@pytest.mark.parametrize("kind", gan.LOSSES)
def test_gradients_match_finite_differences(kind):
    rng = np.random.default_rng(1)
    cfg = gan.config()
    p = random_params(rng, cfg)
    x, z, alpha = rng.normal(0.0, 1.0, (6, 3)), rng.standard_normal((6, 2)), rng.uniform(0.0, 1.0, 6)
    fake = gan.generate(p, z)[0]  # the discriminator's step holds the fakes fixed
    _, grads = gan.d_step(p, x, z, kind, alpha, 10.0, 1.0)
    num = numeric_grads(lambda: gan.dis_loss(p, x, fake, kind, alpha, 10.0, 1.0)[0], p, gan.DIS)
    for k in gan.DIS:
        np.testing.assert_allclose(grads[k], num[k], rtol=1e-6, atol=1e-8, err_msg=k)
    _, grads = gan.g_step(p, z, kind)
    num = numeric_grads(lambda: gan.gen_loss(p, z, kind)[0], p, gan.GEN)
    for k in gan.GEN:
        np.testing.assert_allclose(grads[k], num[k], rtol=1e-6, atol=1e-8, err_msg=k)


def test_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    p = random_params(rng, gan.config())
    x = rng.normal(0.0, 1.0, (5, 3))
    g = gan.input_grad(p, gan.critic(p, x)[1])
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6
        num = (gan.critic(p, x + e)[0] - gan.critic(p, x - e)[0]) / 2e-6
        np.testing.assert_allclose(g[:, j], num, rtol=1e-7, atol=1e-9)


def test_optimal_discriminator_and_theorem_1():
    """eq-dstar maximises V(G, D) pointwise, and V(G, D*) = -log 4 + 2 JSD(p_data || p_g), -log 4 when they agree."""
    x = np.linspace(-8.0, 9.0, 34001)
    dx = x[1] - x[0]
    pd, pg = npdf(x, 0.0, 1.0), npdf(x, 1.5, 0.7)
    ds = gan.optimal_discriminator(pd, pg)
    best = gan.value_at(pd, pg, ds, dx)
    assert best == pytest.approx(-math.log(4) + 2 * gan.jsd(pd, pg, dx), abs=1e-9)
    assert gan.value_at(pd, pd, np.full_like(x, 0.5), dx) == pytest.approx(-math.log(4), abs=1e-9)
    assert 0 < gan.jsd(pd, pg, dx) < math.log(2)
    rng = np.random.default_rng(3)
    for _ in range(5):  # any other discriminator does worse
        other = np.clip(ds + 0.05 * np.sin(rng.uniform(0.5, 3.0) * x + rng.uniform(0, 6)), 1e-9, 1 - 1e-9)
        assert gan.value_at(pd, pg, other, dx) < best


def test_trained_discriminator_approaches_the_optimal_one():
    rng = np.random.default_rng(0)
    p, state = gan.init(rng, gan.config(1, 1, 1, 16)), {}
    for _ in range(2000):
        x, fake = rng.normal(0.0, 1.0, (256, 1)), rng.normal(1.5, 0.7, (256, 1))
        _, terms, pen = gan.dis_loss(p, x, fake, "nonsat")
        gan.adam(p, gan.dis_backward(p, terms, pen), state, 0.01, b1=0.9)
    grid = np.linspace(-2.0, 3.0, 51)
    d = gan.sigmoid(gan.critic(p, grid[:, None])[0])
    assert np.max(np.abs(d - gan.optimal_discriminator(npdf(grid, 0.0, 1.0), npdf(grid, 1.5, 0.7)))) < 0.04


def test_minimax_generator_saturates_and_nonsaturating_does_not():
    """Early in training D rejects the fakes and log(1 - D(G(z))) is flat; -log D(G(z)) has the same fixed point and
    a gradient two orders of magnitude larger."""
    rng = np.random.default_rng(0)
    cfg = gan.config(1, 8, 1, 8)
    p, state = gan.init(rng, cfg), {}
    for _ in range(300):
        _, g = gan.d_step(p, rng.normal(4.0, 0.5, (128, 1)), rng.standard_normal((128, 1)), "nonsat")
        gan.adam(p, g, state, 0.01, b1=0.9)
    z = rng.standard_normal((1000, 1))
    assert gan.sigmoid(gan.critic(p, gan.generate(p, z)[0])[0]).mean() < 0.05

    def size(kind):
        return math.sqrt(sum(np.sum(g**2) for g in gan.g_step(p, z, kind)[1].values()))

    assert size("nonsat") > 20 * size("minimax")


def test_dirac_gan():
    """Simultaneous steps spiral out, each multiplying the distance to the equilibrium by about
    sqrt(1 + h^2 f'(0)^2); alternating steps circle; the R1 penalty gamma/2 psi^2 makes both converge."""
    h = 0.1
    sim = np.linalg.norm(gan.dirac_gan(1.0, 0.0, h, 2000), axis=1)
    assert np.all(np.diff(sim) > 0) and sim[-1] > 2.5
    near = np.linalg.norm(gan.dirac_gan(1e-3, 0.0, h, 100), axis=1)
    assert near[-1] / near[0] == pytest.approx(math.sqrt(1 + h**2 / 4) ** 100, rel=1e-4)
    alt = np.linalg.norm(gan.dirac_gan(1.0, 0.0, h, 2000, mode="alt"), axis=1)
    assert 0.95 < alt.min() and alt.max() < 1.05
    for mode in ("sim", "alt"):
        assert np.linalg.norm(gan.dirac_gan(1.0, 0.0, h, 500, gamma=1.0, mode=mode)[-1]) < 1e-3


def test_gradient_penalty_critic_estimates_the_wasserstein_distance():
    """Two Gaussians in the plane, one shifted by 2: W1 = 2, and the critic's gradient has norm near 1 between them."""
    rng = np.random.default_rng(0)
    p, state, shift = gan.init(rng, gan.config(1, 1, 2, 16)), {}, np.array([2.0, 0.0])
    for _ in range(1500):
        x, fake = rng.normal(0.0, 0.5, (256, 2)), rng.normal(0.0, 0.5, (256, 2)) + shift
        _, terms, pen = gan.dis_loss(p, x, fake, "wgan_gp", rng.uniform(0.0, 1.0, 256))
        gan.adam(p, gan.dis_backward(p, terms, pen), state, 0.01, b1=0.5, b2=0.9)
    x, fake = rng.normal(0.0, 0.5, (20000, 2)), rng.normal(0.0, 0.5, (20000, 2)) + shift
    est = gan.critic(p, x)[0].mean() - gan.critic(p, fake)[0].mean()
    assert 1.8 < est < 2.5
    line = np.stack([np.linspace(0.0, 2.0, 11), np.zeros(11)], 1)
    n = np.linalg.norm(gan.input_grad(p, gan.critic(p, line)[1]), axis=1)
    assert np.all(np.abs(n - 1.0) < 0.3)


def gauss(rng, n):
    return rng.normal(2.0, 0.5, (n, 1))


def mixture(rng, n):
    return np.where(rng.uniform(size=(n, 1)) < 0.5, -2.0, 2.0) + rng.normal(0.0, 0.2, (n, 1))


@pytest.mark.parametrize("kind", ["nonsat", "minimax"])
def test_generator_learns_a_gaussian(kind):
    rng = np.random.default_rng(0)
    cfg = gan.config(1, 16, 1, 16)
    p = gan.init(rng, cfg)
    gan.train(p, cfg, gauss, rng, 2000, kind, batch=128, lr=2e-3)
    s = gan.sample(p, cfg, rng, 20000)[:, 0]
    assert abs(s.mean() - 2.0) < 0.3 and 0.2 < s.std() < 0.8


@pytest.mark.parametrize("kind", ["nonsat", "r1"])
def test_generator_covers_both_modes(kind):
    rng = np.random.default_rng(0)
    cfg = gan.config(1, 16, 1, 16)
    p = gan.init(rng, cfg)
    gan.train(p, cfg, mixture, rng, 2000, kind, batch=128, lr=2e-3, gamma=1.0)
    s = gan.sample(p, cfg, rng, 20000)[:, 0]
    assert 0.35 < np.mean(s > 0) < 0.65 and s.std() > 1.5


@pytest.mark.parametrize("kind", ["nonsat", "wgan", "wgan_gp"])
def test_generator_learns_a_shifted_plane(kind):
    """With the cross-entropy loss the discriminator's loss ends near log 4, the value of Theorem 1 at the optimum."""
    kw = {"nonsat": {"lr": 2e-3}, "wgan": {"n_dis": 5, "c": 0.1, "opt": "rms", "lr": 5e-3},
          "wgan_gp": {"n_dis": 5, "b1": 0.0, "b2": 0.9, "lr": 1e-3}}[kind]  # fmt: skip
    rng = np.random.default_rng(0)
    cfg = gan.config(2, 16, 2, 16)
    p, mean = gan.init(rng, cfg), np.array([2.0, -1.0])
    trace = gan.train(p, cfg, lambda r, n: r.normal(0.0, 0.5, (n, 2)) + mean, rng, 1500, kind, batch=128, **kw)
    s = gan.sample(p, cfg, rng, 20000)
    assert np.all(np.abs(s.mean(0) - mean) < 0.5) and np.all(s.std(0) < 1.0)
    if kind == "nonsat":
        assert trace[-100:, 0].mean() == pytest.approx(math.log(4), abs=0.05)


def test_worked_example():
    """Two outcomes, p_data = (1/2, 1/2) and p_g = (0.9, 0.1): D* = (5/14, 5/6), V(G, D*) = -1.1828 = -log 4 + 2 JSD."""
    pd, pg = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    ds = gan.optimal_discriminator(pd, pg)
    np.testing.assert_allclose(ds, [5 / 14, 5 / 6], rtol=1e-15)
    assert math.log(ds[1] / (1 - ds[1])) == pytest.approx(math.log(5), rel=1e-12)
    assert gan.value_at(pd, pg, ds, 1.0) == pytest.approx(-1.18280, abs=1e-5)
    assert gan.jsd(pd, pg, 1.0) == pytest.approx(0.10175, abs=1e-5)
    assert gan.value_at(pd, pg, np.full(2, 0.5), 1.0) == pytest.approx(-math.log(4), rel=1e-15)
    a = math.log(0.01 / 0.99)  # a fake that D gives probability 0.01
    assert -gan.sigmoid(a) == pytest.approx(-0.01) and gan.sigmoid(a) - 1 == pytest.approx(-0.99)


@pytest.mark.parametrize("kind", gan.LOSSES)
def test_losses_are_finite_at_extreme_logits(kind):
    """The losses are softplus of the logit, never the log of a sigmoid, so a discriminator that is sure of itself
    gives a large finite loss and a bounded gradient, not inf or nan."""
    rng = np.random.default_rng(4)
    for b in (-800.0, 800.0):
        p = gan.init(rng, gan.config())
        p["dis_b2"][:] = b
        x, z, alpha = rng.normal(0.0, 1.0, (4, 3)), rng.standard_normal((4, 2)), rng.uniform(0.0, 1.0, 4)
        for value, grads in (gan.d_step(p, x, z, kind, alpha), gan.g_step(p, z, kind)):
            assert math.isfinite(value)
            assert all(np.all(np.isfinite(g)) and np.max(np.abs(g)) < 100 for g in grads.values())


def gan_pass(p, z, x):
    """One generator pass on the noise and one discriminator pass on the fakes and the data, as eq-flops counts."""
    fake = gan.generate(p, z)[0]
    return gan.critic(p, np.concatenate([fake, x]))[0]


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    k = c["config"]
    cfg = gan.config(k["d_noise"], k["d_hidden"], k["d_data"], k["d_critic"])
    rng = np.random.default_rng(0)
    z, x = rng.standard_normal((k["B"], k["d_noise"])), rng.standard_normal((k["B"], k["d_data"]))
    return gan_pass, (gan.init(rng, cfg), z, x)
