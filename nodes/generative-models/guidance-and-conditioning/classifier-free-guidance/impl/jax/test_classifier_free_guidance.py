# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Tests of the JAX implementation: the shared fixture with jax.grad, the guided noise as the score of the tilted
density by jax.grad, classifier guidance, the end variance of the guided ODE as a scan, the rescale and the thresholds,
the interval with jax.lax.cond and its count, and a denoiser trained by a scan of Adam steps with a dropped condition."""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import classifier_free_guidance as cg

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=msg)


def test_shared_fixture():
    P = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = cg.config(*(int(v) for v in inp["cfg"]))
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    close(cg.forward(P, cfg, z, t, y), want["eps_hat"])
    assert float(cg.eps_loss(P, cfg, z, t, y, eps)) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, g in cg.eps_loss_grad(P, cfg, z, t, y, eps).items():
        close(g, want[f"d{k}"], rtol=1e-10, msg=k)
    guided = cg.guide(inp["e_c"], inp["e_u"], 3.0)
    close(guided, want["guided"])
    close(cg.rescale(inp["e_c"], guided, 0.7), want["rescaled"])
    close(cg.dynamic_threshold(inp["x0"], 90.0), want["dyn"])
    abar = cg.alpha_bars(1000)
    means, w = inp["means"], jnp.array([0.5, 0.5])
    eps_fn = cg.guided_eps(lambda zz, tt: cg.exact_eps_cond(zz, jnp.full(len(zz), abar[tt]), means, 0.1, 0),
                           lambda zz, tt: cg.exact_eps(zz, jnp.full(len(zz), abar[tt]), means, 0.1, w),
                           2.0, abar, interval=(0.3, 3.0))  # fmt: skip
    out, nfe = cg.ddim_sample(eps_fn, inp["zT"], abar, [999, 600, 300, 100, 0])
    close(out, want["ddim"], rtol=1e-10, atol=1e-12)
    assert int(nfe) == int(want["nfe"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cg.config(k["dim"], k["hidden"], k["temb"], k["classes"])
    assert cg.param_count(cfg) == mod["params"]
    assert cg.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = cg.init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out = cg.forward(p, cfg, jnp.zeros((k["B"], k["dim"])), 3, jnp.full(k["B"], k["classes"]))
    assert out.shape == (k["B"], k["dim"])


def test_guided_noise_is_tilted_score():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    means, s2, z = jax.random.normal(k1, (3, 3)), 0.3, jax.random.normal(k2, (7, 3))
    w = jax.random.uniform(k3, (3,)) + 0.2
    w = w / w.sum()
    abar, c = cg.alpha_bars(1000), 1
    for t in (10, 300, 800):
        a = jnp.full(len(z), abar[t])
        for s in (0.0, 2.5, 7.0):

            def logp(zz, a=a, s=s):
                lc = cg.log_marginal(zz, a, means[c : c + 1], s2, jnp.ones(1))
                return (s * lc + (1 - s) * cg.log_marginal(zz, a, means, s2, w)).sum()

            guided = cg.guide(cg.exact_eps_cond(z, a, means, s2, c), cg.exact_eps(z, a, means, s2, w), s)
            close(guided, -jnp.sqrt(1 - a[:, None]) * jax.grad(logp)(z), rtol=1e-10, atol=1e-12)
            g = cg.classifier_grad(z, a, means, s2, w, jnp.full(len(z), c))
            close(guided, cg.classifier_guided_eps(cg.exact_eps_cond(z, a, means, s2, c), g, a, s - 1.0), atol=1e-12)
            close(guided, cg.classifier_guided_eps(cg.exact_eps(z, a, means, s2, w), g, a, s), atol=1e-12)


def test_guided_ode_variance():
    abar = cg.alpha_bars(1000)
    for s in (1.0, 2.0, 3.0):

        def eps(zz, tt, s=s):
            a = jnp.full(len(zz), abar[tt])
            return cg.guide(cg.gaussian_eps(zz, a, 1.0, 0.25), cg.gaussian_eps(zz, a, 0.0, 1.25), s)

        x, _ = cg.ddim_sample(cg.plain_eps(eps), jnp.array([[0.0], [1.0]]), abar, jnp.arange(999, -1, -1))
        assert float((x[1, 0] - x[0, 0]) ** 2) == pytest.approx(0.25**s * 1.25 ** (1 - s), rel=0.03)


def test_rescale_and_thresholds():
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    e_c, e_u = jax.random.normal(k1, (2, 4, 3, 5, 5))
    out = cg.guide(e_c, e_u, 7.5)
    close(cg.rescale(e_c, out, 0.0), out)
    close(cg.rescale(e_c, out, 1.0).std(axis=(1, 2, 3)), e_c.std(axis=(1, 2, 3)))
    x = 1.6 * jax.random.normal(k2, (5, 40))
    close(cg.dynamic_threshold(x, 0.0), cg.static_threshold(x))
    for p in (50.0, 99.5):
        assert bool((jnp.abs(cg.dynamic_threshold(x, p)) <= 1.0 + 1e-15).all())


def test_interval_count():
    abar = cg.alpha_bars(1000)
    steps = jnp.asarray(np.linspace(999, 0, 32).round().astype(int))
    sig = cg.edm_sigma(abar[steps])
    inside = int(((sig > 0.28) & (sig <= 2.9)).sum())
    _, nfe = cg.ddim_sample(cg.guided_eps(lambda z, t: z, lambda z, t: 0 * z, 2.0, abar, interval=(0.28, 2.9)),
                            jnp.zeros((1, 2)), abar, steps)  # fmt: skip
    assert int(nfe) == len(steps) + inside < 2 * len(steps)


def test_training_and_guided_sampling():
    key = jax.random.PRNGKey(10)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    D, C, s2 = 2, 3, 0.25
    m = jnp.array([[1.5, 0.0], [-0.75, 1.3], [-0.75, -1.3]])
    y = jax.random.randint(k1, (6000,), 0, C)
    x0 = m[y] + math.sqrt(s2) * jax.random.normal(k2, (6000, D))
    cfg = cg.config(D, 64, 16, C)
    abar = cg.alpha_bars(1000)
    p, trace = jax.jit(lambda p, k: cg.train(p, cfg, x0, y, abar, 2500, k))(cg.init(k3, cfg), k4)
    assert float(trace[-200:].mean()) < 0.6 * float(trace[:50].mean())
    zT = jax.random.normal(k5, (3000, D))
    steps = np.linspace(999, 0, 25).round().astype(int)
    w = jnp.full(C, 1.0 / C)
    conf = []
    for s in (1.0, 3.0):
        eps = cg.guided_eps(lambda z, t: cg.forward(p, cfg, z, t, 1), lambda z, t: cg.forward(p, cfg, z, t, C), s, abar)
        x, _ = cg.ddim_sample(eps, zT, abar, steps)
        conf.append(float(jnp.exp(cg.log_classifier(x, jnp.ones(len(x)) - 1e-12, m, s2, w))[:, 1].mean()))
        if s == 1.0:
            assert float(jnp.abs(x.mean(0) - m[1]).max()) < 0.25
    assert conf[1] > conf[0] and conf[1] > 0.97
