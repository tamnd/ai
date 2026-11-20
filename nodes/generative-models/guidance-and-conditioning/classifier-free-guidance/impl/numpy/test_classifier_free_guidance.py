# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Tests of the NumPy reference: the shared fixture, the conventions of eq-cfg, the guided noise as the score of the
tilted density, classifier guidance and its equivalences, the invariance under the parameterization, the perturbed
Gaussian mean, the closed forms of eq-tilt and eq-ode-var against DDIM, the rescale, the thresholds, the interval and
its count of evaluations, the worked example, the trade-off on two overlapping classes, and a denoiser trained with a
dropped condition whose guided samples are more confidently in their class."""

import json
import math
import pathlib

import numpy as np
import pytest

import classifier_free_guidance as cg

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def mixture(rng, B=7, D=3, K=3):
    means = rng.standard_normal((K, D))
    w = rng.random(K) + 0.2
    return means, 0.3, w / w.sum(), rng.standard_normal((B, D))


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = cg.config(*(int(v) for v in inp["cfg"]))
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    close(cg.forward(P, cfg, z, t, y), want["eps_hat"])
    assert cg.eps_loss(P, cfg, z, t, y, eps) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, g in cg.eps_loss_grad(P, cfg, z, t, y, eps).items():
        close(g, want[f"d{k}"], rtol=1e-10, msg=k)
    close(cg.guide(inp["e_c"], inp["e_u"], 3.0), want["guided"])
    close(cg.rescale(inp["e_c"], cg.guide(inp["e_c"], inp["e_u"], 3.0), 0.7), want["rescaled"])
    close(cg.dynamic_threshold(inp["x0"], 90.0), want["dyn"])
    abar = cg.alpha_bars(1000)
    means, w = inp["means"], np.array([0.5, 0.5])
    eps_fn, count = cg.guided_sampler(lambda zz, tt: cg.exact_eps_cond(zz, np.full(len(zz), abar[tt]), means, 0.1, 0),
                                      lambda zz, tt: cg.exact_eps(zz, np.full(len(zz), abar[tt]), means, 0.1, w),
                                      2.0, abar, interval=(0.3, 3.0))  # fmt: skip
    close(cg.ddim_sample(eps_fn, inp["zT"], abar, [999, 600, 300, 100, 0]), want["ddim"], rtol=1e-10, atol=1e-12)
    assert count["nfe"] == int(want["nfe"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cg.config(k["dim"], k["hidden"], k["temb"], k["classes"])
    assert cg.param_count(cfg) == mod["params"]
    assert cg.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = cg.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out = cg.forward(p, cfg, np.zeros((k["B"], k["dim"])), 3, np.full(k["B"], k["classes"]))
    assert out.shape == (k["B"], k["dim"])


def test_conventions():
    rng = np.random.default_rng(0)
    e_c, e_u = rng.standard_normal((2, 5, 4))
    for w in (0.0, 0.3, 2.0):
        close(cg.guide_ho(e_c, e_u, w), cg.guide(e_c, e_u, 1.0 + w))
        close(cg.guide(e_c, e_u, 1.0 + w), (1.0 + w) * e_c + (1.0 - (1.0 + w)) * e_u)  # Imagen's w e_c + (1 - w) e_u
    close(cg.guide(e_c, e_u, 1.0), e_c)
    close(cg.guide(e_c, e_u, 0.0), e_u)


def test_guided_noise_is_tilted_score():
    """With exact models, guided noise = -sqrt(1 - a) grad log [p(z|c)^s p(z)^(1-s)] = the noise of p(z|c) p(c|z)^(s-1)."""
    rng = np.random.default_rng(1)
    means, s2, w, z = mixture(rng)
    abar, c, h = cg.alpha_bars(1000), 1, 1e-5
    for t in (10, 300, 800):
        a = np.full(len(z), abar[t])
        for s in (0.0, 1.0, 2.5, 7.0):

            def logp(zz, a=a, s=s):
                lc = cg.log_marginal(zz, a, means[c : c + 1], s2, np.ones(1))
                return s * lc + (1.0 - s) * cg.log_marginal(zz, a, means, s2, w)

            grad = np.stack([(logp(z + h * e) - logp(z - h * e)) / (2 * h) for e in np.eye(z.shape[1])], 1)
            guided = cg.guide(cg.exact_eps_cond(z, a, means, s2, c), cg.exact_eps(z, a, means, s2, w), s)
            close(guided, -np.sqrt(1.0 - a[:, None]) * grad, rtol=1e-7, atol=1e-8)
            close(guided, cg.classifier_guided_eps(cg.exact_eps_cond(z, a, means, s2, c),
                                                   cg.classifier_grad(z, a, means, s2, w, np.full(len(z), c)), a,
                                                   s - 1.0), rtol=1e-11, atol=1e-12)  # fmt: skip


def test_classifier_guidance_equivalences():
    """Ho and Salimans: guidance at s on the unconditional model equals s - 1 on the conditional one; the classifier
    gradient is the difference of the two scores."""
    rng = np.random.default_rng(2)
    means, s2, w, z = mixture(rng)
    a = np.full(len(z), cg.alpha_bars(1000)[400])
    c = np.full(len(z), 2)
    g = cg.classifier_grad(z, a, means, s2, w, c)
    e_u, e_c = cg.exact_eps(z, a, means, s2, w), cg.exact_eps_cond(z, a, means, s2, 2)
    close(g, -(e_c - e_u) / np.sqrt(1.0 - a[:, None]), rtol=1e-10, atol=1e-12)
    for s in (1.0, 3.0, 10.0):
        close(cg.classifier_guided_eps(e_u, g, a, s), cg.classifier_guided_eps(e_c, g, a, s - 1.0), atol=1e-12)
        close(cg.classifier_guided_eps(e_u, g, a, s), cg.guide(e_c, e_u, s), atol=1e-12)
    lc = cg.log_classifier(z, a, means, s2, w)
    close(np.exp(lc).sum(1), np.ones(len(z)))
    h = 1e-6
    fd = np.stack([(cg.log_classifier(z + h * e, a, means, s2, w)[:, 2] - cg.log_classifier(z - h * e, a, means, s2, w)[:, 2])
                   / (2 * h) for e in np.eye(z.shape[1])], 1)  # fmt: skip
    close(g, fd, rtol=1e-6, atol=1e-7)


def test_parameterization_invariance():
    """The weights of eq-cfg sum to 1, so guiding the x-prediction, the v-prediction or the score gives the same step."""
    rng = np.random.default_rng(3)
    z, e_c, e_u = rng.standard_normal((3, 6, 4))
    a = np.full(6, 0.37)
    for s in (0.5, 2.0, 6.0):
        e = cg.guide(e_c, e_u, s)
        close(cg.eps_to_x0(z, e, a), cg.guide(cg.eps_to_x0(z, e_c, a), cg.eps_to_x0(z, e_u, a), s))
        close(cg.eps_to_v(z, e, a), cg.guide(cg.eps_to_v(z, e_c, a), cg.eps_to_v(z, e_u, a), s))
        close(-e / np.sqrt(1 - a[:, None]), cg.guide(-e_c / np.sqrt(1 - a[:, None]), -e_u / np.sqrt(1 - a[:, None]), s))
        close(cg.x0_to_eps(z, cg.eps_to_x0(z, e, a), a), e)


def test_perturbed_gaussian_mean():
    """log N(x; mu, S) + g.x - log N(x; mu + S g, S) is the same constant at every x (Dhariwal and Nichol, eq. 7-10)."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((3, 3))
    S, mu, g = A @ A.T + 0.5 * np.eye(3), rng.standard_normal(3), rng.standard_normal(3)
    Si = np.linalg.inv(S)

    def logn(x, m):
        return -0.5 * np.einsum("bi,ij,bj->b", x - m, Si, x - m)

    x = rng.standard_normal((50, 3)) * 3
    diff = logn(x, mu) + x @ g - logn(x, cg.perturbed_mean(mu, S, g))
    close(diff, np.full(50, diff[0]), rtol=1e-11, atol=1e-11)
    close(diff[0], mu @ g + 0.5 * g @ S @ g, rtol=1e-11)


def test_tilted_gaussian_and_guided_ode():
    """eq-tilt is the density whose score guidance computes at every noise level; but the guided ODE does not carry
    the data to it: its end variance is v1^s v0^(1-s) (eq-ode-var), below the tilted variance for s > 1."""
    rng = np.random.default_rng(5)
    abar = cg.alpha_bars(1000)
    m1, v1, m0, v0 = 1.0, 0.25, 0.0, 1.25
    z = rng.standard_normal((5, 1))
    for t in (50, 500):
        a = np.full(5, abar[t])
        for s in (1.0, 2.0, 4.0):
            tg = cg.tilted_gaussian(math.sqrt(abar[t]) * m1, abar[t] * v1 + 1 - abar[t],
                                    math.sqrt(abar[t]) * m0, abar[t] * v0 + 1 - abar[t], s)  # fmt: skip
            assert tg is not None
            mt, vt = tg
            score = -(z - mt) / vt
            guided = cg.guide(cg.gaussian_eps(z, a, m1, v1), cg.gaussian_eps(z, a, m0, v0), s)
            close(guided, -math.sqrt(1 - abar[t]) * score, rtol=1e-12, atol=1e-13)
    zT = np.array([[0.0], [1.0]])  # the guided DDIM map is affine, x = A z_T + b, and its end variance is A^2
    steps = np.arange(999, -1, -1)
    for s in (1.0, 2.0, 3.0):
        x = cg.ddim_sample(lambda zz, tt, s=s: cg.guide(cg.gaussian_eps(zz, np.full(len(zz), abar[tt]), m1, v1),
                                                        cg.gaussian_eps(zz, np.full(len(zz), abar[tt]), m0, v0), s),
                           zT, abar, steps)  # fmt: skip
        assert float((x[1, 0] - x[0, 0]) ** 2) == pytest.approx(cg.guided_ode_variance(v1, v0, s), rel=0.03)
        tg = cg.tilted_gaussian(m1, v1, m0, v0, s)
        assert tg is not None
        vt = tg[1]
        if s > 1:
            assert cg.guided_ode_variance(v1, v0, s) < vt
    # Guiding a narrow model with a wider one always normalizes; the reverse stops normalizing at s = 1 / (1 - v1 / v0).
    assert cg.tilted_gaussian(m0, v0, m1, v1, 1.0 / (1.0 - v1 / v0) - 0.1) is not None
    assert cg.tilted_gaussian(m0, v0, m1, v1, 1.0 / (1.0 - v1 / v0) + 0.1) is None


def test_rescale():
    rng = np.random.default_rng(6)
    e_c, e_u = rng.standard_normal((2, 4, 3, 5, 5))
    out = cg.guide(e_c, e_u, 7.5)
    close(cg.rescale(e_c, out, 0.0), out)
    r = cg.rescale(e_c, out, 1.0)
    close(r.std(axis=(1, 2, 3)), e_c.std(axis=(1, 2, 3)))
    sp, so = e_c.std(axis=(1, 2, 3), ddof=1), out.std(axis=(1, 2, 3), ddof=1)  # torch's unbiased std, as in Lin et al.
    phi = 0.7
    close(cg.rescale(e_c, out, phi), phi * out * (sp / so)[:, None, None, None] + (1 - phi) * out)


def test_thresholds():
    rng = np.random.default_rng(7)
    x = 1.6 * rng.standard_normal((5, 40))
    st = cg.static_threshold(x)
    y = rng.uniform(-1, 1, (1000, 40))  # the static threshold is the closest point of the box [-1, 1]^n
    assert np.all(((x[:, None] - y[None]) ** 2).sum(-1) >= ((x - st) ** 2).sum(-1)[:, None] - 1e-12)
    for p in (50.0, 90.0, 99.5):
        d = cg.dynamic_threshold(x, p)
        assert np.all(np.abs(d) <= 1.0 + 1e-15)
        s = np.maximum(np.percentile(np.abs(x), p, axis=1), 1.0)
        kept = np.abs(x) <= s[:, None]
        close(d[kept], (x / s[:, None])[kept])
        assert np.all(kept.sum(1) >= math.floor(p / 100 * (x.shape[1] - 1)) + 1)  # the interpolated percentile
    small = 0.5 * x / np.abs(x).max()
    close(cg.dynamic_threshold(small, 99.5), small)
    close(cg.dynamic_threshold(x, 0.0), st)


def test_interval_and_count():
    abar = cg.alpha_bars(1000)
    steps = np.linspace(999, 0, 32).round().astype(int)
    sig = cg.edm_sigma(abar[steps])
    lo, hi = 0.28, 2.9
    inside = int(((sig > lo) & (sig <= hi)).sum())
    close(cg.interval_scale(sig, 2.0, lo, hi), np.where((sig > lo) & (sig <= hi), 2.0, 1.0))
    z = np.zeros((1, 2))
    for interval, want in ((None, 2 * len(steps)), ((lo, hi), len(steps) + inside)):
        eps, count = cg.guided_sampler(lambda zz, tt: zz, lambda zz, tt: 0 * zz, 2.0, abar, interval=interval)
        cg.ddim_sample(eps, z, abar, steps)
        assert count["nfe"] == want
    assert len(steps) + inside < 2 * len(steps)


def test_worked_example():
    """Two classes at +1 and -1 with variance 1/4, equal weights; z = 0.5 at abar = 0.5."""
    means, s2, w = np.array([[1.0], [-1.0]]), 0.25, np.array([0.5, 0.5])
    z, a = np.array([[0.5]]), np.array([0.5])
    e_c, e_u = cg.exact_eps_cond(z, a, means, s2, 0), cg.exact_eps(z, a, means, s2, w)
    v, r0, r1 = 0.625, 0.5 - math.sqrt(0.5), 0.5 + math.sqrt(0.5)
    pc = 1.0 / (1.0 + math.exp(-(r1**2 - r0**2) / (2 * v)))
    close(e_c, [[math.sqrt(0.5) * r0 / v]])
    close(e_u, [[math.sqrt(0.5) * (pc * r0 + (1 - pc) * r1) / v]])
    close(np.exp(cg.log_classifier(z, a, means, s2, w))[0, 0], pc)
    assert pc == pytest.approx(1 / (1 + math.exp(-4 * 0.5 * math.sqrt(0.5) / (2 * v))), rel=1e-12)
    close(cg.guide(e_c, e_u, 3.0), [[3 * e_c[0, 0] - 2 * e_u[0, 0]]])
    assert pc == pytest.approx(0.7561, abs=1e-4)
    assert float(e_c[0, 0]) == pytest.approx(-0.2343, abs=1e-4)
    assert float(e_u[0, 0]) == pytest.approx(0.1559, abs=1e-4)
    assert float(cg.guide(e_c, e_u, 3.0)[0, 0]) == pytest.approx(-1.0148, abs=1e-4)
    x0 = [float(cg.eps_to_x0(z, e, a)[0, 0]) for e in (e_c, e_u, cg.guide(e_c, e_u, 3.0))]
    close(x0, [0.9414, 0.5512, 1.7219], atol=1e-4)  # the guided x-prediction overshoots the class mean of 1


def test_tradeoff_on_overlapping_classes():
    """Two overlapping classes: the exact classifier's confidence in guided samples grows with s, the spread along the
    axis that separates the classes falls, and the spread across it, where the classes agree, does not change."""
    abar = cg.alpha_bars(1000)
    means, s2, w = np.array([[1.0, 0.0], [-1.0, 0.0]]), 0.5, np.array([0.5, 0.5])
    zT = np.random.default_rng(8).standard_normal((4000, 2))
    conf, spread = [], []
    for s in (0.0, 1.0, 3.0):
        eps, _ = cg.guided_sampler(lambda z, t: cg.exact_eps_cond(z, np.full(len(z), abar[t]), means, s2, 0),
                                   lambda z, t: cg.exact_eps(z, np.full(len(z), abar[t]), means, s2, w), s, abar)  # fmt: skip
        x = cg.ddim_sample(eps, zT, abar, np.linspace(999, 0, 50).round().astype(int))
        conf.append(float(np.exp(cg.log_classifier(x, np.ones(len(x)) - 1e-12, means, s2, w))[:, 0].mean()))
        spread.append((float(x[:, 0].std()), float(x[:, 1].std())))
    assert conf[0] == pytest.approx(0.5, abs=0.03)
    assert conf[0] < conf[1] < conf[2] and conf[2] > 0.9
    assert spread[1][0] == pytest.approx(math.sqrt(s2), rel=0.05)
    assert spread[2][0] < 0.8 * spread[1][0]
    for _, across in spread:
        assert across == pytest.approx(math.sqrt(s2), rel=0.05)


def test_gradient_by_finite_differences():
    rng = np.random.default_rng(9)
    cfg = cg.config(3, 6, 4, 2)
    p = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in cg.init(rng, cfg).items()}
    z, t, y, eps = (
        rng.standard_normal((5, 3)),
        np.array([1, 90, 400, 700, 999]),
        np.array([0, 1, 2, 2, 1]),
        rng.standard_normal((5, 3)),
    )
    g, h = cg.eps_loss_grad(p, cfg, z, t, y, eps), 1e-6
    for k in p:
        for idx in np.ndindex(p[k].shape):
            q = {kk: vv.copy() for kk, vv in p.items()}
            q[k][idx] += h
            up = cg.eps_loss(q, cfg, z, t, y, eps)
            q[k][idx] -= 2 * h
            fd = (up - cg.eps_loss(q, cfg, z, t, y, eps)) / (2 * h)
            assert g[k][idx] == pytest.approx(fd, rel=1e-5, abs=1e-8), (k, idx)
    assert cg.param_count(cfg) == sum(v.size for v in p.values()) == 3 * 6 + 4 * 6 + 3 * 6 + 6 + 6 * 3 + 3
    assert cg.flops_forward(cfg, 5) == 2 * 5 * (3 * 6 + 4 * 6 + 6 * 3)


def test_training_and_guided_sampling():
    rng = np.random.default_rng(10)
    D, C, s2 = 2, 3, 0.25
    m = np.array([[1.5, 0.0], [-0.75, 1.3], [-0.75, -1.3]])
    y = rng.integers(0, C, 6000)
    x0 = m[y] + math.sqrt(s2) * rng.standard_normal((6000, D))
    cfg = cg.config(D, 64, 16, C)
    abar = cg.alpha_bars(1000)
    p, trace = cg.train(cg.init(rng, cfg), cfg, x0, y, abar, 2500, rng)
    assert trace[-200:].mean() < 0.6 * trace[:50].mean()
    zT = rng.standard_normal((3000, D))
    steps = np.linspace(999, 0, 25).round().astype(int)
    w = np.full(C, 1.0 / C)
    unc = cg.ddim_sample(lambda z, t: cg.forward(p, cfg, z, t, C), zT, abar, steps)
    k = ((unc[:, None] - m) ** 2).sum(-1).argmin(1)
    assert np.all(np.bincount(k, minlength=C) / len(k) > 0.2)  # the null model covers all three classes
    conf = []
    for s in (1.0, 3.0):
        eps, _ = cg.guided_sampler(
            lambda z, t: cg.forward(p, cfg, z, t, 1), lambda z, t: cg.forward(p, cfg, z, t, C), s, abar
        )
        x = cg.ddim_sample(eps, zT, abar, steps)
        conf.append(float(np.exp(cg.log_classifier(x, np.ones(len(x)) - 1e-12, m, s2, w))[:, 1].mean()))
        if s == 1.0:
            assert float(np.abs(x.mean(0) - m[1]).max()) < 0.25
    assert conf[1] > conf[0] and conf[1] > 0.97


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: one conditional pass of the noise predictor."""
    k = c["config"]
    cfg = cg.config(k["dim"], k["hidden"], k["temb"], k["classes"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((k["B"], k["dim"]))
    return cg.forward, (cg.init(rng, cfg), cfg, x, 3, rng.integers(0, k["classes"], k["B"]))
