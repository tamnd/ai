# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the worked example, the principal
component autoencoder and its reconstruction error, the KL penalty, the latent scale and the signal-to-noise ratio it
sets, vector quantization with its straight-through gradients, the k-means step and the EMA codebook, the exact noise
predictor against the score of the noisy marginal, DDIM with the exact predictor, classifier-free guidance, the
single-token cross-attention, the hand-written gradient against finite differences, a trained denoiser, and the whole
two-stage pipeline from signals to latents and back."""

import json
import math
import pathlib

import numpy as np
import pytest

import latent_diffusion as ld

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return ld.config(k["positions"], k["channels"], k["hidden"], k["temb"], k["ctx_tokens"], k["attn"], k["classes"])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: the predicted noise for B latents."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    z = rng.standard_normal((k["B"], k["positions"], k["channels"]))
    return ld.forward, (ld.init(rng, cfg), cfg, z, 500, np.zeros(k["B"], int))


def signals(rng, n, N=4, f=4, c=2, C=3, s=0.3, noise=0.01):
    """Signals of length N f whose patches lie near a c-dimensional subspace, with class-dependent latents."""
    A = np.linalg.qr(rng.standard_normal((f, c)))[0].T  # (c, f), orthonormal rows
    means = 1.5 * rng.standard_normal((C, N, c))
    y = rng.integers(0, C, n)
    z = means[y] + s * rng.standard_normal((n, N, c))
    x = (z @ A + 0.5 + noise * rng.standard_normal((n, N, f))).reshape(n, N * f)
    return x, y


def latent_gmm(z, y, C):
    """Class means and one pooled isotropic variance of flattened latents."""
    zf = z.reshape(len(z), -1)
    means = np.stack([zf[y == k].mean(0) for k in range(C)])
    return means, float(((zf - means[y]) ** 2).mean())


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = ld.config(*(int(v) for v in inp["cfg"]))
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    np.testing.assert_allclose(ld.forward(P, cfg, z, t, y), want["eps_hat"], rtol=1e-12, atol=1e-13)
    assert ld.eps_loss(P, cfg, z, t, y, eps) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, g in ld.eps_loss_grad(P, cfg, z, t, y, eps).items():
        np.testing.assert_allclose(g, want[f"d{k}"], err_msg=k, rtol=1e-10, atol=1e-13)
    ae = ld.fit_autoencoder(inp["x"], 4, 2)
    np.testing.assert_allclose(np.abs(ld.encode(ae, inp["x"], 4)), np.abs(want["latent"]), rtol=1e-10, atol=1e-12)
    idx, zq = ld.quantize(inp["z_e"], inp["codebook"])
    np.testing.assert_array_equal(idx, want["idx"])
    g_e, g_cb = ld.vq_grads(inp["z_e"], inp["codebook"], idx, inp["g_zq"], 0.25)
    np.testing.assert_allclose(g_e, want["g_e"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(g_cb, want["g_cb"], rtol=1e-12, atol=1e-13)
    st = ld.ema_update(ld.ema_init(inp["codebook"]), inp["z_e"], idx, 0.99)
    np.testing.assert_allclose(st["e"], want["ema_e"], rtol=1e-12, atol=1e-13)
    abar = ld.alpha_bars(1000)
    out = ld.ddim_sample(lambda zz, tt: ld.exact_eps(zz, np.full(len(zz), abar[tt]), inp["means"], 0.1,
                                                     np.array([0.5, 0.5])), inp["zT"], abar, [999, 600, 300, 0])  # fmt: skip
    np.testing.assert_allclose(out, want["ddim"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert ld.param_count(cfg) == mod["params"]
    assert ld.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = ld.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    z = np.zeros((k["B"], k["positions"], k["channels"]))
    assert ld.forward(p, cfg, z, 3, np.zeros(k["B"], int)).shape == (k["B"], k["positions"] * k["channels"])


def test_worked_example():
    # Two signals of length 4 cut into patches of length f = 2: (1, 1), (3, 3), (-1, -1), (1, 1). They all lie on the
    # line through the mean (1, 1) along (1, 1) / sqrt 2, so one channel keeps everything and halves the length.
    x = np.array([[1.0, 1.0, 3.0, 3.0], [-1.0, -1.0, 1.0, 1.0]])
    ae = ld.fit_autoencoder(x, 2, 1)
    z = ld.encode(ae, x, 2)
    r = 2.0 * math.sqrt(2.0)
    np.testing.assert_allclose(np.abs(z[..., 0]), [[0.0, r], [r, 0.0]], atol=1e-12)
    np.testing.assert_allclose(ld.decode(ae, z), x, atol=1e-12)
    # The scale is the standard deviation of the four numbers 0, +-r, -+r, 0: sqrt(2 r^2 / 4) = 2.
    assert ld.latent_scale(z) == pytest.approx(2.0, rel=1e-12)


def test_autoencoder_is_pca():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 24)) @ rng.standard_normal((24, 24))
    P = ld.patches(x, 6).reshape(-1, 6)
    lam = np.sort(np.linalg.eigvalsh(np.cov(P, rowvar=False, bias=True)))[::-1]
    for c in (1, 3, 6):
        ae = ld.fit_autoencoder(x, 6, c)
        np.testing.assert_allclose(ae["We"].T @ ae["We"], np.eye(c), atol=1e-12)
        err = ((ld.decode(ae, ld.encode(ae, x, 6)) - x) ** 2).sum() / len(P)
        assert err == pytest.approx(lam[c:].sum(), rel=1e-9, abs=1e-12)  # the discarded eigenvalues
    assert ld.encode(ld.fit_autoencoder(x, 6, 2), x, 6).shape == (500, 4, 2)


def test_kl_penalty():
    assert ld.kl_to_standard_normal(np.zeros((3, 2, 2)), np.zeros((3, 2, 2))) == pytest.approx(np.zeros(3))
    rng = np.random.default_rng(1)
    mu, lv = rng.standard_normal((1, 1)), rng.standard_normal((1, 1))
    s = mu + np.exp(0.5 * lv) * rng.standard_normal(400000)
    mc = (-0.5 * (s - mu) ** 2 / np.exp(lv) - 0.5 * lv + 0.5 * s**2).mean()  # E log q - log p, the 2 pi cancel
    assert ld.kl_to_standard_normal(mu, lv)[0] == pytest.approx(mc, abs=5e-3)


def test_latent_scale_sets_the_snr():
    rng = np.random.default_rng(2)
    z = 7.0 * rng.standard_normal((300, 8, 4)) + 0.2
    s = ld.latent_scale(z)
    assert ld.latent_scale(z / s) == pytest.approx(1.0, rel=1e-12)
    abar = ld.alpha_bars(1000)
    # The SNR of eq-snr scales with the variance of z0: an unscaled latent of variance 49 sees a 49 times cleaner
    # signal at every step, so the whole schedule is shifted towards less noise.
    np.testing.assert_allclose(ld.snr(abar, s**2), s**2 * ld.snr(abar), rtol=1e-12)
    t49 = np.searchsorted(-ld.snr(abar, s**2), -1.0)  # the first step with SNR below 1
    t1 = np.searchsorted(-ld.snr(abar), -1.0)
    assert t49 > t1 + 100


def test_quantize_and_straight_through():
    rng = np.random.default_rng(3)
    cb, z = rng.standard_normal((6, 3)), rng.standard_normal((4, 5, 3))
    idx, zq = ld.quantize(z, cb)
    brute = ((z[..., None, :] - cb) ** 2).sum(-1).argmin(-1)
    np.testing.assert_array_equal(idx, brute)
    np.testing.assert_array_equal(zq, cb[idx])
    beta, g_zq = 0.25, rng.standard_normal(z.shape)
    g_e, g_cb = ld.vq_grads(z, cb, idx, g_zq, beta)
    # With the assignment held fixed, the codebook gradient is that of the codebook term, the encoder gradient is the
    # decoder gradient copied straight through plus beta times that of the commitment term.
    h = 1e-6
    for i in np.ndindex(cb.shape):
        d = np.zeros_like(cb)
        d[i] = h
        fd = (ld.vq_terms(z, cb + d, idx, beta)[0] - ld.vq_terms(z, cb - d, idx, beta)[0]) / (2 * h)
        assert g_cb[i] == pytest.approx(fd, abs=1e-8)
    for i in [(0, 0, 0), (2, 3, 1), (3, 4, 2)]:
        d = np.zeros_like(z)
        d[i] = h
        fd = (ld.vq_terms(z + d, cb, idx, beta)[1] - ld.vq_terms(z - d, cb, idx, beta)[1]) / (2 * h)
        assert g_e[i] - g_zq[i] == pytest.approx(fd, abs=1e-8)
    assert ld.vq_terms(z, cb, idx, beta)[1] == pytest.approx(beta * ld.vq_terms(z, cb, idx, beta)[0])


def test_kmeans_and_ema():
    rng = np.random.default_rng(4)
    z = rng.standard_normal((400, 2)) + 3.0 * rng.integers(0, 3, (400, 1))
    cb = z[:5].copy()
    err = []
    for _ in range(6):
        idx, zq = ld.quantize(z, cb)
        err.append(((z - zq) ** 2).sum())
        cb = ld.kmeans_step(z, cb)
    assert all(b <= a + 1e-9 for a, b in zip(err, err[1:], strict=False))
    # After the step, each used code is where the codebook term's gradient vanishes for its assignment.
    idx, _ = ld.quantize(z, cb)
    cb2 = ld.kmeans_step(z, cb)
    _, g_cb = ld.vq_grads(z, cb2, idx, np.zeros_like(z), 0.25)
    assert np.abs(g_cb).max() < 1e-12
    # One EMA step from zero counts and sums puts every used code exactly at its mean: (1 - gamma) cancels in m / N.
    st = ld.ema_update(ld.ema_init(cb), z, idx, 0.99)
    np.testing.assert_allclose(st["e"], cb2, atol=1e-12)
    # With a fixed assignment the EMA stays at the mean; with fresh batches it averages them.
    for _ in range(50):
        st = ld.ema_update(st, z, idx, 0.99)
    np.testing.assert_allclose(st["e"], cb2, atol=1e-10)
    unused = np.setdiff1d(np.arange(len(cb)), idx)
    np.testing.assert_array_equal(st["e"][unused], cb[unused])


def mixture_log_marginal(zt, a, means, s2, w):
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - math.sqrt(a) * means[None]
    lp = np.log(w)[None] - 0.5 * (r**2).sum(-1) / v - 0.5 * zt.shape[1] * math.log(2 * math.pi * v)
    m = lp.max(1, keepdims=True)
    return (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]


def test_exact_eps_is_the_scaled_score():
    rng = np.random.default_rng(5)
    means, s2, w = rng.standard_normal((3, 4)), 0.2, np.array([0.2, 0.5, 0.3])
    abar = ld.alpha_bars(1000)
    for t in (5, 200, 700):
        a = abar[t]
        zt = rng.standard_normal((6, 4))
        e = ld.exact_eps(zt, np.full(6, a), means, s2, w)
        h, score = 1e-5, np.zeros_like(zt)
        for j in range(4):
            d = np.zeros_like(zt)
            d[:, j] = h
            score[:, j] = (mixture_log_marginal(zt + d, a, means, s2, w)
                           - mixture_log_marginal(zt - d, a, means, s2, w)) / (2 * h)  # fmt: skip
        np.testing.assert_allclose(e, -math.sqrt(1.0 - a) * score, rtol=1e-6, atol=1e-7)


def test_ddim_with_the_exact_predictor():
    rng = np.random.default_rng(6)
    means, s2, w = np.array([[2.0, 0.0], [-1.0, 1.5], [0.0, -2.0]]), 0.05, np.array([0.5, 0.3, 0.2])
    abar = ld.alpha_bars(1000)
    steps = np.linspace(999, 0, 100).round().astype(int)
    out = ld.ddim_sample(lambda z, t: ld.exact_eps(z, np.full(len(z), abar[t]), means, s2, w),
                         rng.standard_normal((20000, 2)), abar, steps)  # fmt: skip
    k = ((out[:, None] - means) ** 2).sum(-1).argmin(1)
    np.testing.assert_allclose(np.bincount(k) / len(k), w, atol=0.015)
    for j in range(3):
        np.testing.assert_allclose(out[k == j].mean(0), means[j], atol=0.02)
        assert out[k == j].var(0).mean() == pytest.approx(s2, rel=0.2)


def test_classifier_free_guidance():
    assert ld.guided(np.array([1.0]), np.array([3.0]), 1.0)[0] == 1.0
    assert ld.guided(np.array([1.0]), np.array([3.0]), 0.0)[0] == 3.0
    rng = np.random.default_rng(7)
    means, s2 = np.array([[1.0, 0.0], [-1.0, 0.0]]), 0.5  # two classes that overlap
    abar = ld.alpha_bars(1000)
    steps = np.linspace(999, 0, 50).round().astype(int)
    zT = rng.standard_normal((4000, 2))

    def p_class0(z):
        lp = -((z[:, None] - means) ** 2).sum(-1) / (2 * s2)
        return 1.0 / (1.0 + np.exp(lp[:, 1] - lp[:, 0]))

    conf = []
    for s in (0.0, 1.0, 3.0):

        def eps(z, t, s=s):
            a = np.full(len(z), abar[t])
            return ld.guided(
                ld.exact_eps(z, a, means[:1], s2, np.ones(1)), ld.exact_eps(z, a, means, s2, np.ones(2) / 2), s
            )

        conf.append(p_class0(ld.ddim_sample(eps, zT, abar, steps)).mean())
    assert conf[0] == pytest.approx(0.5, abs=0.03)  # s = 0 samples the unconditional mixture
    assert conf[0] < conf[1] < conf[2]  # a larger guidance scale makes class 0 more certain
    assert conf[2] > 0.9


def test_single_token_cross_attention_is_a_class_bias():
    rng = np.random.default_rng(8)
    cfg = ld.config(3, 2, 5, 4, 1, 3, 2)
    p = ld.init(rng, cfg)
    p0 = dict(p, Wv=np.zeros_like(p["Wv"]))
    z, t = rng.standard_normal((4, 3, 2)), rng.integers(0, 1000, 4)
    for y in range(3):
        yy = np.full(4, y)
        # With one context token the softmax weight is 1, so the attention adds ctx[y] Wv to h before the second tanh,
        # the same vector for every z and t.
        _, cache = ld._forward(p, cfg, z, t, yy)
        _, cache0 = ld._forward(p0, cfg, z, t, yy)
        added = np.arctanh(cache[-1]) - np.arctanh(cache0[-1])
        np.testing.assert_allclose(cache[-2], np.ones((4, 1)))
        np.testing.assert_allclose(added, np.broadcast_to(p["ctx"][y, 0] @ p["Wv"], added.shape), atol=1e-10)


def test_gradient():
    rng = np.random.default_rng(9)
    cfg = ld.config(2, 2, 6, 4, 3, 5, 2)
    p = ld.init(rng, cfg)
    z, t, y = rng.standard_normal((5, 2, 2)), rng.integers(0, 1000, 5), rng.integers(0, 3, 5)
    e = rng.standard_normal((5, 4))
    g = ld.eps_loss_grad(p, cfg, z, t, y, e)
    assert set(g) == set(p)
    h = 1e-6
    for k in p:
        for i in np.ndindex(p[k].shape):
            old = p[k][i]
            p[k][i] = old + h
            a = ld.eps_loss(p, cfg, z, t, y, e)
            p[k][i] = old - h
            b = ld.eps_loss(p, cfg, z, t, y, e)
            p[k][i] = old
            assert g[k][i] == pytest.approx((a - b) / (2 * h), abs=1e-7), (k, i)


def test_training_and_sampling():
    rng = np.random.default_rng(10)
    D, C, s2 = 8, 3, 0.09
    m = 1.5 * rng.standard_normal((C, D))
    y = rng.integers(0, C, 4000)
    z0 = (m[y] + math.sqrt(s2) * rng.standard_normal((4000, D))).reshape(-1, 4, 2)
    cfg = ld.config(4, 2, 64, 16, 2, 8, C)
    abar = ld.alpha_bars(1000)
    p, trace = ld.train(ld.init(rng, cfg), cfg, z0, y, abar, 1500, rng)
    assert trace[-200:].mean() < 0.5 * trace[:50].mean()
    c = 1
    for t, tol in ((300, 0.2), (600, 0.1), (900, 0.1)):  # squared error in 8 dimensions, against E|eps|^2 = 8
        e = rng.standard_normal((2000, D))
        zt = ld.q_sample(m[c] + math.sqrt(s2) * rng.standard_normal((2000, D)), np.full(2000, abar[t]), e)
        ex = ld.exact_eps(zt, np.full(2000, abar[t]), m[c : c + 1], s2, np.ones(1))
        assert ((ld.forward(p, cfg, zt, t, np.full(2000, c)) - ex) ** 2).sum(1).mean() < tol
    steps = np.linspace(999, 0, 25).round().astype(int)
    out = ld.ddim_sample(lambda z, t: ld.forward(p, cfg, z, t, np.full(len(z), c)), rng.standard_normal((2000, D)),
                         abar, steps)  # fmt: skip
    assert np.abs(out.mean(0) - m[c]).max() < 0.35
    assert out.std(0).mean() == pytest.approx(math.sqrt(s2), rel=0.3)


def test_two_stage_pipeline():
    rng = np.random.default_rng(11)
    x, y = signals(rng, 3000)
    f, c, C = 4, 2, 3
    ae = ld.fit_autoencoder(x, f, c)
    z = ld.encode(ae, x, f)
    assert z.shape == (3000, 4, 2)  # 16 numbers per signal become 8: f / c = 2
    assert ((ld.decode(ae, z) - x) ** 2).mean() < 2e-4
    sigma = ld.latent_scale(z)
    means, s2 = latent_gmm(z / sigma, y, C)
    abar = ld.alpha_bars(1000)
    steps = np.linspace(999, 0, 50).round().astype(int)
    for k in range(C):
        u = ld.ddim_sample(lambda zz, t, k=k: ld.exact_eps(zz, np.full(len(zz), abar[t]), means[k : k + 1], s2, np.ones(1)),
                           rng.standard_normal((2000, 8)), abar, steps)  # fmt: skip
        xs = ld.decode(ae, sigma * u.reshape(-1, 4, 2))  # undo the scale, then one pass through the decoder
        np.testing.assert_allclose(xs.mean(0), x[y == k].mean(0), atol=0.05)
        np.testing.assert_allclose(xs.std(0), x[y == k].std(0), rtol=0.2, atol=0.02)
