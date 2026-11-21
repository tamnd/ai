# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Tests of the JAX implementation: the shared fixture, the counts of shapes.json, the worked example, the principal
component autoencoder, the straight-through estimator and the two VQ terms under jax.grad, the k-means step and the
EMA codebook, the exact noise predictor against jax.grad of the noisy marginal, DDIM as a scan with the exact
predictor and with guidance, and a denoiser trained by a scan of Adam steps whose DDIM samples land on their class."""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
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
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (k["B"], k["positions"], k["channels"]))
    return ld.forward, (ld.init(key, cfg), cfg, z, 500, jnp.zeros(k["B"], int))


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol, err_msg=msg)


def test_shared_fixture():
    P = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = ld.config(*(int(v) for v in inp["cfg"]))
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    close(ld.forward(P, cfg, z, t, y), want["eps_hat"])
    assert float(ld.eps_loss(P, cfg, z, t, y, eps)) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, g in ld.eps_loss_grad(P, cfg, z, t, y, eps).items():
        close(g, want[f"d{k}"], rtol=1e-10, msg=k)
    ae = ld.fit_autoencoder(inp["x"], 4, 2)
    close(jnp.abs(ld.encode(ae, inp["x"], 4)), np.abs(want["latent"]), rtol=1e-10, atol=1e-12)
    idx, _ = ld.quantize(inp["z_e"], inp["codebook"])
    np.testing.assert_array_equal(np.asarray(idx), want["idx"])
    g_e, g_cb = ld.vq_grads(inp["z_e"], inp["codebook"], idx, inp["g_zq"], 0.25)
    close(g_e, want["g_e"])
    close(g_cb, want["g_cb"])
    close(ld.ema_update(ld.ema_init(inp["codebook"]), inp["z_e"], idx, 0.99)["e"], want["ema_e"])
    abar = ld.alpha_bars(1000)
    out = ld.ddim_sample(lambda zz, tt: ld.exact_eps(zz, jnp.full(len(zz), abar[tt]), inp["means"], 0.1,
                                                     jnp.array([0.5, 0.5])), inp["zT"], abar, [999, 600, 300, 0])  # fmt: skip
    close(out, want["ddim"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert ld.param_count(cfg) == mod["params"]
    assert ld.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = ld.init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    out = ld.forward(p, cfg, jnp.zeros((k["B"], k["positions"], k["channels"])), 3, jnp.zeros(k["B"], int))
    assert out.shape == (k["B"], k["positions"] * k["channels"])


def test_worked_example():
    x = jnp.array([[1.0, 1.0, 3.0, 3.0], [-1.0, -1.0, 1.0, 1.0]])
    ae = ld.fit_autoencoder(x, 2, 1)
    z = ld.encode(ae, x, 2)
    r = 2.0 * math.sqrt(2.0)
    close(jnp.abs(z[..., 0]), [[0.0, r], [r, 0.0]], atol=1e-12)
    close(ld.decode(ae, z), x, atol=1e-12)
    assert ld.latent_scale(z) == pytest.approx(2.0, rel=1e-12)


def test_autoencoder_is_pca():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, (500, 24)) @ jax.random.normal(k2, (24, 24))
    P = ld.patches(x, 6).reshape(-1, 6)
    lam = jnp.linalg.eigvalsh(jnp.cov(P.T, bias=True))[::-1]
    for c in (1, 3, 6):
        ae = ld.fit_autoencoder(x, 6, c)
        err = ((ld.decode(ae, ld.encode(ae, x, 6)) - x) ** 2).sum() / len(P)
        assert float(err) == pytest.approx(float(lam[c:].sum()), rel=1e-9, abs=1e-12)


def test_straight_through_and_vq_terms():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
    cb, z, gz = jax.random.normal(k1, (6, 3)), jax.random.normal(k2, (4, 5, 3)), jax.random.normal(k3, (4, 5, 3))
    idx, zq = ld.quantize(z, cb)
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(((z[..., None, :] - cb) ** 2).sum(-1).argmin(-1)))
    close(ld.straight_through(z, zq), zq)
    close(jax.grad(lambda ze: (gz * ld.straight_through(ze, zq)).sum())(z), gz)
    # The codebook term moves only the codebook, the commitment term only the encoder.
    close(jax.grad(lambda ze: ld.vq_terms(ze, cb, idx, 0.25)[0])(z), jnp.zeros_like(z))
    close(jax.grad(lambda c: ld.vq_terms(z, c, idx, 0.25)[1])(cb), jnp.zeros_like(cb))
    n = idx.size
    g_e, _ = ld.vq_grads(z, cb, idx, gz, 0.25)
    close(g_e, gz + 0.5 * (z - cb[idx]) / n)


def test_kmeans_and_ema():
    k1, k2 = jax.random.split(jax.random.PRNGKey(4))
    z = jax.random.normal(k1, (400, 2)) + 3.0 * jax.random.randint(k2, (400, 1), 0, 3)
    cb = z[:5]
    err = []
    for _ in range(6):
        _, zq = ld.quantize(z, cb)
        err.append(float(((z - zq) ** 2).sum()))
        cb = ld.kmeans_step(z, cb)
    assert all(b <= a + 1e-9 for a, b in zip(err, err[1:], strict=False))
    idx, _ = ld.quantize(z, cb)
    close(ld.ema_update(ld.ema_init(cb), z, idx, 0.99)["e"], ld.kmeans_step(z, cb), atol=1e-12)


def test_exact_eps_is_the_scaled_score():
    k1, k2 = jax.random.split(jax.random.PRNGKey(5))
    means, s2, w = jax.random.normal(k1, (3, 4)), 0.2, jnp.array([0.2, 0.5, 0.3])
    abar = ld.alpha_bars(1000)
    for t in (5, 200, 700):
        a = abar[t]

        def logp(z, a=a):
            r = z[None, :] - jnp.sqrt(a) * means
            return jax.nn.logsumexp(jnp.log(w) - 0.5 * (r**2).sum(-1) / (a * s2 + 1.0 - a))

        zt = jax.random.normal(jax.random.fold_in(k2, t), (6, 4))
        close(ld.exact_eps(zt, jnp.full(6, a), means, s2, w), -jnp.sqrt(1.0 - a) * jax.vmap(jax.grad(logp))(zt),
              rtol=1e-10, atol=1e-12)  # fmt: skip


def test_ddim_and_guidance():
    key = jax.random.PRNGKey(6)
    means, s2, w = jnp.array([[2.0, 0.0], [-1.0, 1.5], [0.0, -2.0]]), 0.05, jnp.array([0.5, 0.3, 0.2])
    abar = ld.alpha_bars(1000)
    steps = np.linspace(999, 0, 100).round().astype(int)
    out = ld.ddim_sample(lambda z, t: ld.exact_eps(z, jnp.full(len(z), abar[t]), means, s2, w),
                         jax.random.normal(key, (20000, 2)), abar, steps)  # fmt: skip
    k = ((out[:, None] - means) ** 2).sum(-1).argmin(1)
    close(jnp.bincount(k, length=3) / len(k), w, rtol=0, atol=0.015)
    means, s2 = jnp.array([[1.0, 0.0], [-1.0, 0.0]]), 0.5
    zT = jax.random.normal(jax.random.fold_in(key, 1), (4000, 2))
    conf = []
    for s in (0.0, 1.0, 3.0):

        def eps(z, t, s=s):
            a = jnp.full(len(z), abar[t])
            return ld.guided(
                ld.exact_eps(z, a, means[:1], s2, jnp.ones(1)), ld.exact_eps(z, a, means, s2, jnp.ones(2) / 2), s
            )

        x = ld.ddim_sample(eps, zT, abar, np.linspace(999, 0, 50).round().astype(int))
        conf.append(float(jax.nn.sigmoid(2 * x[:, 0] / s2).mean()))
    assert conf[0] == pytest.approx(0.5, abs=0.03)
    assert conf[0] < conf[1] < conf[2] and conf[2] > 0.9


def test_training_and_sampling():
    key = jax.random.PRNGKey(10)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    D, C, s2 = 8, 3, 0.09
    m = 1.5 * jax.random.normal(k1, (C, D))
    y = jax.random.randint(k2, (4000,), 0, C)
    z0 = (m[y] + math.sqrt(s2) * jax.random.normal(k3, (4000, D))).reshape(-1, 4, 2)
    cfg = ld.config(4, 2, 64, 16, 2, 8, C)
    abar = ld.alpha_bars(1000)
    p, trace = jax.jit(lambda p, k: ld.train(p, cfg, z0, y, abar, 1500, k))(ld.init(k4, cfg), k5)
    assert float(trace[-200:].mean()) < 0.5 * float(trace[:50].mean())
    c = 1
    out = ld.ddim_sample(lambda z, t: ld.forward(p, cfg, z, t, jnp.full(len(z), c)), jax.random.normal(k6, (2000, D)),
                         abar, np.linspace(999, 0, 25).round().astype(int))  # fmt: skip
    assert float(jnp.abs(out.mean(0) - m[c]).max()) < 0.35
    assert float(out.std(0).mean()) == pytest.approx(math.sqrt(s2), rel=0.3)
