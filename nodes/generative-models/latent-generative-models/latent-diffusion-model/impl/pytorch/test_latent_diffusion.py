# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Tests of the PyTorch implementation: the shared fixture, the counts of shapes.json, the worked example, the
principal component autoencoder, the straight-through estimator against the NumPy reference's hand-written gradients,
the k-means step and the EMA codebook, the exact noise predictor against autograd's score of the noisy marginal, DDIM
with it, classifier-free guidance, the denoiser's gradient by gradcheck, and a denoiser trained with torch.optim.Adam
whose DDIM samples land on the class it was asked for."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

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
    g = torch.Generator().manual_seed(0)
    z = torch.randn(k["B"], k["positions"], k["channels"], generator=g)
    return ld.Denoiser(cfg, g), (z, torch.full((k["B"],), 500), torch.zeros(k["B"], dtype=torch.long))


def T(a):
    return torch.as_tensor(np.asarray(a))


def load_model(cfg, P):
    model = ld.Denoiser(cfg)
    with torch.no_grad():
        for k, v in P.items():
            getattr(model, k).copy_(T(v))
    return model


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: T(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: T(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    cfg = ld.config(*(int(v) for v in inp["cfg"]))
    model = load_model(cfg, P)
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    torch.testing.assert_close(model(z, t, y), want["eps_hat"], rtol=1e-12, atol=1e-13)
    loss = ld.eps_loss(model, z, t, y, eps)
    assert float(loss.detach()) == pytest.approx(float(want["loss"]), rel=1e-12)
    loss.backward()
    for k, prm in model.named_parameters():
        torch.testing.assert_close(prm.grad, want[f"d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    ae = ld.fit_autoencoder(inp["x"], 4, 2)
    torch.testing.assert_close(ld.encode(ae, inp["x"], 4).abs(), want["latent"].abs(), rtol=1e-10, atol=1e-12)
    idx, _ = ld.quantize(inp["z_e"], inp["codebook"])
    torch.testing.assert_close(idx, want["idx"])
    g_e, g_cb = ld.vq_grads(inp["z_e"], inp["codebook"], idx, inp["g_zq"], 0.25)
    torch.testing.assert_close(g_e, want["g_e"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(g_cb, want["g_cb"], rtol=1e-12, atol=1e-13)
    st = ld.ema_update(ld.ema_init(inp["codebook"]), inp["z_e"], idx, 0.99)
    torch.testing.assert_close(st["e"], want["ema_e"], rtol=1e-12, atol=1e-13)
    abar = ld.alpha_bars(1000)
    out = ld.ddim_sample(lambda zz, tt: ld.exact_eps(zz, abar[tt].expand(len(zz)), inp["means"], 0.1,
                                                     torch.tensor([0.5, 0.5])), inp["zT"], abar, [999, 600, 300, 0])  # fmt: skip
    torch.testing.assert_close(out, want["ddim"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert ld.param_count(cfg) == mod["params"]
    assert ld.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = ld.Denoiser(cfg, torch.Generator().manual_seed(0))
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    z = torch.zeros(k["B"], k["positions"], k["channels"])
    with torch.no_grad():
        out = model(z, 3, torch.zeros(k["B"], dtype=torch.long))
    assert out.shape == (k["B"], k["positions"] * k["channels"])


def test_worked_example():
    x = torch.tensor([[1.0, 1.0, 3.0, 3.0], [-1.0, -1.0, 1.0, 1.0]])
    ae = ld.fit_autoencoder(x, 2, 1)
    z = ld.encode(ae, x, 2)
    r = 2.0 * math.sqrt(2.0)
    torch.testing.assert_close(z[..., 0].abs(), torch.tensor([[0.0, r], [r, 0.0]]), atol=1e-12, rtol=0)
    torch.testing.assert_close(ld.decode(ae, z), x, atol=1e-12, rtol=0)
    assert ld.latent_scale(z) == pytest.approx(2.0, rel=1e-12)


def test_autoencoder_is_pca():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 24, generator=g) @ torch.randn(24, 24, generator=g)
    P = ld.patches(x, 6).reshape(-1, 6)
    lam = torch.linalg.eigvalsh(torch.cov(P.T, correction=0)).flip(0)
    for c in (1, 3, 6):
        ae = ld.fit_autoencoder(x, 6, c)
        err = ((ld.decode(ae, ld.encode(ae, x, 6)) - x) ** 2).sum() / len(P)
        assert float(err) == pytest.approx(float(lam[c:].sum()), rel=1e-9, abs=1e-12)


def test_straight_through():
    g = torch.Generator().manual_seed(3)
    cb, z = torch.randn(6, 3, generator=g), torch.randn(4, 5, 3, generator=g)
    idx, zq = ld.quantize(z, cb)
    torch.testing.assert_close(idx, ((z[..., None, :] - cb) ** 2).sum(-1).argmin(-1))
    # The forward value is z_q; the gradient passes to z_e unchanged.
    ze = z.clone().requires_grad_()
    st = ld.straight_through(ze, zq)
    torch.testing.assert_close(st, zq)
    gz = torch.randn(z.shape, generator=g)
    (st * gz).sum().backward()
    torch.testing.assert_close(ze.grad, gz)
    # The codebook moves only by the codebook term, the encoder by the decoder gradient plus the commitment term.
    beta = 0.25
    g_e, g_cb = ld.vq_grads(z, cb, idx, gz, beta)
    n = idx.numel()
    torch.testing.assert_close(g_e, gz + 2 * beta * (z - cb[idx]) / n)
    want = torch.zeros_like(cb).index_add_(0, idx.reshape(-1), (-2.0 * (z - cb[idx]) / n).reshape(-1, 3))
    torch.testing.assert_close(g_cb, want)


def test_kmeans_and_ema():
    g = torch.Generator().manual_seed(4)
    z = torch.randn(400, 2, generator=g) + 3.0 * torch.randint(0, 3, (400, 1), generator=g)
    cb = z[:5].clone()
    err = []
    for _ in range(6):
        _, zq = ld.quantize(z, cb)
        err.append(float(((z - zq) ** 2).sum()))
        cb = ld.kmeans_step(z, cb)
    assert all(b <= a + 1e-9 for a, b in zip(err, err[1:], strict=False))
    idx, _ = ld.quantize(z, cb)
    st = ld.ema_update(ld.ema_init(cb), z, idx, 0.99)
    torch.testing.assert_close(st["e"], ld.kmeans_step(z, cb), atol=1e-12, rtol=0)


def test_exact_eps_is_the_scaled_score():
    g = torch.Generator().manual_seed(5)
    means, s2, w = torch.randn(3, 4, generator=g), 0.2, torch.tensor([0.2, 0.5, 0.3])
    abar = ld.alpha_bars(1000)
    for t in (5, 200, 700):
        a = abar[t]
        zt = torch.randn(6, 4, generator=g).requires_grad_()
        v = a * s2 + 1.0 - a
        r = zt[:, None, :] - a.sqrt() * means[None]
        logp = torch.logsumexp(w.log()[None] - 0.5 * (r**2).sum(-1) / v, 1).sum()
        (score,) = torch.autograd.grad(logp, zt)
        torch.testing.assert_close(ld.exact_eps(zt.detach(), a.expand(6), means, s2, w), -(1.0 - a).sqrt() * score)


def test_ddim_and_guidance():
    g = torch.Generator().manual_seed(6)
    means, s2, w = torch.tensor([[2.0, 0.0], [-1.0, 1.5], [0.0, -2.0]]), 0.05, torch.tensor([0.5, 0.3, 0.2])
    abar = ld.alpha_bars(1000)
    steps = torch.linspace(999, 0, 100).round().long()
    out = ld.ddim_sample(lambda z, t: ld.exact_eps(z, abar[t].expand(len(z)), means, s2, w),
                         torch.randn(20000, 2, generator=g), abar, steps)  # fmt: skip
    k = ((out[:, None] - means) ** 2).sum(-1).argmin(1)
    torch.testing.assert_close(torch.bincount(k) / len(k), w, atol=0.015, rtol=0)
    # Guidance: two overlapping classes; s = 0 is the mixture, a larger s makes class 0 more certain.
    means, s2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), 0.5
    zT = torch.randn(4000, 2, generator=g)
    conf = []
    for s in (0.0, 1.0, 3.0):

        def eps(z, t, s=s):
            a = abar[t].expand(len(z))
            return ld.guided(
                ld.exact_eps(z, a, means[:1], s2, torch.ones(1)), ld.exact_eps(z, a, means, s2, torch.ones(2) / 2), s
            )

        x = ld.ddim_sample(eps, zT, abar, torch.linspace(999, 0, 50).round().long())
        conf.append(float(torch.sigmoid(2 * x[:, 0] / s2).mean()))  # p(class 0 | x) for these two Gaussians
    assert conf[0] == pytest.approx(0.5, abs=0.03)
    assert conf[0] < conf[1] < conf[2] and conf[2] > 0.9


def test_gradcheck():
    cfg = ld.config(2, 2, 4, 4, 3, 3, 2)
    model = ld.Denoiser(cfg, torch.Generator().manual_seed(9))
    g = torch.Generator().manual_seed(10)
    z, t, y = torch.randn(3, 2, 2, generator=g), torch.tensor([4, 400, 900]), torch.tensor([0, 2, 1])
    eps = torch.randn(3, 4, generator=g)
    names = [k for k, _ in model.named_parameters()]

    def f(*ps):
        return (
            torch.func.functional_call(model, dict(zip(names, ps, strict=True)), (z, t, y))
            .sub(eps)
            .pow(2)
            .sum(1)
            .mean()
        )

    assert torch.autograd.gradcheck(f, tuple(p.detach().clone().requires_grad_() for p in model.parameters()))


def test_training_and_sampling():
    g = torch.Generator().manual_seed(10)
    D, C, s2 = 8, 3, 0.09
    m = 1.5 * torch.randn(C, D, generator=g)
    y = torch.randint(0, C, (4000,), generator=g)
    z0 = (m[y] + math.sqrt(s2) * torch.randn(4000, D, generator=g)).reshape(-1, 4, 2)
    model = ld.Denoiser(ld.config(4, 2, 64, 16, 2, 8, C), g)
    abar = ld.alpha_bars(1000)
    trace = ld.train(model, z0, y, abar, 1500, g)
    assert np.mean(trace[-200:]) < 0.5 * np.mean(trace[:50])
    c = 1
    with torch.no_grad():
        out = ld.ddim_sample(lambda z, t: model(z, t, torch.full((len(z),), c)), torch.randn(2000, D, generator=g),
                             abar, torch.linspace(999, 0, 25).round().long())  # fmt: skip
    assert float((out.mean(0) - m[c]).abs().max()) < 0.35
    assert float(out.std(0).mean()) == pytest.approx(math.sqrt(s2), rel=0.3)
