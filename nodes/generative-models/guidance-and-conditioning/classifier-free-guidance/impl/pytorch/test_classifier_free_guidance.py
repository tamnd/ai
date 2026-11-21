# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Tests of the PyTorch implementation: the shared fixture with autograd gradients, gradcheck through the denoiser,
the guided noise as the score of the tilted density by autograd, classifier guidance, the end variance of the guided
ODE, the rescale and the thresholds, the interval count, and a denoiser trained with a dropped condition."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

import classifier_free_guidance as cg

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    a = a.detach().numpy() if isinstance(a, torch.Tensor) else a
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    model = cg.Denoiser(*(int(v) for v in inp["cfg"]), params=P)
    z, t, y, eps = inp["z"], inp["t"], inp["y"], inp["eps"]
    close(model(z, t, y), want["eps_hat"])
    loss = cg.eps_loss(model, z, t, y, eps)
    assert loss.item() == pytest.approx(float(want["loss"]), rel=1e-12)
    loss.backward()
    for k, prm in model.named_parameters():
        assert prm.grad is not None
        close(prm.grad, want[f"d{k}"], rtol=1e-10, msg=k)
    guided = cg.guide(inp["e_c"], inp["e_u"], 3.0)
    close(guided, want["guided"])
    close(cg.rescale(inp["e_c"], guided, 0.7), want["rescaled"])
    close(cg.dynamic_threshold(inp["x0"], 90.0), want["dyn"])
    abar = cg.alpha_bars(1000)
    means, w = inp["means"], torch.tensor([0.5, 0.5])
    eps_fn, count = cg.guided_sampler(lambda zz, tt: cg.exact_eps_cond(zz, abar[tt].expand(len(zz)), means, 0.1, 0),
                                      lambda zz, tt: cg.exact_eps(zz, abar[tt].expand(len(zz)), means, 0.1, w),
                                      2.0, abar, interval=(0.3, 3.0))  # fmt: skip
    close(cg.ddim_sample(eps_fn, inp["zT"], abar, [999, 600, 300, 100, 0]), want["ddim"], rtol=1e-10, atol=1e-12)
    assert count["nfe"] == int(want["nfe"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    assert cg.param_count(k["dim"], k["hidden"], k["temb"], k["classes"]) == mod["params"]
    assert cg.flops_forward(k["dim"], k["hidden"], k["temb"], k["B"]) == mod["flops_forward"]
    model = cg.Denoiser(k["dim"], k["hidden"], k["temb"], k["classes"])
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    with torch.no_grad():
        out = model(torch.zeros(k["B"], k["dim"]), 3, k["classes"])
    assert out.shape == (k["B"], k["dim"])


def test_gradcheck():
    """gradcheck of the loss through the input and every parameter, by torch.func.functional_call."""
    torch.manual_seed(1)
    model = cg.Denoiser(3, 4, 4, 2)
    t, y, eps = torch.tensor([1, 200, 500, 999]), torch.tensor([0, 1, 2, 2]), torch.randn(4, 3)
    names = [k for k, _ in model.named_parameters()]

    def loss(z, *ps):
        out = torch.func.functional_call(model, dict(zip(names, ps, strict=True)), (z, t, y))
        return ((out - eps) ** 2).sum(1).mean()

    args = (torch.randn(4, 3), *model.parameters())
    assert torch.autograd.gradcheck(loss, tuple(v.detach().clone().requires_grad_(True) for v in args))


def test_guided_noise_is_tilted_score():
    torch.manual_seed(2)
    means, s2 = torch.randn(3, 3), 0.3
    w = torch.rand(3) + 0.2
    w = w / w.sum()
    z = torch.randn(7, 3)
    abar, c = cg.alpha_bars(1000), 1
    for t in (10, 300, 800):
        a = abar[t].expand(len(z))
        for s in (0.0, 2.5, 7.0):
            zz = z.clone().requires_grad_(True)
            logp = s * cg.log_marginal(zz, a, means[c : c + 1], s2, torch.ones(1)) + (1 - s) * cg.log_marginal(
                zz, a, means, s2, w
            )
            (grad,) = torch.autograd.grad(logp.sum(), zz)
            guided = cg.guide(cg.exact_eps_cond(z, a, means, s2, c), cg.exact_eps(z, a, means, s2, w), s)
            close(guided, -(1 - a[:, None]).sqrt() * grad, rtol=1e-10, atol=1e-12)
            g = cg.classifier_grad(z, a, means, s2, w, torch.full((len(z),), c))
            close(guided, cg.classifier_guided_eps(cg.exact_eps_cond(z, a, means, s2, c), g, a, s - 1.0), atol=1e-12)
            close(guided, cg.classifier_guided_eps(cg.exact_eps(z, a, means, s2, w), g, a, s), atol=1e-12)


def test_guided_ode_variance():
    abar = cg.alpha_bars(1000)
    zT = torch.tensor([[0.0], [1.0]])
    for s in (1.0, 2.0, 3.0):
        x = cg.ddim_sample(lambda zz, tt, s=s: cg.guide(cg.gaussian_eps(zz, abar[tt].expand(len(zz)), 1.0, 0.25),
                                                        cg.gaussian_eps(zz, abar[tt].expand(len(zz)), 0.0, 1.25), s),
                           zT, abar, list(range(999, -1, -1)))  # fmt: skip
        assert float((x[1, 0] - x[0, 0]) ** 2) == pytest.approx(0.25**s * 1.25 ** (1 - s), rel=0.03)


def test_rescale_and_thresholds():
    torch.manual_seed(3)
    e_c, e_u = torch.randn(2, 4, 3, 5, 5)
    out = cg.guide(e_c, e_u, 7.5)
    close(cg.rescale(e_c, out, 0.0), out)
    close(cg.rescale(e_c, out, 1.0).std(dim=(1, 2, 3)), e_c.std(dim=(1, 2, 3)))
    x = 1.6 * torch.randn(5, 40)
    for p in (0.0, 50.0, 99.5):
        d = cg.dynamic_threshold(x, p)
        assert bool((d.abs() <= 1.0 + 1e-15).all())
        close(d, cg.static_threshold(x) if p == 0.0 else d)
        close(d.numpy(), np.clip(x.numpy(), -1, 1) if p == 0 else d.numpy())
    s = np.maximum(np.percentile(np.abs(x.numpy()), 99.5, axis=1, keepdims=True), 1.0)
    close(cg.dynamic_threshold(x, 99.5), np.clip(x.numpy(), -s, s) / s)


def test_interval_count():
    abar = cg.alpha_bars(1000)
    steps = [int(v) for v in np.linspace(999, 0, 32).round()]
    sig = cg.edm_sigma(abar[steps])
    inside = int(((sig > 0.28) & (sig <= 2.9)).sum())
    eps, count = cg.guided_sampler(lambda z, t: z, lambda z, t: 0 * z, 2.0, abar, interval=(0.28, 2.9))
    cg.ddim_sample(eps, torch.zeros(1, 2), abar, steps)
    assert count["nfe"] == len(steps) + inside < 2 * len(steps)


def test_training_and_guided_sampling():
    gen = torch.Generator().manual_seed(10)
    torch.manual_seed(10)
    D, C, s2 = 2, 3, 0.25
    m = torch.tensor([[1.5, 0.0], [-0.75, 1.3], [-0.75, -1.3]])
    y = torch.randint(0, C, (6000,), generator=gen)
    x0 = m[y] + math.sqrt(s2) * torch.randn(6000, D, generator=gen)
    model = cg.Denoiser(D, 64, 16, C)
    abar = cg.alpha_bars(1000)
    trace = cg.train(model, x0, y, abar, 2500, gen)
    assert float(trace[-200:].mean()) < 0.6 * float(trace[:50].mean())
    zT = torch.randn(3000, D, generator=gen)
    steps = [int(v) for v in np.linspace(999, 0, 25).round()]
    w = torch.full((C,), 1.0 / C)
    conf = []
    with torch.no_grad():
        for s in (1.0, 3.0):
            eps, _ = cg.guided_sampler(lambda z, t: model(z, t, 1), lambda z, t: model(z, t, C), s, abar)
            x = cg.ddim_sample(eps, zT, abar, steps)
            conf.append(float(cg.log_classifier(x, torch.ones(len(x)) - 1e-12, m, s2, w)[:, 1].exp().mean()))
            if s == 1.0:
                assert float((x.mean(0) - m[1]).abs().max()) < 0.25
    assert conf[1] > conf[0] and conf[1] > 0.97
