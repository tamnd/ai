# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Tests of the PyTorch version: the shared fixture and the autograd gradients of the simple loss, the counts of
shapes.json, gradcheck through the network, the posterior mean as the reverse mean at the true noise, the DDIM marginals
and determinism, and a short training run on two Gaussian modes."""

import json
import pathlib

import numpy as np
import pytest
import torch

from diffusion import (
    EpsModel,
    config,
    ddim_sample,
    ddim_step,
    ddpm_sample,
    ddpm_step,
    flops_forward,
    guided_eps,
    param_count,
    posterior,
    q_sample,
    reverse_mean,
    schedule,
    simple_loss,
    time_embedding,
    train,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
MODES = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])


def two_modes(n, generator=None):
    k = torch.randint(0, 2, (n,), generator=generator)
    return MODES[k] + 0.3 * torch.randn(n, 2, generator=generator)


def random_model(cfg, seed, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    model = EpsModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return model


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    cfg = config()
    model = EpsModel.from_params(params, cfg)
    x0, t, eps, z = inp["x0"], inp["t"], inp["eps"], inp["z"]
    torch.testing.assert_close(time_embedding(t, cfg["d_time"]), want["emb"], rtol=1e-12, atol=1e-14)
    xt = q_sample(cfg, x0, t, eps)
    torch.testing.assert_close(xt, want["xt"], rtol=1e-12, atol=1e-14)
    out = model(xt, t)
    torch.testing.assert_close(out, want["eps_pred"], rtol=1e-12, atol=1e-14)
    loss = simple_loss(model, x0, t, eps)
    loss.backward()
    torch.testing.assert_close(loss.detach(), want["loss"], rtol=1e-12, atol=1e-14)
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, want[f"d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    pm, pv = posterior(cfg, xt, x0, t)
    torch.testing.assert_close(pm, want["post_mean"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(pv, want["post_var"], rtol=1e-12, atol=1e-14)
    out = out.detach()
    torch.testing.assert_close(reverse_mean(cfg, xt, t, out), want["mu"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(ddpm_step(cfg, xt, t, out, z), want["x_prev"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        ddim_step(cfg, xt, t, t // 2, out, z, 0.5),
        want["x_ddim"],
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["d_time"], k["n_steps"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = EpsModel(cfg)
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    assert model(torch.zeros(k["B"], k["d_data"]), torch.ones(k["B"], dtype=torch.long)).shape == (k["B"], k["d_data"])


def test_gradcheck():
    model = random_model(config(), 3)
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(4), requires_grad=True)
    t = torch.tensor([1, 3, 6, 10])
    assert torch.autograd.gradcheck(lambda v: model(v, t), (x,))


def test_reverse_mean_at_the_true_noise_is_the_posterior_mean():
    cfg = config()
    g = torch.Generator().manual_seed(5)
    x0, eps = torch.randn(6, 3, generator=g), torch.randn(6, 3, generator=g)
    t = torch.tensor([2, 3, 5, 7, 9, 10])
    xt = q_sample(cfg, x0, t, eps)
    torch.testing.assert_close(
        reverse_mean(cfg, xt, t, eps),
        posterior(cfg, xt, x0, t)[0],
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_ddim_step_keeps_the_forward_marginals(eta):
    cfg = config(d_data=1)
    g = torch.Generator().manual_seed(6)
    n = 400_000
    x0, eps = torch.full((n, 1), -0.6), torch.randn(n, 1, generator=g)
    t, s = torch.full((n,), 8), torch.full((n,), 2)
    xs = ddim_step(cfg, q_sample(cfg, x0, t, eps), t, s, eps, torch.randn(n, 1, generator=g), eta)
    ab = schedule(cfg)["abar"][1].item()
    assert xs.mean().item() == pytest.approx(ab**0.5 * -0.6, abs=0.005)
    assert xs.var().item() == pytest.approx(1 - ab, rel=0.01)


def test_ddim_is_deterministic_and_guidance_at_zero_is_conditional():
    model = random_model(config(), 7)
    xT = torch.randn(3, 3, generator=torch.Generator().manual_seed(8))
    torch.manual_seed(0)
    a = ddim_sample(model, xT, 5)
    torch.manual_seed(1)
    torch.testing.assert_close(ddim_sample(model, xT, 5), a, rtol=0, atol=0)
    ec, eu = torch.randn(3, 3), torch.randn(3, 3)
    torch.testing.assert_close(guided_eps(ec, eu, 0.0), ec)
    torch.testing.assert_close(guided_eps(ec, eu, 2.0) - ec, 2.0 * (ec - eu))


def test_learns_two_modes():
    torch.manual_seed(1)
    model = EpsModel(config(2, 64, 16, 100, 1e-3, 0.1))
    g = torch.Generator().manual_seed(2)
    trace = train(model, two_modes, 2000, generator=g)
    assert np.mean(trace[-200:]) < 0.56  # the exact noise prediction scores about 0.50
    xs = ddpm_sample(model, 2000, torch.Generator().manual_seed(3))
    dist = torch.cdist(xs, MODES).min(1).values
    assert 0.4 < (xs[:, 0] > 0).double().mean().item() < 0.6
    assert (dist < 1.0).double().mean().item() > 0.95
