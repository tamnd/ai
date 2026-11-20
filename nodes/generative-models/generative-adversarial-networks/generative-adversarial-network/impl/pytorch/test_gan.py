# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Tests of the PyTorch version: the shared fixture and the autograd gradients of the five losses, the counts of
shapes.json, gradcheck through the gradient penalties, the fakes held fixed in the discriminator's step, and a short
training run that moves the generator onto a Gaussian."""

import json
import pathlib

import numpy as np
import pytest
import torch

from gan import DIS, GAN, GEN, LOSSES, config, param_count, train

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return GAN.from_params(params, config()), inp, want


def test_shared_fixture_forward():
    model, inp, want = fixture()
    fake = model.generate(inp["z"])
    torch.testing.assert_close(fake, want["fake"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.critic(inp["x"]), want["a_real"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.critic(fake), want["a_fake"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.input_grad(inp["x"]), want["grad_real"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", LOSSES)
def test_shared_fixture_gradients(kind):
    model, inp, want = fixture()
    for side, names in (("d", DIS), ("g", GEN)):
        model.zero_grad()
        if side == "d":
            value = model.dis_loss(inp["x"], inp["z"], kind, inp["alpha"], 10.0, 1.0)
        else:
            value = model.gen_loss(inp["z"], kind)
        torch.testing.assert_close(value, want[f"{kind}/{side}/loss"], rtol=1e-12, atol=1e-13)
        value.backward()
        for k in names:
            torch.testing.assert_close(model.p[k].grad, want[f"{kind}/{side}/d{k}"], rtol=1e-10, atol=1e-12, msg=k)
        if side == "d":  # the fakes are held fixed, so the discriminator's step leaves the generator alone
            assert all(model.p[k].grad is None for k in GEN)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_noise"], k["d_hidden"], k["d_data"], k["d_critic"])
    assert param_count(cfg) == (mod["params_gen"], mod["params_dis"])
    model = GAN(cfg)
    assert sum(p.numel() for p in model.gen_params()) == mod["params_gen"]
    assert sum(p.numel() for p in model.dis_params()) == mod["params_dis"]


@pytest.mark.parametrize("kind", ["wgan_gp", "r1"])
def test_gradcheck_through_the_penalty(kind):
    g = torch.Generator().manual_seed(0)
    model = GAN(config(), generator=g)
    x, z, alpha = torch.randn(3, 3, generator=g), torch.randn(3, 2, generator=g), torch.rand(3, generator=g)
    names = list(DIS)

    def f(*ws):
        saved = {k: model.p[k] for k in names}
        for k, w in zip(names, ws, strict=True):
            model.p._parameters[k] = w
        try:
            return model.dis_loss(x, z, kind, alpha, 10.0, 1.0)
        finally:
            for k in names:
                model.p._parameters[k] = saved[k]

    ws = tuple(model.p[k].detach().clone().requires_grad_(True) for k in names)
    assert torch.autograd.gradcheck(f, ws)


def test_training_moves_the_generator_onto_a_gaussian():
    g = torch.Generator().manual_seed(0)
    model = GAN(config(1, 16, 1, 16), generator=g)

    def data(n, gen):
        return 2.0 + 0.5 * torch.randn(n, 1, generator=gen)

    trace = train(model, data, 1500, "nonsat", batch=128, lr=2e-3, generator=g)
    with torch.no_grad():
        s = model.generate(torch.randn(20000, 1, generator=g))[:, 0]
    assert abs(s.mean().item() - 2.0) < 0.3 and 0.2 < s.std().item() < 0.8
    assert np.mean([d for d, _ in trace[-100:]]) == pytest.approx(np.log(4), abs=0.1)
