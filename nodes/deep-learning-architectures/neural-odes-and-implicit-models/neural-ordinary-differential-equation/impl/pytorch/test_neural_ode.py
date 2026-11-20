# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, gradcheck and the adjoint backward pass."""

import json
import pathlib

import numpy as np
import pytest
import torch

from neural_ode import NAMES, NeuralODE, config, mse_loss, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return config(k["d"], k["d_hidden"], k["d_state"], k["n_steps"])


def test_shared_fixture():
    raw = dict(np.load(FIXTURES / "params-small.npz"))
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    model = NeuralODE.from_params(raw, cfg)
    x = torch.tensor(inp["x"], requires_grad=True)
    y = model(x)
    np.testing.assert_allclose(y.detach().numpy(), want["y"], rtol=1e-12, atol=1e-14)
    loss = mse_loss(y[:, : cfg["d"]], torch.as_tensor(inp["t"]))
    assert loss.item() == pytest.approx(float(want["loss"]), rel=1e-12)
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(getattr(model, k).grad.numpy(), want[f"d{k}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    model = NeuralODE(cfg, torch.Generator().manual_seed(0))
    y = model(torch.zeros(k["B"], k["d"]))
    assert list(y.shape) == c["blocks"]["solve"]["outputs"]["y"]
    assert param_count(cfg) == c["model"]["params"] == sum(p.numel() for p in model.parameters())


def test_gradcheck():
    model = NeuralODE(config(2, 3, d_state=3, n_steps=2), torch.Generator().manual_seed(1))
    x = torch.randn(3, 2, generator=torch.Generator().manual_seed(2), requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: model(inp), (x,))


def test_adjoint_forward_equals_direct():
    model = NeuralODE(config(2, 4, d_state=3, n_steps=5), torch.Generator().manual_seed(3))
    x = torch.randn(4, 2, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(model(x, adjoint=True), model(x), rtol=0, atol=0)


def test_adjoint_gradient_converges_to_autograd():
    # The adjoint gradient differs from autograd through the steps by the error of the solver, which shrinks with dt.
    torch.manual_seed(5)
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(3, 2, generator=gen)
    c = torch.randn(3, 3, generator=gen)
    base = NeuralODE(config(2, 4, d_state=3, n_steps=2), torch.Generator().manual_seed(7))
    gaps = []
    for n in (2, 4, 8):
        model = NeuralODE.from_params({k: getattr(base, k).detach() for k in NAMES}, config(2, 4, d_state=3, n_steps=n))
        grads = []
        for adjoint in (False, True):
            xx = x.clone().requires_grad_(True)
            model.zero_grad()
            (model(xx, adjoint=adjoint) * c).sum().backward()
            grads.append([xx.grad.clone()] + [getattr(model, k).grad.clone() for k in NAMES])
        gaps.append(max((a - b).abs().max().item() for a, b in zip(*grads, strict=True)))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


def test_training_reduces_loss():
    gen = torch.Generator().manual_seed(8)
    model = NeuralODE(config(2, 8, d_state=3, n_steps=4), gen)
    x = torch.randn(16, 2, generator=gen)
    t = torch.stack([x[:, 0] * x[:, 1], torch.sin(x[:, 0])], 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = mse_loss(model(x, adjoint=True)[:, :2], t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
