# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Tests of the PyTorch version: the shared fixture and the autograd gradients of the negative log-likelihood, the
counts of shapes.json, the inverse, the log-determinant against torch.autograd.functional.jacobian, gradcheck, the
planar and autoregressive layers, and a short maximum-likelihood run on the banana density."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from flow import Flow, ar_density, ar_sample, config, flops_forward, param_count, planar, train, u_hat

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return Flow.from_params(params, config()), {k: torch.as_tensor(v) for k, v in params.items()}, inp, want


def random_flow(cfg, seed, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    model = Flow(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return model


def banana(n, generator=None):
    x2 = 2 * torch.randn(n, generator=generator)
    return torch.stack([x2**2 / 4 + torch.randn(n, generator=generator), x2], 1)


def test_shared_fixture():
    model, p, inp, want = fixture()
    z, ld = model(inp["x"])
    torch.testing.assert_close(z, want["z"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(ld, want["logdet"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.log_prob(inp["x"]), want["log_prob"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.inverse(inp["u"]), want["x_inv"], rtol=1e-12, atol=1e-13)
    x = inp["x"].clone().requires_grad_(True)
    loss = model.nll(x)
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=1e-13)
    loss.backward()
    torch.testing.assert_close(x.grad, want["dx"], rtol=1e-10, atol=1e-12)
    for k, layer in enumerate(model.layers):
        for n in ("W1", "b1", "Ws", "bs", "Wt", "bt"):
            torch.testing.assert_close(getattr(layer, n).grad, want[f"dc{k}_{n}"], rtol=1e-10, atol=1e-12)
    w, u, b = p["planar_w"], p["planar_u"], p["planar_b"][0]
    f, pld = planar(inp["x"], w, u_hat(w, u), b)
    torch.testing.assert_close(f, want["planar_f"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(pld, want["planar_logdet"], rtol=1e-12, atol=1e-13)
    q = {k[3:]: v for k, v in p.items() if k.startswith("ar_")}
    au, ald = ar_density(q, inp["x"])
    torch.testing.assert_close(au, want["ar_u"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(ald, want["ar_logdet"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(ar_sample(q, inp["u"]), want["ar_x"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["n_layers"])
    assert param_count(cfg) == mod["params"]
    assert sum(p.numel() for p in Flow(cfg).parameters()) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]


def test_inverse_and_logdet():
    model = random_flow(config(4, 6, 3), 1)
    x = torch.randn(20, 4, generator=torch.Generator().manual_seed(2))
    z, ld = model(x)
    torch.testing.assert_close(model.inverse(z), x, rtol=1e-12, atol=1e-12)
    for row, lr in zip(x[:4], ld[:4], strict=True):
        J = torch.autograd.functional.jacobian(lambda v: model(v[None])[0][0], row)
        torch.testing.assert_close(lr, torch.linalg.slogdet(J)[1], rtol=1e-10, atol=1e-12)


def test_zero_init_is_the_identity():
    model = Flow(config(5, 7, 4))
    x = torch.randn(10, 5)
    z, ld = model(x)
    torch.testing.assert_close(z, x, rtol=0, atol=0)
    torch.testing.assert_close(ld, torch.zeros(10), rtol=0, atol=0)


def test_gradcheck():
    model = random_flow(config(3, 4, 3), 3)
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(4), requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: model.log_prob(v), (x,))


def test_planar_fix():
    g = torch.Generator().manual_seed(5)
    w, u = torch.randn(3, generator=g), torch.randn(3, generator=g)
    u = u - (w @ u + 3.0) * w / (w @ w)
    uh = u_hat(w, u)
    torch.testing.assert_close(w @ uh, torch.nn.functional.softplus(w @ u) - 1)
    z = torch.randn(4, 3, generator=g)
    for row, lr in zip(z, planar(z, w, uh, 0.3)[1], strict=True):
        J = torch.autograd.functional.jacobian(lambda v: planar(v[None], w, uh, 0.3)[0][0], row)
        torch.testing.assert_close(lr, torch.linalg.slogdet(J)[1], rtol=1e-10, atol=1e-12)


def test_autoregressive_round_trip():
    g = torch.Generator().manual_seed(6)
    D = 4
    q = {"Wm": 0.7 * torch.randn(D, D, generator=g), "bm": torch.randn(D, generator=g),
         "Wa": 0.4 * torch.randn(D, D, generator=g), "ba": 0.3 * torch.randn(D, generator=g)}  # fmt: skip
    u = torch.randn(8, D, generator=g)
    x = ar_sample(q, u)
    torch.testing.assert_close(ar_density(q, x)[0], u, rtol=1e-12, atol=1e-12)
    assert (ar_sample(q, u, passes=D - 1) - x).abs().max() > 1e-3


def test_learns_a_banana():
    torch.manual_seed(1)
    model = Flow(config(2, 32, 4))
    g = torch.Generator().manual_seed(2)
    train(model, banana, 1000, generator=g)
    held = banana(20000, torch.Generator().manual_seed(9))
    with torch.no_grad():
        nll = model.nll(held).item()
    entropy = math.log(2 * math.pi * math.e) + 0.5 * math.log(4)
    assert nll == pytest.approx(entropy, abs=0.1)
    assert nll < 3.9  # a Gaussian fit to the banana is above 4
