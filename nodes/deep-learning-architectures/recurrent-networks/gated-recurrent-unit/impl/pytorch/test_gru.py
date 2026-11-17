# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradient, torch.nn.GRU, and training."""

import json
import pathlib

import numpy as np
import pytest
import torch

from gru import GRU, config, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RESETS = ("after", "before")


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


def fixture(reset):
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return GRU.from_params(params, cfg_of(SHAPES[0]), reset), inp, want


@pytest.mark.parametrize("reset", RESETS)
def test_shared_fixture(reset):
    model, inp, want = fixture(reset)
    sfx = "" if reset == "after" else "_before"
    h, cache = model(inp["x"], inp["h0"])
    torch.testing.assert_close(h, want["h" + sfx], rtol=1e-10, atol=1e-12)
    for name in ("h_T", "r", "z", "n"):
        torch.testing.assert_close(cache[name], want[name + sfx], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("reset", RESETS)
def test_shared_fixture_gradients(reset):
    # Autograd of the same loss as fixtures/make.py, against the backward pass of the NumPy reference.
    model, inp, want = fixture(reset)
    sfx = "" if reset == "after" else "_before"
    x, h0 = (inp[k].clone().requires_grad_(True) for k in ("x", "h0"))
    h, cache = model(x, h0)
    loss = (h * inp["r"]).sum() + (cache["h_T"] * inp["r_h"]).sum()
    loss.backward()
    got = {"dx": x.grad, "dh0": h0.grad} | {"d" + k: p.grad for k, p in model.named_parameters()}
    for name, v in got.items():
        torch.testing.assert_close(v, want[name + sfx], rtol=1e-10, atol=1e-12, msg=name)


def test_worked_example_leaky_integrator():
    model = GRU(config(1, 1))
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.0, 0.0, 1.0]]))
        model.U.zero_()
        model.b_i.copy_(torch.tensor([0.0, 2.0, 0.0]))
        model.b_h.zero_()
    h, _ = model(torch.ones(1, 10, 1), torch.zeros(1, 1))
    torch.testing.assert_close(h[0, 0, 0], torch.tensor(0.0907842488), rtol=0, atol=1e-10)
    torch.testing.assert_close(h[0, 7, 0], torch.tensor(0.4857076084), rtol=0, atol=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    if k["H"] > 300:
        pytest.skip("the wide configs only check the counts")
    torch.manual_seed(0)
    for reset in RESETS:
        model = GRU(cfg_of(c), reset)
        h, cache = model(torch.randn(k["B"], k["T"], k["I"]))
        assert list(h.shape) == blocks["layer"]["outputs"]["h"]
        assert list(cache["h_T"].shape) == blocks["layer"]["outputs"]["h_T"]
        for name, shape in blocks["cell"]["internals"].items():
            assert list(cache[name].shape) == shape, name


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["config"]["H"] <= 300:
        assert sum(p.numel() for p in GRU(cfg).parameters()) == c["model"]["params"]
        assert sum(p.numel() for p in torch.nn.GRU(cfg["input_size"], cfg["hidden_size"]).parameters()) == (
            c["model"]["params"]
        )


def test_matches_torch_nn_gru():
    torch.manual_seed(3)
    model = GRU(config(3, 5))
    lib = model.to_torch_gru()
    x, h0 = torch.randn(4, 7, 3), torch.randn(4, 5)
    h, cache = model(x, h0)
    out, h_n = lib(x, h0[None])
    torch.testing.assert_close(h, out, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(cache["h_T"], h_n[0], rtol=1e-12, atol=1e-13)
    # The placement of Cho et al. is a different function with the same weights.
    model.reset = "before"
    h_b, _ = model(x, h0)
    assert (h_b - out).abs().max() > 1e-3


@pytest.mark.parametrize("reset", RESETS)
def test_gradcheck(reset):
    torch.manual_seed(1)
    model = GRU(config(2, 3), reset)
    names = [name for name, _ in model.named_parameters()]
    ps = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    x, h0 = torch.randn(2, 4, 2, requires_grad=True), torch.randn(2, 3, requires_grad=True)

    def f(x, h0, *q):
        return torch.func.functional_call(model, dict(zip(names, q)), (x, h0))[0]

    assert torch.autograd.gradcheck(f, (x, h0, *ps))


def test_learns_to_carry_a_bit_over_30_steps():
    # The first input is a sign, the next 29 are noise; a linear readout of h_30 must give back the sign.
    torch.manual_seed(0)
    T, H = 30, 8

    def batch(n):
        s = torch.randint(0, 2, (n,)) * 2.0 - 1.0
        x = torch.randn(n, T, 1) * 0.5
        x[:, 0, 0] = s
        return x, (s > 0).double()

    model, head = GRU(config(1, H)), torch.nn.Linear(H, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=0.05)
    for _ in range(300):
        x, y = batch(64)
        _, cache = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(cache["h_T"])[:, 0], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.0)
        opt.step()
    with torch.no_grad():
        x, y = batch(1000)
        _, cache = model(x)
        acc = ((head(cache["h_T"])[:, 0] > 0).double() == y).double().mean().item()
    assert acc > 0.95, acc
