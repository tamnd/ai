# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/long-short-term-memory/ (ai://YVW8D3).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradient, and torch.nn.LSTM."""

import json
import pathlib

import numpy as np
import pytest
import torch

from lstm import LSTM, config, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return LSTM.from_params(params, cfg_of(SHAPES[0])), inp, want


def test_shared_fixture():
    model, inp, want = fixture()
    h, cache = model(inp["x"], inp["h0"], inp["c0"])
    torch.testing.assert_close(h, want["h"], rtol=1e-10, atol=1e-12)
    for name in ("c", "h_T", "c_T", "i", "f", "g", "o"):
        torch.testing.assert_close(cache[name], want[name], rtol=1e-10, atol=1e-12)


def test_shared_fixture_gradients():
    # Autograd of the same loss as fixtures/make.py, against the backward pass of the NumPy reference.
    model, inp, want = fixture()
    x, h0, c0 = (inp[k].clone().requires_grad_(True) for k in ("x", "h0", "c0"))
    h, cache = model(x, h0, c0)
    loss = (h * inp["r"]).sum() + (cache["h_T"] * inp["r_h"]).sum() + (cache["c_T"] * inp["r_c"]).sum()
    loss.backward()
    got = {
        "dx": x.grad,
        "dh0": h0.grad,
        "dc0": c0.grad,
        "dW": model.W.grad,
        "dU": model.U.grad,
        "db": model.b.grad,
    }
    for name, v in got.items():
        torch.testing.assert_close(v, want[name], rtol=1e-10, atol=1e-12, msg=name)


def test_worked_example():
    model = LSTM(config(1, 1))
    with torch.no_grad():
        model.W.copy_(torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        model.U.zero_()
        model.b.zero_()
    h, c, g = model.cell(torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1))
    torch.testing.assert_close(c, torch.tensor([[1.1118556566]]), rtol=0, atol=1e-10)
    torch.testing.assert_close(h, torch.tensor([[0.4023586789]]), rtol=0, atol=1e-10)
    torch.testing.assert_close(g["f"], torch.tensor([[0.7310585786]]), rtol=0, atol=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    torch.manual_seed(0)
    model = LSTM(cfg_of(c))
    h, cache = model(torch.randn(k["B"], k["T"], k["I"]))
    assert list(h.shape) == blocks["layer"]["outputs"]["h"]
    assert list(cache["c"].shape) == blocks["layer"]["outputs"]["c"]
    for name, shape in blocks["cell"]["internals"].items():
        assert list(cache[name].shape) == shape, name


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    model = LSTM(cfg_of(c))
    assert param_count(cfg_of(c)) == c["model"]["params"] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agrees_with_torch_lstm(seed):
    # The second oracle: torch.nn.LSTM with the same weights, gates stacked as (i, f, g, o).
    torch.manual_seed(seed)
    model = LSTM(config(3, 5), forget_bias=1.0)
    lib = model.to_torch_lstm()
    x, h0, c0 = torch.randn(2, 7, 3), torch.randn(2, 5), torch.randn(2, 5)
    h, cache = model(x, h0, c0)
    out, (h_n, c_n) = lib(x, (h0[None], c0[None]))
    torch.testing.assert_close(h, out, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["h_T"], h_n[0], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["c_T"], c_n[0], rtol=1e-10, atol=1e-12)


def test_torch_lstm_gate_order_matters():
    # Swapping the forget and the input block of the stack changes the output, so the agreement above checks the order.
    torch.manual_seed(3)
    model = LSTM(config(2, 3), forget_bias=2.0)
    lib = model.to_torch_lstm()
    with torch.no_grad():
        named = dict(lib.named_parameters())
        for p in (named["weight_ih_l0"], named["weight_hh_l0"], named["bias_ih_l0"]):
            p.copy_(torch.cat([p[3:6], p[0:3], p[6:]]))
    x = torch.randn(1, 4, 2)
    assert (model(x)[0] - lib(x)[0]).abs().max() > 1e-3


def test_gradcheck_one_step():
    torch.manual_seed(4)
    model = LSTM(config(2, 3))

    def step(x, h, c, W, U, b):
        model.W.data, model.U.data, model.b.data = W, U, b
        z = x @ W + h @ U + b
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        c_new = torch.sigmoid(zf) * c + torch.sigmoid(zi) * torch.tanh(zg)
        return torch.sigmoid(zo) * torch.tanh(c_new), c_new

    args = [
        torch.randn(2, 2),
        torch.randn(2, 3),
        torch.randn(2, 3),
        model.W.detach().clone(),
        model.U.detach().clone(),
        model.b.detach().clone(),
    ]
    args = [a.requires_grad_(True) for a in args]
    assert torch.autograd.gradcheck(step, args, eps=1e-6, atol=1e-8)
    # The same step through the module.
    h, c, _ = model.cell(args[0], args[1], args[2])
    h2, c2 = step(*args)
    torch.testing.assert_close(h, h2)
    torch.testing.assert_close(c, c2)


def test_gradcheck_sequence():
    torch.manual_seed(5)
    model = LSTM(config(2, 3), forget_bias=1.0)
    x = torch.randn(1, 4, 2, requires_grad=True)
    c0 = torch.randn(1, 3, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, c0: model(x, None, c0)[0], (x, c0), eps=1e-6, atol=1e-8)


def test_copy_when_forget_is_one_and_input_is_zero():
    torch.manual_seed(6)
    model = LSTM(config(3, 4))
    with torch.no_grad():
        model.b[:4] = -800.0
        model.b[4:8] = 800.0
    c0 = torch.randn(2, 4)
    _, cache = model(3.0 * torch.randn(2, 20, 3), torch.randn(2, 4), c0)
    assert torch.all(cache["i"] == 0.0) and torch.all(cache["f"] == 1.0)
    assert torch.equal(cache["c_T"], c0)


def test_gates_in_unit_interval():
    torch.manual_seed(7)
    model = LSTM(config(3, 4))
    _, cache = model(4.0 * torch.randn(8, 10, 3))
    for name in ("i", "f", "o"):
        assert torch.all((cache[name] > 0) & (cache[name] < 1)), name


def test_learns():
    # The layer and a linear readout learn to output, at the last step, the sign of the first input of a sequence of
    # 12 random numbers: the answer must be kept in a cell over 11 steps. Mean squared error on 64 sequences,
    # 300 steps of Adam; a constant prediction leaves the loss near 1.
    torch.manual_seed(0)
    x = torch.randn(64, 12, 1)
    y = torch.sign(x[:, 0, 0])
    model = LSTM(config(1, 8), forget_bias=1.0)
    readout = torch.nn.Linear(8, 1)
    opt = torch.optim.Adam([*model.parameters(), *readout.parameters()], lr=2e-2)
    losses = []
    for _ in range(300):
        h, _ = model(x)
        loss = ((readout(h[:, -1]).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 0.5
    assert losses[-1] < 0.05, losses[-1]


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_config_rejects_bad_sizes():
    assert config(1, 1) == {"input_size": 1, "hidden_size": 1}
    with pytest.raises(ValueError):
        config(0, 4)
    with pytest.raises(ValueError):
        config(3, 0)


def test_init_forget_bias():
    torch.manual_seed(3)
    m0 = LSTM(config(2, 5))
    torch.manual_seed(3)
    m1 = LSTM(config(2, 5), forget_bias=1.0)
    want = torch.cat([torch.zeros(5), torch.ones(5), torch.zeros(10)])
    torch.testing.assert_close((m1.b - m0.b).detach(), want)
    assert torch.equal(m0.W, m1.W) and torch.equal(m0.U, m1.U)
