# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://FKJKTR).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradient, torch.nn.RNN, and training."""

import json
import pathlib

import numpy as np
import pytest
import torch

from rnn import NONLINEARITIES, RNN, clip_by_norm, config, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    return config(c["config"]["input_size"], c["config"]["hidden_size"])


def fixture(nl):
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return RNN.from_params(params, cfg_of(SHAPES[0]), nl), inp, want


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_fixture(nl):
    model, inp, want = fixture(nl)
    sfx = "" if nl == "tanh" else "_relu"
    h, cache = model(inp["x"], inp["h0"])
    torch.testing.assert_close(h, want["h" + sfx], rtol=1e-10, atol=1e-12)
    for name in ("h_T", "a"):
        torch.testing.assert_close(cache[name], want[name + sfx], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_shared_fixture_gradients(nl):
    # Autograd of the same loss as fixtures/make.py, against the backward pass of the NumPy reference.
    model, inp, want = fixture(nl)
    sfx = "" if nl == "tanh" else "_relu"
    x, h0 = (inp[k].clone().requires_grad_(True) for k in ("x", "h0"))
    h, cache = model(x, h0)
    loss = (h * inp["r"]).sum() + (cache["h_T"] * inp["r_h"]).sum()
    loss.backward()
    got = {"dx": x.grad, "dh0": h0.grad} | {"d" + k: p.grad for k, p in model.named_parameters()}
    for name, v in got.items():
        torch.testing.assert_close(v, want[name + sfx], rtol=1e-10, atol=1e-12, msg=name)


def test_worked_example():
    x = torch.zeros(1, 10, 1)
    x[0, 0, 0] = 1.0
    for nl, h10, grad in (("tanh", 0.1892358120, 0.0419487706), ("relu", 0.9**9, 0.9**9)):
        model = RNN(config(1, 1), nl)
        with torch.no_grad():
            model.W.fill_(1.0)
            model.U.fill_(0.9)
            model.b_i.zero_()
            model.b_h.zero_()
        x.requires_grad_(True)
        _, cache = model(x, torch.zeros(1, 1))
        (g,) = torch.autograd.grad(cache["h_T"].sum(), x)
        torch.testing.assert_close(cache["h_T"][0, 0], torch.tensor(h10), rtol=0, atol=1e-10)
        torch.testing.assert_close(g[0, 0, 0], torch.tensor(grad), rtol=0, atol=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    k, blocks = c["config"], c["blocks"]
    if k["H"] > 300:
        pytest.skip("the wide configs only check the counts")
    torch.manual_seed(0)
    for nl in NONLINEARITIES:
        h, cache = RNN(cfg_of(c), nl)(torch.randn(k["B"], k["T"], k["I"]))
        assert list(h.shape) == blocks["layer"]["outputs"]["h"]
        assert list(cache["h_T"].shape) == blocks["layer"]["outputs"]["h_T"]
        assert list(cache["a"].shape) == blocks["step"]["internals"]["a"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["config"]["H"] <= 300:
        assert sum(p.numel() for p in RNN(cfg).parameters()) == c["model"]["params"]
        lib = torch.nn.RNN(cfg["input_size"], cfg["hidden_size"])
        assert sum(p.numel() for p in lib.parameters()) == c["model"]["params"]
        gru, lstm = torch.nn.GRU(cfg["input_size"], cfg["hidden_size"]), torch.nn.LSTM(cfg["input_size"], cfg["hidden_size"])
        assert sum(p.numel() for p in gru.parameters()) == c["model"]["gru_params"]
        assert sum(p.numel() for p in lstm.parameters()) == c["model"]["lstm_params"]


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_matches_torch_nn_rnn(nl):
    torch.manual_seed(3)
    model = RNN(config(3, 5), nl)
    lib = model.to_torch_rnn()
    x, h0 = torch.randn(4, 7, 3), torch.randn(4, 5)
    h, cache = model(x, h0)
    out, h_n = lib(x, h0[None])
    torch.testing.assert_close(h, out, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(cache["h_T"], h_n[0], rtol=1e-12, atol=1e-13)
    # A square layer also runs with the weights copied without the transpose, and computes another function.
    square = RNN(config(5, 5), nl)
    wrong = square.to_torch_rnn()
    with torch.no_grad():
        wrong.weight_ih_l0.copy_(square.W)
        wrong.weight_hh_l0.copy_(square.U)
    x5 = torch.randn(4, 7, 5)
    assert not torch.allclose(square(x5, h0)[0], wrong(x5, h0[None])[0], atol=1e-6)


@pytest.mark.parametrize("nl", NONLINEARITIES)
def test_gradcheck(nl):
    torch.manual_seed(1)
    model = RNN(config(2, 3), nl)
    names = [name for name, _ in model.named_parameters()]
    ps = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    x, h0 = torch.randn(2, 4, 2, requires_grad=True), torch.randn(2, 3, requires_grad=True)

    def f(x, h0, *q):
        return torch.func.functional_call(model, dict(zip(names, q)), (x, h0))[0]

    assert torch.autograd.gradcheck(f, (x, h0, *ps))


def test_clip_by_norm_is_torch_clip_grad_norm():
    # Algorithm 1 of Pascanu et al. keeps the direction and caps the norm; torch.nn.utils.clip_grad_norm_ is the same.
    torch.manual_seed(2)
    grads = [torch.randn(3, 4) * 10, torch.randn(4)]
    clipped, norm = clip_by_norm(grads, 1.0)
    torch.testing.assert_close(torch.sqrt(sum((g * g).sum() for g in clipped)), torch.tensor(1.0))
    for g, c in zip(grads, clipped):
        torch.testing.assert_close(c * norm, g)  # the same direction
    ps = [torch.nn.Parameter(torch.zeros_like(g)) for g in grads]
    for p, g in zip(ps, grads):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(ps, 1.0)
    for p, c in zip(ps, clipped):
        torch.testing.assert_close(p.grad, c, rtol=1e-6, atol=1e-12)  # torch adds 1e-6 to the norm
    small_grads, _ = clip_by_norm([g / 1000 for g in grads], 1.0)
    for g, c in zip(grads, small_grads):
        torch.testing.assert_close(c, g / 1000)  # below the threshold nothing changes


def test_gradient_vanishes_with_distance():
    # At the default initialisation the gradient of the last state with respect to earlier inputs falls
    # geometrically: the input 60 steps back gets a gradient many orders of magnitude smaller than the last one.
    torch.manual_seed(0)
    model = RNN(config(1, 16))
    x = torch.randn(8, 60, 1, requires_grad=True)
    _, cache = model(x)
    (g,) = torch.autograd.grad(cache["h_T"].sum(), x)
    per_step = g.abs().mean(dim=(0, 2))  # (T,)
    assert per_step[0] < 1e-6 * per_step[-1]


def test_learns_to_carry_a_bit_over_5_steps():
    # The first input is a sign, the next 4 are noise; a linear readout of h_5 must give back the sign.
    torch.manual_seed(0)
    T, H = 5, 8

    def batch(n):
        s = torch.randint(0, 2, (n,)) * 2.0 - 1.0
        x = torch.randn(n, T, 1) * 0.5
        x[:, 0, 0] = s
        return x, (s > 0).double()

    model, head = RNN(config(1, H)), torch.nn.Linear(H, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=0.05)
    for _ in range(200):
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
