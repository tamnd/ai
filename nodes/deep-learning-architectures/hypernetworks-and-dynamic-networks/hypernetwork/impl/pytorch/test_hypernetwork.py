# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Tests of the PyTorch hypernetworks: the shared fixture and its gradients for the static network, the HyperRNN and
the HyperLSTM, the counts of the shape contract, gradcheck, the initialization of Appendix A.2.3, and a small
experiment in which a network with generated kernels learns with a fifth of the parameters of an ordinary one."""

import json
import pathlib

import numpy as np
import pytest
import torch

import hypernetwork as M

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]


def fixture():
    p, i, e = (dict(np.load(FIX / f"{n}-small.npz")) for n in ("params", "input", "expected"))
    return p, i, e


def part(d, prefix):
    return {k[len(prefix) + 1 :]: v for k, v in d.items() if k.startswith(prefix + ".")}


def test_static_fixture():
    P, I, E = fixture()
    model = M.HyperConvNet.from_params(part(P, "static"))
    e = part(E, "static")
    x = torch.as_tensor(I["static.x"]).requires_grad_(True)
    logits = model(x)
    assert np.allclose(logits.detach().numpy(), e["logits"], atol=1e-12)
    loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(I["static.y"]))
    assert np.isclose(loss.item(), e["loss"], atol=1e-12)
    loss.backward()
    for k, t in model.named_like_numpy().items():
        assert np.allclose(t.grad.numpy(), e["d" + k], atol=1e-12), k
    assert np.allclose(x.grad.numpy(), e["dx"], atol=1e-12)


def test_hyper_rnn_fixture():
    P, I, E = fixture()
    model = M.HyperRNN.from_params(part(P, "rnn"))
    e = part(E, "rnn")
    x = torch.as_tensor(I["rnn.x"]).requires_grad_(True)
    out = model(x)
    assert np.allclose(out.detach().numpy(), e["out"], atol=1e-12)
    loss = 0.5 * ((out - torch.as_tensor(I["rnn.target"])) ** 2).mean()
    assert np.isclose(loss.item(), e["loss"], atol=1e-12)
    loss.backward()
    for k, t in model.named_parameters():
        assert np.allclose(t.grad.numpy(), e["d" + k], atol=1e-12), k
    assert np.allclose(x.grad.numpy(), e["dx"], atol=1e-12)


@pytest.mark.parametrize("ln", [True, False])
def test_hyper_lstm_fixture(ln):
    P, I, E = fixture()
    model = M.HyperLSTM.from_params(part(P, "lstm"), ln=ln)
    out = model(torch.as_tensor(I["rnn.x"]))
    assert np.allclose(out.detach().numpy(), E["lstm.out_ln" if ln else "lstm.out"], atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, m = c["config"], c["model"]
    model = M.HyperConvNet(k["c_in"], k["n_basic"], k["mult"], k["fsize"], k["depth"], k["n_z"], k["classes"])
    assert sum(t.numel() for t in model.parameters()) == m["params"]
    assert sum(t.numel() for t in model.hyper.parameters()) == m["hyper_params"]
    ws = model.kernels()
    assert sum(w.numel() for w in ws) == m["generated_weights"]
    if c["kind"] != "named":
        x = torch.randn(k["B"], k["c_in"], k["size"], k["size"], dtype=torch.double)
        assert list(model(x).shape) == c["blocks"]["head"]["outputs"]["logits"]
        assert list(ws[0].shape) == c["blocks"]["conv"]["internals"]["w"]


def test_gradcheck():
    torch.manual_seed(0)
    net = M.HyperConvNet(1, 2, 1, 3, 2, 2, 2)
    x = torch.randn(1, 1, 4, 4, dtype=torch.double, requires_grad=True)
    z = net.hyper.z.detach().clone().requires_grad_(True)
    W_out = net.hyper.W_out.detach().clone().requires_grad_(True)
    call = lambda x, z, W: torch.func.functional_call(net, {"hyper.z": z, "hyper.W_out": W}, (x,))  # noqa: E731
    assert torch.autograd.gradcheck(call, (x, z, W_out))
    rnn = M.HyperRNN(2, 3, 2, 2)
    s = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(rnn, (s,))
    lstm = M.HyperLSTM(2, 3, 2, 2)
    with torch.no_grad():
        for t in lstm.parameters():
            t.add_(0.3 * torch.randn_like(t))
    assert torch.autograd.gradcheck(lstm, (s,))


def test_the_initialization_of_appendix_a23():
    """Every weight scaling vector starts at 0.1 and stays there until W_hh moves from zero; one step of Adam on any
    loss moves it, so the scaling becomes a function of the input."""
    torch.manual_seed(1)
    cell = M.HyperLSTM(5, 8, 4, 4)
    x = torch.randn(3, 6, 5, dtype=torch.double)
    _, ds = cell(x, return_scales=True)
    assert torch.allclose(ds, torch.full_like(ds, 0.1))
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    out = cell(x)
    (out**2).sum().backward()
    opt.step()
    _, ds = cell(x, return_scales=True)
    assert ds.std() > 1e-4


def test_dropout_is_inside_the_main_lstm_only():
    """Eq. 13: dropout on phi(g_t) of the main LSTM; in eval mode the output does not depend on it."""
    torch.manual_seed(2)
    cell = M.HyperLSTM(3, 6, 4, 2, dropout=0.5)
    x = torch.randn(2, 5, 3, dtype=torch.double)
    cell.eval()
    a = cell(x)
    cell.dropout = 0.0
    assert torch.allclose(a, cell(x))


def bars(g, n, S=8, noise=0.3):
    """Images of 8 x 8 pixels with a bar of 3 pixels at a random place, horizontal, vertical, or along either
    diagonal, the class, under Gaussian noise of standard deviation 0.3."""
    y = torch.randint(0, 4, (n,), generator=g)
    r, c = torch.randint(1, S - 1, (n,), generator=g), torch.randint(1, S - 1, (n,), generator=g)
    s = torch.arange(-1, 2)
    step = torch.tensor([[0, 1], [1, 0], [1, 1], [1, -1]])[y]  # (n, 2)
    rows, cols = r[:, None] + s * step[:, :1], c[:, None] + s * step[:, 1:]
    x = torch.zeros(n, 1, S, S, dtype=torch.double)
    x[torch.arange(n)[:, None], 0, rows, cols] = 1.0
    return x + noise * torch.randn(x.shape, generator=g, dtype=torch.double), y


def train(model, seed, steps=300):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = bars(g, 64)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = bars(torch.Generator().manual_seed(99), 512)
    with torch.no_grad():
        return (model(x).argmax(-1) == y).double().mean().item()


def test_generated_kernels_learn_with_fewer_parameters():
    """Four generated convolutions of 8 channels, each tiled from 2 x 2 basic kernels of 4 channels, N_z = 4: 472
    parameters, against 2,452 for the same network with ordinary kernels. With seed 0 they reach 0.896 and 0.762 on
    512 held-out images, where chance is 0.25; the Experiments section of the page has seeds 0 to 2."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    hyper = M.HyperConvNet(1, 4, 2, 3, 4, 4, 4)
    torch.manual_seed(0)
    plain = M.ConvNet(1, 8, 3, 4, 4)
    assert sum(t.numel() for t in hyper.parameters()) == 472
    assert sum(t.numel() for t in plain.parameters()) == 2452
    acc_hyper, acc_plain = train(hyper, 0), train(plain, 0)
    assert acc_hyper > 0.7 and acc_plain > 0.7, (acc_hyper, acc_plain)
