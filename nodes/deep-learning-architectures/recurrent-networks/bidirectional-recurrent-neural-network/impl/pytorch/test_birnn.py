# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Tests of the PyTorch version: the shared fixture and its gradients, torch.nn.LSTM and torch.nn.RNN with
bidirectional=True on packed sequences, gradcheck, and a task that needs the future."""

import json
import pathlib

import numpy as np
import pytest
import torch

from birnn import BiRNN, config, nll, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return config(k["input_size"], k["hidden_size"], k["n_layers"], k["n_classes"], "lstm", k["directions"] == 2)


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    model = BiRNN.from_params(params, config())
    x = inp["x"].clone().requires_grad_(True)
    logits = model(x, inp["lengths"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    loss = nll(logits, inp["labels"], inp["lengths"])
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=0)
    loss.backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)
    torch.testing.assert_close(x.grad, want["dx"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        model = BiRNN(cfg)
        assert sum(p.numel() for p in model.parameters()) == c["model"]["params"]
        k = c["config"]
        logits = model(torch.randn(k["B"], k["T"], k["input_size"]))
        assert list(logits.shape) == c["blocks"]["output"]["outputs"]["logits"]
        # torch.nn.LSTM keeps two biases per direction and layer where this network has one.
        D = 2 if cfg["bidirectional"] else 1
        extra = cfg["n_layers"] * D * 4 * cfg["hidden_size"]
        lib = sum(p.numel() for p in model.to_torch().parameters())
        assert lib == c["model"]["params"] - (D * cfg["hidden_size"] * cfg["n_classes"] + cfg["n_classes"]) + extra


@pytest.mark.parametrize("cell", ["lstm", "tanh"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_matches_torch_on_packed_sequences(cell, bidirectional):
    torch.manual_seed(0)
    cfg = config(5, 6, 3, 4, cell, bidirectional)
    model = BiRNN(cfg, scale=0.5)
    lengths = torch.tensor([7, 5, 5, 2, 1])
    x = torch.randn(5, 7, 5)
    net = model.to_torch()
    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    out, _ = net(packed)
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=7)
    torch.testing.assert_close(model.features(x, lengths), out, rtol=1e-12, atol=1e-13)
    # Without packing, torch.nn.LSTM starts the backward direction at step T, inside the padding of short sequences.
    if bidirectional:
        padded, _ = net(x)
        assert (padded[3, :2] - out[3, :2]).abs().max() > 1e-3


def test_gradcheck():
    torch.manual_seed(1)
    cfg = config(2, 3, 2, 3)
    model = BiRNN(cfg, scale=0.5)
    names = [n for n, _ in model.named_parameters()]
    ps = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    x, labels, lengths = torch.randn(3, 4, 2), torch.randint(0, 3, (3, 4)), torch.tensor([4, 2, 3])

    def f(*q):
        return nll(torch.func.functional_call(model, dict(zip(names, q)), (x, lengths)), labels, lengths)

    assert torch.autograd.gradcheck(f, ps)


def neighbour_task(n, T, generator):
    """Tokens in {0, 1, 2}, one-hot; the label of step t is (x_{t-1} + x_{t+1}) mod 3, a missing neighbour being 0."""
    tok = torch.randint(0, 3, (n, T), generator=generator)
    padded = torch.nn.functional.pad(tok, (1, 1))
    labels = (padded[:, :-2] + padded[:, 2:]) % 3
    return torch.nn.functional.one_hot(tok, 3).double(), labels


def train(model, delay, steps=300):
    # With delay d the network reads d more zero inputs and the label of step t is read at step t + d.
    g = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)

    def scores(x):
        x = torch.nn.functional.pad(x, (0, 0, 0, delay))
        return model(x)[:, delay:]

    for _ in range(steps):
        x, labels = neighbour_task(64, 8, g)
        loss = nll(scores(x), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, labels = neighbour_task(1000, 8, g)
        # The last step is left out: its label is x_{T-1} mod 3, which a forward network knows.
        return (scores(x).argmax(-1) == labels)[:, :-1].double().mean().item()


def test_the_future_is_needed():
    # The label needs x_{t+1}, which a forward network has not read at step t: it stays near chance (1/3), the
    # bidirectional network learns the task, and so does the forward network whose output is delayed by one step.
    # Two layers each: with one layer the two directions meet only in the affine output, which adds a score of the
    # past to a score of the future, and in this run the one-layer bidirectional network stays near chance too.
    accs = {}
    for name, L, bidirectional, delay in [("bi", 2, True, 0), ("uni", 2, False, 0), ("delay", 2, False, 1), ("bi1", 1, True, 0)]:
        torch.manual_seed(0)
        accs[name] = train(BiRNN(config(3, 16, L, 3, "lstm", bidirectional), scale=0.3), delay=delay)
    assert accs["bi"] > 0.95 and accs["delay"] > 0.95, accs
    assert accs["uni"] < 0.4 and accs["bi1"] < 0.5, accs
