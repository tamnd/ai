# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Tests of the NumPy reference: shapes and counts, the two directions, the context of each output, padding and the
gradient of eq-loss."""

import json
import pathlib

import numpy as np
import pytest

from birnn import (
    backward,
    config,
    flops,
    forward,
    init,
    loss,
    nll,
    param_count,
    param_shapes,
    reverse_padded,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, cell="lstm"):
    k = c["config"]
    return config(k["input_size"], k["hidden_size"], k["n_layers"], k["n_classes"], cell, k["directions"] == 2)


def batch(rng, cfg, B, T):
    return rng.normal(size=(B, T, cfg["input_size"])), rng.integers(0, cfg["n_classes"], (B, T))


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    assert sum(int(np.prod(s)) for s in param_shapes(cfg).values()) == c["model"]["params"]
    if c["kind"] != "named":
        k = c["config"]
        assert flops(cfg, k["B"], k["T"]) == c["model"]["flops_forward"]
        rng = np.random.default_rng(0)
        x, _ = batch(rng, cfg, k["B"], k["T"])
        logits, cache = forward(init(rng, cfg), cfg, x)
        assert list(logits.shape) == c["blocks"]["output"]["outputs"]["logits"]
        assert list(cache["top"].shape) == c["blocks"]["output"]["inputs"]["top"]


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    logits, cache = forward(params, cfg, inp["x"], inp["lengths"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(nll(logits, inp["labels"], inp["lengths"]), want["loss"], rtol=1e-13)
    grads, dx = backward(params, cfg, cache, inp["labels"])
    for k, v in grads.items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-12, atol=1e-14, err_msg=k)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-12, atol=1e-14)


def test_reverse_padded_is_an_involution():
    x = np.arange(12).reshape(2, 6)
    r = reverse_padded(x, [6, 3])
    assert r.tolist() == [[5, 4, 3, 2, 1, 0], [8, 7, 6, 9, 10, 11]]
    assert np.array_equal(reverse_padded(r, [6, 3]), x)


@pytest.mark.parametrize("cell", ["lstm", "tanh"])
def test_each_direction_alone_is_a_unidirectional_rnn(cell):
    # Schuster and Paliwal: without the backward states the network is a forward RNN; without the forward states it
    # is an RNN with a reversed time axis. With zero weights a direction's state is exactly zero (tanh(0) = 0 and,
    # for an LSTM, c = sigmoid(0) c + sigmoid(0) tanh(0) = 0), so the other direction alone reaches the output.
    rng = np.random.default_rng(1)
    bi, uni = config(cell=cell, n_layers=1), config(cell=cell, n_layers=1, bidirectional=False)
    H = bi["hidden_size"]
    x, _ = batch(rng, bi, 3, 7)
    for keep, drop, rows in (("fwd", "bwd", slice(None, H)), ("bwd", "fwd", slice(H, None))):
        params = init(rng, bi, scale=0.5)
        for n in "WUb":
            params[f"l0_{drop}_{n}"] = np.zeros_like(params[f"l0_{drop}_{n}"])
        one = {f"l0_fwd_{n}": params[f"l0_{keep}_{n}"] for n in "WUb"}
        one |= {"out_W": params["out_W"][rows], "out_b": params["out_b"]}
        seq = x if keep == "fwd" else x[:, ::-1]
        want = forward(one, uni, seq)[0]
        want = want if keep == "fwd" else want[:, ::-1]
        np.testing.assert_allclose(forward(params, bi, x)[0], want, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_context_of_each_output(bidirectional):
    # A unidirectional output at step t depends on x_1..x_t only; a bidirectional one on the whole sequence.
    rng = np.random.default_rng(2)
    cfg = config(bidirectional=bidirectional)
    params = init(rng, cfg, scale=0.5)
    x, _ = batch(rng, cfg, 1, 6)
    base = forward(params, cfg, x)[0]
    for s in range(6):
        y = x.copy()
        y[0, s] += 1.0
        changed = np.abs(forward(params, cfg, y)[0] - base).max(axis=-1)[0] > 1e-12
        want = [True] * 6 if bidirectional else [t >= s for t in range(6)]
        assert changed.tolist() == want, (s, changed)


@pytest.mark.parametrize("cell", ["lstm", "tanh"])
def test_padding(cell):
    # A padded sequence gives the outputs of the same sequence alone, whatever the padding holds: the backward
    # direction starts at its last real step. Reading the padding as data changes the outputs.
    rng = np.random.default_rng(3)
    cfg = config(cell=cell)
    params = init(rng, cfg, scale=0.5)
    x, labels = batch(rng, cfg, 3, 7)
    lengths = np.array([7, 4, 1])
    logits = forward(params, cfg, x, lengths)[0]
    total = 0.0
    for b, n in enumerate(lengths):
        alone = forward(params, cfg, x[b : b + 1, :n])[0]
        np.testing.assert_allclose(logits[b, :n], alone[0], rtol=1e-13, atol=1e-14)
        total += nll(alone, labels[b : b + 1, :n])
    np.testing.assert_allclose(nll(logits, labels, lengths), total / 3, rtol=1e-13)
    y = x.copy()
    y[1, 4:] = 1e3
    np.testing.assert_allclose(forward(params, cfg, y, lengths)[0], logits, rtol=0, atol=0)
    assert np.abs(forward(params, cfg, y)[0][1, :4] - logits[1, :4]).max() > 1e-3


@pytest.mark.parametrize("cell,bidirectional", [("lstm", True), ("tanh", True), ("lstm", False)])
def test_gradient(cell, bidirectional):
    rng = np.random.default_rng(4)
    cfg = config(2, 3, 2, 4, cell, bidirectional)
    params = init(rng, cfg, scale=0.5)
    x, labels = batch(rng, cfg, 3, 5)
    lengths = np.array([5, 3, 2])
    logits, cache = forward(params, cfg, x, lengths)
    grads, dx = backward(params, cfg, cache, labels)
    eps = 1e-6
    for name, p in params.items():
        num = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = loss(params, cfg, x, labels, lengths)
            p[idx] = old - eps
            down = loss(params, cfg, x, labels, lengths)
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grads[name], num, rtol=1e-5, atol=1e-8, err_msg=name)
    num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        up = loss(params, cfg, x, labels, lengths)
        x[idx] = old - eps
        down = loss(params, cfg, x, labels, lengths)
        x[idx] = old
        num[idx] = (up - down) / (2 * eps)
    np.testing.assert_allclose(dx, num, rtol=1e-5, atol=1e-8)
    assert np.all(dx[1, 3:] == 0) and np.all(dx[2, 2:] == 0)  # no gradient reaches the padding


def test_worked_example():
    # The worked example of the page: one tanh unit per direction, input weight 1, recurrent weight 1/2, no bias,
    # on x = (1, 0, -1); the score is the sum of the two states.
    cfg = config(1, 1, 1, 2, "tanh", True)
    params = {f"l0_{d}_{n}": np.array(v) for d in ("fwd", "bwd") for n, v in (("W", [[1.0]]), ("U", [[0.5]]), ("b", [0.0]))}
    params |= {"out_W": np.array([[1.0, 0.0], [1.0, 0.0]]), "out_b": np.zeros(2)}
    x = np.array([[[1.0], [0.0], [-1.0]]])
    logits, cache = forward(params, cfg, x)
    np.testing.assert_allclose(cache["top"][0, :, 0], [0.7616, 0.3634, -0.6741], atol=1e-4)
    np.testing.assert_allclose(cache["top"][0, :, 1], [0.6741, -0.3634, -0.7616], atol=1e-4)
    np.testing.assert_allclose(logits[0, :, 0], [1.4357, 0.0, -1.4357], atol=1e-4)
    # A fourth, padded step holding 5: with the length the outputs do not change; read as data, it reaches step 1.
    x4 = np.concatenate([x, [[[5.0]]]], axis=1)
    np.testing.assert_allclose(forward(params, cfg, x4, [3])[0][0, :3], logits[0], rtol=0, atol=1e-15)
    _, wrong = forward(params, cfg, x4)
    np.testing.assert_allclose(wrong["top"][0, :3, 1], [0.7096, -0.2270, -0.4622], atol=1e-4)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    x, _ = batch(rng, cfg, k["B"], k["T"])
    return forward, (init(rng, cfg), cfg, x)


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_bad_configs():
    with pytest.raises(ValueError):
        config(cell="gru")
    with pytest.raises(ValueError):
        config(n_classes=1)
    with pytest.raises(ValueError):
        forward(init(np.random.default_rng(0), config()), config(), np.zeros((2, 3, 3)), [3, 0])
