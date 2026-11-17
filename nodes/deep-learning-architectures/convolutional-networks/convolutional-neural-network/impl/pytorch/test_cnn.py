# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""Tests of the PyTorch version: agreement with torch.nn.functional.conv2d, the shared fixture, autograd against
a numeric gradient, equivariance and the shape contract."""

import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cnn import CNN, config, conv2d, param_count

torch.set_default_dtype(torch.float64)
HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
NAMES = ["conv1", "pool1", "conv2", "pool2"]


def cfg_of(c):
    k = c["config"]
    return config(k["channels"], k["c1"], k["c2"], k["f1"], k["f2"], k["classes"], k["size"])


def test_worked_example():
    x = torch.arange(9.0).reshape(1, 1, 3, 3)
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
    assert conv2d(x, w)[0, 0].tolist() == [[27.0, 37.0], [57.0, 67.0]]
    assert conv2d(x, w, pad=1, stride=2)[0, 0].tolist() == [[0.0, 11.0], [30.0, 67.0]]


@pytest.mark.parametrize(("k", "s", "p", "mode"), [(1, 1, 0, "zeros"), (3, 1, 1, "zeros"), (3, 2, 1, "zeros"),
                                                   (5, 3, 2, "zeros"), (3, 1, 1, "circular"), (4, 2, 2, "circular")])
def test_matches_torch_conv2d(k, s, p, mode):
    g = torch.Generator().manual_seed(k * 100 + s * 10 + p)
    x, w, b = torch.randn(2, 3, 9, 7, generator=g), torch.randn(4, 3, k, k, generator=g), torch.randn(4, generator=g)
    if mode == "zeros":
        ref = F.conv2d(x, w, b, stride=s, padding=p)
    else:
        ref = F.conv2d(F.pad(x, (p, p, p, p), mode="circular"), w, b, stride=s)
        layer = torch.nn.Conv2d(3, 4, k, stride=s, padding=p, padding_mode="circular")
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)
        torch.testing.assert_close(layer(x), ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(conv2d(x, w, b, s, p, mode), ref, rtol=1e-12, atol=1e-12)


def test_gradient_numeric():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 2, 6, 6, generator=g, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, generator=g, requires_grad=True)
    b = torch.randn(3, generator=g, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x_, w_, b_: conv2d(x_, w_, b_, 2, 1), (x, w, b))


def test_equivariance_circular():
    g = torch.Generator().manual_seed(8)
    x, w = torch.randn(1, 2, 8, 8, generator=g), torch.randn(3, 2, 3, 3, generator=g)
    for t in [(1, 0), (2, -3), (7, 5)]:
        torch.testing.assert_close(conv2d(torch.roll(x, t, (2, 3)), w, pad=1, mode="circular"),
                                   torch.roll(conv2d(x, w, pad=1, mode="circular"), t, (2, 3)), rtol=1e-12, atol=1e-12)


def test_shared_fixture():
    flat = dict(np.load(FIXTURES / "params-small.npz"))
    inputs, expected = dict(np.load(FIXTURES / "input-small.npz")), dict(np.load(FIXTURES / "expected-small.npz"))
    params = {}
    for path, v in flat.items():
        layer, leaf = path.split("/")
        params.setdefault(layer, {})[leaf] = v
    cfg = cfg_of(next(c for c in SHAPES if c["kind"] == "example"))
    logits, cache = CNN.from_params(params, cfg)(torch.as_tensor(inputs["x"]))
    torch.testing.assert_close(logits, torch.as_tensor(expected["logits"]), rtol=1e-10, atol=1e-12)
    for name in NAMES:
        torch.testing.assert_close(cache[name], torch.as_tensor(expected[name]), rtol=1e-10, atol=1e-12)
    lx, lw, lb = (torch.as_tensor(inputs[k]) for k in ("layer_x", "layer_w", "layer_b"))
    torch.testing.assert_close(conv2d(lx, lw, lb, 2, 1), torch.as_tensor(expected["layer_zeros"]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(conv2d(lx, lw, lb, 2, 1, "circular"), torch.as_tensor(expected["layer_circular"]),
                               rtol=1e-10, atol=1e-12)


def test_shape_contract():
    for c in SHAPES:
        cfg = cfg_of(c)
        net = CNN(cfg)
        assert param_count(cfg) == sum(p.numel() for p in net.parameters()) == c["model"]["params"], c["name"]
        logits, cache = net(torch.randn(*c["blocks"]["conv1"]["inputs"]["x"]))
        assert list(logits.shape) == c["blocks"]["fc3"]["outputs"]["logits"]
        for name in NAMES:
            assert list(cache[name].shape) == c["blocks"][name]["outputs"]["y"], (c["name"], name)


def test_config_defaults_and_errors():
    cfg = config()
    assert cfg == {"channels": 1, "c1": 6, "c2": 16, "f1": 120, "f2": 84, "classes": 10, "size": 32}
    assert param_count(cfg) == 61706 == sum(p.numel() for p in CNN(cfg).parameters())
    assert config(size=16)["size"] == 16
    for bad in ({"size": 12}, {"size": 14}, {"size": 15}, {"channels": 0}, {"classes": 0}):
        with pytest.raises(ValueError):
            config(**bad)
