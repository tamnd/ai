# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, gradcheck, XOR and training."""

import json
import pathlib

import numpy as np
import pytest
import torch

from multilayer_perceptron import MLP, config, mse_loss, param_count, sgd_step

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
X_XOR = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T_XOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


def cfg_of(c, activation="tanh"):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["d_out"], k["n_hidden"], activation)


def test_shared_fixture():
    raw = dict(np.load(FIXTURES / "params-small.npz"))
    params = {k: [raw[f"{k}/{i}"] for i in range(3)] for k in ("W", "b")}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    model = MLP.from_params(params, cfg_of(SHAPES[0]))
    x = torch.tensor(inp["x"], requires_grad=True)
    y = model(x)
    np.testing.assert_allclose(y.detach().numpy(), want["y"], rtol=1e-10, atol=1e-12)
    loss = mse_loss(y, torch.as_tensor(inp["t"]))
    assert loss.item() == pytest.approx(float(want["loss"]), rel=1e-10)
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), want["dx"], rtol=1e-10, atol=1e-12)
    for i in range(3):
        np.testing.assert_allclose(model.W[i].grad.numpy(), want[f"dW/{i}"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(model.b[i].grad.numpy(), want[f"db/{i}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    model = MLP(cfg, torch.Generator().manual_seed(0))
    y = model(torch.zeros(k["B"], k["d_in"]))
    assert list(y.shape) == c["blocks"]["output"]["outputs"]["y"]
    assert param_count(cfg) == c["model"]["params"] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu"])
def test_gradcheck(activation):
    model = MLP(config(3, 4, 2, 2, activation), torch.Generator().manual_seed(1))
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(2), requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: model(inp), (x,))


def test_xor_network():
    params = {
        "W": [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[1.0, -2.0]])],
        "b": [np.array([0.0, -1.0]), np.array([0.0])],
    }
    model = MLP.from_params(params, config(2, 2, 1, 1, "relu"))
    torch.testing.assert_close(model(X_XOR), T_XOR, rtol=0, atol=0)


def test_linear_collapse():
    model = MLP(config(3, 5, 2, 2, "identity"), torch.Generator().manual_seed(3))
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(4))
    W = model.W[2] @ model.W[1] @ model.W[0]
    b = model.W[2] @ (model.W[1] @ model.b[0] + model.b[1]) + model.b[2]
    torch.testing.assert_close(model(x), x @ W.T + b)


def test_batch_independence():
    model = MLP(config(3, 5, 2, 2, "gelu"), torch.Generator().manual_seed(5))
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(model(x), torch.cat([model(x[i : i + 1]) for i in range(5)]))


def test_training_reduces_loss():
    model = MLP(config(2, 4, 1, 1, "tanh"), torch.Generator().manual_seed(0))
    first = None
    for _ in range(500):
        loss = mse_loss(model(X_XOR), T_XOR)
        first = loss.item() if first is None else first
        loss.backward()
        sgd_step(model, 0.5)
    assert mse_loss(model(X_XOR), T_XOR).item() < 0.01 * first


def test_config_errors():
    with pytest.raises(ValueError):
        config(3, 0, 2, 1)
    with pytest.raises(ValueError):
        config(3, 4, 2, 1, "softplus")
