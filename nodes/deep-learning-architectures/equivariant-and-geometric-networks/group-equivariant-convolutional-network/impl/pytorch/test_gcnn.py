# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://1REXCC).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, equivariance and invariance."""

import json
import pathlib

import numpy as np
import pytest
import torch

from gcnn import NAMES, config, forward, gconv, init, lift, loss_fn, param_count, rotate, rotate_p4

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["c_in"], k["c1"], k["c2"], k["k"], k["n_classes"], **kw)


def test_shared_fixture():
    params = {k: torch.tensor(v, requires_grad=True) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    x = torch.tensor(inp["x"], requires_grad=True)
    labels = torch.tensor(inp["labels"])
    np.testing.assert_allclose(forward(params, x, cfg).detach().numpy(), want["logits"], rtol=1e-12, atol=1e-14)
    loss = loss_fn(params, x, labels, cfg)
    assert loss.item() == pytest.approx(float(want["loss"]), rel=1e-12)
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(params[k].grad.numpy(), want[f"d{k}"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(torch.Generator().manual_seed(0), cfg)
    x = torch.zeros(k["B"], k["c_in"], k["N"], k["N"], dtype=torch.float64)
    h1 = lift(x, params["W1"])
    assert list(h1.shape) == c["blocks"]["lift"]["outputs"]["h1"]
    assert list(gconv(h1, params["W2"]).shape) == c["blocks"]["gconv"]["outputs"]["h2"]
    assert list(forward(params, x, cfg).shape) == c["blocks"]["head"]["outputs"]["logits"]
    assert param_count(cfg) == c["model"]["params"] == sum(v.numel() for v in params.values())


@pytest.mark.parametrize("mode", ["zero", "circular"])
def test_layers_are_equivariant_and_logits_invariant(mode):
    gen = torch.Generator().manual_seed(1)
    cfg = config(2, 3, 2, 3, 4, mode=mode)
    params = init(gen, cfg)
    x = torch.randn(2, 2, 6, 6, generator=gen, dtype=torch.float64)
    h1 = lift(x, params["W1"], mode)
    for r in range(1, 4):
        torch.testing.assert_close(lift(rotate(x, r), params["W1"], mode), rotate_p4(h1, r), rtol=0, atol=1e-12)
        torch.testing.assert_close(gconv(rotate_p4(h1, r), params["W2"], mode), rotate_p4(gconv(h1, params["W2"], mode), r),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(forward(params, rotate(x, r), cfg), forward(params, x, cfg), rtol=0, atol=1e-12)


def test_float32_training_reduces_loss():
    gen = torch.Generator().manual_seed(2)
    cfg = config(1, 4, 4, 3, 2)
    params = {k: v.float().requires_grad_() for k, v in init(gen, cfg).items()}
    x = torch.randn(8, 1, 6, 6, generator=gen)
    x[4:] = torch.rot90(x[:4], 1, dims=(2, 3)).flip(-1)  # the mirror images, which p4 does not relate
    labels = torch.tensor([0] * 4 + [1] * 4)
    opt = torch.optim.Adam(params.values(), lr=0.05)
    first = loss_fn(params, x, labels, cfg).item()
    for _ in range(150):
        opt.zero_grad()
        loss_fn(params, x, labels, cfg).backward()
        opt.step()
    assert loss_fn(params, x, labels, cfg).item() < 0.5 * first
