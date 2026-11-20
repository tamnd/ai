# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradients against finite
differences, the guards of the gradients, and a short training run on a copy task."""

import json
import pathlib

import numpy as np
import pytest
import torch

from ntm import bits_per_sequence, config, cosine, forward, init, param_count, sharpen

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("x_dim", "y_dim", "hidden", "n_loc", "width", "n_read", "n_write", "shift")


def cfg_of(c):
    return config(**{k: c["config"][k] for k in KEYS})


def copy_batch(gen, batch, items, bits):
    t = 2 * items + 1
    seq = torch.randint(0, 2, (batch, items, bits), generator=gen).double()
    x = torch.zeros(batch, t, bits + 1, dtype=torch.float64)
    x[:, :items, :bits] = seq
    x[:, items, bits] = 1.0
    targets = torch.zeros(batch, t, bits, dtype=torch.float64)
    targets[:, items + 1 :] = seq
    mask = torch.zeros(batch, t, dtype=torch.float64)
    mask[:, items + 1 :] = 1.0
    return x, targets, mask


def test_shared_fixture():
    params = {k: torch.tensor(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits, ws, mem = forward(params, torch.tensor(inp["x"]), cfg, trace=True)
    np.testing.assert_allclose(logits.numpy(), want["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ws.numpy(), want["weights"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mem.numpy(), want["memory"], rtol=1e-10, atol=1e-12)
    loss = bits_per_sequence(logits, torch.tensor(inp["targets"]), torch.tensor(inp["mask"]))
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(torch.Generator().manual_seed(0), cfg)
    assert param_count(cfg) == c["model"]["params"] == sum(v.numel() for v in params.values())
    with torch.no_grad():
        logits, ws, mem = forward(params, torch.zeros(k["B"], k["T"], k["X"], dtype=torch.float64), cfg, trace=True)
    assert list(logits.shape) == [k["B"], k["T"], k["Y"]]
    assert list(ws.shape[2:]) == c["blocks"]["address"]["outputs"]["w"]
    assert list(mem.shape) == c["blocks"]["write"]["outputs"]["memory"]


def test_gradcheck():
    """Every parameter, through the 4 stages of the addressing, the write and the read, over 4 steps."""
    cfg = config(x_dim=2, y_dim=2, hidden=3, n_loc=4, width=2, n_read=1, n_write=1, shift=1)
    gen = torch.Generator().manual_seed(1)
    params = init(gen, cfg, scale=1.5)
    x = torch.randn(2, 4, 2, generator=gen, dtype=torch.float64)
    targets = torch.randint(0, 2, (2, 4, 2), generator=gen).double()
    mask = torch.ones(2, 4, dtype=torch.float64)

    def loss(*vals):
        return bits_per_sequence(forward(dict(zip(params, vals, strict=True)), x, cfg), targets, mask)

    assert torch.autograd.gradcheck(loss, tuple(params.values()), eps=1e-6, atol=1e-6)


def test_cosine_gradient_needs_the_floor():
    """A row of zeros in the memory: the derivative of the square root at 0 is infinite, and times the 0 of the
    inner derivative it is NaN. The floor TINY on the squared length keeps the gradient finite."""
    mem = torch.zeros(1, 3, 2, dtype=torch.float64, requires_grad=True)
    key = torch.tensor([[1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    cosine(mem, key).sum().backward()
    assert torch.isfinite(mem.grad).all() and torch.isfinite(key.grad).all()
    mem.grad = None
    naive = torch.einsum("bnm,bm->bn", mem, key) / (torch.sqrt((mem * mem).sum(-1)) * key.norm() + 1e-8)
    naive.sum().backward()
    assert torch.isnan(mem.grad).all()


def test_sharpening_a_weighting_with_zeros():
    """After a one-hot shift a weighting has exact zeros; 0 ** gamma stays 0 and its gradient is finite."""
    w = torch.tensor([[0.0, 0.25, 0.75]], dtype=torch.float64)
    gamma = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    out = sharpen(w, gamma)
    out[0, 1].backward()
    assert out[0, 0] == 0 and torch.isfinite(gamma.grad).all()


def test_training_on_a_copy_task_reduces_the_loss():
    """A few hundred Adam steps on random copy sequences of 3 items of 3 bits; the bits per sequence on held-out
    sequences fall well below the 9 bits of guessing."""
    torch.manual_seed(0)
    cfg = config(x_dim=4, y_dim=3, hidden=32, n_loc=8, width=6, n_read=1, n_write=1, shift=1)
    gen = torch.Generator().manual_seed(2)
    params = init(gen, cfg)
    opt = torch.optim.Adam(params.values(), lr=1e-2)
    test = copy_batch(torch.Generator().manual_seed(3), 64, 3, 3)
    with torch.no_grad():
        before = float(bits_per_sequence(forward(params, test[0], cfg), test[1], test[2]))
    for _ in range(400):
        x, targets, mask = copy_batch(gen, 32, 3, 3)
        loss = bits_per_sequence(forward(params, x, cfg), targets, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 10.0)
        opt.step()
    with torch.no_grad():
        after = float(bits_per_sequence(forward(params, test[0], cfg), test[1], test[2]))
    assert before > 8.0 and after < 0.5 * before
