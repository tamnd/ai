# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
"""Tests of the PyTorch version: the shared fixture of the NumPy reference, exact agreement with full storage, the
counts of forwards and saved activations, and agreement with torch.utils.checkpoint.checkpoint_sequential."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint_sequential

from gradient_checkpointing import Chain, saved_activations, segment_sizes

torch.set_default_dtype(torch.float64)

# The shared fixture: three layers of width 2 and a batch of 2, with the outputs of the NumPy reference.
W = [[[0.5, -0.25], [0.75, 0.5]], [[-0.5, 1.0], [0.25, 0.25]], [[1.0, 0.5], [-0.75, 0.5]]]
B = [[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]]
X = [[1.0, -1.0], [0.5, 2.0]]
Y = [[-0.16063384074179027, 0.5653530619863063], [0.6832818232403335, -0.11798989504262712]]
DX = [[-0.22916355262070015, -0.22191162909668455], [0.01690646129811597, 0.05965517658867096]]
DB = [
    [0.03557438564506993, -0.3067257121934922],
    [-0.1884971146317812, 0.19116142814553794],
    [0.2077863024765885, 0.26830532190736106],
]
DW0 = [[0.0770465591221502, -0.2844074265075516], [-0.345644802043623, 0.5402402512942772]]


def grads(model, x):
    x = x.clone().requires_grad_()
    y = model(x)
    (0.5 * (y**2).sum()).backward()
    return y.detach(), x.grad, model.W.grad, model.b.grad


@pytest.mark.parametrize("segments", [None, 1, 2, 3])
def test_shared_fixture(segments):
    model = Chain.from_params({"W": torch.tensor(W), "b": torch.tensor(B)}, segments)
    y, dx, dW, db = grads(model, torch.tensor(X))
    assert dW is not None
    for got, want in [(y, Y), (dx, DX), (db, DB), (dW[0], DW0)]:
        torch.testing.assert_close(got, torch.tensor(want), rtol=1e-12, atol=1e-14)


def random_model(n, d, segments, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = {"W": torch.randn(n, d, d, generator=g) / d**0.5, "b": 0.1 * torch.randn(n, d, generator=g)}
    return Chain.from_params(params, segments), torch.randn(2, d, generator=g)


@pytest.mark.parametrize("n", [1, 4, 9])
def test_equal_to_full_storage(n):
    full, x = random_model(n, 3, None)
    want = grads(full, x)
    assert full.forwards == n
    for k in range(1, n + 1):
        model, _ = random_model(n, 3, k)
        got = grads(model, x)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert model.forwards == 2 * n  # torch recomputes every layer of every segment


@pytest.mark.parametrize(("n", "k"), [(9, 3), (10, 3), (4, 4), (5, 1)])
def test_saved_activations(n, k):
    full, x = random_model(n, 3, None)
    _, saved_full = saved_activations(full, x)
    assert saved_full == n + 1  # h_0 to h_n
    model, _ = random_model(n, 3, k)
    _, saved = saved_activations(model, x)
    assert saved == k  # the input of each segment


def test_agrees_with_checkpoint_sequential():
    full, x = random_model(10, 3, None)
    model, _ = random_model(10, 3, 3)
    fns = [lambda h, i=i: full.layer(i, h) for i in range(10)]
    xr = x.clone().requires_grad_()
    y = checkpoint_sequential(fns, 3, xr, use_reentrant=False)
    (0.5 * (y**2).sum()).backward()
    want = (y.detach(), xr.grad, full.W.grad, full.b.grad)
    for a, b in zip(grads(model, x), want, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-15)


def test_segment_sizes_and_errors():
    assert segment_sizes(10, 3) == [4, 3, 3]
    assert segment_sizes(9, 9) == [1] * 9
    with pytest.raises(ValueError):
        Chain(4, 2, 5)
    with pytest.raises(ValueError):
        Chain(4, 2, 0)
    with pytest.raises(ValueError, match="at least 1"):
        Chain(0, 2)
    with pytest.raises(ValueError, match="at least 1"):
        Chain(3, 0)
    assert Chain(1, 1, 1).W.shape == (1, 1, 1)
