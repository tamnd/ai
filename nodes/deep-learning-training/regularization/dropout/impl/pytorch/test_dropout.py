# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Tests of the PyTorch reference: the shared fixture, the gradient, and agreement with torch.nn.Dropout."""

import pytest
import torch
from dropout import Dropout

torch.set_default_dtype(torch.float64)

# The worked example of the page.
H = [2.0, 4.0, -1.0, 3.0]
EXAMPLES = [
    (0.5, [1, 0, 1, 1], [4.0, 0.0, -2.0, 6.0], [2.0, 0.0, 2.0, 2.0]),
    (0.2, [1, 1, 0, 1], [2.5, 5.0, 0.0, 3.75], [1.25, 1.25, 0.0, 1.25]),
]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]]
M = [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 1]]
DY = [[1.0, 2.0, -1.0, 0.5, 3.0], [-2.0, 1.0, 0.0, 4.0, -1.0], [0.25, -0.5, 2.0, 1.0, -3.0]]
P = 0.4


def t(v):
    return torch.tensor(v, dtype=torch.float64)


@pytest.mark.parametrize(("p", "mask", "y", "dx"), EXAMPLES)
def test_worked_example(p, mask, y, dx):
    h = t(H).requires_grad_()
    out = Dropout(p)(h, mask=t(mask))
    torch.testing.assert_close(out, t(y), rtol=1e-12, atol=0)
    out.sum().backward()
    torch.testing.assert_close(h.grad, t(dx), rtol=1e-12, atol=0)


def test_shared_fixture():
    # The forward pass and the autograd gradient match the NumPy forward and backward: y = X M / (1 - p), dx = DY M / (1 - p).
    x = t(X).requires_grad_()
    y = Dropout(P)(x, mask=t(M))
    torch.testing.assert_close(y, t(X) * t(M) / 0.6, rtol=1e-10, atol=1e-12)
    (y * t(DY)).sum().backward()
    torch.testing.assert_close(x.grad, t(DY) * t(M) / 0.6, rtol=1e-8, atol=1e-10)


def test_gradient():
    layer = Dropout(P)
    mask = t(M)
    x = t(X).requires_grad_()
    assert torch.autograd.gradcheck(lambda v: layer(v, mask=mask), (x,))


def test_eval_is_identity():
    layer = Dropout(0.5).eval()
    x = t(X)
    torch.testing.assert_close(layer(x), x, rtol=0, atol=0)


def test_masks_differ():
    layer = Dropout(0.5)
    g = torch.Generator().manual_seed(1)
    x = torch.ones(64)
    assert not torch.equal(layer(x, generator=g), layer(x, generator=g))


def test_expectation():
    # The tolerance is 5 standard errors.
    p, n = 0.3, 200_000
    g = torch.Generator().manual_seed(2)
    h = t(H)
    y = Dropout(p)(h.expand(n, 4), generator=g)
    se = h.abs() * (p / (1 - p) / n) ** 0.5
    assert torch.all((y.mean(dim=0) - h).abs() <= 5 * se)


def test_agrees_with_torch_dropout():
    # torch.nn.Dropout uses the same convention: p is the probability to drop, and the kept entries are scaled by 1 / (1 - p).
    torch.manual_seed(3)
    p = 0.3
    x = torch.ones(100_000)
    ours = Dropout(p)(x)
    theirs = torch.nn.Dropout(p)(x)
    assert set(torch.unique(theirs).tolist()) == set(torch.unique(ours).tolist()) == {0.0, 1 / (1 - p)}
    assert abs((ours == 0).double().mean().item() - (theirs == 0).double().mean().item()) < 0.01
    assert torch.equal(torch.nn.Dropout(p).eval()(x), Dropout(p).eval()(x))


def test_functional_dropout_needs_training_flag():
    # The pitfall of the page: torch.nn.functional.dropout drops units unless training=False is given.
    torch.manual_seed(4)
    x = torch.ones(1000)
    assert torch.any(torch.nn.functional.dropout(x, 0.5) == 0)
    assert torch.equal(torch.nn.functional.dropout(x, 0.5, training=False), x)


def test_p_zero_is_identity():
    x = t(X)
    assert torch.equal(Dropout(0.0)(x), x)


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_invalid_p(p):
    with pytest.raises(ValueError):
        Dropout(p)


def test_from_params():
    assert Dropout.from_params({}, 0.25).p == 0.25
    with pytest.raises(ValueError):
        Dropout.from_params({"w": 1}, 0.25)


def test_mask_shape():
    with pytest.raises(ValueError):
        Dropout(0.5)(t(X), mask=torch.ones(3))
