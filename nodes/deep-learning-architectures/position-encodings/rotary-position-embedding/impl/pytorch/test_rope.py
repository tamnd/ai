# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
"""Tests of the PyTorch reference: the worked example, the shared fixture, the complex form and the gradient."""

import math

import pytest
import torch

from rope import RotaryEmbedding

torch.set_default_dtype(torch.float64)

# The worked example of the page: d = 4, theta = (1, 0.01).
Q = [[1.0, 2.0, 0.0, 1.0]]
K = [[2.0, 1.0, 1.0, 0.0]]
Q_AT_1 = [-1.1426, 1.9221, -0.0100, 1.0000]
SCORES = [
    ((0, 0), 4.0),
    ((5, 5), 4.0),
    ((1, 0), -0.3732),
    ((2, 1), -0.3732),
    ((7, 6), -0.3732),
    ((0, 1), 4.6956),
    ((3, 0), -4.4133),
    ((10, 7), -4.4133),
]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 0.25, -0.75, 1.0], [1.0, 3.0, -2.5, 0.0, 4.0, -1.0], [-1.25, 0.5, 1.5, -3.0, 2.25, 0.5]]
POS = [0.0, 3.0, 17.0]
BASE = 1000.0  # theta = (1, 0.1, 0.01) for d = 6


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def score(rope, q, k, m, n):
    return float((rope(t(q), [m]) * rope(t(k), [n])).sum())


def test_worked_example():
    rope = RotaryEmbedding.from_params({}, 4)
    torch.testing.assert_close(rope.frequencies(), t([1.0, 0.01]), rtol=1e-15, atol=0)
    torch.testing.assert_close(rope(t(Q), [1])[0], t(Q_AT_1), rtol=0, atol=5e-5)
    for (m, n), s in SCORES:
        assert round(score(rope, Q, K, m, n), 4) == s


def test_shared_fixture():
    rope = RotaryEmbedding(6, BASE)
    y = rope(t(X), POS)
    torch.testing.assert_close(y[0], t(X[0]), rtol=0, atol=0)
    for row, m in [(1, 3.0), (2, 17.0)]:
        for i, a in enumerate([m, m / 10, m / 100]):
            x1, x2 = X[row][2 * i], X[row][2 * i + 1]
            expect = t([x1 * math.cos(a) - x2 * math.sin(a), x1 * math.sin(a) + x2 * math.cos(a)])
            torch.testing.assert_close(y[row, 2 * i : 2 * i + 2], expect, rtol=1e-12, atol=1e-12)


def test_complex_form_agrees():
    rope = RotaryEmbedding(6, BASE)
    x = t(X)
    z = torch.view_as_complex(x.reshape(3, 3, 2).contiguous())
    a = torch.outer(t(POS), rope.frequencies())
    y = torch.view_as_real(z * torch.polar(torch.ones_like(a), a)).reshape(3, 6)  # eq: eq-rope-complex
    torch.testing.assert_close(rope(x, POS), y, rtol=1e-12, atol=1e-12)


def test_relative_and_norm():
    rope = RotaryEmbedding(6, BASE)
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 6, generator=g), torch.randn(1, 6, generator=g)
    for m, n, shift in [(4, 1, 100), (0, 9, -37), (250, 250, 3)]:
        a = (rope(q, [m]) * rope(k, [n])).sum()
        b = (rope(q, [m + shift]) * rope(k, [n + shift])).sum()
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(rope(q, [m]).norm(), q.norm(), rtol=1e-12, atol=0)


def test_gradient_is_inverse_rotation():
    rope = RotaryEmbedding(6, BASE)
    x = t(X).requires_grad_()
    dy = torch.arange(18.0).reshape(3, 6) / 7 - 1
    (rope(x, POS) * dy).sum().backward()
    # R^T dy is the rotation of dy by the opposite angles, that is by the negative positions.
    torch.testing.assert_close(x.grad, rope(dy, [-p for p in POS]), rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(lambda v: rope(v, POS), (t(X).requires_grad_(),))


def test_interpolation():
    plain, pi = RotaryEmbedding(6, BASE), RotaryEmbedding(6, BASE, scale=4.0)
    torch.testing.assert_close(pi(t(X), POS), plain(t(X), [p / 4 for p in POS]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("d", "base", "scale"), [(3, 10000.0, 1.0), (4, 1.0, 1.0), (4, 10.0, 0.0)])
def test_bad_config(d, base, scale):
    with pytest.raises(ValueError):
        RotaryEmbedding(d, base, scale)


def test_errors():
    with pytest.raises(ValueError):
        RotaryEmbedding(6, BASE)(t(X)[:, :4], POS)
    with pytest.raises(ValueError):
        RotaryEmbedding.from_params({"w": 1}, 4)
