# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://3FZB17).
"""Tests of the PyTorch reference: the worked example and the shared fixture of the NumPy tests, autograd against the
closed-form gradient, its orthogonality to the input, the scaling invariance, pRMSNorm, and agreement with
torch.nn.functional.rms_norm."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rms_normalization import RMSNorm, input_gradient, partial_count

torch.set_default_dtype(torch.float64)

# The worked example of the page.
WA = [4.0, 0.0, -2.0, 4.0]
WY = [4 / 3, 0.0, -2 / 3, 4 / 3]
WDX = [5 / 27, 0.0, 2 / 27, -4 / 27]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 1.0], [1.0, 3.0, -2.5, 0.0], [-1.25, 0.5, 1.5, 4.0]]
DY = [[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 3.0], [0.25, -0.5, 2.0, -1.0]]
GAIN = [1.5, 0.5, 2.0, -1.0]
Y = [[0.547722411446, -0.547722411446, 2.921186194378, -0.730296548595],
     [0.74420831594, 0.74420831594, -2.480694386468, 0.0],
     [-0.837218274818, 0.111629103309, 1.339549239709, -1.786065652945]]
DX = [[1.351048478578, -0.036514418464, -0.438178474444, 0.146059037075],
      [-1.442619208327, 0.385461709308, -0.114493558885, -1.488416631881],
      [0.429128650266, -0.21630310143, 1.472043658581, -0.390875571733]]
DGAIN = [-0.766665859426, -0.814102117212, -0.12104385748, -1.420917378648]


def t(v):
    return torch.as_tensor(v, dtype=torch.float64)


def layer(n=4, eps=1e-6, p=1.0, gain=None):
    m = RMSNorm(n, eps=eps, p=p)
    if gain is not None:
        with torch.no_grad():
            m.gain.copy_(t(gain))
    return m


def autograd(m, x, dy):
    x = x.clone().requires_grad_(True)
    y = m(x)
    (y * dy).sum().backward()
    return y.detach(), x.grad


def test_worked_example():
    m = layer(eps=0.0)
    y, dx = autograd(m, t(WA), t([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(y, t(WY), atol=1e-12, rtol=0)
    torch.testing.assert_close(dx, t(WDX), atol=1e-12, rtol=0)
    torch.testing.assert_close(m.gain.grad, t([4 / 3, 0.0, 0.0, 0.0]), atol=1e-12, rtol=0)
    torch.testing.assert_close(layer(eps=0.0, p=0.5)(t(WA)), t(WA) / 8 ** 0.5, atol=1e-12, rtol=0)


def test_shared_fixture():
    m = layer(gain=GAIN)
    y, dx = autograd(m, t(X), t(DY))
    torch.testing.assert_close(y, t(Y), atol=1e-11, rtol=0)
    torch.testing.assert_close(dx, t(DX), atol=1e-11, rtol=0)
    torch.testing.assert_close(m.gain.grad, t(DGAIN), atol=1e-11, rtol=0)


@pytest.mark.parametrize("p", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("eps", [0.0, 1e-3, 0.5])
def test_autograd_matches_closed_form(p, eps):
    g = torch.Generator().manual_seed(20261116)
    x, dy, gain = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g), torch.randn(8, generator=g)
    m = layer(8, eps=eps, p=p, gain=gain)
    _, dx = autograd(m, x, dy)
    torch.testing.assert_close(dx, input_gradient(x, dy, gain, eps=eps, p=p), atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda v: m(v), (x[:2].clone().requires_grad_(True),))


def test_no_gain():
    m = RMSNorm(4, eps=1e-3, elementwise_affine=False)
    assert m.gain is None and len(list(m.parameters())) == 0
    x = t(X)
    torch.testing.assert_close(m(x), x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-3))
    _, dx = autograd(m, x, t(DY))
    torch.testing.assert_close(dx, input_gradient(x, t(DY), None, eps=1e-3), atol=1e-12, rtol=1e-10)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=2, max_size=10),
       st.lists(st.floats(-10, 10, allow_nan=False), min_size=10, max_size=10),
       st.floats(1e-2, 1e2), st.floats(0.05, 1.0))
def test_orthogonal_and_invariant(a, d, delta, p):
    n = len(a)
    x, dy = t(a), t(d[:n])
    m = layer(n, eps=0.0, p=p)
    if x[: m.k].pow(2).mean() < 1e-3:
        return
    y, dx = autograd(m, x, dy)
    assert abs(float(x @ dx)) <= 1e-9 * (1 + float(x.abs().sum() * (dx.abs().sum() + dy.abs().sum())))
    torch.testing.assert_close(m(delta * x).detach(), y, atol=1e-9, rtol=1e-9)
    _, dx2 = autograd(m, delta * x, dy)
    torch.testing.assert_close(dx2, dx / delta, atol=1e-9, rtol=1e-8)


def test_partial_uses_only_first_k():
    m = layer(eps=1e-6, p=0.5)
    assert m.k == 2
    x = t(X)
    x2 = x.clone()
    x2[:, 2:] *= 10.0
    torch.testing.assert_close(m(x2)[:, :2], m(x)[:, :2], atol=1e-12, rtol=0)
    r = torch.sqrt(x[:, :2].pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(m(x), x / r, atol=1e-12, rtol=0)


@pytest.mark.parametrize("eps", [1e-6, 0.3])
def test_matches_torch_rms_norm(eps):
    m = layer(eps=eps, gain=GAIN)
    ref = torch.nn.functional.rms_norm(t(X), (4,), t(GAIN), eps=eps)
    torch.testing.assert_close(m(t(X)), ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(("n", "p", "k"), [(16, 0.0625, 1), (10, 0.3, 3), (5, 1.0, 5), (7, 0.5, 4), (3, 1e-12, 1),
                                           (512, 0.0625, 32)])
def test_partial_count(n, p, k):
    assert partial_count(n, p) == k


def test_errors_and_defaults():
    with pytest.raises(ValueError, match="p must be in"):
        partial_count(4, 0.0)
    with pytest.raises(ValueError, match="p must be in"):
        partial_count(4, 1.5)
    with pytest.raises(ValueError, match="n must be positive"):
        partial_count(0, 0.5)
    with pytest.raises(ValueError, match="eps must be non-negative"):
        RMSNorm(4, eps=-1e-3)
    assert RMSNorm(4, eps=0.0).eps == 0.0
    with pytest.raises(ValueError, match="last axis of x must have size 4"):
        RMSNorm(4)(torch.ones(2, 3))
    with pytest.raises(ValueError, match="last axis"):
        RMSNorm(1)(torch.tensor(1.0))
    m = RMSNorm(4)
    assert m.eps == 1e-6 and m.k == 4 and m.n == 4
    torch.testing.assert_close(m.gain.detach(), torch.ones(4))
    torch.testing.assert_close(input_gradient(t(WA), t([1.0, 0, 0, 0]), eps=0.0), t(WDX), atol=1e-12, rtol=0)
