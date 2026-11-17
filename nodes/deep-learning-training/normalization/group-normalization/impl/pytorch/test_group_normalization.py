# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
"""Tests of the PyTorch reference: the worked example and the shared fixture of the NumPy tests, autograd against the
closed-form gradient, the special cases G = 1 and G = C, batch normalization with one example, independence from the
batch, and agreement with torch.nn.GroupNorm, torch.nn.InstanceNorm2d and torch.nn.LayerNorm."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from group_normalization import GroupNorm, input_gradient, normalize_over

torch.set_default_dtype(torch.float64)

# The worked example of the page.
WX = [[[0.0, 0.0], [4.0, 4.0], [0.0, 6.0], [6.0, 8.0]]]
WY = [[[-1.0, -1.0], [1.0, 1.0], [-5 / 3, 1 / 3], [1 / 3, 1.0]]]
WDY = [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]
WDX = [[[0.0, 0.0], [0.0, 0.0], [1 / 54, -2 / 54], [-2 / 54, 3 / 54]]]

# The shared fixture of the NumPy tests.
X = [[[float(((3 * c + s + 12 * n) * 7) % 11 - 5) / 2 for s in range(3)] for c in range(4)] for n in range(2)]
DY = [[[float(((3 * c + s + 12 * n) * 5) % 13 - 6) / 3 for s in range(3)] for c in range(4)] for n in range(2)]
GAMMA = [1.5, 0.5, 2.0, -1.0]
BETA = [0.25, -0.5, 0.0, 1.0]
Y = [
    [
        [-1.83957437545, 1.294787187725, -0.496276562661],
        [0.296028333505, -0.300992916624, -0.898014166752],
        [2.723934680039, 0.302659408893, -2.118615862252],
        [-0.059307931126, 1.151329704447, 2.361967340019],
    ],
    [
        [0.679447736958, -1.381901400442, 2.225459590008],
        [-0.528629849131, -1.215746228264, -0.013292564781],
        [0.572596982611, -2.175868533922, 2.633946120011],
        [1.057259698261, 2.431492456528, 0.026585129561],
    ],
]
DX = [
    [
        [-1.546738738476, -0.271417818487, 1.345267995596],
        [-0.464020876747, 0.157629520456, 0.779279917658],
        [-0.595563434834, 1.533063655466, -1.584405675048],
        [0.239252746634, -0.658714251997, 1.066366959779],
    ],
    [
        [0.824164796071, -1.299515880715, -0.675098402439],
        [0.751322212964, 0.459951880533, -0.060824606413],
        [2.098897024156, -1.013029383241, 0.310143554903],
        [0.633799043173, 0.041299359266, -2.071109598257],
    ],
]
DGAMMA = [3.818310466865, -1.425345532937, 3.623724250235, 2.827542113775]
DBETA = [-8 / 3, 4 / 3, 1.0, 2 / 3]


def t(v):
    return torch.as_tensor(v, dtype=torch.float64)


def layer(g=2, c=4, eps=1e-5, gamma=None, beta=None, affine=True):
    m = GroupNorm(g, c, eps=eps, affine=affine)
    with torch.no_grad():
        if gamma is not None:
            params(m)[0].copy_(t(gamma))
        if beta is not None:
            params(m)[1].copy_(t(beta))
    return m


def params(m):
    assert m.weight is not None and m.bias is not None
    return m.weight, m.bias


def grads(m):
    w, b = params(m)
    assert w.grad is not None and b.grad is not None
    return w.grad, b.grad


def autograd(m, x, dy):
    x = x.clone().requires_grad_(True)
    y = m(x)
    (y * dy).sum().backward()
    return y.detach(), x.grad


def test_worked_example():
    m = layer(eps=0.0, affine=False)
    assert m.weight is None and m.bias is None
    y, dx = autograd(m, t(WX), t(WDY))
    torch.testing.assert_close(y, t(WY), atol=1e-12, rtol=0)
    torch.testing.assert_close(dx, t(WDX), atol=1e-12, rtol=0)
    torch.testing.assert_close(input_gradient(t(WX), t(WDY), 2, eps=0.0), t(WDX), atol=1e-12, rtol=0)


def test_shared_fixture():
    m = layer(gamma=GAMMA, beta=BETA)
    y, dx = autograd(m, t(X), t(DY))
    torch.testing.assert_close(y, t(Y), atol=1e-11, rtol=0)
    torch.testing.assert_close(dx, t(DX), atol=1e-11, rtol=0)
    torch.testing.assert_close(grads(m)[0], t(DGAMMA), atol=1e-11, rtol=0)
    torch.testing.assert_close(grads(m)[1], t(DBETA), atol=1e-11, rtol=0)


def test_defaults_and_errors():
    m = GroupNorm(2, 4)
    assert m.eps == 1e-5 and m.num_groups == 2 and m.num_channels == 4
    torch.testing.assert_close(params(m)[0].detach(), torch.ones(4), atol=0, rtol=0)
    torch.testing.assert_close(params(m)[1].detach(), torch.zeros(4), atol=0, rtol=0)
    for args in [(3, 4), (0, 4), (2, 0)]:
        with pytest.raises(ValueError):
            GroupNorm(*args)
    with pytest.raises(ValueError):
        GroupNorm(2, 4, eps=-1.0)
    with pytest.raises(ValueError):
        m(torch.zeros(2, 3, 5))
    with pytest.raises(ValueError):
        m(torch.zeros(4))
    with pytest.raises(ValueError):
        normalize_over(torch.zeros(4), 1)
    with pytest.raises(ValueError):
        normalize_over(torch.zeros(2, 6), 4)
    with pytest.raises(ValueError):
        normalize_over(torch.zeros(2, 6), 0)


@pytest.mark.parametrize("g", [1, 2, 4, 8])
@pytest.mark.parametrize("shape", [(3, 8, 4, 5), (2, 8, 3), (2, 8, 2, 3, 2)])
def test_matches_torch_group_norm(g, shape):
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(*shape, generator=gen) * 2 + 1
    m = layer(g=g, c=8, gamma=torch.randn(8, generator=gen), beta=torch.randn(8, generator=gen))
    ref = torch.nn.GroupNorm(g, 8, eps=1e-5)
    with torch.no_grad():
        ref.weight.copy_(params(m)[0])
        ref.bias.copy_(params(m)[1])
    torch.testing.assert_close(m(x), ref(x), atol=1e-10, rtol=0)


@settings(max_examples=30, deadline=None)
@given(st.integers(1, 3), st.sampled_from([2, 4, 6]), st.integers(1, 4), st.integers(2, 4), st.integers(0, 2**31 - 1))
def test_special_cases(n, c, h, w, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, h, w, generator=gen) * 3 - 1
    ln = torch.nn.LayerNorm([int(c), int(h), int(w)], eps=1e-5, elementwise_affine=False)
    inorm = torch.nn.InstanceNorm2d(c, eps=1e-5, affine=False)
    torch.testing.assert_close(layer(g=1, c=c, affine=False)(x), ln(x), atol=1e-10, rtol=0)
    torch.testing.assert_close(layer(g=c, c=c, affine=False)(x), inorm(x), atol=1e-10, rtol=0)
    torch.testing.assert_close(normalize_over(x, 1), ln(x), atol=1e-10, rtol=0)
    torch.testing.assert_close(normalize_over(x, c), inorm(x), atol=1e-10, rtol=0)
    bn = torch.nn.functional.batch_norm(x, None, None, training=True, eps=1e-5)
    torch.testing.assert_close(normalize_over(x, c, over_batch=True), bn, atol=1e-10, rtol=0)
    bn1 = torch.nn.functional.batch_norm(x[:1], None, None, training=True, eps=1e-5)
    torch.testing.assert_close(bn1, inorm(x[:1]), atol=1e-10, rtol=0)


@pytest.mark.parametrize("g", [1, 2, 4])
@pytest.mark.parametrize("eps", [0.0, 1e-3, 0.5])
def test_autograd_matches_closed_form(g, eps):
    gen = torch.Generator().manual_seed(20261116)
    x, dy, gamma = (
        torch.randn(3, 4, 2, 3, generator=gen),
        torch.randn(3, 4, 2, 3, generator=gen),
        torch.randn(4, generator=gen),
    )
    m = layer(g=g, eps=eps, gamma=gamma)
    _, dx = autograd(m, x, dy)
    torch.testing.assert_close(dx, input_gradient(x, dy, g, gamma=gamma, eps=eps), atol=1e-10, rtol=0)
    _, dx0 = autograd(layer(g=g, eps=eps, affine=False), x, dy)
    torch.testing.assert_close(dx0, input_gradient(x, dy, g, eps=eps), atol=1e-10, rtol=0)
    assert torch.autograd.gradcheck(
        lambda a: layer(g=g, eps=eps + 1e-3, gamma=gamma)(a), (x[:1].clone().requires_grad_(True),)
    )


def test_batch_independence():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(5, 4, 3, 3, generator=gen)
    other = torch.randn(5, 4, 3, 3, generator=gen) * 10 + 4
    m = layer(gamma=GAMMA, beta=BETA)
    full = m(x)
    for k in range(5):
        torch.testing.assert_close(full[k : k + 1], m(x[k : k + 1]), atol=1e-12, rtol=0)
    mixed = torch.cat([x[:1], other[1:]])
    torch.testing.assert_close(m(mixed)[:1], full[:1], atol=1e-12, rtol=0)
    bn = torch.nn.functional.batch_norm
    assert (bn(mixed, None, None, training=True)[0] - bn(x, None, None, training=True)[0]).abs().max() > 1e-3
