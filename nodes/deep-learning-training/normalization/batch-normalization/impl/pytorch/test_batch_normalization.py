# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
"""Tests of the PyTorch module: the worked example, the gradient, and agreement with torch.nn.BatchNorm1d and
torch.nn.BatchNorm2d in training and evaluation mode, for the outputs, the gradients and the running statistics."""

import pytest
import torch

from batch_normalization import BatchNorm

torch.set_default_dtype(torch.float64)

# The worked example of the page.
WX = [[2.0], [4.0], [5.0], [6.0], [8.0]]
WY = [[-2.0], [0.0], [1.0], [2.0], [4.0]]
WG = [[1.0], [0.0], [0.0], [0.0], [0.0]]
WDX = [[0.35], [-0.35], [-0.2], [-0.05], [0.25]]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0], [1.0, 3.0, -2.5], [-1.25, 0.5, 1.5], [2.0, -0.75, 0.25]]
DY = [[1.0, 2.0, -1.0], [-2.0, 1.0, 0.0], [0.25, -0.5, 2.0], [1.5, 0.5, -3.0]]
GAMMA = [1.5, 0.5, 2.0]
BETA = [0.1, -0.2, 0.3]


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def x4():
    return (((torch.arange(48.0) * 7) % 11 - 5).reshape(4, 3, 2, 2) / 2).double()


def dy4():
    return (((torch.arange(48.0) * 5) % 13 - 6).reshape(4, 3, 2, 2) / 3).double()


def pair(cls, momentum=0.1, eps=1e-5):
    ours, ref = BatchNorm(3, eps=eps, momentum=momentum), cls(3, eps=eps, momentum=momentum)
    with torch.no_grad():
        for m in (ours, ref):
            m.weight.copy_(t(GAMMA))
            m.bias.copy_(t(BETA))
    return ours, ref


def test_worked_example():
    bn = BatchNorm(1, eps=0.0)
    with torch.no_grad():
        bn.weight.fill_(2.0)
        bn.bias.fill_(1.0)
    x = t(WX).requires_grad_()
    y = bn(x)
    torch.testing.assert_close(y, t(WY), atol=1e-12, rtol=0)
    y.backward(t(WG))
    torch.testing.assert_close(x.grad, t(WDX), atol=1e-12, rtol=0)
    torch.testing.assert_close(bn.weight.grad, t([-1.5]), atol=1e-12, rtol=0)
    torch.testing.assert_close(bn.bias.grad, t([1.0]), atol=1e-12, rtol=0)
    torch.testing.assert_close(bn.running_mean, t([0.5]))
    torch.testing.assert_close(bn.running_var, t([1.4]))


def test_gradient_finite_differences():
    bn = BatchNorm(3, eps=1e-3)
    for x in (t(X), x4()):
        def loss(a, g, b):
            y = torch.func.functional_call(bn, {"weight": g, "bias": b}, (a,))
            return torch.sum(y * torch.sin(a))

        inputs = (x.clone().requires_grad_(), t(GAMMA).requires_grad_(), t(BETA).requires_grad_())
        assert torch.autograd.gradcheck(loss, inputs)


def test_backward_formula():
    bn = BatchNorm(3)
    with torch.no_grad():
        bn.weight.copy_(t(GAMMA))
    x = t(X).requires_grad_()
    bn(x).backward(t(DY))
    m, g = 4, t(DY)
    mean, var = t(X).mean(0), t(X).var(0, unbiased=False)
    s = torch.sqrt(var + bn.eps)
    xhat = (t(X) - mean) / s
    dx = t(GAMMA) / (m * s) * (m * g - g.sum(0) - xhat * (g * xhat).sum(0))
    torch.testing.assert_close(x.grad, dx, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("momentum", [0.1, 0.5, None])
def test_running_stats_match_torch(momentum):
    for cls, x, dy in [(torch.nn.BatchNorm1d, t(X), t(DY)), (torch.nn.BatchNorm2d, x4(), dy4())]:
        ours, ref = pair(cls, momentum)
        for k in range(3):
            xa, xb = (x * (k + 1) + k).requires_grad_(), (x * (k + 1) + k).requires_grad_()
            ya, yb = ours(xa), ref(xb)
            torch.testing.assert_close(ya, yb, rtol=1e-10, atol=1e-12)
            ya.backward(dy)
            yb.backward(dy)
            torch.testing.assert_close(xa.grad, xb.grad, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(ours.weight.grad, ref.weight.grad, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(ours.bias.grad, ref.bias.grad, rtol=1e-9, atol=1e-11)
        torch.testing.assert_close(ours.running_mean, ref.running_mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ours.running_var, ref.running_var, rtol=1e-12, atol=1e-12)
        assert int(ours.num_batches_tracked) == int(ref.num_batches_tracked) == 3


def test_eval_uses_running_stats():
    ours, ref = pair(torch.nn.BatchNorm1d)
    ours(t(X) + 1.0)
    ref(t(X) + 1.0)
    ours.eval()
    ref.eval()
    y = ours(t(X))
    torch.testing.assert_close(y, ref(t(X)), rtol=1e-10, atol=1e-12)
    expected = t(GAMMA) * (t(X) - ours.running_mean) / torch.sqrt(ours.running_var + ours.eps) + t(BETA)
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ours(t(X)[1:2]), y[1:2], rtol=1e-12, atol=1e-12)
    before = ours.running_mean.clone()
    ours(t(X) * 5)
    torch.testing.assert_close(ours.running_mean, before, rtol=0, atol=0)


def test_momentum_convention():
    alpha = 0.1
    ours = BatchNorm(3, momentum=alpha)
    rho = 1 - alpha  # the Keras or Flax momentum
    mean, var = torch.zeros(3), torch.ones(3)
    for k in range(4):
        x = t(X) * (k + 1) - k
        ours(x)
        mean = rho * mean + (1 - rho) * x.mean(0)
        var = rho * var + (1 - rho) * x.var(0, unbiased=True)
    torch.testing.assert_close(ours.running_mean, mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ours.running_var, var, rtol=1e-12, atol=1e-12)


def test_conv_per_channel():
    ours, ref = pair(torch.nn.BatchNorm2d, eps=1e-12)  # PyTorch rejects eps = 0 in training
    y = ours(x4())
    torch.testing.assert_close(y, ref(x4()), rtol=1e-10, atol=1e-12)
    z = (y - t(BETA).view(1, 3, 1, 1)) / t(GAMMA).view(1, 3, 1, 1)
    torch.testing.assert_close(z.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-12, rtol=0)
    torch.testing.assert_close(z.var(dim=(0, 2, 3), unbiased=False), torch.ones(3), atol=1e-12, rtol=0)


def test_single_value_rejected():
    with pytest.raises(ValueError, match="more than 1 value"):
        BatchNorm(3)(t(X)[:1])
    with pytest.raises(ValueError, match="more than 1 value"):
        torch.nn.BatchNorm1d(3)(t(X)[:1])
    assert BatchNorm(3)(x4()[:1]).shape == (1, 3, 2, 2)


def test_fold():
    torch.manual_seed(20261114)
    conv = torch.nn.Conv2d(2, 3, 3, padding=1)
    bn = BatchNorm(3)
    with torch.no_grad():
        bn.weight.copy_(t(GAMMA))
        bn.bias.copy_(t(BETA))
        bn.running_mean.copy_(t([0.3, -0.1, 0.7]))
        bn.running_var.copy_(t([1.2, 0.4, 2.5]))
    bn.eval()
    z = torch.randn(2, 2, 5, 5)
    w2, b2 = bn.fold(conv.weight, conv.bias)
    folded = torch.nn.functional.conv2d(z, w2, b2, padding=1)
    torch.testing.assert_close(folded, bn(conv(z)), rtol=1e-10, atol=1e-12)
