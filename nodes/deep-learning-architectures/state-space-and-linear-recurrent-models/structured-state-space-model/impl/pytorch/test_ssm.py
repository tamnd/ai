# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Tests of the PyTorch layer: the two modes agree, the impulse response, causality, the discretizations and gradients."""

import math

import numpy as np
import pytest
import torch

from ssm import DiagonalSSM

torch.set_default_dtype(torch.float64)
METHODS = ["zoh", "bilinear"]


def layer(seed, method, h=3, n=4):
    torch.manual_seed(seed)
    m = DiagonalSSM(h, n, method, dt_min=0.01, dt_max=0.5)
    with torch.no_grad():
        m.log_a_re.add_(0.3 * torch.randn(h, n))
        m.a_im.add_(torch.randn(h, n))
        m.b.add_(0.5 * torch.randn(h, n, dtype=torch.complex128))
    return m


def reference(m, u):
    """A plain loop over the definition x_k = Ab x_(k-1) + Bb u_k, y_k = 2 Re(C x_k) + D u_k, in NumPy."""
    ab, bb = (t.detach().numpy() for t in m.discrete())
    c, d = m.c.detach().numpy(), m.d.detach().numpy()
    u = u.numpy()
    x = np.zeros((u.shape[0], m.h, m.n), dtype=complex)
    y = np.empty_like(u)
    for t in range(u.shape[1]):
        x = ab * x + bb * u[:, t, :, None]
        y[:, t] = 2 * np.einsum("hn,bhn->bh", c, x).real + d * u[:, t]
    return y


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("length", [1, 9, 33])
def test_recurrence_equals_convolution(method, length):
    m = layer(length, method)
    u = torch.randn(2, length, 3)
    conv, scan = m(u), m(u, mode="scan")
    assert torch.allclose(conv, scan, atol=1e-10)
    assert np.allclose(conv.detach().numpy(), reference(m, u), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_impulse_response_is_kernel(method):
    m = layer(1, method)
    u = torch.zeros(3, 25, 3)
    for h in range(3):
        u[h, 0, h] = 1.0
    k = m.kernel(25).detach()
    y = m(u).detach()
    for h in range(3):
        expected = k[h].clone()
        expected[0] += m.d[h].detach()
        assert torch.allclose(y[h, :, h], expected, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_causality(method):
    m = layer(2, method)
    u = torch.randn(2, 20, 3)
    v = u.clone()
    v[:, 11:] += torch.randn(2, 9, 3)
    for mode in ("conv", "scan"):
        a, b = m(u, mode=mode).detach(), m(v, mode=mode).detach()
        assert torch.allclose(a[:, :11], b[:, :11], atol=1e-12)
        assert not torch.allclose(a[:, 11:], b[:, 11:])


def test_zoh_scalar_and_bilinear_stable():
    m = DiagonalSSM(1, 1, "zoh")
    with torch.no_grad():
        m.log_a_re.fill_(math.log(0.7))
        m.a_im.zero_()
        m.log_dt.fill_(math.log(0.25))
        m.b.fill_(1.3)
    ab, bb = m.discrete()
    assert math.isclose(ab.real.item(), math.exp(-0.7 * 0.25)) and abs(ab.imag.item()) < 1e-15
    assert math.isclose(bb.real.item(), (math.exp(-0.7 * 0.25) - 1) / -0.7 * 1.3)
    m = layer(3, "bilinear", h=4, n=8)
    with torch.no_grad():
        m.log_dt.copy_(torch.tensor([-6.0, 0.0, 4.0, 9.0]))
        m.a_im.mul_(100.0)
    assert m.discrete()[0].abs().max() < 1.0


def test_step_and_gradients():
    m = layer(4, "zoh", h=2, n=3)
    u = torch.randn(1, 12, 2)
    full = m(u).detach()
    x = torch.zeros(1, 2, 3, dtype=torch.complex128)
    for t in range(12):
        x, y_t = m.step(x, u[:, t])
        assert torch.allclose(y_t.detach(), full[:, t], atol=1e-10)
    loss_c = (m(u) ** 2).sum()
    g_c = torch.autograd.grad(loss_c, [m.log_dt, m.log_a_re, m.c])
    loss_s = (m(u, mode="scan") ** 2).sum()
    g_s = torch.autograd.grad(loss_s, [m.log_dt, m.log_a_re, m.c])
    for a, b in zip(g_c, g_s):
        assert torch.allclose(a, b, atol=1e-8)


def test_from_params_and_errors():
    params = {
        "log_dt": np.array([math.log(0.1)]),
        "log_a_re": np.array([[math.log(0.5)]]),
        "a_im": np.array([[0.0]]),
        "b": np.array([[1.0 + 0j]]),
        "c": np.array([[0.5 + 0j]]),
        "d": np.array([0.0]),
    }
    m = DiagonalSSM.from_params(params)
    k = m.kernel(3).detach().numpy()
    ab = math.exp(-0.05)
    assert np.allclose(k, [(ab - 1) / -0.5 * ab**i for i in range(3)])
    with pytest.raises(ValueError):
        DiagonalSSM(method="euler")
    with pytest.raises(ValueError):
        m(torch.zeros(1, 2, 1), mode="fft")
