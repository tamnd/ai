# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://RA5R92).
"""Tests of the PyTorch modules: the worked example, autograd against eq-ln-backward, the Jacobian, agreement with
torch.nn.LayerNorm, torch.nn.functional.layer_norm, torch.nn.RMSNorm and torch.nn.functional.group_norm in outputs and
gradients, the same output in training and evaluation mode, independence from the batch, and the two placements."""

import pytest
import torch

from layer_normalization import LayerNorm, Residual, RMSNorm

torch.set_default_dtype(torch.float64)

# The worked example of the page.
WA = [3.0, -4.0, 7.0, -1.0, 5.0]
WGAIN = [2.0, 1.0, 2.0, 1.0, 4.0]
WBIAS = [0.0, 1.0, 0.0, 1.0, 0.0]
WY = [0.5, -0.5, 2.5, 0.25, 3.0]
WDY = [0.0, 1.0, 0.0, 0.0, 0.0]
WDX = [-0.625 / 20, 1.75 / 20, 0.875 / 20, -2.125 / 20, 0.125 / 20]

# The shared fixture of the NumPy tests.
X = [[0.5, -1.5, 2.0, 1.0], [1.0, 3.0, -2.5, 0.0], [-1.25, 0.5, 1.5, 4.0]]
DY = [[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 3.0], [0.25, -0.5, 2.0, -1.0]]
GAIN = [1.5, 0.5, 2.0, -1.0]
BIAS = [0.1, -0.2, 0.3, 0.0]


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def x3():
    return ((torch.arange(24.0) * 7) % 11 - 5).reshape(2, 3, 4) / 2


def dy3():
    return ((torch.arange(24.0) * 5) % 13 - 6).reshape(2, 3, 4) / 3


def set_params(ours, ref, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ("weight", "bias"):
            a, b = getattr(ours, name, None), getattr(ref, name, None)
            assert (a is None) == (b is None), name
            if a is not None:
                v = torch.randn(a.shape, generator=g, dtype=a.dtype)
                a.copy_(v)
                b.copy_(v)


def test_worked_example():
    ln = LayerNorm(5, eps=0.0)
    with torch.no_grad():
        ln.weight.copy_(t(WGAIN))
        ln.bias.copy_(t(WBIAS))
    x = t(WA).requires_grad_()
    y = ln(x)
    torch.testing.assert_close(y, t(WY), atol=1e-12, rtol=0)
    y.backward(t(WDY))
    torch.testing.assert_close(x.grad, t(WDX), atol=1e-12, rtol=0)
    torch.testing.assert_close(ln.weight.grad, t([0.0, -1.5, 0.0, 0.0, 0.0]), atol=1e-12, rtol=0)
    torch.testing.assert_close(ln.bias.grad, t(WDY), atol=1e-12, rtol=0)
    j = torch.autograd.functional.jacobian(LayerNorm(5, eps=0.0, elementwise_affine=False), t(WA))
    torch.testing.assert_close(torch.linalg.matrix_norm(j, 2), t(0.25))
    centered = t(WA) - t(WA).mean()
    torch.testing.assert_close(RMSNorm(5, eps=0.0)(centered), LayerNorm(5, eps=0.0)(t(WA)), atol=1e-12, rtol=0)


def test_backward_formula():
    ln = LayerNorm(4)
    with torch.no_grad():
        ln.weight.copy_(t(GAIN))
        ln.bias.copy_(t(BIAS))
    x, g = t(X).requires_grad_(), t(DY)
    ln(x).backward(g)
    h = 4
    s = torch.sqrt(x.detach().var(dim=-1, unbiased=False, keepdim=True) + ln.eps)
    xhat = (x.detach() - x.detach().mean(-1, keepdim=True)) / s
    u = g * t(GAIN)
    expected = (h * u - u.sum(-1, keepdim=True) - xhat * (u * xhat).sum(-1, keepdim=True)) / (h * s)
    torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(3), rtol=0, atol=1e-12)
    torch.testing.assert_close(ln.weight.grad, (g * xhat).sum(0), rtol=1e-12, atol=1e-12)


def test_gradcheck():
    for m in (LayerNorm((3, 4), eps=1e-3), RMSNorm((3, 4), eps=1e-3)):
        set_params(m, type(m)((3, 4)))
        x = x3().requires_grad_()
        assert torch.autograd.gradcheck(lambda z, m=m: m(z), (x,))


def test_input_gradient_orthogonal_without_eps():
    ln = LayerNorm(4, eps=0.0, elementwise_affine=False)
    x = t(X).requires_grad_()
    y = ln(x)
    y.backward(t(DY))
    torch.testing.assert_close((x.grad * y.detach()).sum(-1), torch.zeros(3), rtol=0, atol=1e-12)
    torch.testing.assert_close(x.grad.sum(-1), torch.zeros(3), rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape", [(4,), (3, 4)])
@pytest.mark.parametrize(("affine", "bias"), [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_matches_torch_layernorm(shape, affine, bias, dtype):
    x, dy = (t(X), t(DY)) if shape == (4,) else (x3(), dy3())
    x, dy = x.to(dtype), dy.to(dtype)
    ours = LayerNorm(shape, elementwise_affine=affine, bias=bias).to(dtype)
    ref = torch.nn.LayerNorm(shape, elementwise_affine=affine, bias=bias, dtype=dtype)
    set_params(ours, ref)
    tol = {"rtol": 1e-12, "atol": 1e-12} if dtype == torch.float64 else {"rtol": 1e-5, "atol": 1e-5}
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, yb = ours(xa), ref(xb)
    torch.testing.assert_close(ya, yb, **tol)
    ya.backward(dy)
    yb.backward(dy)
    torch.testing.assert_close(xa.grad, xb.grad, **tol)
    if affine:
        torch.testing.assert_close(ours.weight.grad, ref.weight.grad, **tol)
    if bias:
        torch.testing.assert_close(ours.bias.grad, ref.bias.grad, **tol)
    f = torch.nn.functional.layer_norm(x, shape, ref.weight, ref.bias, eps=1e-5)
    torch.testing.assert_close(ya, f, **tol)


@pytest.mark.parametrize("shape", [(4,), (3, 4)])
@pytest.mark.parametrize("eps", [None, 1e-6, 0.5])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_matches_torch_rmsnorm(shape, eps, dtype):
    x, dy = (t(X), t(DY)) if shape == (4,) else (x3(), dy3())
    x, dy = x.to(dtype) * 1e-4, dy.to(dtype)  # small inputs, where the default eps matters
    ours, ref = RMSNorm(shape, eps=eps).to(dtype), torch.nn.RMSNorm(shape, eps=eps, dtype=dtype)
    set_params(ours, ref)
    tol = {"rtol": 1e-12, "atol": 1e-12} if dtype == torch.float64 else {"rtol": 1e-5, "atol": 1e-5}
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, yb = ours(xa), ref(xb)
    torch.testing.assert_close(ya, yb, **tol)
    ya.backward(dy)
    yb.backward(dy)
    torch.testing.assert_close(xa.grad, xb.grad, rtol=tol["rtol"], atol=tol["atol"] * 1e4)
    torch.testing.assert_close(ours.weight.grad, ref.weight.grad, **tol)


def test_rmsnorm_default_eps_is_not_layernorm_eps():
    x = t(X) * 1e-4
    a, b = RMSNorm(4)(x), RMSNorm(4, eps=1e-5)(x)
    assert (a - b).abs().max() > 1e-2


def test_train_equals_eval():
    for ours, ref in [(LayerNorm(4), torch.nn.LayerNorm(4)), (RMSNorm(4), torch.nn.RMSNorm(4))]:
        set_params(ours, ref)
        for m in (ours, ref):
            m.train()
            y_train = m(t(X))
            for k in range(3):
                m(t(X) * (k + 2) - k)
            m.eval()
            torch.testing.assert_close(m(t(X)), y_train, rtol=0, atol=0)
        assert not list(ours.buffers()) and not list(ref.buffers())


def test_batch_independence():
    ln = LayerNorm(4)
    set_params(ln, torch.nn.LayerNorm(4))
    y = ln(t(X))
    for i in range(3):
        torch.testing.assert_close(ln(t(X)[i:i + 1]), y[i:i + 1], rtol=0, atol=1e-15)
    other = t(X).clone()
    other[1:] = 100 * torch.randn(2, 4, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(ln(other)[0], y[0], rtol=0, atol=1e-15)


def test_per_element_affine():
    ln = LayerNorm((3, 4))
    assert ln.weight.shape == (3, 4) and ln.bias.shape == (3, 4)
    assert torch.nn.LayerNorm((3, 4)).weight.shape == (3, 4)
    assert torch.nn.GroupNorm(1, 3).weight.shape == (3,)


def test_group_norm_with_one_group_is_layer_norm():
    x = ((torch.arange(96.0) * 7) % 13 - 6).reshape(2, 3, 4, 4) / 3
    ours = LayerNorm((3, 4, 4), elementwise_affine=False)(x)
    torch.testing.assert_close(ours, torch.nn.functional.group_norm(x, 1, eps=1e-5), rtol=1e-12, atol=1e-12)


def test_placements():
    d = 4
    zero = torch.nn.Linear(d, d)
    with torch.no_grad():
        zero.weight.zero_()
        zero.bias.zero_()
    norm = LayerNorm(d, eps=0.0, elementwise_affine=False)
    torch.testing.assert_close(Residual(zero, norm, "pre")(t(X)), t(X), rtol=0, atol=0)
    torch.testing.assert_close(Residual(zero, norm, "post")(t(X)), norm(t(X)), rtol=0, atol=0)
    with pytest.raises(ValueError):
        Residual(zero, norm, "middle")


def test_pre_ln_stream_grows_post_ln_stays():
    """Post-LN outputs have the norm sqrt(d) at every depth; the Pre-LN residual stream is not normalized and grows
    with depth at initialization, as in Lemma 2 of Xiong et al."""
    torch.manual_seed(20261115)
    d, n, depth = 64, 512, 8
    pre = post = torch.randn(n, d)
    norms = []
    with torch.no_grad():
        for _ in range(depth):
            f = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, d))
            pre = Residual(f, LayerNorm(d, eps=0.0), "pre")(pre)
            post = Residual(f, LayerNorm(d, eps=0.0), "post")(post)
            torch.testing.assert_close(post.norm(dim=-1), torch.full((n,), d ** 0.5))
            norms.append(float((pre ** 2).sum(-1).mean()))
    assert all(b > a for a, b in zip(norms, norms[1:])), norms


def test_shape_errors():
    with pytest.raises(ValueError):
        LayerNorm(0)
    with pytest.raises(ValueError):
        LayerNorm(4)(torch.zeros(3, 5))
    with pytest.raises(ValueError):
        RMSNorm((3, 4))(torch.zeros(4))
