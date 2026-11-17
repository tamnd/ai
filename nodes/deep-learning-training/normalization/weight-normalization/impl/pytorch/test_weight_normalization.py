# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://5PC0VR).
"""Tests of the PyTorch reference: the shared fixture and the worked examples of the NumPy tests, autograd against
the closed form of eq-grad, the orthogonality of the gradient in v, the data-dependent initialization, the power
iteration against torch.linalg, and agreement with torch.nn.utils.parametrizations.weight_norm and spectral_norm."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from weight_normalization import (
    SpectralNormLinear,
    WeightNormLinear,
    data_dependent_init,
    power_iteration,
    weight_grads,
)

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
X = [[float(((4 * n + k) * 7) % 11 - 5) / 2 for k in range(4)] for n in range(3)]
V = [[float(((4 * j + k) * 5) % 9 - 4) / 3 for k in range(4)] for j in range(2)]
G = [1.5, -0.5]
B = [0.25, -1.0]
DY = [[float(((2 * n + j) * 5) % 7 - 3) / 2 for j in range(2)] for n in range(3)]
Y = [[5.453364296299, -2.734454765433], [-2.35168214815, -0.315346803119], [1.893167672515, -0.908712907082]]
DX = [
    [1.825741858351, -0.684653196881, 1.323662847304, -1.186732207928],
    [-0.182574185835, 0.273861278753, -0.091287092918, 0.36514837167],
    [-1.551880579598, 0.273861278753, -1.186732207928, 0.639009650423],
]
DV = [
    [-1.519930097077, 1.150217370761, -0.985900603509, -5.093819784798],
    [0.620752231839, -0.178009831189, 0.65270271436, 0.607059167902],
]
DG = [-3.560196623784, 4.746928831711]
DB = [0.0, 0.5]
INIT_G = [0.470148599148, 0.485706576147]
INIT_B = [-0.443491154839, -0.310371189432]


def close(a, b, atol=1e-11):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), rtol=0.0, atol=atol)


def fixture_layer():
    layer = WeightNormLinear(4, 2)
    with torch.no_grad():
        layer.v.copy_(torch.tensor(V))
        layer.g.copy_(torch.tensor(G))
        layer.b.copy_(torch.tensor(B))
    return layer


def test_shared_fixture():
    layer = fixture_layer()
    x = torch.tensor(X, requires_grad=True)
    y = layer(x)
    y.backward(torch.tensor(DY))
    assert close(y.detach(), Y)
    assert close(x.grad, DX) and close(layer.v.grad, DV) and close(layer.g.grad, DG) and close(layer.b.grad, DB)
    data_dependent_init(layer, torch.tensor(X))
    assert close(layer.g.detach(), INIT_G) and close(layer.b.detach(), INIT_B)
    assert close(layer.v.detach(), V, atol=0.0)


def test_worked_examples():
    layer = WeightNormLinear(2, 1)
    with torch.no_grad():
        layer.v.copy_(torch.tensor([[3.0, 4.0]]))
        layer.g.fill_(2.0)
    assert close(layer.weight().detach(), [[1.2, 1.6]])
    grad_v, grad_g = weight_grads(layer.v.detach(), layer.g.detach(), torch.tensor([[1.0, 0.0]]))
    assert close(grad_g, [0.6]) and close(grad_v, [[0.256, -0.192]])
    x = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    data_dependent_init(layer, x)
    assert close(layer.g.detach(), [0.4]) and close(layer.b.detach(), [-1.4])
    assert close(layer(x).detach().ravel(), [-1.4, -0.2, 0.2, 1.4])
    w = torch.diag(torch.tensor([3.0, 1.0]))
    u0 = torch.tensor([1.0, 1.0]) / 2**0.5
    for k in (1, 2, 3):
        sigma = power_iteration(w, u0, k)[0]
        assert abs(9.0 - sigma**2 - 8.0 / (9.0 ** (2 * k - 1) + 1.0)) < 1e-12


def test_init_and_bad_inputs():
    layer = WeightNormLinear(400, 300, generator=torch.Generator().manual_seed(0))
    assert layer.v.shape == (300, 400) and bool((layer.g == 1).all()) and bool((layer.b == 0).all())
    assert abs(float(layer.v.detach().std()) - 0.05) < 1e-3
    assert abs(float(WeightNormLinear(400, 300, scale=1.0).v.detach().std()) - 1.0) < 0.01
    assert WeightNormLinear(1, 1).v.shape == (1, 1)
    for a, b in [(0, 3), (3, 0)]:
        with pytest.raises(ValueError):
            WeightNormLinear(a, b)
    layer = fixture_layer()
    with pytest.raises(ValueError):
        layer(torch.tensor(X)[0])
    with pytest.raises(ValueError):
        layer(torch.tensor(X)[:, :3])
    with pytest.raises(ValueError):
        data_dependent_init(layer, torch.tensor(X)[:1])
    with pytest.raises(ValueError):
        data_dependent_init(layer, torch.tensor(X)[0])
    const = WeightNormLinear(2, 1)
    with torch.no_grad():
        const.v.copy_(torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        data_dependent_init(const, torch.tensor([[2.0, 1.0], [2.0, 5.0]]))
    with pytest.raises(ValueError):
        power_iteration(torch.ones(2, 3), torch.ones(2), n_iter=0)
    with pytest.raises(ValueError):
        SpectralNormLinear(3, 2, n_iter=0)


def test_edges_and_defaults():
    one = WeightNormLinear(1, 1)
    with torch.no_grad():
        one.v.fill_(2.0)
    data_dependent_init(one, torch.tensor([[1.0], [3.0]]))
    assert close(one.g.detach(), [1.0]) and close(one.b.detach(), [-2.0])
    with pytest.raises(ValueError):
        data_dependent_init(fixture_layer(), torch.tensor(X)[:1], eps=1e-3)
    w, u0 = torch.diag(torch.tensor([3.0, 1.0])), torch.tensor([1.0, 1.0]) / 2**0.5
    assert power_iteration(w, u0)[0] == power_iteration(w, u0, 1)[0]
    assert abs(power_iteration(w, u0)[0] ** 2 - 8.2) < 1e-12
    layer = SpectralNormLinear(400, 300, generator=torch.Generator().manual_seed(1))
    assert abs(float(layer.weight.detach().std()) - 400**-0.5) < 1e-3
    assert abs(float(layer.u.norm()) - 1.0) < 1e-12 and abs(float(layer.v.norm()) - 1.0) < 1e-12
    v = layer.weight.detach().T @ layer.u
    assert close(layer.v, v / v.norm(), 1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(1e-3, 10.0))
def test_autograd_matches_closed_form(seed, eta):
    gen = torch.Generator().manual_seed(seed)
    layer = WeightNormLinear(5, 3, scale=1.0, generator=gen)
    with torch.no_grad():
        layer.g.copy_(torch.randn(3, generator=gen))
    x, dy = torch.randn(4, 5, generator=gen), torch.randn(4, 3, generator=gen)
    layer(x).backward(dy)
    grad_v, grad_g = weight_grads(layer.v.detach(), layer.g.detach(), dy.T @ x)
    assert close(layer.v.grad, grad_v, 1e-10) and close(layer.g.grad, grad_g, 1e-10)
    assert close((layer.v.grad * layer.v.detach()).sum(dim=1), torch.zeros(3), 1e-10)
    v = layer.v.detach()
    new = v - eta * layer.v.grad
    assert close(new.pow(2).sum(dim=1), v.pow(2).sum(dim=1) + eta**2 * layer.v.grad.pow(2).sum(dim=1), 1e-8)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(2, 9), st.sampled_from([0.0, 1e-3, 0.5]))
def test_init_moments(seed, n, eps):
    gen = torch.Generator().manual_seed(seed)
    layer = WeightNormLinear(4, 3, generator=gen)
    x = torch.randn(n, 4, generator=gen) * 3 + 1
    t = x @ layer.v.detach().T / layer.v.detach().norm(dim=1)
    data_dependent_init(layer, x, eps=eps)
    y = layer(x).detach()
    var = t.var(dim=0, unbiased=False)
    assert close(y.mean(dim=0), torch.zeros(3), 1e-10)
    assert torch.allclose(y.var(dim=0, unbiased=False), var / (var + eps), rtol=1e-9)


def test_against_torch_weight_norm():
    ref = torch.nn.Linear(4, 2)
    with torch.no_grad():
        ref.bias.copy_(torch.tensor(B))
    ref = torch.nn.utils.parametrizations.weight_norm(ref, name="weight", dim=0)
    with torch.no_grad():
        ref.parametrizations.weight.original0.copy_(torch.tensor(G).reshape(2, 1))
        ref.parametrizations.weight.original1.copy_(torch.tensor(V))
    layer = fixture_layer()
    x = torch.tensor(X)
    assert close(layer(x).detach(), ref(x).detach())
    assert close(layer.weight().detach(), ref.weight.detach())


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(2, 6), st.integers(2, 6))
def test_power_iteration_against_svd(seed, rows, cols):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=gen)
    s = torch.linalg.svdvals(w)
    u0 = torch.randn(rows, generator=gen)
    assert power_iteration(w, u0, 1)[0] <= float(s[0]) + 1e-12
    sigma, u, v = power_iteration(w, u0, 1000)
    assert abs(sigma - float(s[0])) <= 1e-6 * float(s[0])
    assert close(w @ v, sigma * u, 1e-5 * float(s[0]))


def test_spectral_norm_layer_matches_torch():
    gen = torch.Generator().manual_seed(5)
    mine = SpectralNormLinear(6, 4, generator=gen)
    ref = torch.nn.Linear(6, 4)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(torch.randn(4, generator=gen))
        mine.bias.copy_(ref.bias)
    # spectral_norm starts u and v from the global generator, and one torch step updates u from v, then v from u,
    # where eq-power updates v from u, then u from v. So the two v of one step differ by half a step, by up to 1e-8
    # after the 15 steps of the registration. Many training steps first make u and v converge, and then one step of
    # either order gives the same u, v and sigma to rounding.
    torch.manual_seed(5)
    ref = torch.nn.utils.parametrizations.spectral_norm(ref, n_power_iterations=1)
    with torch.no_grad():
        for _ in range(100):
            ref(torch.zeros(1, 6))
        mine.u.copy_(ref.parametrizations.weight[0]._u)
    x = torch.randn(3, 6, generator=gen)
    out_mine, out_ref = mine(x), ref(x)
    assert close(out_mine.detach(), out_ref.detach(), 1e-10)
    assert close(mine.u, ref.parametrizations.weight[0]._u, 1e-10)
    assert close(mine.v, ref.parametrizations.weight[0]._v, 1e-10)
    out_mine.sum().backward()
    out_ref.sum().backward()
    assert close(mine.weight.grad, ref.parametrizations.weight.original.grad, 1e-10)
    mine.eval()
    u_before = mine.u.clone()
    w_eval = mine.normalized_weight().detach()
    assert close(mine.u, u_before, 0.0)
    assert abs(float(torch.linalg.matrix_norm(w_eval, ord=2)) - 1.0) < 0.05


def test_spectral_norm_layer_converges():
    gen = torch.Generator().manual_seed(9)
    layer = SpectralNormLinear(8, 5, n_iter=1, generator=gen)
    x = torch.randn(2, 8, generator=gen)
    for _ in range(200):
        layer(x)
    assert abs(float(torch.linalg.matrix_norm(layer.normalized_weight().detach(), ord=2)) - 1.0) < 1e-9
    deep = SpectralNormLinear(8, 5, n_iter=300, generator=torch.Generator().manual_seed(9))
    deep(x)
    assert abs(float(torch.linalg.matrix_norm(deep.normalized_weight().detach(), ord=2)) - 1.0) < 1e-9
