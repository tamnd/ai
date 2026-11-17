# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://5PC0VR).
"""Tests of the NumPy reference: the worked examples of the page, a shared fixture, the gradients against finite
differences, the projection form, the orthogonality of the gradient and the growth of ||v||, the moments after the
data-dependent initialization, mean-only batch normalization, the power iteration against the SVD, and agreement
with PyTorch when it is installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weight_normalization import (
    backward,
    data_dependent_init,
    forward,
    init,
    mean_only_backward,
    mean_only_forward,
    projection,
    spectral_norm,
    spectral_normalize,
    weight,
    weight_grads,
)

# The shared fixture: N = 3, in = 4, out = 2. The PyTorch tests hold the same numbers.
X = ((np.arange(12.0) * 7) % 11 - 5).reshape(3, 4) / 2
V = ((np.arange(8.0) * 5) % 9 - 4).reshape(2, 4) / 3
G = np.array([1.5, -0.5])
B = np.array([0.25, -1.0])
DY = ((np.arange(6.0) * 5) % 7 - 3).reshape(3, 2) / 2
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


def fd(f, a, h=1e-6):
    """Central finite differences of the scalar function f at the array a."""
    out = np.zeros_like(a)
    for i in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[i] = h
        out[i] = (f(a + e) - f(a - e)) / (2 * h)
    return out


def test_worked_example_gradient():
    v, g, grad_w = np.array([[3.0, 4.0]]), np.array([2.0]), np.array([[1.0, 0.0]])
    assert np.allclose(weight(v, g), [[1.2, 1.6]])
    grad_v, grad_g = weight_grads(v, g, grad_w)
    assert np.allclose(grad_g, [0.6]) and np.allclose(grad_v, [[0.256, -0.192]])
    assert np.isclose(grad_v[0] @ v[0], 0.0)
    assert np.allclose(projection([1.2, 1.6]) @ grad_w[0] * 2 / 5, grad_v[0])
    v_new = v[0] - 1.0 * grad_v[0]
    assert np.allclose(v_new, [2.744, 4.192]) and np.isclose(v_new @ v_new, 25.1024)


def test_worked_example_init():
    params = {"v": np.array([[3.0, 4.0]]), "g": np.array([7.0]), "b": np.array([9.0])}
    x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    new = data_dependent_init(params, x)
    assert np.allclose(new["g"], [0.4]) and np.allclose(new["b"], [-1.4])
    assert np.allclose(forward(new, x)[0].ravel(), [-1.4, -0.2, 0.2, 1.4])


def test_worked_example_power():
    w = np.diag([3.0, 1.0])
    u0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    for k in (1, 2, 3):
        sigma, u, v = spectral_norm(w, u0, k)
        assert np.isclose(9.0 - sigma**2, 8.0 / (9.0 ** (2 * k - 1) + 1.0), rtol=1e-9, atol=1e-15)
        assert np.isclose(np.linalg.norm(u), 1.0) and np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(spectral_norm(w, u0, 1)[0], np.sqrt(8.2))
    assert np.allclose(u0, np.array([1.0, 1.0]) / np.sqrt(2.0))


def test_shared_fixture():
    params = {"v": V, "g": G, "b": B}
    y, cache = forward(params, X)
    dx, grads = backward(params, cache, DY)
    assert np.allclose(y, Y, atol=1e-11)
    assert np.allclose(dx, DX, atol=1e-11)
    assert np.allclose(grads["v"], DV, atol=1e-11)
    assert np.allclose(grads["g"], DG, atol=1e-11)
    assert np.allclose(grads["b"], DB, atol=1e-11)
    new = data_dependent_init(params, X)
    assert np.allclose(new["g"], INIT_G, atol=1e-11) and np.allclose(new["b"], INIT_B, atol=1e-11)


def test_init():
    p = init(400, 300, np.random.default_rng(0))
    assert p["v"].shape == (300, 400) and p["g"].shape == (300,) and p["b"].shape == (300,)
    assert np.all(p["g"] == 1.0) and np.all(p["b"] == 0.0)
    assert abs(p["v"].std() - 0.05) < 1e-3 and abs(p["v"].mean()) < 1e-3
    assert abs(init(400, 300, np.random.default_rng(0), scale=1.0)["v"].std() - 1.0) < 0.01
    assert init(1, 1, np.random.default_rng(0))["v"].shape == (1, 1)
    for a, b in [(0, 3), (3, 0)]:
        with pytest.raises(ValueError):
            init(a, b, np.random.default_rng(0))


def test_bad_inputs():
    params = {"v": V, "g": G, "b": B}
    with pytest.raises(ValueError):
        forward(params, X[0])
    with pytest.raises(ValueError):
        forward(params, X[:, :3])
    with pytest.raises(ValueError):
        weight(np.array([[1.0, 0.0], [0.0, 0.0]]), np.ones(2))
    with pytest.raises(ValueError):
        data_dependent_init(params, X[:1])
    with pytest.raises(ValueError):
        data_dependent_init(params, X[0])
    with pytest.raises(ValueError):
        data_dependent_init({"v": np.array([[1.0, 0.0]])}, np.array([[2.0, 1.0], [2.0, 5.0]]))
    w = np.ones((2, 3))
    with pytest.raises(ValueError):
        spectral_norm(w, np.ones(3))
    with pytest.raises(ValueError):
        spectral_norm(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        spectral_norm(w, np.ones(2), n_iter=0)


def test_lists_defaults_and_edges():
    vl, gl, xl, dyl = V.tolist(), G.tolist(), X.tolist(), DY.tolist()
    assert np.allclose(weight(vl, gl), weight(V, G), rtol=0.0, atol=0.0)
    params = {"v": V, "g": G, "b": B}
    y, cache = forward(params, xl)
    assert np.allclose(y, Y, rtol=0.0, atol=1e-11)
    grads = backward(params, cache, dyl)[1]
    assert np.allclose(grads["v"], DV, rtol=0.0, atol=1e-11) and np.allclose(grads["g"], DG, rtol=0.0, atol=1e-11)
    assert np.allclose(weight_grads(vl, gl, (DY.T @ X).tolist())[0], DV, rtol=0.0, atol=1e-11)
    assert np.allclose(data_dependent_init(params, xl)["g"], INIT_G, rtol=0.0, atol=1e-11)
    assert np.allclose(mean_only_forward(xl, 0.0), X - X.mean(axis=0), rtol=0.0, atol=0.0)
    assert np.allclose(mean_only_backward(xl)[0], X - X.mean(axis=0), rtol=0.0, atol=0.0)
    # a row of norm exactly 1 is valid, and one input feature is enough
    assert np.allclose(weight([[0.0, 1.0]], [2.0]), [[0.0, 2.0]])
    one = data_dependent_init({"v": np.array([[2.0]])}, np.array([[1.0], [3.0]]))
    assert np.allclose([one["g"][0], one["b"][0]], [1.0, -2.0])
    # one example is refused even when eps would make sigma positive
    with pytest.raises(ValueError):
        data_dependent_init(params, X[:1], eps=1e-3)
    # the default is one round of power iteration, far from converged here
    w, u0 = [[3.0, 0.0], [0.0, 1.0]], [2**-0.5, 2**-0.5]
    assert spectral_norm(w, u0)[0] == spectral_norm(w, u0, 1)[0]
    assert abs(spectral_norm(w, u0)[0] ** 2 - 8.2) < 1e-12
    w_sn, u1 = spectral_normalize(w, u0)
    assert np.allclose(w_sn, np.diag([3.0, 1.0]) / 8.2**0.5) and np.allclose(u1, spectral_norm(w, u0, 1)[1])


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(0.01, 100.0))
def test_length_and_scale_invariance(seed, c):
    rng = np.random.default_rng(seed)
    v, g = rng.normal(size=(3, 5)), rng.normal(size=3)
    w = weight(v, g)
    assert np.allclose(np.linalg.norm(w, axis=1), np.abs(g))
    assert np.allclose(weight(c * v, g), w)


@pytest.mark.parametrize("shape", [(4, 3, 2), (1, 5, 1), (6, 2, 3)])
def test_gradient_finite_differences(shape):
    n, k, m = shape
    rng = np.random.default_rng(n * 100 + k * 10 + m)
    params = {"v": rng.normal(size=(m, k)), "g": rng.normal(size=m), "b": rng.normal(size=m)}
    x, dy = rng.normal(size=(n, k)), rng.normal(size=(n, m))
    y, cache = forward(params, x)
    dx, grads = backward(params, cache, dy)

    def loss(name):
        return lambda a: float(np.sum(dy * forward({**params, name: a}, x)[0]))

    for name in ("v", "g", "b"):
        assert np.allclose(grads[name], fd(loss(name), params[name]), rtol=1e-6, atol=1e-8)
    assert np.allclose(dx, fd(lambda a: float(np.sum(dy * forward(params, a)[0])), x), rtol=1e-6, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**31 - 1), st.floats(1e-3, 10.0))
def test_projection_orthogonality_and_growth(seed, eta):
    rng = np.random.default_rng(seed)
    v, g, grad_w = rng.normal(size=(3, 4)), rng.normal(size=3), rng.normal(size=(3, 4))
    grad_v, grad_g = weight_grads(v, g, grad_w)
    w = weight(v, g)
    for j in range(3):
        m = projection(w[j])
        assert np.allclose(m, m.T) and np.allclose(m @ m, m) and np.allclose(m @ w[j], 0.0)
        assert np.allclose(grad_v[j], g[j] / np.linalg.norm(v[j]) * m @ grad_w[j])
        assert np.isclose(grad_v[j] @ v[j], 0.0, atol=1e-10)
        assert np.isclose(grad_g[j], grad_w[j] @ v[j] / np.linalg.norm(v[j]))
        new = v[j] - eta * grad_v[j]
        assert np.isclose(new @ new, v[j] @ v[j] + eta**2 * grad_v[j] @ grad_v[j])


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(2, 9), st.sampled_from([0.0, 1e-3, 0.5]))
def test_init_moments(seed, n, eps):
    rng = np.random.default_rng(seed)
    params = {"v": rng.normal(0.0, 0.05, size=(3, 4)), "g": np.ones(3), "b": np.zeros(3)}
    x = rng.normal(size=(n, 4)) * 3 + 1
    new = data_dependent_init(params, x, eps=eps)
    assert np.array_equal(new["v"], params["v"]) and new["v"] is not params["v"]
    y = forward(new, x)[0]
    t = x @ params["v"].T / np.linalg.norm(params["v"], axis=1)
    var = t.var(axis=0)
    assert np.allclose(y.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(y.var(axis=0), var / (var + eps), rtol=1e-9)


def test_init_default_eps():
    params = {"v": V, "g": G, "b": B}
    y = forward(data_dependent_init(params, X), X)[0]
    assert np.allclose(y.var(axis=0), 1.0, rtol=1e-12)
    two = data_dependent_init(params, X[:2])
    assert np.allclose(forward(two, X[:2])[0].var(axis=0), 1.0)


def test_mean_only():
    rng = np.random.default_rng(3)
    t, b, dt = rng.normal(size=(5, 3)) + 2, rng.normal(size=3), rng.normal(size=(5, 3))
    out = mean_only_forward(t, b)
    assert np.allclose(out.mean(axis=0), b) and np.allclose(out - out.mean(axis=0), t - t.mean(axis=0))
    grad_t, grad_b = mean_only_backward(dt)
    assert np.allclose(grad_t, fd(lambda a: float(np.sum(dt * mean_only_forward(a, b))), t), atol=1e-8)
    assert np.allclose(grad_b, fd(lambda a: float(np.sum(dt * mean_only_forward(t, a))), b), atol=1e-8)
    assert np.allclose(grad_t.sum(axis=0), 0.0)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(2, 6), st.integers(2, 6))
def test_power_iteration_against_svd(seed, rows, cols):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(rows, cols))
    s = np.linalg.svd(w, compute_uv=False)
    u0 = rng.normal(size=rows)
    one = spectral_norm(w, u0, 1)[0]
    assert one <= s[0] + 1e-12
    sigma, u, v = spectral_norm(w, u0, 1000)
    assert np.isclose(sigma, s[0], rtol=1e-6)
    assert np.allclose(w @ v, sigma * u, atol=1e-5 * s[0])
    w_sn, u_new = spectral_normalize(w, u0, 1000)
    assert np.isclose(np.linalg.svd(w_sn, compute_uv=False)[0], 1.0, rtol=1e-6)
    assert np.allclose(np.abs(u_new), np.abs(u), atol=1e-6)


def test_power_iteration_recycle():
    rng = np.random.default_rng(7)
    w = rng.normal(size=(4, 6))
    u = rng.normal(size=4)
    for _ in range(300):
        w = w + 1e-3 * rng.normal(size=w.shape)
        sigma, u, _ = spectral_norm(w, u, 1)
    assert np.isclose(sigma, np.linalg.norm(w, 2), rtol=1e-4)


def test_against_torch():
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(4, 2).double()
    with torch.no_grad():
        lin.bias.copy_(torch.tensor(B))
    lin = torch.nn.utils.parametrizations.weight_norm(lin, name="weight", dim=0)
    with torch.no_grad():
        lin.parametrizations.weight.original0.copy_(torch.tensor(G).reshape(2, 1))
        lin.parametrizations.weight.original1.copy_(torch.tensor(V))
    xt = torch.tensor(X, requires_grad=True)
    out = lin(xt)
    out.backward(torch.tensor(DY))
    assert np.allclose(out.detach().numpy(), Y, atol=1e-11)
    assert np.allclose(xt.grad.numpy(), DX, atol=1e-11)
    assert np.allclose(lin.parametrizations.weight.original1.grad.numpy(), DV, atol=1e-11)
    assert np.allclose(lin.parametrizations.weight.original0.grad.numpy().ravel(), DG, atol=1e-11)
    w = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
    ref = float(torch.linalg.matrix_norm(torch.tensor(w), ord=2))
    assert np.isclose(spectral_norm(w, np.array([1.0, 0.0]), 50)[0], ref, rtol=1e-12)
