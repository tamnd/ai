# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rms-normalization/ (ai://NEW-rms-normalization).
"""Tests of the NumPy reference: the worked example, the gradient against finite differences, its orthogonality to
the input, the scaling invariance, the link to layer normalization, pRMSNorm, and agreement with
torch.nn.functional.rms_norm when PyTorch is installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rms_normalization import backward, config, forward, init, layer_norm, partial_count

# The worked example of the page: n = 4, eps = 0, gain 1.
WA = np.array([4.0, 0.0, -2.0, 4.0])
WY = [4 / 3, 0.0, -2 / 3, 4 / 3]
WDY = np.array([1.0, 0.0, 0.0, 0.0])
WDX = np.array([5.0, 0.0, 2.0, -4.0]) / 27

# The shared fixture. The PyTorch tests hold the same numbers.
X = np.array([[0.5, -1.5, 2.0, 1.0], [1.0, 3.0, -2.5, 0.0], [-1.25, 0.5, 1.5, 4.0]])
DY = np.array([[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 3.0], [0.25, -0.5, 2.0, -1.0]])
GAIN = np.array([1.5, 0.5, 2.0, -1.0])
Y = [[0.547722411446, -0.547722411446, 2.921186194378, -0.730296548595],
     [0.74420831594, 0.74420831594, -2.480694386468, 0.0],
     [-0.837218274818, 0.111629103309, 1.339549239709, -1.786065652945]]
DX = [[1.351048478578, -0.036514418464, -0.438178474444, 0.146059037075],
      [-1.442619208327, 0.385461709308, -0.114493558885, -1.488416631881],
      [0.429128650266, -0.21630310143, 1.472043658581, -0.390875571733]]
DGAIN = [-0.766665859426, -0.814102117212, -0.12104385748, -1.420917378648]
X3 = ((np.arange(24.0) * 7) % 11 - 5).reshape(2, 3, 4) / 2
DY3 = ((np.arange(24.0) * 5) % 13 - 6).reshape(2, 3, 4) / 3

vectors = hnp.arrays(np.float64, st.integers(2, 12), elements=st.floats(-10, 10, allow_nan=False))


def numeric_grad(fun, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[idx] = h
        g[idx] = (fun(a + e) - fun(a - e)) / (2 * h)
    return g


def test_worked_example():
    cfg = config(4, eps=0.0)
    p = init(cfg)
    y, cache = forward(p, WA, cfg)
    np.testing.assert_allclose(y, WY, atol=1e-12)
    np.testing.assert_allclose(cache["r"], [3.0], atol=1e-12)
    dx, grads = backward(p, cache, WDY)
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    np.testing.assert_allclose(grads["gain"], [4 / 3, 0, 0, 0], atol=1e-12)
    assert abs(WA @ dx) < 1e-12
    dx2, _ = backward(p, forward(p, 2 * WA, cfg)[1], WDY)
    np.testing.assert_allclose(dx2, WDX / 2, atol=1e-12)
    np.testing.assert_allclose(forward(p, WA, config(4, eps=0.0, p=0.5))[0], WA / np.sqrt(8.0), atol=1e-12)


def test_worked_example_layer_norm_gap():
    y = forward({}, WA, config(4, eps=0.0, elementwise_affine=False))[0]
    z = layer_norm(WA, eps=0.0)
    np.testing.assert_allclose(z, np.array([2.5, -1.5, -3.5, 2.5]) / np.sqrt(6.75), atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean((y - z) ** 2)), np.sqrt(2 - np.sqrt(3)), atol=1e-12)
    np.testing.assert_allclose(y, np.sqrt(3) / 2 * z + 0.5, atol=1e-12)


def test_shared_fixture():
    cfg = config(4, eps=1e-6)
    p = {"gain": GAIN.copy()}
    y, cache = forward(p, X, cfg)
    np.testing.assert_allclose(y, Y, atol=1e-11)
    dx, grads = backward(p, cache, DY)
    np.testing.assert_allclose(dx, DX, atol=1e-11)
    np.testing.assert_allclose(grads["gain"], DGAIN, atol=1e-11)


@pytest.mark.parametrize(("x", "dy"), [(X, DY), (X3, DY3), (X[0], DY[0])])
@pytest.mark.parametrize("prob", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("eps", [0.0, 1e-3, 0.5])
def test_gradient_finite_differences(x, dy, prob, eps):
    cfg = config(4, eps=eps, p=prob)
    rng = np.random.default_rng(20261116)
    p = {"gain": rng.normal(size=4)}
    _, cache = forward(p, x, cfg)
    dx, grads = backward(p, cache, dy)
    np.testing.assert_allclose(dx, numeric_grad(lambda a: float(np.sum(dy * forward(p, a, cfg)[0])), x),
                               rtol=1e-6, atol=1e-8)
    gnum = numeric_grad(lambda g: float(np.sum(dy * forward({"gain": g}, x, cfg)[0])), p["gain"])
    np.testing.assert_allclose(grads["gain"], gnum, rtol=1e-6, atol=1e-8)


def test_no_gain():
    cfg = config(4, eps=1e-3, elementwise_affine=False)
    assert init(cfg) == {}
    y, cache = forward({}, X, cfg)
    np.testing.assert_allclose(y, X / np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + 1e-3), atol=1e-12)
    dx, grads = backward({}, cache, DY)
    assert grads == {}
    np.testing.assert_allclose(dx, numeric_grad(lambda a: float(np.sum(DY * forward({}, a, cfg)[0])), X),
                               rtol=1e-6, atol=1e-8)


@settings(max_examples=60, deadline=None)
@given(vectors, st.floats(1e-3, 1e3), st.floats(0.05, 1.0))
def test_scale_invariance(a, delta, prob):
    if np.mean(a ** 2) < 1e-6:
        return
    cfg = config(a.size, eps=0.0, p=prob)
    if np.mean(a[: cfg["k"]] ** 2) < 1e-6:
        return
    p = {"gain": np.linspace(-1, 2, a.size)}
    np.testing.assert_allclose(forward(p, delta * a, cfg)[0], forward(p, a, cfg)[0], rtol=1e-9, atol=1e-9)


def test_eps_breaks_scale_invariance():
    cfg = config(4, eps=0.5)
    assert not np.allclose(forward({}, 3 * X, cfg)[0], forward({}, X, cfg)[0])


@settings(max_examples=60, deadline=None)
@given(vectors, vectors, st.floats(0.05, 1.0))
def test_gradient_orthogonal_to_input(a, d, prob):
    n = min(a.size, d.size)
    a, d = a[:n], d[:n]
    cfg = config(n, eps=0.0, p=prob)
    if np.mean(a[: cfg["k"]] ** 2) < 1e-3:
        return
    p = {"gain": np.linspace(0.5, 1.5, n)}
    dx, _ = backward(p, forward(p, a, cfg)[1], d)
    assert abs(a @ dx) <= 1e-9 * (1 + np.abs(a).sum() * np.abs(dx).sum() + np.abs(d).sum())


def test_eps_leak_of_orthogonality():
    eps = 0.25
    cfg = config(4, eps=eps)
    dx, _ = backward({}, forward({}, X[0], cfg)[1], DY[0])
    m = np.mean(X[0] ** 2)
    r = np.sqrt(m + eps)
    np.testing.assert_allclose(X[0] @ dx, (X[0] @ DY[0]) / r * eps / (m + eps), rtol=1e-12)


@settings(max_examples=60, deadline=None)
@given(vectors, st.floats(0.0, 1.0))
def test_layer_norm_is_rms_of_centred_input(a, eps):
    c = a - a.mean()
    if np.mean(c ** 2) + eps < 1e-6:
        return
    cfg = config(a.size, eps=eps)
    g = np.linspace(-2, 2, a.size)
    np.testing.assert_allclose(layer_norm(a, g, eps), forward({"gain": g}, c, cfg)[0], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mu", [-4.0, -0.2, 0.05, 1.0, 7.0])
def test_mean_gap(mu):
    a = X[1] - X[1].mean()
    sigma = np.sqrt(np.mean(a ** 2))
    b = a + mu
    rho = mu / sigma
    y = forward({}, b, config(4, eps=0.0, elementwise_affine=False))[0]
    z = layer_norm(b, eps=0.0)
    np.testing.assert_allclose(forward({}, a, config(4, eps=0.1))[0], layer_norm(a, eps=0.1), atol=1e-12)
    np.testing.assert_allclose(y, (z + rho) / np.sqrt(1 + rho ** 2), atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean((y - z) ** 2)), np.sqrt(2 - 2 / np.sqrt(1 + rho ** 2)), rtol=1e-9)


def test_partial_uses_only_first_k():
    cfg = config(4, eps=1e-6, p=0.5)
    assert cfg["k"] == 2
    x = X.copy()
    x[:, 2:] *= 10.0  # the tail does not enter the statistic
    y, cache = forward({}, x, cfg)
    np.testing.assert_allclose(cache["r"], np.sqrt(np.mean(X[:, :2] ** 2, axis=-1, keepdims=True) + 1e-6))
    np.testing.assert_allclose(y[:, :2], forward({}, X, cfg)[0][:, :2], atol=1e-12)


@pytest.mark.parametrize(("n", "p", "k"), [(16, 0.0625, 1), (10, 0.3, 3), (5, 1.0, 5), (7, 0.5, 4), (3, 1e-12, 1),
                                           (512, 0.0625, 32), (1, 0.4, 1)])
def test_partial_count(n, p, k):
    assert partial_count(n, p) == k


def test_errors():
    with pytest.raises(ValueError, match="p must be in"):
        partial_count(4, 0.0)
    with pytest.raises(ValueError, match="p must be in"):
        partial_count(4, 1.5)
    with pytest.raises(ValueError, match="n must be positive"):
        partial_count(0, 0.5)
    with pytest.raises(ValueError, match="eps must be non-negative"):
        config(4, eps=-1e-3)
    assert config(4, eps=0.0)["eps"] == 0.0
    with pytest.raises(ValueError, match="last axis of x must have size 4"):
        forward({}, np.ones((2, 3)), config(4))
    with pytest.raises(ValueError, match="last axis"):
        forward({}, np.float64(1.0), config(1))
    assert config(4)["eps"] == 1e-6 and config(4)["k"] == 4 and config(4)["affine"] is True
    np.testing.assert_array_equal(init(config(3))["gain"], np.ones(3))


def test_torch_rms_norm_agrees():
    torch = pytest.importorskip("torch")
    for eps in [1e-6, 0.3]:
        y, _ = forward({"gain": GAIN}, X, config(4, eps=eps))
        ref = torch.nn.functional.rms_norm(torch.tensor(X), (4,), torch.tensor(GAIN), eps=eps)
        np.testing.assert_allclose(y, ref.numpy(), rtol=1e-12, atol=1e-12)
