# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/layer-normalization/ (ai://RA5R92).
"""Tests of the NumPy reference: the worked example, the gradients, the Jacobian, the invariances of Section 5 of Ba
et al., independence from the batch, RMSNorm, the two placements in a residual block, and agreement with
torch.nn.LayerNorm, torch.nn.RMSNorm and torch.nn.functional.group_norm when PyTorch is installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from layer_normalization import (backward, config, forward, init, post_ln, pre_ln, rms_backward, rms_config,
                                 rms_forward)

# The worked example of the page: one example with H = 5 summed inputs, eps = 0.
WA = np.array([3.0, -4.0, 7.0, -1.0, 5.0])
WGAIN = np.array([2.0, 1.0, 2.0, 1.0, 4.0])
WBIAS = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
WXHAT = [0.25, -1.5, 1.25, -0.75, 0.75]
WY = [0.5, -0.5, 2.5, 0.25, 3.0]
WDY = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
WDX = np.array([-0.625, 1.75, 0.875, -2.125, 0.125]) / 20

# The shared fixture. The PyTorch and JAX tests hold the same numbers.
X = np.array([[0.5, -1.5, 2.0, 1.0], [1.0, 3.0, -2.5, 0.0], [-1.25, 0.5, 1.5, 4.0]])
DY = np.array([[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 3.0], [0.25, -0.5, 2.0, -1.0]])
GAIN = np.array([1.5, 0.5, 2.0, -1.0])
BIAS = np.array([0.1, -0.2, 0.3, 0.0])
X3 = ((np.arange(24.0) * 7) % 11 - 5).reshape(2, 3, 4) / 2  # normalized_shape (3, 4)
DY3 = ((np.arange(24.0) * 5) % 13 - 6).reshape(2, 3, 4) / 3


def params():
    return {"gain": GAIN.copy(), "bias": BIAS.copy()}


def numeric_grad(fun, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[idx] = h
        g[idx] = (fun(a + e) - fun(a - e)) / (2 * h)
    return g


def jacobian(x, cfg, rms=False):
    fwd, bwd = (rms_forward, rms_backward) if rms else (forward, backward)
    _, cache = fwd({}, x, cfg)
    return np.stack([bwd({}, cache, e)[0] for e in np.eye(x.size)])  # row k is the gradient of y_k


def test_worked_example():
    cfg = config(5, eps=0.0)
    p = {"gain": WGAIN, "bias": WBIAS}
    y, cache = forward(p, WA, cfg)
    np.testing.assert_allclose(cache["z"], WXHAT, atol=1e-12)
    np.testing.assert_allclose(y, WY, atol=1e-12)
    dx, grads = backward(p, cache, WDY)
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    np.testing.assert_allclose(grads["gain"], [0.0, -1.5, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(grads["bias"], WDY, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(jacobian(WA, cfg), 2), 0.25, rtol=1e-12)


def test_rms_worked_example():
    cfg = rms_config(5, eps=0.0)
    centered = WA - WA.mean()
    np.testing.assert_allclose(rms_forward({}, centered, cfg)[0], WXHAT, atol=1e-12)
    np.testing.assert_allclose(rms_forward({}, WA, cfg)[0], WA / np.sqrt(20.0), atol=1e-12)


@pytest.mark.parametrize(("x", "dy", "shape"), [(X, DY, 4), (X3, DY3, (3, 4)), (X[0], DY[0], 4)])
@pytest.mark.parametrize("rms", [False, True])
def test_gradient_finite_differences(x, dy, shape, rms):
    cfg = rms_config(shape, eps=1e-3) if rms else config(shape, eps=1e-3)
    fwd, bwd = (rms_forward, rms_backward) if rms else (forward, backward)
    rng = np.random.default_rng(20261115)
    p = {"gain": rng.normal(size=cfg["shape"])}
    if not rms:
        p["bias"] = rng.normal(size=cfg["shape"])
    _, cache = fwd(p, x, cfg)
    dx, grads = bwd(p, cache, dy)
    np.testing.assert_allclose(dx, numeric_grad(lambda a: float(np.sum(dy * fwd(p, a, cfg)[0])), x),
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(grads["gain"], numeric_grad(
        lambda g: float(np.sum(dy * fwd({**p, "gain": g}, x, cfg)[0])), p["gain"]), rtol=1e-6, atol=1e-8)


def test_statistics_treated_as_constants_are_wrong():
    _, cache = forward(params(), X, config(4))
    dx, _ = backward(params(), cache, DY)
    assert np.abs(dx - GAIN * cache["inv"] * DY).max() > 0.1


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(-10, 10), min_size=3, max_size=12), st.integers(0, 10**6))
def test_input_gradient_sums_to_zero_and_is_orthogonal(values, seed):
    a = np.array(values)
    if a.std() < 1e-3:
        return
    cfg = config(a.size, eps=0.0)
    p = {"gain": np.random.default_rng(seed).normal(size=a.size), "bias": np.zeros(a.size)}
    _, cache = forward(p, a, cfg)
    dx, _ = backward(p, cache, np.random.default_rng(seed + 1).normal(size=a.size))
    scale = np.abs(dx).max() + 1.0
    assert abs(dx.sum()) <= 1e-9 * scale * a.size
    assert abs(np.dot(dx, cache["z"])) <= 1e-9 * scale * a.size


@pytest.mark.parametrize("eps", [0.0, 0.5])
def test_jacobian_spectral_norm(eps):
    rng = np.random.default_rng(1)
    for h in [3, 8, 64]:
        a = rng.normal(size=h) * 3 + 2
        j = jacobian(a, config(h, eps=eps))
        s = np.sqrt(a.var() + eps)
        np.testing.assert_allclose(np.linalg.norm(j, 2), 1.0 / s, rtol=1e-10)
        if eps == 0.0:
            np.testing.assert_allclose(np.linalg.norm(j, 2), np.sqrt(h) / np.linalg.norm(a - a.mean()), rtol=1e-10)
        np.testing.assert_allclose(j, j.T, atol=1e-12)


def test_output_on_sphere():
    y, _ = forward({}, X * 10 + 3, config(4, eps=0.0))
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), 2.0, rtol=1e-12)
    z, _ = rms_forward({}, X * 10 + 3, rms_config(4, eps=0.0))
    np.testing.assert_allclose(np.linalg.norm(z, axis=-1), 2.0, rtol=1e-12)


@pytest.mark.parametrize("rms", [False, True])
def test_batch_independence(rms):
    cfg = rms_config(4) if rms else config(4)
    fwd = rms_forward if rms else forward
    p = {"gain": GAIN} if rms else params()
    y = fwd(p, X, cfg)[0]
    for i in range(X.shape[0]):
        np.testing.assert_allclose(fwd(p, X[i:i + 1], cfg)[0], y[i:i + 1], rtol=1e-14)  # batch size 1
        np.testing.assert_allclose(fwd(p, X[i], cfg)[0], y[i], rtol=1e-14)
    other = X.copy()
    other[1:] = 100 * np.random.default_rng(3).normal(size=(2, 4))
    np.testing.assert_allclose(fwd(p, other, cfg)[0][0], y[0], rtol=1e-14)


def test_train_equals_eval():
    """There is no mode and no state: the same input gives the same output, before and after other batches."""
    cfg = config(4)
    y1 = forward(params(), X, cfg)[0]
    for k in range(3):
        forward(params(), X * (k + 2) - k, cfg)
    np.testing.assert_array_equal(forward(params(), X, cfg)[0], y1)


def test_invariances():
    rng = np.random.default_rng(20261115)
    w, x, gamma = rng.normal(size=(6, 5)), rng.normal(size=(7, 5)), rng.normal(size=5)
    ln, rms = config(6, eps=0.0), rms_config(6, eps=0.0)
    gain = {"gain": rng.normal(size=6)}

    def out(fwd, cfg, w2, x2):
        return fwd(gain, x2 @ w2.T, cfg)[0]

    base, base_rms = out(forward, ln, w, x), out(rms_forward, rms, w, x)
    w2 = 2.5 * w + np.outer(np.ones(6), gamma)  # Ba et al., Eq. 6
    np.testing.assert_allclose(out(forward, ln, w2, x), base, atol=1e-12)
    np.testing.assert_allclose(out(forward, ln, w, x * rng.uniform(0.1, 9, size=(7, 1))), base, atol=1e-12)
    np.testing.assert_allclose(out(rms_forward, rms, 2.5 * w, x), base_rms, atol=1e-12)
    np.testing.assert_allclose(out(rms_forward, rms, w, x * rng.uniform(0.1, 9, size=(7, 1))), base_rms, atol=1e-12)
    one_row = w.copy()
    one_row[0] *= 3.0
    assert np.abs(out(forward, ln, one_row, x) - base).max() > 1e-3
    assert np.abs(out(rms_forward, rms, one_row, x) - base_rms).max() > 1e-3
    assert np.abs(out(forward, ln, w, x + gamma) - base).max() > 1e-3  # dataset re-centering
    assert np.abs(out(rms_forward, rms, w2, x) - base_rms).max() > 1e-3  # RMSNorm is not re-centering invariant


def test_per_element_affine():
    cfg = config((3, 4))
    p = init(None, cfg)
    assert p["gain"].shape == (3, 4) and p["bias"].shape == (3, 4)
    p = {"gain": np.arange(12.0).reshape(3, 4), "bias": -np.arange(12.0).reshape(3, 4)}
    y, cache = forward(p, X3, cfg)
    np.testing.assert_allclose(y, cache["z"] * p["gain"] + p["bias"], rtol=1e-14)
    assert set(init(None, config(4, elementwise_affine=False))) == set()
    assert set(init(None, config(4, bias=False))) == {"gain"}


def test_pre_and_post_placement():
    norm = lambda v: forward({}, v, config(4, eps=0.0))[0]  # noqa: E731
    zero = lambda v: 0.0 * v  # noqa: E731
    np.testing.assert_allclose(pre_ln(X, zero, norm), X, rtol=1e-14)
    np.testing.assert_allclose(post_ln(X, zero, norm), norm(X), rtol=1e-14)
    x = X
    for _ in range(4):
        x = post_ln(x, lambda v: 3.0 * v + 1.0, norm)
        np.testing.assert_allclose(np.linalg.norm(x, axis=-1), 2.0, rtol=1e-12)


def test_matches_torch_layernorm():
    torch = pytest.importorskip("torch")
    for x, dy, shape in [(X, DY, (4,)), (X3, DY3, (3, 4))]:
        for affine, bias in [(True, True), (True, False), (False, True)]:
            cfg = config(shape, elementwise_affine=affine, bias=bias)
            ref = torch.nn.LayerNorm(shape, elementwise_affine=affine, bias=bias, dtype=torch.float64)
            p = init(None, cfg)
            rng = np.random.default_rng(7)
            with torch.no_grad():
                for k, name in [("gain", "weight"), ("bias", "bias")]:
                    if k in p:
                        p[k] = rng.normal(size=shape)
                        getattr(ref, name).copy_(torch.tensor(p[k]))
            y, cache = forward(p, x, cfg)
            xt = torch.tensor(x, requires_grad=True)
            yt = ref(xt)
            np.testing.assert_allclose(y, yt.detach().numpy(), rtol=1e-12, atol=1e-12)
            yt.backward(torch.tensor(dy))
            dx, grads = backward(p, cache, dy)
            np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-10, atol=1e-12)
            if affine:
                np.testing.assert_allclose(grads["gain"], ref.weight.grad.numpy(), rtol=1e-10, atol=1e-12)
            f = torch.nn.functional.layer_norm(torch.tensor(x), shape, eps=1e-5)
            np.testing.assert_allclose(forward({}, x, cfg)[0], f.numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eps", [None, 1e-6])
def test_matches_torch_rmsnorm(eps):
    torch = pytest.importorskip("torch")
    for x, dy, shape in [(X, DY, (4,)), (X3, DY3, (3, 4))]:
        cfg = rms_config(shape, eps=eps)
        ref = torch.nn.RMSNorm(shape, eps=eps, dtype=torch.float64)
        p = {"gain": np.random.default_rng(8).normal(size=shape)}
        with torch.no_grad():
            ref.weight.copy_(torch.tensor(p["gain"]))
        y, cache = rms_forward(p, x, cfg)
        xt = torch.tensor(x, requires_grad=True)
        yt = ref(xt)
        np.testing.assert_allclose(y, yt.detach().numpy(), rtol=1e-12, atol=1e-12)
        yt.backward(torch.tensor(dy))
        dx, grads = rms_backward(p, cache, dy)
        np.testing.assert_allclose(dx, xt.grad.numpy(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(grads["gain"], ref.weight.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_group_norm_with_one_group_is_layer_norm():
    torch = pytest.importorskip("torch")
    x = ((np.arange(96.0) * 7) % 13 - 6).reshape(2, 3, 4, 4) / 3  # (N, C, H, W)
    ours = forward({}, x, config((3, 4, 4)))[0]
    gn = torch.nn.functional.group_norm(torch.tensor(x), 1, eps=1e-5).numpy()
    np.testing.assert_allclose(ours, gn, rtol=1e-12, atol=1e-12)
    gn3 = torch.nn.functional.group_norm(torch.tensor(x), 3, eps=1e-5).numpy()
    assert np.abs(ours - gn3).max() > 1e-2


def test_config_errors():
    with pytest.raises(ValueError):
        config(0)
    with pytest.raises(ValueError):
        config(4, eps=-1.0)
    with pytest.raises(ValueError):
        forward({}, np.zeros((3, 5)), config(4))
    with pytest.raises(ValueError):
        rms_forward({}, np.zeros(3), rms_config((3, 4)))
    assert rms_config(4)["eps"] == np.finfo(np.float64).eps
