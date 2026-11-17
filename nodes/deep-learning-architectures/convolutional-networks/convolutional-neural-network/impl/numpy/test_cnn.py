# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""Tests of the NumPy reference: the worked example, the loop oracle, the size formula, equivariance, the
receptive field, a numeric gradient check, the shape contract and the shared fixture."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cnn import (
    conv2d,
    conv2d_backward,
    conv2d_loop,
    conv_params,
    config,
    flops,
    forward,
    im2col,
    init,
    max_pool,
    out_size,
    param_count,
    receptive_field,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
NAMES = ["conv1", "pool1", "conv2", "pool2"]

# The worked example of the page: a 3x3 input and a 2x2 kernel.
X3 = np.arange(9.0).reshape(1, 1, 3, 3)
W2 = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)


def cfg_of(c):
    k = c["config"]
    return config(k["channels"], k["c1"], k["c2"], k["f1"], k["f2"], k["classes"], k["size"])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    x = rng.normal(size=(k["B"], k["channels"], k["size"], k["size"]))  # (B, C, R, R)
    return forward, (params, x, cfg)


def test_worked_example():
    np.testing.assert_array_equal(conv2d(X3, W2)[0, 0], [[27.0, 37.0], [57.0, 67.0]])
    # True convolution flips the kernel; it gives other numbers on the same input.
    np.testing.assert_array_equal(conv2d(X3, W2[:, :, ::-1, ::-1])[0, 0], [[13.0, 23.0], [43.0, 53.0]])
    np.testing.assert_array_equal(conv2d(X3, W2, pad=1, stride=2)[0, 0], [[0.0, 11.0], [30.0, 67.0]])


@settings(max_examples=40, deadline=None)
@given(n=st.integers(1, 12), k=st.integers(1, 5), s=st.integers(1, 4), p=st.integers(0, 3))
def test_out_size_counts_windows(n, k, s, p):
    if n + 2 * p < k:
        with pytest.raises(ValueError):
            out_size(n, k, s, p)
        return
    starts = [a for a in range(n + 2 * p) if a % s == 0 and a + k <= n + 2 * p]
    assert out_size(n, k, s, p) == len(starts)
    y = conv2d(np.ones((1, 1, n, n)), np.ones((1, 1, k, k)), stride=s, pad=p)
    assert y.shape == (1, 1, len(starts), len(starts))


def test_out_size_examples():
    assert out_size(32, 5) == 28  # LeNet-5, C1
    assert out_size(28, 2, 2) == 14  # LeNet-5, S2
    assert out_size(224, 3, 1, 1) == 224  # VGG, padding 1 keeps the size
    assert out_size(224, 11, 4) == 54 and out_size(227, 11, 4) == 55 and out_size(224, 11, 4, 2) == 55


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), s=st.integers(1, 3), p=st.integers(0, 2), k=st.integers(1, 4),
       circular=st.booleans())
def test_im2col_equals_loop(seed, s, p, k, circular):
    rng = np.random.default_rng(seed)
    x, w, b = rng.normal(size=(2, 3, 6, 5)), rng.normal(size=(4, 3, k, k)), rng.normal(size=4)
    mode = "circular" if circular else "zeros"
    if circular and p > 5:
        return
    np.testing.assert_allclose(conv2d(x, w, b, s, p, mode), conv2d_loop(x, w, b, s, p, mode), rtol=1e-12, atol=1e-12)


def test_im2col_is_a_view():
    x = np.arange(32.0).reshape(1, 2, 4, 4)
    cols = im2col(x, 3)
    assert cols.shape == (1, 2, 2, 2, 3, 3)
    np.testing.assert_array_equal(cols[0, 1, 1, 0], x[0, 1, 1:4, 0:3])


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), t1=st.integers(-6, 6), t2=st.integers(-6, 6), k=st.sampled_from([1, 3, 5]))
def test_equivariance_circular(seed, t1, t2, k):
    rng = np.random.default_rng(seed)
    x, w = rng.normal(size=(1, 2, 8, 8)), rng.normal(size=(3, 2, k, k))
    p = (k - 1) // 2
    f = lambda z: conv2d(z, w, pad=p, mode="circular")  # noqa: E731
    roll = lambda z: np.roll(z, (t1, t2), axis=(2, 3))  # noqa: E731
    np.testing.assert_allclose(f(roll(x)), roll(f(x)), rtol=1e-12, atol=1e-12)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), t1=st.integers(0, 4), t2=st.integers(0, 4), k=st.sampled_from([1, 3, 5]))
def test_equivariance_of_crops(seed, t1, t2, k):
    # A shift is a crop of a larger image at another offset; a valid convolution commutes with it.
    rng = np.random.default_rng(seed)
    big, w = rng.normal(size=(1, 2, 20, 20)), rng.normal(size=(3, 2, k, k))
    a, n = 3, 10
    crop = big[:, :, a + t1 : a + t1 + n, a + t2 : a + t2 + n]
    full = conv2d(big, w)
    np.testing.assert_allclose(conv2d(crop, w), full[:, :, a + t1 : a + t1 + n - k + 1, a + t2 : a + t2 + n - k + 1],
                               rtol=1e-12, atol=1e-12)


def test_zero_padding_breaks_equivariance_at_the_border():
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(1, 1, 8, 8)), rng.normal(size=(1, 1, 3, 3))
    y, ys = conv2d(x, w, pad=1), conv2d(np.roll(x, 1, axis=3), w, pad=1)
    np.testing.assert_allclose(ys[..., 2:7], y[..., 1:6], rtol=1e-12)  # the interior moves with the input
    assert not np.allclose(ys, np.roll(y, 1, axis=3))  # the border does not


def test_stride_gives_periodic_equivariance():
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=(1, 1, 8, 8)), rng.normal(size=(1, 1, 3, 3))
    f = lambda z: conv2d(z, w, stride=2, pad=1, mode="circular")  # noqa: E731
    np.testing.assert_allclose(f(np.roll(x, 2, axis=3)), np.roll(f(x), 1, axis=3), rtol=1e-12)
    y1 = f(np.roll(x, 1, axis=3))
    assert not any(np.allclose(y1, np.roll(f(x), t, axis=3)) for t in range(4))


def test_pooling_invariance():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 1, 8, 8))
    p = max_pool(x)
    np.testing.assert_array_equal(max_pool(np.roll(x, 2, axis=3)), np.roll(p, 1, axis=3))  # periodic-2
    w = rng.normal(size=(4, 1, 3, 3))
    g = lambda z: conv2d(z, w, pad=1, mode="circular").max(axis=(2, 3))  # noqa: E731
    for t in [(1, 0), (0, 3), (5, 7)]:
        np.testing.assert_allclose(g(np.roll(x, t, axis=(2, 3))), g(x), rtol=1e-12)  # global pooling is invariant


@pytest.mark.parametrize(("stack", "expected"), [
    ([(3, 1)] * 2, (5, 1)),
    ([(3, 1)] * 3, (7, 1)),
    ([(5, 1), (2, 2), (5, 1), (2, 2), (5, 1)], (32, 4)),
    ([(11, 4), (3, 2), (5, 1)], (51, 8)),
])
def test_receptive_field(stack, expected):
    assert receptive_field(stack) == expected
    # Measure it: the inputs that one output unit depends on, through kernels of ones.
    n = 64
    x = np.zeros((1, 1, n, n))
    shapes = [x.shape]
    for k, s in stack:
        shapes.append((1, 1, out_size(shapes[-1][2], k, s), out_size(shapes[-1][3], k, s)))
    dy = np.zeros(shapes[-1])
    dy[0, 0, 0, 0] = 1.0
    for (k, s), shp in zip(reversed(stack), reversed(shapes[:-1])):
        dy, _, _ = conv2d_backward(np.zeros(shp), np.ones((1, 1, k, k)), dy, stride=s)
    rows = np.nonzero(dy[0, 0].any(axis=1))[0]
    assert rows.min() == 0 and rows.max() + 1 == expected[0]


@pytest.mark.parametrize(("s", "p"), [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_gradient_numeric(s, p):
    rng = np.random.default_rng(4)
    x, w, b = rng.normal(size=(2, 2, 6, 6)), rng.normal(size=(3, 2, 3, 3)), rng.normal(size=3)
    g = rng.normal(size=conv2d(x, w, b, s, p).shape)
    loss = lambda x_, w_, b_: float((conv2d(x_, w_, b_, s, p) * g).sum())  # noqa: E731
    dx, dw, db = conv2d_backward(x, w, g, s, p)
    eps = 1e-6
    for arr, grad, which in [(x, dx, 0), (w, dw, 1), (b, db, 2)]:
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            args_p, args_m = [x.copy(), w.copy(), b.copy()], [x.copy(), w.copy(), b.copy()]
            args_p[which][idx] += eps
            args_m[which][idx] -= eps
            num[idx] = (loss(*args_p) - loss(*args_m)) / (2 * eps)
        np.testing.assert_allclose(grad, num, rtol=1e-6, atol=1e-7)


def test_parameter_counts():
    assert conv_params(5, 1, 6) == 156  # LeNet-5, C1
    assert conv_params(5, 16, 120) == 48_120  # LeNet-5, C5
    assert 60 * 25 + 16 == 1_516  # LeNet-5, C3: 60 input maps over the 16 maps of Table I
    assert conv_params(3, 1, 1, False) * 2 == 18 and conv_params(5, 1, 1, False) == 25
    assert param_count(config()) == 61_706


def test_shape_contract():
    rng = np.random.default_rng(5)
    for c in SHAPES:
        cfg, B = cfg_of(c), c["config"]["B"]
        assert param_count(cfg) == c["model"]["params"], c["name"]
        assert flops(cfg, B) == c["model"]["flops_forward"], c["name"]
        params = init(rng, cfg)
        assert sum(v.size for p in params.values() for v in p.values()) == c["model"]["params"]
        x = rng.normal(size=c["blocks"]["conv1"]["inputs"]["x"])
        logits, cache = forward(params, x, cfg)
        assert list(logits.shape) == c["blocks"]["fc3"]["outputs"]["logits"]
        for name in NAMES:
            assert list(cache[name].shape) == c["blocks"][name]["outputs"]["y"], (c["name"], name)


def test_shared_fixture():
    flat = dict(np.load(FIXTURES / "params-small.npz"))
    inputs, expected = dict(np.load(FIXTURES / "input-small.npz")), dict(np.load(FIXTURES / "expected-small.npz"))
    params = {}
    for path, v in flat.items():
        layer, leaf = path.split("/")
        params.setdefault(layer, {})[leaf] = v
    cfg = cfg_of(next(c for c in SHAPES if c["kind"] == "example"))
    logits, cache = forward(params, inputs["x"], cfg)
    np.testing.assert_allclose(logits, expected["logits"], rtol=1e-12, atol=1e-12)
    for name in NAMES:
        np.testing.assert_allclose(cache[name], expected[name], rtol=1e-12, atol=1e-12)
    lx, lw, lb = inputs["layer_x"], inputs["layer_w"], inputs["layer_b"]
    np.testing.assert_allclose(conv2d_loop(lx, lw, lb, 2, 1), expected["layer_zeros"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(conv2d_loop(lx, lw, lb, 2, 1, "circular"), expected["layer_circular"], rtol=1e-12,
                               atol=1e-12)


def test_errors():
    with pytest.raises(ValueError):
        out_size(3, 5)
    with pytest.raises(ValueError):
        conv2d(np.zeros((1, 2, 4, 4)), np.zeros((1, 3, 3, 3)))
    with pytest.raises(ValueError):
        config(size=30)


def test_config_errors_and_init_scale():
    assert config(size=16)["size"] == 16
    for bad in ({"size": 12}, {"size": 14}, {"size": 15}, {"channels": 0}, {"f2": 0}):
        with pytest.raises(ValueError):
            config(**bad)
    params = init(np.random.default_rng(0), config(f1=400, f2=400))
    assert abs(params["fc2"]["w"].std() - np.sqrt(2.0 / 400)) < 0.005  # He initialization
    assert abs(params["conv2"]["w"].std() - np.sqrt(2.0 / (25 * 6))) < 0.01
