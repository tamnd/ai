# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Tests of the NumPy reference: the worked example, the shape contract, the shared fixture and the properties."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from resnet import (
    batch_norm,
    bottleneck,
    config,
    conv2d,
    cross_entropy,
    dense_backward,
    dense_forward,
    dense_init,
    depth,
    flops,
    forward,
    init,
    init_state,
    max_pool,
    param_count,
    relu,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]  # forward runs on the example and the samples


def cfg_of(c):
    k = c["config"]
    return config(k["width"], (k["n1"], k["n2"], k["n3"], k["n4"]), k["classes"])


def unflatten(flat):
    tree = {}
    for path, v in flat.items():
        *head, last = path.split("/")
        node = tree
        for part in head:
            node = node.setdefault(part, {})
        node[last] = v

    def lists(t):
        if not isinstance(t, dict):
            return t
        if t and all(k.isdigit() for k in t):
            return [lists(t[str(i)]) for i in range(len(t))]
        return {k: lists(v) for k, v in t.items()}

    return lists(tree)


def leaves(tree):
    items = tree.values() if isinstance(tree, dict) else tree
    for v in items:
        if isinstance(v, (dict, list)):
            yield from leaves(v)
        else:
            yield v


def unit(x, W1, W2, residual=True):
    """The block of the worked example: two 1x1 convolutions on a 1-pixel image, no normalization."""
    f = conv2d(relu(conv2d(x, W1)), W2)
    return relu(f + x) if residual else relu(f)


def as_conv(W):
    return np.asarray(W, dtype=float)[:, :, None, None]


X = np.array([2.0, -1.0]).reshape(1, 2, 1, 1)
W1 = as_conv([[1.0, 1.0], [1.0, -1.0]])
W2 = as_conv([[0.5, 0.0], [0.0, -1.0]])


def test_worked_example():
    # F(x) = W2 relu(W1 x) = W2 (1, 3) = (0.5, -3); the block outputs relu(F(x) + x) = (2.5, 0).
    np.testing.assert_allclose(unit(X, W1, W2).ravel(), [2.5, 0.0], rtol=1e-12)
    np.testing.assert_allclose(unit(X, W1, W2, residual=False).ravel(), [0.5, 0.0], rtol=1e-12)
    # With W2 = 0 the residual block keeps relu(x); the plain block loses x.
    np.testing.assert_allclose(unit(X, W1, 0 * W2).ravel(), [2.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(unit(X, W1, 0 * W2, residual=False).ravel(), [0.0, 0.0], rtol=1e-12)


def test_worked_example_counts():
    # The first block of stage 1 of ResNet-50: 64 -> 64 -> 64 -> 256 and a 64 -> 256 projection.
    convs = 64 * 64 + 9 * 64 * 64 + 64 * 256 + 64 * 256
    norms = 2 * (64 + 64 + 256 + 256)
    assert convs + norms == 18 * 64**2 + 20 * 64 == 75_008
    assert param_count(config()) == 25_557_032


def block_id(s, i):
    return f"s{s + 1}_{'first' if i == 0 else 'rest'}"


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    x = rng.normal(size=(k["B"], 3, k["R"], k["R"]))
    logits, cache = forward(params, x, cfg)
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["pooled"].shape) == blocks["head"]["internals"]["pooled"]
    assert list(cache["stem"]["conv"].shape) == blocks["stem"]["internals"]["conv"]
    assert list(cache["stem"]["y"].shape) == blocks["stem"]["outputs"]["y"]
    for (s, i), got in cache["blocks"].items():
        want = blocks[block_id(s, i)]
        assert list(got["y"].shape) == want["outputs"]["y"], (s, i)
        for name, shape in want["internals"].items():
            assert list(got[name].shape) == shape, (s, i, name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k = cfg_of(c), c["config"]
    assert param_count(cfg) == c["model"]["params"]
    assert flops(cfg, k["B"], k["R"]) == c["model"]["flops_forward"]
    assert depth(cfg) == c["model"]["depth"]
    if c["kind"] != "named":  # the arrays of the paper's models are too large for a test
        assert sum(a.size for a in leaves(init(np.random.default_rng(0), cfg))) == c["model"]["params"]


def test_paper_flops():
    # Table 1 of He et al. (2016) gives 3.8, 7.6 and 11.3 x 10^9 FLOPs, counting multiply-adds. The count here is
    # 3.86, 7.57 and 11.28: the last two round to the table, the first is 0.06 above it (c-flops-table).
    for blocks, want in (((3, 4, 6, 3), 3.8), ((3, 4, 23, 3), 7.6), ((3, 8, 36, 3), 11.3)):
        assert abs(flops(config(blocks=blocks), 1, 224) / 2e9 - want) < 0.07


def test_stride_on_3x3():
    # v1.5 moves the stride to the 3x3: the same parameters and shapes, more FLOPs.
    a, b = config(2, (1, 2, 1, 1), 5), config(2, (1, 2, 1, 1), 5, stride_on="3x3")
    assert param_count(a) == param_count(b) and flops(b, 2, 64) > flops(a, 2, 64)
    params = init(np.random.default_rng(0), a)
    x = np.random.default_rng(1).normal(size=(2, 3, 64, 64))
    la, ca = forward(params, x, a)
    lb, cb = forward(params, x, b)
    assert la.shape == lb.shape and not np.allclose(la, lb)
    assert all(ca["blocks"][key]["y"].shape == cb["blocks"][key]["y"].shape for key in ca["blocks"])


def fixture():
    params = unflatten(dict(np.load(FIXTURES / "params-small.npz")))
    state = unflatten(dict(np.load(FIXTURES / "state-small.npz")))
    x = np.load(FIXTURES / "input-small.npz")["x"]
    return params, state, x, dict(np.load(FIXTURES / "expected-small.npz")), cfg_of(SHAPES[0])


def test_shared_fixture():
    params, state, x, want, cfg = fixture()
    logits, cache = forward(params, x, cfg)
    np.testing.assert_allclose(logits, want["logits_train"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache["stem"]["y"], want["stem_train"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache["blocks"][(1, 0)]["c"], want["s2_first_c_train"], rtol=1e-12, atol=1e-12)
    logits, cache = forward(params, x, cfg, state)
    np.testing.assert_allclose(logits, want["logits_infer"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache["blocks"][(1, 0)]["c"], want["s2_first_c_infer"], rtol=1e-12, atol=1e-12)


def test_conv2d_against_loops():
    rng = np.random.default_rng(3)
    x, W = rng.normal(size=(2, 3, 7, 7)), rng.normal(size=(4, 3, 3, 3))
    for stride, pad in ((1, 0), (1, 1), (2, 1), (2, 3)):
        got = conv2d(x, W, stride, pad)
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        H = (7 + 2 * pad - 3) // stride + 1
        want = np.zeros((2, 4, H, H))
        for i in range(H):
            for j in range(H):
                patch = xp[:, :, i * stride : i * stride + 3, j * stride : j * stride + 3]
                want[:, :, i, j] = np.einsum("bckl,ockl->bo", patch, W)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_max_pool():
    x = np.arange(16.0).reshape(1, 1, 4, 4) - 20.0  # all negative: the padding must not win
    np.testing.assert_array_equal(max_pool(x)[0, 0], [[-15.0, -13.0], [-7.0, -5.0]])


def test_zero_gamma_makes_blocks_identity():
    # c-zero-gamma: with the last scale of each block at 0, a block with the identity shortcut passes its input.
    cfg = config(2, (2, 2, 2, 2), 5)
    params = init(np.random.default_rng(0), cfg, zero_gamma=True)
    _, cache = forward(params, np.random.default_rng(1).normal(size=(2, 3, 32, 32)), cfg)
    for s in range(4):
        np.testing.assert_allclose(cache["blocks"][(s, 1)]["y"], cache["blocks"][(s, 0)]["y"], rtol=0, atol=0)


def test_identity_shortcut_has_no_parameters():
    p = init(np.random.default_rng(0), config(2, (1, 2, 1, 1), 5))["stages"][1]
    assert "proj" in p[0] and set(p[1]) == {"conv1", "bn1", "conv2", "bn2", "conv3", "bn3"}


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_batch_norm_standardizes(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(3.0, 5.0, size=(4, 3, 5, 5))
    p = {"gamma": rng.uniform(0.5, 2.0, 3), "beta": rng.normal(size=3)}
    y = batch_norm(x, p)
    np.testing.assert_allclose(y.mean(axis=(0, 2, 3)), p["beta"], atol=1e-10)
    np.testing.assert_allclose(y.std(axis=(0, 2, 3)), p["gamma"], rtol=1e-4)


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(1, 3))
def test_conv_commutes_with_shifts(seed, shift):
    # A convolution with stride 1 is translation equivariant away from the border.
    rng = np.random.default_rng(seed)
    x, W = rng.normal(size=(1, 2, 12, 12)), rng.normal(size=(3, 2, 3, 3))
    moved = np.roll(x, shift, axis=3)
    a, b = conv2d(x, W), conv2d(moved, W)
    np.testing.assert_allclose(b[..., shift:], a[..., : a.shape[-1] - shift], rtol=1e-10, atol=1e-10)


def test_bottleneck_rest_keeps_shape():
    rng = np.random.default_rng(0)
    p = init(rng, config(2, (2, 1, 1, 1), 5))["stages"][0][1]
    x = relu(rng.normal(size=(2, 8, 5, 5)))
    y, cache = bottleneck(p, x, 1)
    assert y.shape == x.shape and cache["shortcut"] is x


def numeric_grad(f, a, eps=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old = a[idx]
        a[idx] = old + eps
        up = f()
        a[idx] = old - eps
        down = f()
        a[idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("train", [False, True])
def test_dense_backward_matches_finite_differences(residual, train):
    rng = np.random.default_rng(5)
    p, state = dense_init(rng, 6, 4, 2, 3)
    for k in p:  # move the scales and shifts away from 1 and 0
        if k[0] in "gb" and k != "bo":
            p[k] += rng.normal(0.0, 0.3, p[k].shape)
    for k in state:
        state[k] = state[k] + rng.uniform(0.1, 0.5, state[k].shape)
    x, y = rng.normal(size=(8, 6)), rng.integers(0, 3, 8)

    def loss():
        return cross_entropy(dense_forward(p, dict(state), x, residual, train)[0], y)[0]

    logits, cache = dense_forward(p, dict(state), x, residual, train)
    grads = dense_backward(p, cache, cross_entropy(logits, y)[1], residual)
    assert set(grads) == set(p)
    for k in p:
        np.testing.assert_allclose(grads[k], numeric_grad(loss, p[k]), rtol=1e-5, atol=1e-8, err_msg=k)


def test_dense_running_statistics():
    rng = np.random.default_rng(0)
    p, state = dense_init(rng, 5, 4, 1, 2)
    x = rng.normal(2.0, 1.0, size=(64, 5))
    before = {k: v.copy() for k, v in state.items()}
    dense_forward(p, state, x, residual=True, train=True)
    z = x @ p["W0"]
    np.testing.assert_allclose(state["m0"], 0.9 * before["m0"] + 0.1 * z.mean(axis=0), rtol=1e-12)
    dense_forward(p, state, x, residual=True, train=False)  # inference leaves them alone
    np.testing.assert_allclose(state["m0"], 0.9 * before["m0"] + 0.1 * z.mean(axis=0), rtol=1e-12)


def test_init_state_matches_norms():
    params = init(np.random.default_rng(0), config(2, (1, 1, 1, 1), 5))
    state = init_state(params)
    assert set(state["stages"][0][0]) == {"bn1", "bn2", "bn3", "bn_proj"}
    assert state["stem"]["bn"]["var"].shape == (2,)
