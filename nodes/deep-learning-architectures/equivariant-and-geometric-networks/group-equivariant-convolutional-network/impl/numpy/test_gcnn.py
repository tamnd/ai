# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Tests of the NumPy reference: the group action, the equivariance of both layers, the filter bank against the
definition, the invariance of the model, the gradients, the pitfalls, the shapes and the shared fixture."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gcnn import (
    NAMES,
    config,
    corr2d,
    corr2d_backward,
    cross_entropy,
    flops,
    forward,
    gconv,
    gconv_bank,
    gconv_direct,
    init,
    lift,
    loss_and_grads,
    param_count,
    rotate,
    rotate_p4,
    shift,
    train,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["c_in"], k["c1"], k["c2"], k["k"], k["n_classes"], **kw)


def random_params(rng, cfg):
    params = init(rng, cfg)
    for b in ("b1", "b2", "bc"):
        params[b] = rng.normal(0.0, 0.3, params[b].shape)
    return params


# The group action.

def test_rotate_p4_is_an_action_of_order_four():
    f = np.random.default_rng(0).normal(size=(2, 3, 4, 5, 5))
    np.testing.assert_array_equal(rotate_p4(rotate_p4(f)), rotate_p4(f, 2))
    np.testing.assert_array_equal(rotate_p4(f, 4), f)
    np.testing.assert_array_equal(rotate_p4(rotate_p4(f, 3)), f)


def test_shift_adjoint_is_the_opposite_shift():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 6, 6)), rng.normal(size=(2, 6, 6))
    for mode in ("zero", "circular"):
        for a, b in ((1, 2), (-2, 1), (0, -1)):
            assert np.sum(shift(x, a, b, mode) * y) == pytest.approx(np.sum(x * shift(y, -a, -b, mode)))


# The planar correlation is translation equivariant but not rotation equivariant.

def test_planar_correlation_commutes_with_cyclic_translations():
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=(2, 3, 6, 6)), rng.normal(size=(4, 3, 3, 3))
    np.testing.assert_allclose(corr2d(np.roll(x, (2, -1), (2, 3)), w, "circular"),
                               np.roll(corr2d(x, w, "circular"), (2, -1), (2, 3)), atol=1e-12)


@pytest.mark.parametrize("n", [5, 6])
def test_rotated_input_equals_rotated_output_with_counter_rotated_filter(n):
    # [L_r f] * psi = L_r [f * L_{r^-1} psi]: rotating the image does not rotate the planar feature map, unless the
    # filter turns the other way (eq-planar-rotation).
    rng = np.random.default_rng(3)
    x, w = rng.normal(size=(2, 2, n, n)), rng.normal(size=(3, 2, 3, 3))
    np.testing.assert_allclose(corr2d(rotate(x), w), rotate(corr2d(x, rotate(w, -1))), atol=1e-12)
    assert np.abs(corr2d(rotate(x), w) - rotate(corr2d(x, w))).max() > 0.1


def test_planar_correlation_gradients_are_adjoints():
    rng = np.random.default_rng(4)
    x, w, dout = rng.normal(size=(2, 3, 5, 5)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=(2, 4, 5, 5))
    for mode in ("zero", "circular"):
        dx, dw = corr2d_backward(x, w, dout, mode)
        dx2, dw2 = rng.normal(size=x.shape), rng.normal(size=w.shape)
        lin = lambda x, w: np.sum(corr2d(x, w, mode) * dout)  # noqa: E731, linear in each argument
        assert np.sum(dx * dx2) == pytest.approx(lin(dx2, w))
        assert np.sum(dw * dw2) == pytest.approx(lin(x, dw2))


# The two layers are equivariant: rotate the input, and the output turns and cycles.

@pytest.mark.parametrize("mode", ["zero", "circular"])
@pytest.mark.parametrize("n", [5, 6])
def test_lift_is_rotation_equivariant(mode, n):
    rng = np.random.default_rng(5)
    x, W = rng.normal(size=(2, 2, n, n)), rng.normal(size=(3, 2, 3, 3))
    for r in range(4):
        np.testing.assert_allclose(lift(rotate(x, r), W, mode), rotate_p4(lift(x, W, mode), r), atol=1e-12)


@pytest.mark.parametrize("mode", ["zero", "circular"])
@pytest.mark.parametrize("n", [5, 6])
def test_gconv_is_rotation_equivariant(mode, n):
    rng = np.random.default_rng(6)
    f, W = rng.normal(size=(2, 3, 4, n, n)), rng.normal(size=(2, 3, 4, 5, 5))
    for r in range(4):
        np.testing.assert_allclose(gconv(rotate_p4(f, r), W, mode), rotate_p4(gconv(f, W, mode), r), atol=1e-12)


def test_both_layers_are_equivariant_to_cyclic_translations():
    rng = np.random.default_rng(7)
    x, W1, W2 = rng.normal(size=(1, 2, 6, 6)), rng.normal(size=(3, 2, 3, 3)), rng.normal(size=(2, 3, 4, 3, 3))
    t = lambda a: np.roll(a, (1, 3), (-2, -1))  # noqa: E731
    np.testing.assert_allclose(lift(t(x), W1, "circular"), t(lift(x, W1, "circular")), atol=1e-12)
    f = lift(x, W1, "circular")
    np.testing.assert_allclose(gconv(t(f), W2, "circular"), t(gconv(f, W2, "circular")), atol=1e-12)


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(3, 7), st.sampled_from([1, 3]), st.sampled_from(["zero", "circular"]))
def test_filter_bank_equals_the_definition(seed, n, k, mode):
    # One planar correlation with the augmented bank (eq-bank) is the sum over the group of eq-gconv.
    rng = np.random.default_rng(seed)
    f, W = rng.normal(size=(1, 2, 4, n, n)), rng.normal(size=(3, 2, 4, k, k))
    np.testing.assert_allclose(gconv(f, W, mode), gconv_direct(f, W, mode), atol=1e-11)


def test_filter_bank_is_a_permutation_of_the_weights():
    # Cohen and Welling, section 7: each filter of the bank is the stored filter with its entries permuted, so the
    # bank holds every weight exactly 4 times.
    W = np.arange(2 * 3 * 4 * 3 * 3, dtype=float).reshape(2, 3, 4, 3, 3)
    bank = gconv_bank(W)
    assert bank.shape == (8, 12, 3, 3)
    values, counts = np.unique(bank, return_counts=True)
    np.testing.assert_array_equal(values, np.sort(W.ravel()))
    assert set(counts) == {4}


# The model.

@pytest.mark.parametrize("mode", ["zero", "circular"])
def test_logits_are_invariant_to_rotations(mode):
    rng = np.random.default_rng(8)
    cfg = config(2, 3, 4, 3, 5, mode=mode)
    params = random_params(rng, cfg)
    x = rng.normal(size=(3, 2, 7, 7))
    for r in range(1, 4):
        np.testing.assert_allclose(forward(params, rotate(x, r), cfg), forward(params, x, cfg), atol=1e-12)


def test_logits_are_invariant_to_cyclic_translations():
    rng = np.random.default_rng(9)
    cfg = config(1, 3, 3, 3, 4, mode="circular")
    params = random_params(rng, cfg)
    x = rng.normal(size=(2, 1, 6, 6))
    np.testing.assert_allclose(forward(params, np.roll(x, (2, 5), (2, 3)), cfg), forward(params, x, cfg), atol=1e-12)


def test_planar_cnn_of_the_same_shape_is_not_invariant():
    # Drop the rotated copies: a plain CNN with the same layers but one filter per channel.
    rng = np.random.default_rng(10)
    x, w1, w2 = rng.normal(size=(1, 1, 7, 7)), rng.normal(size=(3, 1, 3, 3)), rng.normal(size=(2, 3, 3, 3))
    g = lambda x: np.maximum(corr2d(np.maximum(corr2d(x, w1), 0), w2), 0).mean(axis=(2, 3))  # noqa: E731
    assert np.abs(g(rotate(x)) - g(x)).max() > 1e-3


@pytest.mark.parametrize("mode", ["zero", "circular"])
def test_gradients_match_finite_differences(mode):
    rng = np.random.default_rng(11)
    cfg = config(2, 2, 3, 3, 3, mode=mode)
    params = random_params(rng, cfg)
    x, labels = rng.normal(size=(3, 2, 5, 5)), np.array([0, 2, 1])
    _, grads, dx = loss_and_grads(params, x, labels, cfg)
    eps = 1e-6
    for k in NAMES:
        for _ in range(3):
            idx = tuple(rng.integers(s) for s in params[k].shape)
            up, dn = ({j: v.copy() for j, v in params.items()} for _ in range(2))
            up[k][idx] += eps
            dn[k][idx] -= eps
            num = (loss_and_grads(up, x, labels, cfg)[0] - loss_and_grads(dn, x, labels, cfg)[0]) / (2 * eps)
            assert grads[k][idx] == pytest.approx(num, rel=1e-5, abs=1e-8)
    idx = (1, 0, 2, 3)
    xp, xm = x.copy(), x.copy()
    xp[idx] += eps
    xm[idx] -= eps
    num = (loss_and_grads(params, xp, labels, cfg)[0] - loss_and_grads(params, xm, labels, cfg)[0]) / (2 * eps)
    assert dx[idx] == pytest.approx(num, rel=1e-5, abs=1e-8)


def test_input_gradient_is_equivariant():
    # The loss is invariant, so its gradient with respect to the image turns with the image.
    rng = np.random.default_rng(12)
    cfg = config(1, 2, 2, 3, 3)
    params = random_params(rng, cfg)
    x, labels = rng.normal(size=(2, 1, 5, 5)), np.array([1, 2])
    np.testing.assert_allclose(loss_and_grads(params, rotate(x), labels, cfg)[2],
                               rotate(loss_and_grads(params, x, labels, cfg)[2]), atol=1e-12)


def test_training_reduces_loss_and_keeps_invariance():
    rng = np.random.default_rng(13)
    cfg = config(1, 3, 3, 3, 2)
    params = init(rng, cfg)
    # Two classes of 6 by 6 images: a bar and an L shape, each in a random rotation and position, plus noise.
    shapes = [np.zeros((3, 3)), np.zeros((3, 3))]
    shapes[0][1, :] = 1.0
    shapes[1][:, 0], shapes[1][2, :] = 1.0, 1.0
    x, labels = np.zeros((16, 1, 6, 6)), np.arange(16) % 2
    for i in range(16):
        u, v = rng.integers(0, 4, 2)
        x[i, 0, u:u + 3, v:v + 3] = rotate(shapes[labels[i]], rng.integers(4))
    x += 0.05 * rng.normal(size=x.shape)
    trained, losses = train(params, x, labels, cfg, lr=0.5, steps=150)
    assert losses[-1] < 0.5 * losses[0]
    np.testing.assert_allclose(forward(trained, rotate(x), cfg), forward(trained, x, cfg), atol=1e-12)


# The pitfalls of the page.

def test_bias_per_rotation_plane_breaks_equivariance():
    rng = np.random.default_rng(14)
    x, W = rng.normal(size=(1, 1, 5, 5)), rng.normal(size=(2, 1, 3, 3))
    b = rng.normal(size=(2, 4))[:, :, None, None]  # one bias per channel and rotation
    out = lambda x: lift(x, W) + b  # noqa: E731
    assert np.abs(out(rotate(x)) - rotate_p4(out(x))).max() > 0.1
    b_shared = b[:, :1]  # one bias per channel, shared by its planes, as in the model
    np.testing.assert_allclose(lift(rotate(x), W) + b_shared, rotate_p4(lift(x, W) + b_shared), atol=1e-12)


def test_rotating_filters_without_cycling_breaks_equivariance():
    # The bank of a G-correlation must also cycle the rotation axis of the filter, (s - r) mod 4 in eq-gconv.
    rng = np.random.default_rng(15)
    f, W = rng.normal(size=(1, 2, 4, 5, 5)), rng.normal(size=(2, 2, 4, 3, 3))
    O, C = 2, 2
    bank = np.stack([rotate(W, r) for r in range(4)], axis=1).reshape(4 * O, 4 * C, 3, 3)  # rotated, not cycled
    wrong = lambda f: corr2d(f.reshape(1, 4 * C, 5, 5), bank).reshape(1, O, 4, 5, 5)  # noqa: E731
    assert np.abs(wrong(rotate_p4(f)) - rotate_p4(wrong(f))).max() > 0.1


def test_stride_two_on_an_even_grid_breaks_rotation_equivariance():
    # Subsampling keeps the pixels with even indices. On a grid of odd side 2m + 1 the kept set is symmetric under
    # rot90; on an even side it is not, and the rotated output is read on a shifted grid.
    rng = np.random.default_rng(16)
    W = rng.normal(size=(2, 1, 3, 3))
    for n, ok in ((7, True), (6, False)):
        x = rng.normal(size=(1, 1, n, n))
        out = lambda x: lift(x, W)[..., ::2, ::2]  # noqa: E731
        gap = np.abs(out(rotate(x)) - rotate_p4(out(x))).max()
        assert (gap < 1e-12) == ok


def test_pooling_rotations_early_loses_relative_orientation():
    # Two p4 maps with a feature at the same two places: in fa the two features have orientations 0 and 1 (an L), in
    # fb both have orientation 0 (parallel). Pooling over rotations right away gives the same planar map, so no later
    # layer can tell them apart; a group correlation before the pooling can, because it sees both orientations.
    fa, fb = np.zeros((1, 1, 4, 7, 7)), np.zeros((1, 1, 4, 7, 7))
    fa[0, 0, 0, 2, 2], fa[0, 0, 1, 2, 4] = 1.0, 1.0
    fb[0, 0, 0, 2, 2], fb[0, 0, 0, 2, 4] = 1.0, 1.0
    np.testing.assert_array_equal(fa.max(axis=2), fb.max(axis=2))
    W = np.zeros((1, 1, 4, 5, 5))
    W[0, 0, 0, 2, 2], W[0, 0, 1, 2, 4] = 1.0, 1.0  # orientation 0 here and orientation 1 two pixels to the right
    pa, pb = (np.maximum(gconv(f, W), 0.0).max(axis=2).max() for f in (fa, fb))
    assert pa == pytest.approx(2.0) and pb == pytest.approx(1.0)


# The shape contract and the shared fixture.

@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(np.random.default_rng(0), cfg)
    x = np.zeros((k["B"], k["c_in"], k["N"], k["N"]))
    h1 = lift(x, params["W1"])
    assert list(h1.shape) == c["blocks"]["lift"]["outputs"]["h1"]
    assert list(gconv(h1, params["W2"]).shape) == c["blocks"]["gconv"]["outputs"]["h2"]
    assert list(gconv_bank(params["W2"]).shape) == c["blocks"]["gconv"]["internals"]["bank"]
    assert list(forward(params, x, cfg).shape) == c["blocks"]["head"]["outputs"]["logits"]
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    assert flops(cfg, k["B"], k["N"]) == c["model"]["flops_forward"]
    assert 4 * (params["W1"].size + params["W2"].size) == c["model"]["planar_weights_same_width"]


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits = forward(params, inp["x"], cfg)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-14)
    loss, grads, dx = loss_and_grads(params, inp["x"], inp["labels"], cfg)
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-10, atol=1e-12)
    for k in NAMES:
        np.testing.assert_allclose(grads[k], want[f"d{k}"], rtol=1e-10, atol=1e-12)


def test_cross_entropy_gradient():
    rng = np.random.default_rng(17)
    z, y = rng.normal(size=(4, 3)), np.array([0, 2, 1, 1])
    _, g = cross_entropy(z, y)
    eps = 1e-6
    dz = np.zeros_like(z)
    dz[2, 1] = eps
    assert g[2, 1] == pytest.approx((cross_entropy(z + dz, y)[0] - cross_entropy(z - dz, y)[0]) / (2 * eps), rel=1e-6)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), rng.normal(size=(k["B"], k["c_in"], k["N"], k["N"])), cfg)
