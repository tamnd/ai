# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://NEW-mixup).
"""Tests of the NumPy reference: the worked example, the shared fixture, the two forms of the mixed loss, the pairing
of inputs and labels, the Beta moments, the CutMix box and its area-adjusted weight, and integer inputs."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mixup import (
    beta_variance,
    cutmix_batch,
    cutmix_box,
    cutmix_lambda,
    log_softmax,
    mixed_cross_entropy,
    mixup_batch,
    mixup_targets,
    one_hot,
    sample_cutmix_box,
    sample_lambda,
    soft_cross_entropy,
)

# The shared fixture. The PyTorch tests check these numbers too.
LOGITS = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
Y_A = [0, 2, 3]
Y_B = [1, 2, 0]
FIXTURE = {0.0: 2.3319850661352497, 0.3: 1.9569850661352495, 0.7: 1.4569850661352497, 1.0: 1.0819850661352495}
# An 8-wide, 6-high image, lam = 0.64 and the centre (1, 5): a 3 x 3 box in the corner, weight 0.8125.
CUTMIX = {"height": 6, "width": 8, "lam": 0.64, "center": (1.0, 5.0), "box": (0, 3, 3, 6), "adjusted": 0.8125}


def test_worked_example():
    x, _, _ = mixup_batch([[2.0, 0.0], [0.0, 4.0]], [0, 2], 0.7, [1, 0])
    np.testing.assert_allclose(x[0], [1.4, 1.2], atol=1e-15)
    q = mixup_targets([0, 2], 3, 0.7, [1, 0])
    np.testing.assert_allclose(q[0], [0.7, 0.0, 0.3], atol=1e-15)
    z = [[1.0, 0.5, -0.5]]
    np.testing.assert_allclose(np.exp(log_softmax(z))[0], [0.546549, 0.331499, 0.121952], atol=1e-6)
    assert soft_cross_entropy(z, [[0.7, 0.0, 0.3]]) == pytest.approx(1.054131, abs=1e-6)
    assert mixed_cross_entropy(z, [0], [2], 0.7) == pytest.approx(1.054131, abs=1e-6)
    assert beta_variance(0.2) == pytest.approx(1 / 5.6, abs=1e-15)
    assert beta_variance(1.0) == pytest.approx(1 / 12, abs=1e-15)
    box = cutmix_box(32, 32, 0.75, (4.0, 28.0))
    assert box == (0, 12, 20, 32)
    assert cutmix_lambda(box, 32, 32) == 0.859375


@pytest.mark.parametrize("lam", sorted(FIXTURE))
def test_fixture(lam):
    assert mixed_cross_entropy(LOGITS, Y_A, Y_B, lam) == pytest.approx(FIXTURE[lam], abs=1e-12)
    q = lam * one_hot(Y_A, 4) + (1 - lam) * one_hot(Y_B, 4)
    assert soft_cross_entropy(LOGITS, q) == pytest.approx(FIXTURE[lam], abs=1e-12)


def test_cutmix_fixture():
    box = cutmix_box(CUTMIX["height"], CUTMIX["width"], CUTMIX["lam"], CUTMIX["center"])
    assert box == CUTMIX["box"]
    assert cutmix_lambda(box, CUTMIX["height"], CUTMIX["width"]) == CUTMIX["adjusted"]


@settings(max_examples=200, deadline=None)
@given(
    st.integers(1, 12),
    st.integers(2, 9),
    st.floats(0.0, 1.0),
    st.integers(0, 2**32 - 1),
)
def test_two_forms_of_the_loss_agree(b, k, lam, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=(b, k))
    y = rng.integers(k, size=b)
    perm = rng.permutation(b)
    q = mixup_targets(y, k, lam, perm)
    np.testing.assert_allclose(q.sum(axis=-1), 1.0, atol=1e-12)
    assert soft_cross_entropy(z, q) == pytest.approx(mixed_cross_entropy(z, y, y[perm], lam), abs=1e-10)


def test_labels_follow_inputs():
    # Inputs that encode their own label: the mixed input must equal the mixed target.
    rng = np.random.default_rng(0)
    k, b = 5, 16
    y = rng.integers(k, size=b)
    x = one_hot(y, k)
    for _ in range(20):
        lam = sample_lambda(0.4, rng)
        perm = rng.permutation(b)
        xm, ya, yb = mixup_batch(x, y, lam, perm)
        np.testing.assert_allclose(xm, mixup_targets(y, k, lam, perm), atol=1e-15)
        np.testing.assert_array_equal(ya, y)
        np.testing.assert_array_equal(yb, y[perm])
        np.testing.assert_allclose(xm, lam * one_hot(ya, k) + (1 - lam) * one_hot(yb, k), atol=1e-15)


def test_endpoints():
    x = np.arange(12.0).reshape(4, 3)
    perm = [3, 2, 1, 0]
    np.testing.assert_array_equal(mixup_batch(x, [0, 1, 2, 3], 1.0, perm)[0], x)
    np.testing.assert_array_equal(mixup_batch(x, [0, 1, 2, 3], 0.0, perm)[0], x[perm])
    np.testing.assert_allclose(mixup_batch(x, [0, 1, 2, 3], 0.25, perm)[0], 0.25 * x + 0.75 * x[perm])


@pytest.mark.parametrize("lam", [-0.1, 1.1, float("nan")])
def test_lam_out_of_range(lam):
    with pytest.raises(ValueError):
        mixup_batch(np.zeros((2, 1)), [0, 1], lam, [1, 0])
    with pytest.raises(ValueError):
        mixup_targets([0, 1], 2, lam, [1, 0])
    with pytest.raises(ValueError):
        mixed_cross_entropy([[0.0, 0.0]], [0], [1], lam)
    with pytest.raises(ValueError):
        cutmix_box(4, 4, lam, (2.0, 2.0))


@pytest.mark.parametrize("perm", [[0, 0, 1], [0, 1], [0, 1, 3]])
def test_bad_permutation(perm):
    with pytest.raises(ValueError):
        mixup_batch(np.zeros((3, 2)), [0, 1, 2], 0.5, perm)
    with pytest.raises(ValueError):
        mixup_targets([0, 1, 2], 3, 0.5, perm)
    with pytest.raises(ValueError):
        cutmix_batch(np.zeros((3, 1, 4, 4)), [0, 1, 2], (0, 2, 0, 2), perm)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan")])
def test_alpha_must_be_positive(alpha):
    with pytest.raises(ValueError):
        sample_lambda(alpha, np.random.default_rng(0))


@pytest.mark.parametrize("alpha", [0.1, 0.2, 1.0, 4.0, 32.0])
def test_beta_moments(alpha):
    rng = np.random.default_rng(1)
    lam = np.array([sample_lambda(alpha, rng) for _ in range(40000)])
    assert np.all((lam >= 0) & (lam <= 1))
    tol = 5 * math.sqrt(beta_variance(alpha) / lam.size)
    assert abs(lam.mean() - 0.5) < tol
    assert abs(lam.var() - beta_variance(alpha)) < tol
    assert beta_variance(alpha) == pytest.approx(1 / (4 * (2 * alpha + 1)), abs=1e-15)


def test_small_alpha_is_erm():
    rng = np.random.default_rng(2)
    lam = np.array([sample_lambda(0.01, rng) for _ in range(20000)])
    assert np.mean((lam < 0.01) | (lam > 0.99)) > 0.9
    assert beta_variance(1e-12) == pytest.approx(0.25, abs=1e-10)


@settings(max_examples=300, deadline=None)
@given(
    st.integers(1, 40),
    st.integers(1, 40),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
)
def test_cutmix_lambda_is_pixel_fraction(h, w, lam, fx, fy):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 1, h, w))
    box = cutmix_box(h, w, lam, (fx * w, fy * h))
    mixed, ya, yb, adjusted = cutmix_batch(x, [4, 7], box, [1, 0])
    from_a = (mixed[0, 0] == x[0, 0]).mean()
    assert from_a == pytest.approx(adjusted, abs=1e-12)
    assert adjusted == pytest.approx(cutmix_lambda(box, h, w), abs=0)
    assert list(ya) == [4, 7] and list(yb) == [7, 4]
    # The second image of the batch received the box of the first.
    x1, x2, y1, y2 = box
    np.testing.assert_array_equal(mixed[1, 0, y1:y2, x1:x2], x[0, 0, y1:y2, x1:x2])


@settings(max_examples=300, deadline=None)
@given(st.integers(1, 64), st.integers(1, 64), st.floats(0.0, 1.0), st.integers(0, 2**32 - 1))
def test_box_inside_image(h, w, lam, seed):
    x1, x2, y1, y2 = sample_cutmix_box(h, w, lam, np.random.default_rng(seed))
    assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h
    assert all(isinstance(v, int) for v in (x1, x2, y1, y2))


@pytest.mark.parametrize("lam", [0.0, 0.19, 0.36, 0.75, 0.91, 1.0])
def test_box_area_before_clipping(lam):
    # A box centred in a large image is not clipped and covers 1 - lam of it, up to rounding.
    h, w = 400, 600
    x1, x2, y1, y2 = cutmix_box(h, w, lam, (w / 2, h / 2))
    assert x2 - x1 == pytest.approx(w * math.sqrt(1 - lam), abs=1)
    assert y2 - y1 == pytest.approx(h * math.sqrt(1 - lam), abs=1)
    assert cutmix_lambda((x1, x2, y1, y2), h, w) == pytest.approx(lam, abs=2 * (h + w) / (h * w))


def test_unadjusted_lambda_is_wrong_near_edges():
    # The sampled lam overstates the patch whenever the box runs off the image.
    lam = 0.5
    corner = cutmix_box(32, 32, lam, (0.0, 0.0))
    assert cutmix_lambda(corner, 32, 32) == pytest.approx(1 - 11 * 11 / 1024)
    assert cutmix_lambda(corner, 32, 32) - lam > 0.38
    # With lam ~ U(0, 1) and a uniform centre the mean adjusted weight is about 163 / 240, not 1 / 2.
    rng = np.random.default_rng(4)
    vals = [cutmix_lambda(sample_cutmix_box(224, 224, rng.uniform(), rng), 224, 224) for _ in range(20000)]
    assert np.mean(vals) == pytest.approx(163 / 240, abs=6e-3)


def test_integer_inputs_are_mixed_in_float():
    x = np.array([[250, 10]], dtype=np.uint8).repeat(2, axis=0)
    x[1] = [10, 250]
    mixed, _, _ = mixup_batch(x, [0, 1], 0.5, [1, 0])
    assert mixed.dtype.kind == "f"
    np.testing.assert_allclose(mixed, [[130.0, 130.0], [130.0, 130.0]])
    # CutMix copies pixels and keeps the dtype.
    img = np.arange(32, dtype=np.uint8).reshape(2, 1, 4, 4)
    out, _, _, _ = cutmix_batch(img, [0, 1], (0, 2, 0, 2), [1, 0])
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0, :2, :2], img[1, 0, :2, :2])
    np.testing.assert_array_equal(out[0, 0, 2:, :], img[0, 0, 2:, :])
    np.testing.assert_array_equal(img[0, 0, 0], [0, 1, 2, 3])
