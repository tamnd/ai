# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://88YEPR).
"""Tests of the NumPy reference: the worked example, each image operation, the magnitude map, the RandAugment and
TrivialAugment samplers, the search-space counts, the hit probability and the expected Cutout area."""

import math

import numpy as np
import pytest

from randaugment import (
    OPS,
    RANGES,
    apply_op,
    autoaugment_space,
    cutout,
    cutout_expected_area,
    identity,
    magnitude_to_arg,
    posterize,
    prob_applied,
    randaugment,
    randaugment_grid,
    randaugment_sample,
    rotate,
    solarize,
    translate_x,
    translate_y,
    trivialaugment,
    trivialaugment_sample,
)

# The shared fixture of the page. The PyTorch tests check the same numbers.
IMG = np.array(
    [[10, 50, 90, 130], [170, 210, 250, 30], [70, 110, 150, 190], [230, 20, 60, 100]],
    dtype=np.uint8,
)
POSTERIZE_4 = [
    [0, 48, 80, 128],
    [160, 208, 240, 16],
    [64, 96, 144, 176],
    [224, 16, 48, 96],
]
SOLARIZE_128 = [
    [10, 50, 90, 125],
    [85, 45, 5, 30],
    [70, 110, 105, 65],
    [25, 20, 60, 100],
]
CUTOUT_2_AT_0_3 = [
    [10, 50, 0, 0],
    [170, 210, 250, 30],
    [70, 110, 150, 190],
    [230, 20, 60, 100],
]
TRANSLATE_X_1 = [
    [0, 10, 50, 90],
    [0, 170, 210, 250],
    [0, 70, 110, 150],
    [0, 230, 20, 60],
]


def rand_image(rng, h, w, c=None):
    shape = (h, w) if c is None else (h, w, c)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_worked_example():
    np.testing.assert_array_equal(posterize(IMG, 4), POSTERIZE_4)
    np.testing.assert_array_equal(solarize(IMG, 128), SOLARIZE_128)
    np.testing.assert_array_equal(cutout(IMG, 2, (0, 3)), CUTOUT_2_AT_0_3)
    np.testing.assert_array_equal(translate_x(IMG, 1), TRANSLATE_X_1)
    np.testing.assert_array_equal(rotate(IMG, 90), np.rot90(IMG))
    assert magnitude_to_arg("posterize", 9) == 4
    assert magnitude_to_arg("solarize", 9) == pytest.approx(25.6)
    assert magnitude_to_arg("rotate", 9) == pytest.approx(27.0)
    assert prob_applied(14, 2) == pytest.approx(27 / 196)
    assert cutout_expected_area(16, 32) == 196.0


def test_images_are_validated():
    for bad in [IMG.astype(np.float32), IMG[0], IMG[None, None]]:
        with pytest.raises(ValueError):
            identity(bad)
    with pytest.raises(ValueError):
        rotate(IMG.astype(np.int16), 10)


def test_identity_copies():
    out = identity(IMG)
    np.testing.assert_array_equal(out, IMG)
    out[0, 0] = 99
    assert IMG[0, 0] == 10


@pytest.mark.parametrize("h,w", [(4, 4), (5, 5), (6, 6), (7, 7)])
def test_rotate_right_angles(h, w):
    img = rand_image(np.random.default_rng(h), h, w, 3)
    np.testing.assert_array_equal(rotate(img, 0), img)
    np.testing.assert_array_equal(rotate(img, 90), np.rot90(img, 1))
    np.testing.assert_array_equal(rotate(img, -90), np.rot90(img, -1))
    np.testing.assert_array_equal(rotate(img, 180), img[::-1, ::-1])
    np.testing.assert_array_equal(rotate(img, 270), np.rot90(img, 3))


def test_rotate_non_square_and_fill():
    img = rand_image(np.random.default_rng(1), 4, 6)
    np.testing.assert_array_equal(rotate(img, 180), img[::-1, ::-1])
    out = rotate(np.full((5, 5), 200, np.uint8), 45, fill=7)
    assert out[2, 2] == 200
    assert out[0, 0] == out[0, 4] == out[4, 0] == out[4, 4] == 7
    assert out[0, 2] == 200
    assert out.dtype == np.uint8


def test_rotate_small_angle_nearest():
    img = np.arange(49, dtype=np.uint8).reshape(7, 7)
    out = rotate(img, 10)
    assert out[3, 3] == img[3, 3]
    # At 10 degrees the point one step right of the centre moves less than half a pixel, so it keeps its value.
    assert out[3, 4] == img[3, 4] and out[3, 2] == img[3, 2]
    # Three steps right of the centre it moves by about 0.52 pixel up, so the pixel below supplies it.
    assert out[3, 6] == img[4, 6]
    # The top-centre pixel is fed from its right, since the content turns counter-clockwise.
    assert out[0, 3] == img[0, 4]


def test_translate():
    img = rand_image(np.random.default_rng(2), 5, 6, 3)
    out = translate_x(img, 2)
    np.testing.assert_array_equal(out[:, 2:], img[:, :4])
    assert (out[:, :2] == 0).all()
    out = translate_x(img, -2, fill=9)
    np.testing.assert_array_equal(out[:, :4], img[:, 2:])
    assert (out[:, 4:] == 9).all()
    out = translate_y(img, 3)
    np.testing.assert_array_equal(out[3:], img[:2])
    assert (out[:3] == 0).all()
    out = translate_y(img, -1)
    np.testing.assert_array_equal(out[:4], img[1:])
    assert (out[4:] == 0).all()
    np.testing.assert_array_equal(translate_x(img, 0), img)
    assert (translate_x(img, 6) == 0).all() and (translate_x(img, -6, fill=3) == 3).all()
    assert (translate_y(img, 5) == 0).all()
    np.testing.assert_array_equal(translate_y(img, 4)[4], img[0])


def test_posterize():
    vals = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(posterize(vals[None], 8)[0], vals)
    np.testing.assert_array_equal(posterize(vals[None], 1)[0], np.where(vals >= 128, 128, 0))
    np.testing.assert_array_equal(posterize(vals[None], 6)[0], vals // 4 * 4)
    for bad in [0, 9]:
        with pytest.raises(ValueError):
            posterize(IMG, bad)


def test_solarize():
    vals = np.arange(256, dtype=np.uint8)[None]
    np.testing.assert_array_equal(solarize(vals, 256), vals)
    np.testing.assert_array_equal(solarize(vals, 0), 255 - vals)
    out = solarize(vals, 100)[0]
    assert out[99] == 99 and out[100] == 155 and out[255] == 0
    assert out.dtype == np.uint8


def test_cutout():
    img = np.full((6, 6, 3), 255, np.uint8)
    out = cutout(img, 4, (3, 3), fill=1)
    assert (out[1:5, 1:5] == 1).all() and (out.sum(axis=2) == 765).sum() == 36 - 16
    out = cutout(img, 4, (0, 5))
    assert (out[0:2, 3:6] == 0).all() and (out == 0).sum() == 6 * 3
    out = cutout(img, 4, (5, 0))
    assert (out[3:6, 0:2] == 0).all() and (out == 0).sum() == 6 * 3
    np.testing.assert_array_equal(cutout(img, 0, (2, 2)), img)
    assert (cutout(img, 20, (2, 2)) == 0).all()
    assert img.min() == 255


def test_magnitude_map():
    assert [magnitude_to_arg("rotate", m) for m in (0, 5, 10)] == [0.0, 15.0, 30.0]
    assert magnitude_to_arg("translate_x", 3) == pytest.approx(3.0)
    assert magnitude_to_arg("translate_y", 10) == 10.0
    assert [magnitude_to_arg("posterize", m) for m in (0, 4, 10)] == [8, 6, 4]
    assert [magnitude_to_arg("solarize", m) for m in (0, 10)] == [256.0, 0.0]
    assert magnitude_to_arg("cutout", 5) == pytest.approx(0.1)
    assert magnitude_to_arg("identity", 7) == 0.0
    assert magnitude_to_arg("rotate", 15, m_max=30) == pytest.approx(15.0)
    assert magnitude_to_arg("posterize", 30, m_max=30) == 4
    for op, m in [("rotate", -1), ("rotate", 11), ("blur", 3)]:
        with pytest.raises(ValueError):
            magnitude_to_arg(op, m)
    assert set(OPS) == set(RANGES)


@pytest.mark.parametrize("op", OPS)
def test_apply_op_dispatch(op):
    img = rand_image(np.random.default_rng(3), 20, 20, 3)
    rng = np.random.default_rng(4)
    out = apply_op(img, op, 5, rng)
    arg = magnitude_to_arg(op, 5)
    if op == "identity":
        expected = [img]
    elif op == "rotate":
        expected = [rotate(img, arg), rotate(img, -arg)]
    elif op == "translate_x":
        expected = [translate_x(img, 5), translate_x(img, -5)]
    elif op == "translate_y":
        expected = [translate_y(img, 5), translate_y(img, -5)]
    elif op == "posterize":
        expected = [posterize(img, 6)]
    elif op == "solarize":
        expected = [solarize(img, 128.0)]
    else:
        expected = [cutout(img, 2, (r, c)) for r in range(20) for c in range(20)]
        assert not np.array_equal(out, img)
    assert any(np.array_equal(out, e) for e in expected)


def test_cutout_op_centre_and_size():
    img = np.full((20, 20), 255, np.uint8)
    rng = np.random.default_rng(5)
    rows, sizes = set(), set()
    for _ in range(400):
        out = apply_op(img, "cutout", 10, rng)
        r, c = np.nonzero(out == 0)
        rows.update(r.tolist())
        sizes.add(int((out == 0).sum()))
        assert int(r.max() - r.min()) <= 3 and int(c.max() - c.min()) <= 3
    assert sizes.issuperset({16, 4, 8}) and max(sizes) == 16
    assert rows == set(range(20))


def test_signs_are_balanced():
    img = np.zeros((9, 9), np.uint8)
    img[4, 4] = 255
    rng = np.random.default_rng(6)
    right = sum(int(apply_op(img, "translate_x", 2, rng)[4, 6] == 255) for _ in range(2000))
    assert 900 < right < 1100
    down = sum(int(apply_op(img, "translate_y", 2, rng)[6, 4] == 255) for _ in range(2000))
    assert 900 < down < 1100


def test_randaugment_sample():
    rng = np.random.default_rng(7)
    pol = randaugment_sample(OPS, 3, 9, rng)
    assert len(pol) == 3 and all(m == 9 and op in OPS for op, m in pol)
    assert randaugment_sample(OPS, 0, 9, rng) == []
    with pytest.raises(ValueError):
        randaugment_sample(OPS, -1, 9, rng)
    counts = {op: 0 for op in OPS}
    repeats = 0
    for _ in range(7000):
        pol = randaugment_sample(OPS, 2, 1, rng)
        repeats += pol[0][0] == pol[1][0]
        for op, _ in pol:
            counts[op] += 1
    for c in counts.values():
        assert abs(c - 2000) < 200
    # With replacement: the two draws agree with probability 1/K = 1/7.
    assert abs(repeats / 7000 - 1 / 7) < 0.02


def test_randaugment_applies_the_sampled_ops_in_order():
    img = rand_image(np.random.default_rng(8), 12, 12, 3)
    for seed in range(20):
        out = randaugment(img, 3, 7, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        ref = img
        for op, m in randaugment_sample(OPS, 3, 7, rng):
            ref = apply_op(ref, op, m, rng)
        np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(randaugment(img, 0, 7, np.random.default_rng(0)), img)
    out = randaugment(img, 2, 10, np.random.default_rng(0), ops=("solarize",))
    np.testing.assert_array_equal(out, img)


def test_trivialaugment():
    rng = np.random.default_rng(9)
    ops, mags = set(), []
    for _ in range(6200):
        op, m = trivialaugment_sample(OPS, rng)
        ops.add(op)
        mags.append(m)
    assert ops == set(OPS)
    counts = np.bincount(mags, minlength=31)
    assert len(counts) == 31 and counts.min() > 130 and counts.max() < 270
    op, m = trivialaugment_sample(("rotate",), rng, m_max=0)
    assert (op, m) == ("rotate", 0)
    img = rand_image(np.random.default_rng(10), 8, 8)
    for seed in range(20):
        out = trivialaugment(img, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        op, m = trivialaugment_sample(OPS, rng)
        np.testing.assert_array_equal(out, apply_op(img, op, m, rng, 30))
    out = trivialaugment(img, np.random.default_rng(0), ops=("posterize",), m_max=1)
    assert np.array_equal(out, img) or np.array_equal(out, posterize(img, 4))


def test_search_space_counts():
    assert autoaugment_space() == 1760**10
    assert f"{autoaugment_space():.1e}" == "2.9e+32"
    assert autoaugment_space(n_sub=1) == (16 * 10 * 11) ** 2
    assert autoaugment_space(16, 10, 11, 2, 25) == 1760**50
    assert autoaugment_space(14, 31, 11, 2, 5) == (14 * 31 * 11) ** 10
    assert randaugment_grid([1, 2, 3], [5, 7, 9, 11, 13, 15]) == 18
    assert randaugment_grid([1, 2, 2], [0, 1]) == 4
    assert 14**2 == 196


def test_prob_applied_monte_carlo():
    rng = np.random.default_rng(11)
    for k, n in [(14, 1), (14, 2), (14, 3), (5, 4)]:
        ops = tuple(range(k))
        hits = sum(0 in [op for op, _ in randaugment_sample(ops, n, 0, rng)] for _ in range(4000))
        p = prob_applied(k, n)
        assert abs(hits / 4000 - p) < 4 * math.sqrt(p * (1 - p) / 4000)
    assert prob_applied(14, 0) == 0.0
    assert prob_applied(1, 3) == 1.0


@pytest.mark.parametrize(
    "s,length",
    [
        (0, 6),
        (2, 6),
        (4, 6),
        (6, 6),
        (8, 6),
        (12, 6),
        (4, 9),
        (10, 9),
        (8, 32),
        (16, 32),
    ],
)
def test_cutout_area_on_the_pixel_grid(s, length):
    img = np.full((length, length), 255, np.uint8)
    total = sum(int((cutout(img, s, (r, c)) == 0).sum()) for r in range(length) for c in range(length))
    assert total / length**2 == pytest.approx(cutout_expected_area(s, length), abs=1e-12)


def test_cutout_area_rejects():
    for s, length in [(3, 8), (18, 8), (-2, 8)]:
        with pytest.raises(ValueError):
            cutout_expected_area(s, length)
    assert cutout_expected_area(16, 8) == 64.0
