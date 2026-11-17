# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/randaugment/ (ai://NEW-randaugment).
"""Tests of the PyTorch port: the worked example shared with the NumPy tests, each image operation on (H, W) and
(C, H, W) tensors, the magnitude map and the RandAugment and TrivialAugment samplers."""

import math

import pytest
import torch

from randaugment import (
    OPS,
    RANGES,
    apply_op,
    cutout,
    identity,
    magnitude_to_arg,
    posterize,
    randaugment,
    randaugment_sample,
    rotate,
    solarize,
    translate_x,
    translate_y,
    trivialaugment,
    trivialaugment_sample,
)

# The shared fixture of the page. The NumPy tests check the same numbers.
IMG = torch.tensor(
    [[10, 50, 90, 130], [170, 210, 250, 30], [70, 110, 150, 190], [230, 20, 60, 100]],
    dtype=torch.uint8,
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


def t(rows):
    return torch.tensor(rows, dtype=torch.uint8)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def rand_image(seed, *shape):
    return torch.randint(0, 256, shape, generator=gen(seed), dtype=torch.uint8)


def test_worked_example():
    assert torch.equal(posterize(IMG, 4), t(POSTERIZE_4))
    assert torch.equal(solarize(IMG, 128), t(SOLARIZE_128))
    assert torch.equal(cutout(IMG, 2, (0, 3)), t(CUTOUT_2_AT_0_3))
    assert torch.equal(translate_x(IMG, 1), t(TRANSLATE_X_1))
    assert torch.equal(rotate(IMG, 90), torch.rot90(IMG, 1, (0, 1)))
    assert magnitude_to_arg("posterize", 9) == 4
    assert magnitude_to_arg("solarize", 9) == pytest.approx(25.6)
    assert magnitude_to_arg("rotate", 9) == pytest.approx(27.0)


def test_images_are_validated():
    for bad in [IMG.float(), IMG[0], IMG[None, None], IMG.numpy()]:
        with pytest.raises(ValueError):
            identity(bad)


def test_identity_copies():
    out = identity(IMG)
    assert torch.equal(out, IMG)
    out[0, 0] = 99
    assert int(IMG[0, 0]) == 10


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_rotate_right_angles(n):
    img = rand_image(n, 3, n, n)
    assert torch.equal(rotate(img, 0), img)
    assert torch.equal(rotate(img, 90), torch.rot90(img, 1, (1, 2)))
    assert torch.equal(rotate(img, -90), torch.rot90(img, -1, (1, 2)))
    assert torch.equal(rotate(img, 180), img.flip(1, 2))
    assert torch.equal(rotate(img, 270), torch.rot90(img, 3, (1, 2)))


def test_rotate_fill_and_nearest():
    out = rotate(torch.full((5, 5), 200, dtype=torch.uint8), 45, fill=7)
    assert int(out[2, 2]) == 200 and int(out[0, 2]) == 200
    assert int(out[0, 0]) == int(out[0, 4]) == int(out[4, 0]) == int(out[4, 4]) == 7
    img = torch.arange(49, dtype=torch.uint8).reshape(7, 7)
    out = rotate(img, 10)
    assert out[3, 4] == img[3, 4] and out[3, 6] == img[4, 6] and out[0, 3] == img[0, 4]
    img = rand_image(1, 4, 6)
    assert torch.equal(rotate(img, 180), img.flip(0, 1))


def test_translate():
    img = rand_image(2, 3, 5, 6)
    out = translate_x(img, 2)
    assert torch.equal(out[..., 2:], img[..., :4]) and (out[..., :2] == 0).all()
    out = translate_x(img, -2, fill=9)
    assert torch.equal(out[..., :4], img[..., 2:]) and (out[..., 4:] == 9).all()
    out = translate_y(img, 3)
    assert torch.equal(out[:, 3:], img[:, :2]) and (out[:, :3] == 0).all()
    out = translate_y(img, -1)
    assert torch.equal(out[:, :4], img[:, 1:]) and (out[:, 4:] == 0).all()
    assert torch.equal(translate_x(img, 0), img)
    assert (translate_x(img, 6) == 0).all() and (translate_y(img, -5, fill=3) == 3).all()
    assert torch.equal(translate_y(img, 4)[:, 4], img[:, 0])
    assert torch.equal(translate_y(IMG, 1)[1:], IMG[:3]) and (translate_y(IMG, 1)[0] == 0).all()


def test_posterize_and_solarize():
    vals = torch.arange(256).to(torch.uint8)[None]
    assert torch.equal(posterize(vals, 8), vals)
    assert torch.equal(posterize(vals, 1)[0], torch.where(vals[0] >= 128, 128, 0).to(torch.uint8))
    assert torch.equal(posterize(vals, 6), vals // 4 * 4)
    for bad in [0, 9]:
        with pytest.raises(ValueError):
            posterize(IMG, bad)
    assert torch.equal(solarize(vals, 256), vals)
    assert torch.equal(solarize(vals, 0), 255 - vals)
    out = solarize(vals, 100)[0]
    assert int(out[99]) == 99 and int(out[100]) == 155 and int(out[255]) == 0
    assert out.dtype == torch.uint8


def test_cutout():
    img = torch.full((3, 6, 6), 255, dtype=torch.uint8)
    out = cutout(img, 4, (3, 3), fill=1)
    assert (out[:, 1:5, 1:5] == 1).all() and int((out == 1).sum()) == 3 * 16
    out = cutout(img, 4, (0, 5))
    assert (out[:, 0:2, 3:6] == 0).all() and int((out == 0).sum()) == 3 * 6
    out = cutout(img, 4, (5, 0))
    assert (out[:, 3:6, 0:2] == 0).all() and int((out == 0).sum()) == 3 * 6
    assert torch.equal(cutout(img, 0, (2, 2)), img)
    assert (cutout(img, 20, (2, 2)) == 0).all() and int(img.min()) == 255


def test_magnitude_map():
    assert [magnitude_to_arg("rotate", m) for m in (0, 5, 10)] == [0.0, 15.0, 30.0]
    assert [magnitude_to_arg("posterize", m) for m in (0, 4, 10)] == [8, 6, 4]
    assert [magnitude_to_arg("solarize", m) for m in (0, 10)] == [256.0, 0.0]
    assert magnitude_to_arg("translate_y", 10) == 10.0 and magnitude_to_arg("cutout", 5) == pytest.approx(0.1)
    assert magnitude_to_arg("rotate", 15, m_max=30) == pytest.approx(15.0) and magnitude_to_arg("identity", 3) == 0
    for op, m in [("rotate", -1), ("rotate", 11), ("blur", 3)]:
        with pytest.raises(ValueError):
            magnitude_to_arg(op, m)
    assert set(OPS) == set(RANGES)


@pytest.mark.parametrize("op", OPS)
def test_apply_op_dispatch(op):
    img = rand_image(3, 3, 20, 20)
    out = apply_op(img, op, 5, gen(4))
    arg = magnitude_to_arg(op, 5)
    expected = {
        "identity": lambda: [img],
        "rotate": lambda: [rotate(img, arg), rotate(img, -arg)],
        "translate_x": lambda: [translate_x(img, 5), translate_x(img, -5)],
        "translate_y": lambda: [translate_y(img, 5), translate_y(img, -5)],
        "posterize": lambda: [posterize(img, 6)],
        "solarize": lambda: [solarize(img, 128.0)],
        "cutout": lambda: [cutout(img, 2, (r, c)) for r in range(20) for c in range(20)],
    }[op]()
    assert any(torch.equal(out, e) for e in expected)
    if op == "cutout":
        assert not torch.equal(out, img)


def test_signs_and_cutout_centres():
    img = torch.zeros((9, 9), dtype=torch.uint8)
    img[4, 4] = 255
    g = gen(6)
    right = sum(int(apply_op(img, "translate_x", 2, g)[4, 6] == 255) for _ in range(1000))
    down = sum(int(apply_op(img, "translate_y", 2, g)[6, 4] == 255) for _ in range(1000))
    assert 420 < right < 580 and 420 < down < 580
    white = torch.full((20, 20), 255, dtype=torch.uint8)
    rows, cols, sizes = set(), set(), set()
    for _ in range(400):
        out = apply_op(white, "cutout", 10, g)
        r, c = torch.nonzero(out == 0, as_tuple=True)
        rows.update(r.tolist())
        cols.update(c.tolist())
        sizes.add(int((out == 0).sum()))
    assert rows == cols == set(range(20)) and max(sizes) == 16 and {4, 8} <= sizes


def test_randaugment_sample():
    g = gen(7)
    pol = randaugment_sample(OPS, 3, 9, g)
    assert len(pol) == 3 and all(m == 9 and op in OPS for op, m in pol)
    assert randaugment_sample(OPS, 0, 9, g) == []
    with pytest.raises(ValueError):
        randaugment_sample(OPS, -1, 9, g)
    counts = {op: 0 for op in OPS}
    repeats = 0
    for _ in range(3500):
        pol = randaugment_sample(OPS, 2, 1, g)
        repeats += pol[0][0] == pol[1][0]
        for op, _ in pol:
            counts[op] += 1
    assert all(abs(c - 1000) < 150 for c in counts.values())
    assert abs(repeats / 3500 - 1 / 7) < 0.025
    hits = sum("rotate" in [op for op, _ in randaugment_sample(OPS, 3, 0, g)] for _ in range(4000))
    p = 1 - (6 / 7) ** 3
    assert abs(hits / 4000 - p) < 4 * math.sqrt(p * (1 - p) / 4000)


def test_randaugment_applies_the_sampled_ops_in_order():
    img = rand_image(8, 3, 12, 12)
    for seed in range(20):
        out = randaugment(img, 3, 7, gen(seed))
        g = gen(seed)
        ref = img
        for op, m in randaugment_sample(OPS, 3, 7, g):
            ref = apply_op(ref, op, m, g)
        assert torch.equal(out, ref)
    assert torch.equal(randaugment(img, 0, 7, gen(0)), img)
    assert torch.equal(randaugment(img, 2, 10, gen(0), ops=("solarize",)), img)


def test_trivialaugment():
    g = gen(9)
    ops, mags = set(), []
    for _ in range(6200):
        op, m = trivialaugment_sample(OPS, g)
        ops.add(op)
        mags.append(m)
    assert ops == set(OPS)
    counts = torch.bincount(torch.tensor(mags), minlength=31)
    assert len(counts) == 31 and int(counts.min()) > 130 and int(counts.max()) < 270
    assert trivialaugment_sample(("rotate",), g, m_max=0) == ("rotate", 0)
    img = rand_image(10, 8, 8)
    for seed in range(20):
        out = trivialaugment(img, gen(seed))
        g = gen(seed)
        op, m = trivialaugment_sample(OPS, g)
        assert torch.equal(out, apply_op(img, op, m, g, 30))
    out = trivialaugment(img, gen(0), ops=("posterize",), m_max=1)
    assert torch.equal(out, img) or torch.equal(out, posterize(img, 4))
