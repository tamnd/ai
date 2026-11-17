# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
"""Tests of expected_calibration_error.py: the vectors, the worked example of the page, the properties of the
derivations, temperature scaling, the edge cases and the errors. Random cases use a seeded generator."""

import json
import pathlib
from fractions import Fraction

import numpy as np
import pytest

from expected_calibration_error import (
    bin_index,
    fit_temperature,
    nll,
    reliability_bins,
    score,
    softmax,
)

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = next(d for d in (HERE / "vectors", HERE.parents[1] / "vectors") if d.is_dir())
WE_P = [[0.9, 0.05, 0.05], [0.1, 0.85, 0.05], [0.95, 0.03, 0.02], [0.7, 0.2, 0.1],
        [0.2, 0.65, 0.15], [0.3, 0.25, 0.45], [0.5, 0.3, 0.2], [0.4, 0.35, 0.25]]  # fmt: skip
WE_Y = [0, 1, 2, 0, 0, 2, 1, 0]


def near(x):
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def rows(name):
    with open(VECTORS / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def random_case(rng, n, k, sharp=3.0):
    z = rng.normal(size=(n, k)) * sharp
    p = softmax(z)
    y = rng.integers(0, k, size=n)
    return z, p, y


def by_definition(p, y, m):
    """sum_m |B_m|/n |acc(B_m) - conf(B_m)| with the bins found by a loop over the exact edges m / M."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    total = 0.0
    for b in range(m):
        lo, hi = b / m, (b + 1) / m
        inb = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if inb.any():
            total += inb.sum() / len(y) * abs(np.mean(pred[inb] == y[inb]) - np.mean(conf[inb]))
    return total


@pytest.mark.parametrize("v", rows("edge.jsonl"), ids=lambda v: v["id"])
def test_vectors(v):
    got = score(v["predictions"], v["references"], **v["options"])
    for key in ("ece", "mce", "accuracy", "confidence", "bin_accuracy", "bin_confidence"):
        assert got[key] == near(v["want"][key]), key
    assert got["bin_count"] == v["want"]["bin_count"]


def test_worked_example():
    got = score(WE_P, WE_Y, n_bins=5)
    assert got["bin_count"] == [0, 1, 2, 2, 3]
    assert got["bin_accuracy"] == near([0.0, 1.0, 0.5, 0.5, 2 / 3])
    assert got["bin_confidence"] == near([0.0, 0.4, 0.475, 0.675, 0.9])
    assert got["ece"] == near(1.7 / 8)
    assert got["mce"] == near(0.6)
    assert got["accuracy"] == near(5 / 8)
    assert got["confidence"] == near(0.675)
    # one bin gives |accuracy - mean confidence|
    assert score(WE_P, WE_Y, n_bins=1)["ece"] == near(abs(5 / 8 - 0.675))


def test_matches_definition():
    rng = np.random.default_rng(20261116)
    for _ in range(200):
        n, k, m = int(rng.integers(1, 60)), int(rng.integers(2, 6)), int(rng.integers(1, 40))
        _, p, y = random_case(rng, n, k, sharp=float(rng.uniform(0.1, 6)))
        assert score(p, y, n_bins=m)["ece"] == near(by_definition(p, y, m))


def test_bin_edges():
    for m in (1, 2, 3, 5, 7, 10, 15, 100):
        edges = np.arange(m + 1) / m
        idx = bin_index(edges, m)
        assert idx.tolist() == [0] + list(range(m))  # each edge closes its bin; 0 joins bin 0
        assert bin_index(np.nextafter(edges[:-1], 2.0), m).tolist() == list(range(m))
    assert bin_index(np.array([0.3, 0.30000000000000004]), 10).tolist() == [2, 3]


def test_properties():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n, k, m = int(rng.integers(1, 80)), int(rng.integers(2, 6)), int(rng.integers(1, 20))
        _, p, y = random_case(rng, n, k, sharp=float(rng.uniform(0.1, 6)))
        got = score(p, y, n_bins=m)
        assert 0.0 <= got["ece"] <= got["mce"] + 1e-12 <= 1.0 + 1e-12
        assert sum(got["bin_count"]) == n
        # a coarser binning never has a larger empirical ECE (triangle inequality on the refined bins)
        finer = score(p, y, n_bins=m * int(rng.integers(2, 5)))["ece"]
        assert got["ece"] <= finer + 1e-12
        assert score(p, y, n_bins=1)["ece"] == near(abs(got["accuracy"] - got["confidence"]))
        # the bin statistics are weighted averages of the overall ones
        w = np.array(got["bin_count"]) / n
        assert float(w @ np.array(got["bin_accuracy"])) == near(got["accuracy"])
        assert float(w @ np.array(got["bin_confidence"])) == near(got["confidence"])


def test_permutation_invariant():
    rng = np.random.default_rng(3)
    _, p, y = random_case(rng, 50, 4)
    perm = rng.permutation(50)
    a, b = score(p, y), score(p[perm], y[perm])
    assert a["ece"] == near(b["ece"]) and a["mce"] == near(b["mce"]) and a["bin_count"] == b["bin_count"]


def test_reliability_bins():
    b = reliability_bins(WE_P, WE_Y, 5)
    assert b["count"].tolist() == [0, 1, 2, 2, 3]
    assert b["gap_sum"] == near([0.0, 0.6, 1 - 0.95, 1 - 1.35, 2 - 2.7])


def test_temperature_keeps_argmax_and_accuracy():
    rng = np.random.default_rng(11)
    for _ in range(100):
        z, _, y = random_case(rng, 40, 5)
        base = score(softmax(z), y)
        for t in (0.1, 0.5, 2.0, 10.0):
            q = softmax(z, t)
            assert np.array_equal(q.argmax(axis=1), z.argmax(axis=1))
            assert score(q, y)["accuracy"] == base["accuracy"]
        # the confidence does not increase with T
        c = [float(softmax(z, t).max(axis=1).mean()) for t in (0.25, 0.5, 1, 2, 4)]
        assert all(a >= b - 1e-15 for a, b in zip(c, c[1:]))


def test_temperature_limits():
    z = np.array([[2.0, 1.0, 0.0]])
    assert softmax(z, 1e6)[0] == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert softmax(z, 1e-3)[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert softmax(z, 1.0)[0] == near(np.exp([2.0, 1.0, 0.0]) / np.exp([2.0, 1.0, 0.0]).sum())


def test_fit_temperature_minimizes_nll():
    rng = np.random.default_rng(5)
    for t0 in (0.5, 1.0, 2.5):
        z = rng.normal(size=(4000, 10)) * 3
        y = np.array([rng.choice(10, p=q) for q in softmax(z, t0)])
        t = fit_temperature(z, y)
        assert abs(t / t0 - 1) < 0.1
        grid = np.exp(np.linspace(np.log(t / 2), np.log(t * 2), 81))
        assert nll(z, y, t) <= min(nll(z, y, g) for g in grid) + 1e-12
        # eq-stationary: the mean true-class logit equals the mean softmax-weighted logit at beta = 1/T
        q = softmax(z, t)
        assert float(np.mean(np.sum(q * z, axis=1) - z[np.arange(len(y)), y])) == pytest.approx(0.0, abs=1e-9)


def test_fit_temperature_overconfident_model_gets_t_above_1():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(3000, 5)) * 2
    y = np.array([rng.choice(5, p=q) for q in softmax(z)])
    z_over = 3.0 * z  # the same ranking, three times sharper
    t = fit_temperature(z_over, y)
    assert t == pytest.approx(3.0, rel=0.1)
    before, after = score(softmax(z_over), y), score(softmax(z_over, t), y)
    assert after["accuracy"] == before["accuracy"]
    assert after["ece"] < before["ece"]


def test_fit_temperature_ends():
    z = np.array([[5.0, 0.0], [0.0, 5.0]])
    assert fit_temperature(z, [0, 1]) == 0.05  # all correct: the NLL falls as T -> 0
    assert fit_temperature(z, [1, 0]) == 100.0  # all wrong: the NLL falls as T grows
    assert fit_temperature(z, [0, 1], t_min=0.5, t_max=2.0) == 0.5


def test_exact_rational_worked_example():
    want = Fraction(0)
    for lo, members in ((1, [(0.4, 1)]), (2, [(0.45, 1), (0.5, 0)]), (3, [(0.7, 1), (0.65, 0)]),
                        (4, [(0.9, 1), (0.85, 1), (0.95, 0)])):  # fmt: skip
        want += abs(sum(Fraction(a) - Fraction(c) for c, a in members))
    assert score(WE_P, WE_Y, n_bins=5)["ece"] == near(float(want / 8))


@pytest.mark.parametrize(
    "args, err",
    [
        (([[0.5, 0.5]], [0], 0), ValueError),
        (([[0.5, 0.5]], [0], 2.0), TypeError),
        (([[0.5, 0.5]], [0], True), TypeError),
        (([[0.5, 0.6]], [0], 15), ValueError),
        (([[1.5, -0.5]], [0], 15), ValueError),
        (([[np.nan, 1.0]], [0], 15), ValueError),
        (([], [], 15), ValueError),
        (([[0.5, 0.5]], [2], 15), ValueError),
        (([[0.5, 0.5]], [-1], 15), ValueError),
        (([[0.5, 0.5]], [0.0], 15), TypeError),
        (([[0.5, 0.5]], [0, 1], 15), ValueError),
        (([0.5, 0.5], [0], 15), ValueError),
    ],
)
def test_errors(args, err):
    with pytest.raises(err):
        score(*args)


def test_temperature_errors():
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], 0.0)
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], -1.0)
    with pytest.raises(ValueError):
        nll([[1.0, np.inf]], [0])
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 2.0]], [0], t_min=2.0, t_max=1.0)


def test_shapes_and_defaults():
    one = score([[0.6, 0.4]], [0], n_bins=5)
    assert one["ece"] == near(0.4) and one["bin_count"] == [0, 0, 1, 0, 0]
    assert score(WE_P, WE_Y) == score(WE_P, WE_Y, n_bins=15)
    assert len(reliability_bins(WE_P, WE_Y)["count"]) == 15
    for bad in (np.zeros((0, 3)), np.zeros((2, 0))):
        with pytest.raises(ValueError):
            score(bad, np.zeros(len(bad), dtype=np.int64))
        with pytest.raises(ValueError):
            softmax(bad)
    assert softmax([[0.0, 0.0, 0.0]])[0] == near([1 / 3] * 3)


def test_probability_tolerance():
    assert score([[0.5, 0.5 + 5e-7]], [1])["accuracy"] == 1.0  # within 1e-6
    for row in ([0.5, 0.5 + 5e-6], [1.0 + 5e-7, 0.0]):
        with pytest.raises(ValueError):
            score([row], [0])


def test_labels_checked_against_classes():
    z = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    with pytest.raises(ValueError):
        nll(z, [2, 0, 0])
    with pytest.raises(ValueError):
        fit_temperature(z, [2, 0, 0])


def test_temperature_argument_types():
    z = np.array([[2.0, 1.0, 0.0]])
    assert softmax(z, Fraction(1, 2)) == near(softmax(z, 0.5))
    for t_min, t_max in ((0.0, 1.0), (1.0, np.inf), (1.0, 1.0)):
        with pytest.raises(ValueError):
            fit_temperature(z, [0], t_min=t_min, t_max=t_max)


def test_fit_temperature_flat_logits_and_large_scale():
    # equal logits: the NLL does not depend on T, the slope is 0 everywhere, and the fit returns t_max
    assert fit_temperature(np.zeros((4, 3)), [0, 1, 2, 0]) == 100.0
    rng = np.random.default_rng(17)
    z = rng.normal(size=(2000, 4)) * 2
    y = np.array([rng.choice(4, p=q) for q in softmax(z)])
    # logits 30 times larger overflow exp without the max shift at beta = 1/t_min = 20
    assert fit_temperature(30 * z, y) == pytest.approx(30 * fit_temperature(z, y), rel=1e-6)
