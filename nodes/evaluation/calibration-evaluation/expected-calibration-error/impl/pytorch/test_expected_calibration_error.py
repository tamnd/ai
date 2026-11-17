# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
"""Tests of expected_calibration_error.py in PyTorch: the vectors, the worked example of the page, the properties of
the derivations, temperature scaling, the edge cases and the errors. Random cases use a seeded torch.Generator."""

import json
import math
import pathlib
from fractions import Fraction

import pytest
import torch

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


def randint(g, lo, hi):
    return int(torch.randint(lo, hi, (1,), generator=g))


def random_case(g, n, k, sharp=3.0):
    z = torch.randn(n, k, generator=g, dtype=torch.float64) * sharp
    return z, softmax(z), torch.randint(0, k, (n,), generator=g)


def sample_labels(g, q):
    return torch.multinomial(q, 1, generator=g).squeeze(1)


def by_definition(p, y, m):
    """sum_m |B_m|/n |acc(B_m) - conf(B_m)| with the bins found by a loop over the exact edges m / M."""
    conf, pred = p.max(dim=1).values, p.argmax(dim=1)
    total = 0.0
    for b in range(m):
        lo, hi = b / m, (b + 1) / m
        inb = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        if bool(inb.any()):
            acc = float((pred[inb] == y[inb]).double().mean())
            total += float(inb.sum()) / len(y) * abs(acc - float(conf[inb].mean()))
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
    assert score(torch.tensor(WE_P, dtype=torch.float64), torch.tensor(WE_Y), n_bins=1)["ece"] == near(abs(5 / 8 - 0.675))


def test_matches_definition():
    g = torch.Generator().manual_seed(20261116)
    for _ in range(200):
        n, k, m = randint(g, 1, 60), randint(g, 2, 6), randint(g, 1, 40)
        sharp = 0.1 + 6 * float(torch.rand(1, generator=g))
        _, p, y = random_case(g, n, k, sharp)
        assert score(p, y, n_bins=m)["ece"] == near(by_definition(p, y, m))


def test_bin_edges():
    for m in (1, 2, 3, 5, 7, 10, 15, 100):
        edges = torch.arange(m + 1, dtype=torch.float64) / m
        assert bin_index(edges, m).tolist() == [0] + list(range(m))
        up = torch.nextafter(edges[:-1], torch.full_like(edges[:-1], 2.0))
        assert bin_index(up, m).tolist() == list(range(m))
    assert bin_index(torch.tensor([0.3, 0.30000000000000004], dtype=torch.float64), 10).tolist() == [2, 3]


def test_properties():
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        n, k, m = randint(g, 1, 80), randint(g, 2, 6), randint(g, 1, 20)
        sharp = 0.1 + 6 * float(torch.rand(1, generator=g))
        _, p, y = random_case(g, n, k, sharp)
        got = score(p, y, n_bins=m)
        assert 0.0 <= got["ece"] <= got["mce"] + 1e-12 <= 1.0 + 1e-12
        assert sum(got["bin_count"]) == n
        finer = score(p, y, n_bins=m * randint(g, 2, 5))["ece"]
        assert got["ece"] <= finer + 1e-12
        assert score(p, y, n_bins=1)["ece"] == near(abs(got["accuracy"] - got["confidence"]))
        w = torch.tensor(got["bin_count"], dtype=torch.float64) / n
        assert float(w @ torch.tensor(got["bin_accuracy"], dtype=torch.float64)) == near(got["accuracy"])
        assert float(w @ torch.tensor(got["bin_confidence"], dtype=torch.float64)) == near(got["confidence"])


def test_permutation_invariant():
    g = torch.Generator().manual_seed(3)
    _, p, y = random_case(g, 50, 4)
    perm = torch.randperm(50, generator=g)
    a, b = score(p, y), score(p[perm], y[perm])
    assert a["ece"] == near(b["ece"]) and a["mce"] == near(b["mce"]) and a["bin_count"] == b["bin_count"]


def test_reliability_bins():
    b = reliability_bins(WE_P, WE_Y, 5)
    assert b["count"].tolist() == [0, 1, 2, 2, 3]
    assert b["gap_sum"].tolist() == near([0.0, 0.6, 1 - 0.95, 1 - 1.35, 2 - 2.7])


def test_temperature_keeps_argmax_and_accuracy():
    g = torch.Generator().manual_seed(11)
    for _ in range(100):
        z, _, y = random_case(g, 40, 5)
        base = score(softmax(z), y)
        for t in (0.1, 0.5, 2.0, 10.0):
            q = softmax(z, t)
            assert torch.equal(q.argmax(dim=1), z.argmax(dim=1))
            assert score(q, y)["accuracy"] == base["accuracy"]
        c = [float(softmax(z, t).max(dim=1).values.mean()) for t in (0.25, 0.5, 1, 2, 4)]
        assert all(a >= b - 1e-15 for a, b in zip(c, c[1:]))


def test_temperature_limits():
    z = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    assert softmax(z, 1e6)[0].tolist() == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert softmax(z, 1e-3)[0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    e = [math.exp(2.0), math.exp(1.0), 1.0]
    assert softmax(z, 1.0)[0].tolist() == near([x / sum(e) for x in e])


def test_fit_temperature_minimizes_nll():
    g = torch.Generator().manual_seed(5)
    for t0 in (0.5, 1.0, 2.5):
        z = torch.randn(4000, 10, generator=g, dtype=torch.float64) * 3
        y = sample_labels(g, softmax(z, t0))
        t = fit_temperature(z, y)
        assert abs(t / t0 - 1) < 0.1
        grid = torch.exp(torch.linspace(math.log(t / 2), math.log(t * 2), 81, dtype=torch.float64)).tolist()
        assert nll(z, y, t) <= min(nll(z, y, s) for s in grid) + 1e-12
        q = softmax(z, t)
        slope = float(torch.mean(torch.sum(q * z, dim=1) - z[torch.arange(len(y)), y]))
        assert slope == pytest.approx(0.0, abs=1e-9)


def test_fit_temperature_overconfident_model_gets_t_above_1():
    g = torch.Generator().manual_seed(9)
    z = torch.randn(3000, 5, generator=g, dtype=torch.float64) * 2
    y = sample_labels(g, softmax(z))
    z_over = 3.0 * z
    t = fit_temperature(z_over, y)
    assert t == pytest.approx(3.0, rel=0.1)
    before, after = score(softmax(z_over), y), score(softmax(z_over, t), y)
    assert after["accuracy"] == before["accuracy"]
    assert after["ece"] < before["ece"]


def test_fit_temperature_ends():
    z = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float64)
    assert fit_temperature(z, [0, 1]) == 0.05
    assert fit_temperature(z, [1, 0]) == 100.0
    assert fit_temperature(z, [0, 1], t_min=0.5, t_max=2.0) == 0.5


def test_nll_matches_log_softmax():
    g = torch.Generator().manual_seed(13)
    z, _, y = random_case(g, 30, 4)
    for t in (0.5, 1.0, 3.0):
        logp = torch.log_softmax(z / t, dim=1)
        assert nll(z, y, t) == near(float(-logp[torch.arange(30), y].mean()))


def test_exact_rational_worked_example():
    want = Fraction(0)
    for members in ([(0.4, 1)], [(0.45, 1), (0.5, 0)], [(0.7, 1), (0.65, 0)], [(0.9, 1), (0.85, 1), (0.95, 0)]):
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
        (([[math.nan, 1.0]], [0], 15), ValueError),
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


def test_empty_input():
    with pytest.raises(ValueError):
        score(torch.zeros(0, 3), torch.zeros(0, dtype=torch.int64))


def test_temperature_errors():
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], 0.0)
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], -1.0)
    with pytest.raises(ValueError):
        nll([[1.0, math.inf]], [0])
    with pytest.raises(ValueError):
        fit_temperature([[1.0, 2.0]], [0], t_min=2.0, t_max=1.0)


def test_shapes_and_defaults():
    one = score([[0.6, 0.4]], [0], n_bins=5)
    assert one["ece"] == near(0.4) and one["bin_count"] == [0, 0, 1, 0, 0]
    assert score(WE_P, WE_Y) == score(WE_P, WE_Y, n_bins=15)
    assert len(reliability_bins(WE_P, WE_Y)["count"]) == 15
    for bad in (torch.zeros(0, 3, dtype=torch.float64), torch.zeros(2, 0, dtype=torch.float64)):
        with pytest.raises(ValueError):
            score(bad, torch.zeros(bad.shape[0], dtype=torch.int64))
        with pytest.raises(ValueError):
            softmax(bad)
    assert softmax([[0.0, 0.0, 0.0]])[0].tolist() == near([1 / 3] * 3)


def test_tensor_n_bins():
    assert score(WE_P, WE_Y, n_bins=torch.tensor(5)) == score(WE_P, WE_Y, n_bins=5)
    with pytest.raises(TypeError):
        score(WE_P, WE_Y, n_bins=torch.tensor(5.0))


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
    z = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    assert softmax(z, Fraction(1, 2)).tolist() == softmax(z, 0.5).tolist()
    for t_min, t_max in ((0.0, 1.0), (1.0, math.inf), (1.0, 1.0)):
        with pytest.raises(ValueError):
            fit_temperature(z, [0], t_min=t_min, t_max=t_max)


def test_fit_temperature_flat_logits_and_large_scale():
    assert fit_temperature(torch.zeros(4, 3, dtype=torch.float64), [0, 1, 2, 0]) == 100.0
    g = torch.Generator().manual_seed(17)
    z = torch.randn(2000, 4, generator=g, dtype=torch.float64) * 2
    y = sample_labels(g, softmax(z))
    assert fit_temperature(30 * z, y) == pytest.approx(30 * fit_temperature(z, y), rel=1e-6)
