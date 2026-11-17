# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/pass-at-k/ (ai://NEW-pass-at-k).
"""Tests of pass_at_k.py: the vectors, the worked example of the page, the derivation, the edge cases and the errors."""

import itertools
import json
import math
import pathlib
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pass_at_k import ESTIMATORS, estimate, plug_in, score

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = next(d for d in (HERE / "vectors", HERE.parents[1] / "vectors") if d.is_dir())


def near(x):
    return pytest.approx(x, rel=1e-9, abs=1e-12)


def rows(name):
    with open(VECTORS / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def exact(n, c, k):
    """1 - C(n - c, k) / C(n, k) as a fraction; math.comb is 0 when n - c < k."""
    return 1 - Fraction(math.comb(n - c, k), math.comb(n, k))


@pytest.mark.parametrize("v", rows("edge.jsonl"), ids=lambda v: v["id"])
def test_vectors(v):
    got = score(v["predictions"], v["references"], **v["options"])
    assert got["pass_at_k"] == near(v["want"]["pass_at_k"])
    assert got["per_problem"] == near(v["want"]["per_problem"])


def test_worked_example():
    got = score([0, 1, 3], [5, 5, 5], k=2)
    assert got["per_problem"] == near([0.0, 0.4, 0.9])
    assert got["pass_at_k"] == near(13 / 30)
    # the product of eq-product-form for c = 3: (1 - 2/3)(1 - 2/4)(1 - 2/5) = 1/10
    assert estimate(5, 3, 2) == near(1 - (1 / 3) * (1 / 2) * (3 / 5))
    plug = score([0, 1, 3], [5, 5, 5], k=2, estimator="plug-in")
    assert plug["per_problem"] == near([0.0, 0.36, 0.84])
    assert plug["pass_at_k"] == near(0.4)
    assert score([0, 1, 3], [5, 5, 5])["pass_at_k"] == near(4 / 15)
    assert rows("edge.jsonl")[0]["want"]["pass_at_k"] == near(got["pass_at_k"])


@settings(max_examples=300, deadline=None)
@given(st.integers(1, 80).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n), st.integers(1, n))))
def test_product_form(nck):
    n, c, k = nck
    assert estimate(n, c, k) == near(float(exact(n, c, k)))


def test_unbiased():
    """E_c[estimate] = 1 - (1 - p)^k for c ~ Binomial(n, p), exactly, by summing over c (derivations/)."""
    for n, k, p in [
        (5, 2, Fraction(1, 3)),
        (10, 3, Fraction(1, 2)),
        (20, 7, Fraction(1, 10)),
        (6, 6, Fraction(2, 5)),
    ]:
        mean = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * exact(n, c, k) for c in range(n + 1))
        assert mean == 1 - (1 - p) ** k
        approx = sum(
            math.comb(n, c) * float(p) ** c * float(1 - p) ** (n - c) * estimate(n, c, k) for c in range(n + 1)
        )
        assert approx == near(float(1 - (1 - p) ** k))


def test_monte_carlo():
    """A sampled check: 200,000 draws of c ~ Binomial(200, 0.03); the mean estimate of pass@10 is within 4 standard
    errors of 1 - 0.97^10, and the plug-in mean is more than 10 standard errors below it."""
    rng = np.random.default_rng(20261115)
    n, k, p = 200, 10, 0.03
    cs = rng.binomial(n, p, size=200_000)
    table = np.array([estimate(n, c, k) for c in range(n + 1)])
    plug = np.array([plug_in(n, c, k) for c in range(n + 1)])
    est = table[cs]
    true = 1 - (1 - p) ** k
    se = est.std() / math.sqrt(len(est))
    assert abs(est.mean() - true) < 4 * se
    assert plug[cs].mean() < true - 10 * se


def test_plug_in_is_biased_low():
    """For 2 <= k <= n and 0 < p < 1, E[1 - (1 - c/n)^k] < 1 - (1 - p)^k (Jensen's inequality, derivations/)."""
    for n, k, p in [
        (5, 2, Fraction(1, 3)),
        (200, 10, Fraction(3, 100)),
        (8, 8, Fraction(1, 2)),
    ]:
        mean = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * (1 - (1 - Fraction(c, n)) ** k) for c in range(n + 1))
        assert mean < 1 - (1 - p) ** k
    for n, c in [(10, 3), (7, 0), (7, 7)]:  # equal at k = 1
        assert plug_in(n, c, 1) == near(c / n) and estimate(n, c, 1) == near(c / n)


def test_special_cases():
    for n in [1, 2, 7, 50]:
        for k in range(1, n + 1):
            assert estimate(n, 0, k) == 0.0
            assert estimate(n, n, k) == 1.0
            assert estimate(n, 1, k) == near(k / n)
            for c in range(n + 1):
                if n - c < k:
                    assert estimate(n, c, k) == 1.0
        for c in range(n + 1):
            assert estimate(n, c, n) == (1.0 if c > 0 else 0.0)  # k = n: any correct sample


def test_range_and_monotone():
    for n in [1, 5, 30, 200]:
        for k in range(1, n + 1, max(1, n // 10)):
            col = [estimate(n, c, k) for c in range(n + 1)]
            assert all(0.0 <= v <= 1.0 for v in col)
            assert all(a <= b + 1e-15 for a, b in itertools.pairwise(col))  # more correct samples, higher estimate
        for c in range(n + 1):
            row = [estimate(n, c, k) for k in range(1, n + 1)]
            assert all(a <= b + 1e-15 for a, b in itertools.pairwise(row))  # larger k, higher estimate


def test_large_n_is_stable():
    """C(5000, 1000) does not fit in a float64, but the product does not form it."""
    with pytest.raises(OverflowError):
        float(math.comb(5000, 1000))
    for n, c, k in [
        (5000, 37, 1000),
        (2000, 5, 1000),
        (10_000, 1, 1),
        (100_000, 3, 50_000),
    ]:
        assert estimate(n, c, k) == near(float(exact(n, c, k)))


def test_mean_over_problems():
    c, n = [3, 0, 50, 1], [10, 20, 200, 5]
    got = score(c, n, k=5)
    assert got["pass_at_k"] == near(np.mean([estimate(nj, cj, 5) for cj, nj in zip(c, n, strict=True)]))
    assert score(c[::-1], n[::-1], k=5)["pass_at_k"] == near(got["pass_at_k"])
    assert score(np.array(c), np.array(n), k=np.int64(5))["per_problem"] == near(got["per_problem"])
    assert isinstance(got["pass_at_k"], float) and all(isinstance(v, float) for v in got["per_problem"])


def test_k_larger_than_n_raises():
    """Chen et al. (2021), Figure 3 returns 1.0 for k > n even when no sample passes; score refuses instead."""
    assert estimate(3, 0, 5) == 1.0  # the shortcut n - c < k of Figure 3
    with pytest.raises(ValueError, match="at least k samples"):
        score([0], [3], k=5)
    with pytest.raises(ValueError, match="problem 1"):
        score([1, 1], [10, 4], k=5)
    assert score([0], [5], k=5)["pass_at_k"] == 0.0  # k = n is allowed


@pytest.mark.parametrize(
    "preds, refs, opts, match",
    [
        ([1, 2], [5], {}, "2 counts of correct samples but 1 sample counts"),
        ([], [], {}, "at least one problem"),
        ([1], [5], {"k": 0}, "k must be at least 1"),
        ([1], [5], {"k": -2}, "k must be at least 1"),
        ([1], [5], {"k": 1.0}, "k must be an integer"),
        ([1], [5], {"k": True}, "k must be an integer"),
        ([6], [5], {}, "need 0 <= c <= n"),
        ([-1], [5], {}, "need 0 <= c <= n"),
        ([1.0], [5], {}, "a count of correct samples must be an integer"),
        ([1], [5.0], {}, "a sample count must be an integer"),
        ([False], [5], {}, "a count of correct samples must be an integer"),
        ([1], [5], {"estimator": "biased"}, "estimator must be one of"),
    ],
)
def test_errors(preds, refs, opts, match):
    kind = TypeError if "must be an integer" in match else ValueError
    with pytest.raises(kind, match=match):
        score(preds, refs, **opts)


def test_estimators():
    assert ESTIMATORS == ("unbiased", "plug-in")
    assert score([1], [200], k=100)["pass_at_k"] == near(0.5)
    assert score([1], [200], k=100, estimator="plug-in")["pass_at_k"] == near(1 - 0.995**100)
