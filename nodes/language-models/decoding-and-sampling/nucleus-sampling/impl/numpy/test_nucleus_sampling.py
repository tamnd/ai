# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
"""Tests of the NumPy reference: the worked example of the node in exact fractions, the effect of temperature on the
nucleus, the flat and peaked cases, min-p, the crossing token, ties, the nucleus as a smallest set and its mass
bounds, the variants, sampling frequencies, and the fixture shared with the PyTorch tests."""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from nucleus_sampling import (
    entropy,
    eta_mask,
    min_p_mask,
    nucleus_distribution,
    renormalize,
    sample_next,
    softmax,
    tempered,
    top_k_mask,
    top_p_mask,
    total_variation,
    typical_mask,
)

P = np.array([0.5, 0.2, 0.15, 0.1, 0.05])

# The shared fixture. The PyTorch tests check the same logits, p and kept sets.
FIXTURE_LOGITS = [
    [2.0, 1.0, 0.5, 0.0, -1.0, -3.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 0.0, -1.0, -1.0, -2.0, -4.0],
]
FIXTURE_TOP_P = 0.9
FIXTURE_KEPT = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0]]


def test_worked_example():
    mask = top_p_mask(P, 0.8)
    assert mask.tolist() == [True, True, True, False, False]
    new = nucleus_distribution(P, 0.8)
    fr = [Fraction(x).limit_denominator(1000) for x in new]
    assert fr == [Fraction(10, 17), Fraction(4, 17), Fraction(3, 17), 0, 0]
    assert new.sum() == pytest.approx(1.0)
    assert total_variation(P, new) == pytest.approx(0.15)
    top2 = renormalize(P, top_k_mask(P, 2))
    assert top2 == pytest.approx([5 / 7, 2 / 7, 0, 0, 0])


def test_temperature_shrinks_nucleus():
    logits = np.log(P)
    cold = tempered(logits, 0.5)
    assert cold == pytest.approx([0.769, 0.123, 0.069, 0.031, 0.008], abs=6e-4)
    assert top_p_mask(tempered(logits, 1.0), 0.8).sum() == 3
    assert top_p_mask(cold, 0.8).sum() == 2
    # Tempering after truncation keeps the set and only reweights it; tempering first changes the set.
    after = top_p_mask(P, 0.8)
    first = top_p_mask(cold, 0.8)
    assert after.sum() == 3 and first.sum() == 2
    reweighted = renormalize(tempered(logits, 0.5), after)
    assert (reweighted > 0).sum() == 3


def test_flat_and_peaked():
    flat = np.full(50, 0.02)
    assert top_p_mask(flat, 0.95).sum() == 48
    peaked = np.array([0.96, 0.02, 0.01, 0.01])
    assert top_p_mask(peaked, 0.95).tolist() == [True, False, False, False]
    # top-k with a fixed k keeps 2 of the flat tokens and 3 junk tokens of the peaked one
    assert top_k_mask(flat, 2).sum() == 2
    assert top_k_mask(peaked, 3).sum() == 3


def test_min_p_worked_example():
    mask = min_p_mask(P, 0.25)
    assert 0.25 * P.max() == pytest.approx(0.125)
    assert mask.tolist() == [True, True, True, False, False]


def test_includes_crossing_token():
    # The token that crosses p is kept.
    assert top_p_mask(P, 0.8).sum() == 3
    # Keeping only the tokens whose cumulative sum stays below p would return nothing when the top token exceeds p.
    peaked = np.array([0.96, 0.02, 0.01, 0.01])
    wrong = np.cumsum(np.sort(peaked)[::-1]) < 0.9
    assert wrong.sum() == 0
    assert top_p_mask(peaked, 0.9).sum() == 1


def test_ties_are_deterministic():
    probs = np.array([0.1, 0.3, 0.3, 0.3])
    first = top_p_mask(probs, 0.5)
    assert first.tolist() == [False, True, True, False]
    for _ in range(5):
        assert top_p_mask(probs.copy(), 0.5).tolist() == first.tolist()
    assert top_k_mask(probs, 1).tolist() == [False, True, False, False]


def test_limits():
    assert top_p_mask(P, 0.4).sum() == 1
    assert top_p_mask(P, 1.0).sum() == 5
    with pytest.raises(ValueError):
        top_p_mask(P, 0.0)
    with pytest.raises(ValueError):
        tempered([0.0, 1.0], 0.0)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(0.0, 1.0), min_size=1, max_size=8).filter(
        lambda v: sum(v) > 1e-3
    ),
    st.floats(0.01, 1.0),
)
def test_smallest_set_and_mass_bounds(weights, p):
    probs = np.array(weights) / sum(weights)
    mask = top_p_mask(probs, p)
    m = int(mask.sum())
    kept = probs[mask].sum()
    assert kept >= p - 1e-9
    q = np.sort(probs)[::-1]
    assert kept < p + q[m - 1] + 1e-9
    smallest = min(
        r
        for r in range(1, len(probs) + 1)
        for s in itertools.combinations(range(len(probs)), r)
        if probs[list(s)].sum() >= p - 1e-9
    )
    assert m == smallest
    assert total_variation(probs, renormalize(probs, mask)) == pytest.approx(
        1 - kept, abs=1e-12
    )


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-5.0, 5.0), min_size=2, max_size=10), st.floats(0.05, 0.99))
def test_nucleus_grows_with_p_and_t(logits, p):
    probs = softmax(logits)
    grid = np.linspace(0.05, 1.0, 20)
    sets = [set(np.flatnonzero(top_p_mask(probs, g))) for g in grid]
    assert all(a <= b for a, b in itertools.pairwise(sets))
    sizes = [
        top_p_mask(tempered(logits, t), p).sum() for t in (0.2, 0.5, 1.0, 2.0, 5.0)
    ]
    assert all(a <= b for a, b in itertools.pairwise(sizes))
    hs = [entropy(tempered(logits, t)) for t in (0.2, 0.5, 1.0, 2.0, 5.0)]
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(hs))


def test_variants_keep_the_top_token():
    rng = np.random.default_rng(0)
    for _ in range(100):
        probs = rng.dirichlet(np.full(20, 0.3))
        top = int(np.argmax(probs))
        for mask in (
            eta_mask(probs, 3e-4),
            typical_mask(probs, 0.2),
            min_p_mask(probs, 0.1),
            top_p_mask(probs, 0.1),
        ):
            assert mask.any()
        assert eta_mask(probs, 3e-4)[top] and min_p_mask(probs, 0.1)[top]


def test_eta_threshold():
    # A peaked distribution with a long flat tail: eta keeps more of the plausible tokens than top-p 0.95.
    probs = np.array([0.96] + [0.004] * 10)
    eps = 9e-4
    eta = min(eps, np.sqrt(eps) * np.exp(-entropy(probs)))
    assert eta_mask(probs, eps).sum() == 1 + int((probs[1:] > eta).sum())
    assert eta_mask(probs, eps).sum() > top_p_mask(probs, 0.95).sum()


def test_typical_prefers_typical_tokens():
    probs = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
    h = entropy(probs)
    mask = typical_mask(probs, 0.1)
    best = int(np.argmin(np.abs(h + np.log(probs))))
    assert mask[best]
    # the top token, 0.4 against e^-H of about 0.22, is less typical than the 0.15 tokens
    assert not mask[0]


def test_sampling_frequencies():
    rng = np.random.default_rng(1)
    logits = np.log(P)
    counts = np.zeros(5)
    for _ in range(20000):
        token, _dist = sample_next(logits, rng, top_p=0.8)
        counts[token] += 1
    assert counts[3] == 0 and counts[4] == 0
    assert counts / counts.sum() == pytest.approx(
        [10 / 17, 4 / 17, 3 / 17, 0, 0], abs=0.015
    )


def test_fixture():
    for logits, kept in zip(FIXTURE_LOGITS, FIXTURE_KEPT):
        assert (
            np.flatnonzero(top_p_mask(softmax(logits), FIXTURE_TOP_P)).tolist() == kept
        )
