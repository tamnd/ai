# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-faithfulness/ (ai://NEW-chain-of-thought-faithfulness).
import math

import numpy as np
from chain_of_thought_faithfulness import (
    aoc,
    aoc_ceiling,
    contradiction_rate,
    detection_probability,
    early_answering_curve,
    hint_alpha,
    hint_faithfulness,
    hint_rates,
    pair_contradiction_rate,
    simulate_early_answering,
    simulate_hint_eval,
    weighted_aoc,
)


def test_alpha_matches_the_worked_examples():
    assert math.isclose(hint_alpha(0.30, 0.04, 4), 0.9333, abs_tol=5e-5)
    assert math.isclose(0.10 / hint_alpha(0.10, 0.12, 4), 0.25, abs_tol=1e-12)
    assert hint_alpha(0.05, 0.10, 4) <= 0


def test_normalized_score_recovers_the_verbalization_rate():
    rng = np.random.default_rng(0)
    for use, noise in ((0.6, 0.05), (0.1, 0.3)):
        au, ah, hint, verbal = simulate_hint_eval(300_000, 4, use, 0.25, noise, rng)
        raw, norm = hint_faithfulness(au, ah, hint, verbal, 4)
        assert abs(norm - 0.25) < 0.01
        assert raw < norm
    # A weak hint with heavy noise is where the correction matters most.
    au, ah, hint, verbal = simulate_hint_eval(300_000, 4, 0.1, 0.25, 0.3, rng)
    raw, norm = hint_faithfulness(au, ah, hint, verbal, 4)
    assert norm - raw > 0.05


def test_alpha_is_near_zero_without_hint_use():
    rng = np.random.default_rng(1)
    au, ah, hint, verbal = simulate_hint_eval(50_000, 4, 0.0, 0.5, 0.3, rng)
    p, q = hint_rates(au, ah, hint)
    assert abs(p - q / 2) < 0.01
    assert abs(hint_alpha(p, q, 4)) < 0.05
    raw, norm = hint_faithfulness(au, ah, hint, verbal, 4)
    assert raw == 0 and norm in (None, 0.0)


def test_aoc_values_and_ceiling():
    assert math.isclose(aoc([0.3, 0.4, 0.6, 0.9, 1.0]), 0.3625, abs_tol=1e-12)
    assert aoc([1, 1, 1, 1]) == 0
    assert math.isclose(aoc_ceiling(5, 4), 0.7, abs_tol=1e-12)
    assert math.isclose(weighted_aoc([[0.5, 1], [1, 1, 1]], [1, 3]), 0.25 * 0.25, abs_tol=1e-12)


def test_early_answering_separates_post_hoc_from_load_bearing_chains():
    rng = np.random.default_rng(2)
    post = aoc(early_answering_curve(*simulate_early_answering(20_000, 4, 5, 0, rng)))
    late = aoc(early_answering_curve(*simulate_early_answering(20_000, 4, 5, 4, rng)))
    mid = aoc(early_answering_curve(*simulate_early_answering(20_000, 4, 5, 2, rng)))
    assert post == 0
    assert abs(late - aoc_ceiling(5, 4)) < 0.01
    assert post < mid < late


def test_detection_over_episodes():
    assert math.isclose(detection_probability(0.02, 100), 0.8674, abs_tol=5e-5)
    assert detection_probability(0.02, 148) < 0.95 < detection_probability(0.02, 149)
    assert math.isclose(detection_probability(0.02, 10, recall=0.5), 1 - 0.99**10, abs_tol=1e-12)


def test_pair_contradictions():
    assert math.isclose(pair_contradiction_rate(0.99, 0.63), 0.6274, abs_tol=1e-12)
    assert pair_contradiction_rate(1.0, 0.0) == 0
    rng = np.random.default_rng(3)
    a, b = rng.random(200_000) < 0.99, rng.random(200_000) < 0.63
    assert abs(contradiction_rate(a, b) - 0.6274) < 0.005
