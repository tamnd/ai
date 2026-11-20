# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/test-time-scaling/ (ai://NEW-test-time-scaling).
import itertools
import math

import numpy as np
from test_time_scaling import (
    allocate_budget,
    best_n_fooled,
    best_of_n,
    best_of_n_fooled,
    beta_mixture_coverage,
    coverage_curve,
    coverage_limit,
    exact_coverage,
    fit_exponentiated_power_law,
    flops_matched_multiplier,
    majority_vote,
    pass_at_k,
    random_problems,
    scaling_curves,
    vote_accuracy,
    vote_limit,
    weighted_vote,
)


def test_pass_at_k_is_unbiased():
    for n, p in ((12, 0.1), (30, 0.4)):
        for k in (1, 3, n):
            mean = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * pass_at_k(n, c, k) for c in range(n + 1))
            assert abs(mean - (1 - (1 - p) ** k)) < 1e-10


def test_pass_at_k_matches_binomial_form_and_estimates_coverage():
    for n, c, k in ((10, 3, 4), (100, 1, 50), (100, 0, 7), (5, 5, 2)):
        want = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert abs(pass_at_k(n, c, k) - want) < 1e-12
    rng = np.random.default_rng(0)
    p = rng.beta(0.6, 3.0, size=400)
    correct = rng.binomial(200, p)
    ks = [1, 4, 16, 64]
    assert np.allclose(coverage_curve(200, correct, ks), exact_coverage(p, ks), atol=0.03)


def test_beta_mixture_follows_exponentiated_power_law():
    alpha, beta = 0.5, 2.0
    assert abs(beta_mixture_coverage(alpha, beta, 1) - 0.2) < 1e-12
    ks = np.logspace(3, 6, 20)
    a, b = fit_exponentiated_power_law(ks, [beta_mixture_coverage(alpha, beta, k) for k in ks])
    assert abs(b + alpha) < 0.01 and a < 0
    rng = np.random.default_rng(1)
    p = rng.beta(alpha, beta, size=300_000)
    assert abs(exact_coverage(p, [10])[0] - beta_mixture_coverage(alpha, beta, 10)) < 3e-3


def test_selection_rules():
    answers = np.array([3, 1, 1, 3, 2, 3])
    assert majority_vote(answers) == 3
    assert majority_vote(np.array([5, 4, 4, 5])) == 5  # tie goes to the first seen
    scores = np.array([0.1, 0.9, 0.8, 0.1, 0.95, 0.1])
    assert weighted_vote(answers, scores) == 1
    assert best_of_n(answers, scores) == 2


def test_voting_saturates_while_coverage_keeps_growing():
    rng = np.random.default_rng(2)
    assert vote_accuracy(np.array([0.4, 0.3, 0.3]), 1001, 4000, rng) > 0.99
    assert vote_accuracy(np.array([0.35, 0.45, 0.2]), 1001, 4000, rng) < 0.01
    qs = random_problems(40, 4, 0.5, rng)
    curves = scaling_curves(qs, [1, 4, 16, 64, 256], 200, rng)
    cov, vote = curves["coverage"], curves["vote"]
    assert (np.diff(cov) >= -1e-9).all()
    assert cov[-1] - vote[-1] > 0.2
    assert abs(vote[-1] - vote_limit(qs)) < 0.08
    assert cov[-1] <= coverage_limit(qs) + 1e-9
    assert abs(vote[-1] - vote[-2]) < 0.05


def test_best_of_n_with_a_foolable_verifier_peaks():
    p, eps = 0.3, 0.001
    n_star = best_n_fooled(p, eps)
    assert abs(n_star - 16.5) < 0.05
    vals = [best_of_n_fooled(p, eps, n) for n in range(1, 2000)]
    assert int(np.argmax(vals)) + 1 == 16
    assert vals[0] < vals[15] and vals[999] < vals[15]


def test_allocation_is_optimal_and_skips_impossible_problems():
    rng = np.random.default_rng(3)
    for _ in range(25):
        p = list(rng.choice([0.0, 0.05, 0.2, 0.5, 0.9], size=4))
        budget = int(rng.integers(1, 10))
        k = allocate_budget(p, budget)
        got = sum(1 - (1 - pi) ** ki for pi, ki in zip(p, k))
        best = max(
            sum(1 - (1 - pi) ** ki for pi, ki in zip(p, ks))
            for ks in itertools.product(range(budget + 1), repeat=4)
            if sum(ks) == budget
        )
        assert abs(got - best) < 1e-12
    k = allocate_budget([0.9, 0.05, 0.0], 30)
    assert k[2] == 0 and k[1] > k[0]


def test_flops_matched_multiplier():
    assert flops_matched_multiplier(14, 1.0, 1.0) == 53
    assert abs(flops_matched_multiplier(14, 1.0, 10.0) - 17.9) < 1e-12
    assert flops_matched_multiplier(14, 10.0, 1.0) == 404
    assert flops_matched_multiplier(1, 5.0, 1.0) == 1
