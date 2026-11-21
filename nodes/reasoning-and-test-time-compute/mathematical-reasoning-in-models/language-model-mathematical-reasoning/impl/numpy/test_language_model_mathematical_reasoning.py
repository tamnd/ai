# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-mathematical-reasoning/ (ai://NEW-language-model-mathematical-reasoning).
"""Tests for the templated problems, the toy solvers and the accuracy statistics."""

import itertools

import numpy as np
from language_model_mathematical_reasoning import (
    add_noop,
    answer,
    chain_success,
    gap_se,
    instantiate,
    keyword_solve,
    majority_vote,
    make_template,
    noisy_solve,
    pass_at_k,
    render,
    set_sd,
    vote_accuracy,
)


def test_instances_keep_the_template_and_valid_numbers():
    rng = np.random.default_rng(0)
    t = make_template(3, rng)
    base = instantiate(t, rng)
    renamed = instantiate(t, rng, base=base, numbers=False)
    assert [k for k, _ in renamed["clauses"]] == t
    assert answer(renamed) == answer(base)
    for _ in range(50):
        p = instantiate(t, rng)
        assert 0 < answer(p) < 1000


def test_keyword_solver_survives_new_names_and_numbers_but_not_noops():
    rng = np.random.default_rng(1)
    plain, noop = [], []
    for _ in range(300):
        p = instantiate(make_template(int(rng.integers(1, 5)), rng), rng)
        plain.append(keyword_solve(render(p)) == answer(p))
        q = add_noop(p, rng)
        assert answer(q) == answer(p)
        noop.append(keyword_solve(render(q)) == answer(q))
    assert all(plain)
    assert np.mean(noop) < 0.05


def test_noisy_solver_follows_the_chain_rule():
    rng = np.random.default_rng(2)
    for steps in (1, 4, 8):
        t = ["start"] + ["add"] * steps
        hits = [noisy_solve(p, 0.9, rng) == answer(p) for p in (instantiate(t, rng) for _ in range(4000))]
        assert abs(np.mean(hits) - chain_success(0.9, steps)) < 0.03


def test_spread_across_sets_comes_from_instance_dependence():
    rng = np.random.default_rng(3)
    ps = rng.uniform(0.3, 1.0, 100)
    accs = [np.mean(rng.random(100) < ps) for _ in range(4000)]
    assert abs(np.std(accs) - set_sd(ps)) < 0.003
    assert set_sd(np.round(ps)) == 0.0
    assert set_sd(ps) <= np.sqrt(ps.mean() * (1 - ps.mean()) / 100)
    assert abs(100 * gap_se(0.8, 1205, 1319) - 1.59) < 0.01


def test_vote_accuracy_matches_simulation_and_a_systematic_error_wins():
    rng = np.random.default_rng(4)
    for q, w, k in [(0.4, 0.0, 5), (0.4, 0.2, 7), (0.4, 0.5, 9)]:
        wins = 0
        for t in range(6000):
            u = rng.random(k)
            ans = ["right" if x < q else "wrong" if x < q + w else f"other{t}-{i}" for i, x in enumerate(u)]
            wins += majority_vote(ans, rng) == "right"
        assert abs(wins / 6000 - vote_accuracy(q, w, k)) < 0.025
    assert vote_accuracy(0.4, 0.0, 41) > 0.99
    assert vote_accuracy(0.4, 0.5, 41) < vote_accuracy(0.4, 0.5, 1) == 0.4


def test_pass_at_k_matches_enumeration():
    for n, c, k in [(6, 2, 3), (8, 1, 4), (7, 0, 2)]:
        draws = list(itertools.combinations([1] * c + [0] * (n - c), k))
        assert abs(pass_at_k(n, c, k) - np.mean([any(d) for d in draws])) < 1e-12
