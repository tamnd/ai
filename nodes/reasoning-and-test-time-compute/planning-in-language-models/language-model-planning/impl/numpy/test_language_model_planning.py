# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-planning/ (ai://NEW-language-model-planning).
import math

import numpy as np
import pytest
from language_model_planning import (
    HAND,
    TABLE,
    apply,
    bfs_plan,
    count_states,
    format_plan,
    generate_and_test,
    hand_empty_states,
    noisy_proposer,
    noisy_verifier,
    parse_plan,
    plan_length_bound,
    random_state,
    reachable,
    sound_loop_success,
    sound_verifier,
    table_to_stack,
    unsound_loop_correct,
    validate,
    verifier_precision,
)


def test_state_counts_match_enumeration():
    for n, empty, total in ((3, 13, 22), (4, 73, 125), (5, 501, 866)):
        assert hand_empty_states(n) == empty
        assert count_states(n) == total
        states = reachable(tuple([TABLE] * n))
        assert len(states) == total
        assert sum(HAND not in s for s in states) == empty


def test_validator_reports_the_first_error_like_val():
    init = (1, TABLE, TABLE)  # block 0 on block 1
    goal = {1: 0}
    bad = [("pick-up", 1), ("stack", 1, 0)]
    report = validate(init, goal, bad)
    assert report == {"valid": False, "executable": False, "step": 0, "unmet": ["clear 1"]}
    short = [("unstack", 0, 1), ("put-down", 0)]
    report = validate(init, goal, short)
    assert report["executable"] and not report["valid"] and report["unmet"] == [1]
    good = [*short, ("pick-up", 1), ("stack", 1, 0)]
    assert validate(init, goal, good)["valid"]


def test_optimal_plans_respect_the_bound():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(3, 6))
        init, goal_state = random_state(n, rng), random_state(n, rng)
        goal = dict(enumerate(goal_state))
        plan = bfs_plan(init, goal)
        assert validate(init, goal, plan)["valid"]
        assert len(plan) <= plan_length_bound(init, goal_state) <= 4 * (n - 1)
    for n in (2, 3, 4, 5):
        init, goal = table_to_stack(n)
        assert len(bfs_plan(init, goal)) == 2 * (n - 1)


def test_renaming_does_not_change_the_problem():
    init, goal = table_to_stack(4)
    plan = bfs_plan(init, goal)
    mystery = {"pick-up": "attack", "unstack": "feast", "put-down": "succumb", "stack": "overcome"}
    names = ["x7", "q2", "z9", "m4"]
    text = format_plan(plan, mystery, names)
    assert "stack" not in text
    assert parse_plan(text, mystery, names) == plan
    assert validate(init, goal, parse_plan(text, mystery, names))["valid"]
    with pytest.raises(ValueError):
        parse_plan(text)


def test_sound_loop_returns_only_valid_plans():
    rng = np.random.default_rng(1)
    init, goal = (TABLE, 0, 1, TABLE, 3), {0: 4, 4: 1, 1: TABLE}
    propose = noisy_proposer(init, goal, 0.15, rng)
    single = np.mean([validate(init, goal, propose())["valid"] for _ in range(3000)])
    assert 0.2 < single < 0.8
    wins = 0
    for _ in range(1500):
        plan, _rounds = generate_and_test(propose, sound_verifier(init, goal), 5)
        if plan is not None:
            assert validate(init, goal, plan)["valid"]
            wins += 1
    assert abs(wins / 1500 - sound_loop_success(single, 5)) < 0.04


def test_unsound_verifier_caps_the_loop_at_its_precision():
    assert math.isclose(verifier_precision(0.4, 0.982, 0.8445), 0.4367, abs_tol=5e-5)
    assert math.isclose(unsound_loop_correct(0.4, 0.982, 0.8445, 15), 0.4367, abs_tol=5e-5)
    assert math.isclose(sound_loop_success(0.4, 15), 0.99953, abs_tol=5e-6)
    rng = np.random.default_rng(2)
    init, goal = (TABLE, 0, 1, TABLE, 3), {0: 4, 4: 1, 1: TABLE}
    propose = noisy_proposer(init, goal, 0.15, rng)
    p = np.mean([validate(init, goal, propose())["valid"] for _ in range(3000)])
    correct = 0
    for _ in range(2000):
        verify = noisy_verifier(init, goal, 0.98, 0.8, rng)
        plan, _ = generate_and_test(propose, verify, 10)
        correct += plan is not None and validate(init, goal, plan)["valid"]
    assert abs(correct / 2000 - unsound_loop_correct(p, 0.98, 0.8, 10)) < 0.04
    assert correct / 2000 < p + 0.1


def test_apply_rejects_inexecutable_actions():
    assert apply((TABLE, TABLE), ("stack", 0, 1)) is None
    assert apply((TABLE, TABLE), ("pick-up", 0)) == (HAND, TABLE)
