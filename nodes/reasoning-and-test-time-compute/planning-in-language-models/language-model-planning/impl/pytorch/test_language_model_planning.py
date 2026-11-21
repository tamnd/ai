# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-planning/ (ai://NEW-language-model-planning).
import random

import pytest
import torch
from language_model_planning import TABLE, generate_and_test, optimal_plan, random_problem, success, train, valid


@pytest.fixture(scope="module")
def model():
    return train(sizes=(3, 4), problems=3000, epochs=30, seed=0)


def test_optimal_plans_are_valid_and_table_to_stack_is_two_per_block():
    rng = random.Random(0)
    for _ in range(20):
        on, goal = random_problem(rng.choice([3, 4, 5]), rng)
        assert valid(on, goal, optimal_plan(on, goal))
    for n in (2, 3, 4, 5):
        on = [TABLE] * n + [None] * (6 - n)
        goal = [b + 1 for b in range(n - 1)] + [TABLE] + [None] * (6 - n)
        assert len(optimal_plan(on, goal)) == 2 * (n - 1)


def test_policy_plans_well_on_its_training_sizes(model):
    assert success(model, 3, 200) > 0.85
    assert success(model, 4, 200) > 0.7


def test_policy_degrades_on_larger_problems(model):
    in_dist = success(model, 4, 200)
    larger = success(model, 6, 200)
    assert larger < in_dist - 0.25


def test_validator_loop_lifts_success_and_returns_only_valid_plans(model):
    greedy = success(model, 6, 200)
    looped = success(model, 6, 200, k=8, temperature=1.0)
    assert looped > greedy + 0.1
    rng = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    for _ in range(50):
        on, goal = random_problem(6, rng)
        plan = generate_and_test(model, on, goal, 8, 1.0, gen)
        assert plan is None or valid(on, goal, plan)
