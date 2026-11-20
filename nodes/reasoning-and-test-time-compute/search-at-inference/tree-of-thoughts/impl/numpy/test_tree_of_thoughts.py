# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/tree-of-thoughts/ (ai://NEW-tree-of-thoughts).
import math

import numpy as np

from tree_of_thoughts import (
    SimulatedLM,
    bfs,
    bfs_calls,
    chain,
    chain_success,
    dfs,
    mcts,
    moves,
    problem,
    prune_survival,
    random_problems,
    self_eval_score,
    solvable,
    tree_success,
    uct,
)


def test_game_of_24_tree_size():
    p = problem([4, 9, 10, 13])
    assert solvable(p) and not solvable(problem([1, 1, 1, 1]))
    # 6 pairs x 6 operations, then 3 x 6, then 1 x 6: 36 * 18 * 6 = 3888 leaves, fewer when a step makes a zero
    # that a later step would divide by (4 9 10 13 gives 13 - 9 = 4, then 4 - 4 = 0).
    leaves = lambda p: sum(len(moves(s2)) for _, s in moves(p) for _, s2 in moves(s))
    first = moves(problem([1, 2, 4, 7]))
    assert len(first) == 36 and sum(len(moves(s)) for _, s in first) == 36 * 18
    assert leaves(problem([1, 2, 4, 7])) == 3888 and leaves(p) == 3880


def test_step_selection_beats_whole_chains():
    q, steps, k = 0.4, 3, 5
    assert math.isclose(chain_success(q, steps), 0.064)
    assert math.isclose(chain_success(q, steps, 5), 0.2816, abs_tol=1e-4)
    assert math.isclose(tree_success(q, steps, k), 0.7844, abs_tol=1e-4)
    # Monte Carlo of the abstract model: each proposed step is right with probability q.
    rng = np.random.default_rng(0)
    right = rng.random((20000, steps, k)) < q
    assert abs(right.any(axis=2).all(axis=1).mean() - tree_success(q, steps, k)) < 0.01
    assert abs(right[:, :, 0].all(axis=1).mean() - chain_success(q, steps)) < 0.01


def test_pruning_with_false_negatives():
    assert math.isclose(prune_survival(0.1, 10), 0.3487, abs_tol=1e-4)
    assert math.isclose(prune_survival(0.1, 3), 0.729)
    rng = np.random.default_rng(1)
    assert abs((rng.random((20000, 10)) >= 0.1).all(axis=1).mean() - 0.3487) < 0.01


def test_bfs_call_counts_match_the_bound():
    assert bfs_calls(3, 5, 5, 3) == (11, 55, 165)
    rng = np.random.default_rng(2)
    lm = SimulatedLM(rng, noise=0.0)
    bfs(problem([4, 9, 10, 13]), lm, k=5, b=5)
    assert lm.proposals == 11 and lm.values <= 55


def test_search_beats_chains_and_noise_hurts():
    rng = np.random.default_rng(3)
    probs = random_problems(60, rng)
    cot = np.mean([chain(p, rng) for p in probs])
    exact = np.mean([bfs(p, SimulatedLM(rng, 0.0), k=100, b=1) for p in probs])
    clean = np.mean([bfs(p, SimulatedLM(rng, 0.0), k=10, b=5) for p in probs])
    noisy = np.mean([bfs(p, SimulatedLM(rng, 0.3, value_samples=1), k=10, b=5) for p in probs])
    assert exact == 1.0
    assert clean > cot + 0.4
    assert noisy < clean - 0.15
    assert np.mean([dfs(p, SimulatedLM(rng, 0.0), k=10) for p in probs]) > cot + 0.4


def test_uct_and_mcts():
    a, b = uct(0.6, 5, 10), uct(0.2, 1, 10)
    assert math.isclose(a, 1.2786, abs_tol=1e-4) and math.isclose(b, 1.7174, abs_tol=1e-4)
    assert b > a and uct(0.0, 0, 10) == math.inf
    rng = np.random.default_rng(4)
    probs = random_problems(40, rng)
    assert np.mean([mcts(p, rng, iterations=300) for p in probs]) > 0.3


def test_self_evaluation_score():
    assert math.isclose(self_eval_score(0.9, 0.2), math.sqrt(0.18))
    assert self_eval_score(0.5, 0.9) > self_eval_score(0.9, 0.2)
    assert self_eval_score(0.9, 0.2, lam=0.0) == 0.9
