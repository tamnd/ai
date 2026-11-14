# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Independent fixtures and an exact dynamic-programming oracle."""

import numpy as np

from q_learning import epsilon_greedy, update


def test_worked_example_and_one_entry():
    Q = np.array([[2.0, -7.0], [4.0, 3.0], [8.0, 9.0]])
    before = Q.copy()
    target = update(Q, 0, 0, 1.0, 1, 0.25, 0.5)
    assert target == 3.0
    assert Q[0, 0] == 2.25
    before[0, 0] = 2.25
    np.testing.assert_array_equal(Q, before)


def test_terminal_ignores_nonzero_next_row():
    Q = np.array([[2.0, 0.0], [100.0, 200.0]])
    target = update(Q, 0, 0, -1.0, 1, 0.5, 0.9, terminal=True)
    assert target == -1.0
    assert Q[0, 0] == 0.5


def test_episodic_task_accepts_unit_discount():
    Q = np.array([[0.0, 0.0], [9.0, 8.0]])
    target = update(Q, 0, 0, 2.0, 1, 0.5, 1.0, terminal=True)
    assert target == 2.0
    assert Q[0, 0] == 1.0


def test_time_limit_truncation_bootstraps():
    Q = np.array([[0.0, 0.0], [3.0, 1.0]])
    target = update(Q, 0, 0, -1.0, 1, 1.0, 0.9, terminal=False)
    assert np.isclose(target, 1.7)
    assert np.isclose(Q[0, 0], 1.7)


def test_max_not_sampled_action():
    Q = np.array([[0.0, 0.0], [4.0, -5.0]])
    target = update(Q, 0, 0, 1.0, 1, 1.0, 0.5)
    assert target == 3.0
    assert Q[0, 0] == 3.0


def value_iteration_oracle(transitions, rewards, gamma, tolerance=1e-13):
    """Solve the finite MDP from its full model, without using update()."""
    values = np.zeros(transitions.shape[0])
    while True:
        action_values = rewards + gamma * np.einsum("sak,k->sa", transitions, values)
        next_values = action_values.max(axis=1)
        if np.max(np.abs(next_values - values)) < tolerance:
            return action_values
        values = next_values


def test_against_independent_value_iteration_oracle():
    # A finite deterministic MDP: action 0 moves to the rewarding state,
    # and action 1 stays. The oracle sums the infinite discounted return.
    transitions = np.zeros((2, 2, 2))
    transitions[0, 0, 1] = 1
    transitions[0, 1, 0] = 1
    transitions[1, :, 1] = 1
    rewards = np.array([[0.0, -1.0], [1.0, 0.0]])
    gamma = 0.8
    expected = value_iteration_oracle(transitions, rewards, gamma)
    Q = np.zeros((2, 2))
    for _ in range(2000):
        for s in range(2):
            for a in range(2):
                s_next = int(np.argmax(transitions[s, a]))
                update(Q, s, a, rewards[s, a], s_next, 0.1, gamma)
    np.testing.assert_allclose(Q, expected, atol=0.01, rtol=0)


def test_epsilon_greedy_extremes_and_ties():
    Q = np.array([[2.0, 2.0, 0.0]])
    rng = np.random.default_rng(17)
    assert {epsilon_greedy(Q, 0, 0.0, rng) for _ in range(30)} == {0}
    actions = [epsilon_greedy(Q, 0, 1.0, rng) for _ in range(3000)]
    np.testing.assert_allclose(np.bincount(actions, minlength=3) / 3000, [1/3]*3, atol=0.04)


def test_epsilon_greedy_partial_exploration_frequency():
    Q = np.array([[0.0, 3.0, 1.0]])
    rng = np.random.default_rng(19)
    actions = [epsilon_greedy(Q, 0, 0.1, rng) for _ in range(10000)]
    frequencies = np.bincount(actions, minlength=3) / 10000
    np.testing.assert_allclose(frequencies, [1/30, 0.9 + 1/30, 1/30], atol=0.015)


def test_invalid_hyperparameters_raise():
    Q = np.zeros((2, 2))
    for alpha in (0, 1.1):
        with np.testing.assert_raises(ValueError):
            update(Q.copy(), 0, 0, 0, 1, alpha, 0.9)
    for gamma in (-0.1, 1.1):
        with np.testing.assert_raises(ValueError):
            update(Q.copy(), 0, 0, 0, 1, 0.5, gamma)
    rng = np.random.default_rng(1)
    for epsilon in (-0.1, 1.1):
        with np.testing.assert_raises(ValueError):
            epsilon_greedy(Q, 0, epsilon, rng)


def test_stochastic_transitions_against_value_iteration_oracle():
    transitions = np.array([
        [[0.25, 0.75], [0.8, 0.2]],
        [[0.1, 0.9], [0.6, 0.4]],
    ])
    rewards = np.array([[0.0, -0.4], [1.0, 0.2]])
    gamma = 0.7
    expected = value_iteration_oracle(transitions, rewards, gamma)
    Q = np.zeros((2, 2))
    rng = np.random.default_rng(23)
    visits = np.zeros((2, 2), dtype=int)
    for _ in range(300000):
        s = int(rng.integers(2))
        a = int(rng.integers(2))
        visits[s, a] += 1
        s_next = int(rng.choice(2, p=transitions[s, a]))
        update(Q, s, a, rewards[s, a], s_next, 1.0 / visits[s, a] ** 0.7, gamma)
    np.testing.assert_allclose(Q, expected, atol=0.03, rtol=0)


def test_optimistic_initial_value_guides_greedy_choice():
    Q = np.array([[0.0, 2.0]])
    rng = np.random.default_rng(7)
    assert epsilon_greedy(Q, 0, 0.0, rng) == 1
    update(Q, 0, 1, 0.0, 0, 1.0, 0.0, terminal=True)
    assert epsilon_greedy(Q, 0, 0.0, rng) == 0
