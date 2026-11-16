# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Tests of the NumPy reference: the worked example, the shared fixture, the operator and an independent oracle.

The oracle is policy iteration with exact linear solves. It never calls the code under test.
"""

import numpy as np
import pytest

from q_learning import bellman_optimality, epsilon_greedy, greedy, q_learning_update, sarsa_update

# The worked example of the page: two states, two actions, alpha = 0.5, gamma = 0.9.
Q0 = np.array([[1.0, 2.0], [4.0, 3.0]])
ALPHA, GAMMA = 0.5, 0.9

# The shared fixture: a 3 by 2 table and four transitions (s, a, r, s_next, terminal), applied in order
# with alpha = 0.25 and gamma = 0.5. Each row gives the target and the new value of q[s, a].
FIX_Q = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])
FIX_STEPS = [
    (0, 1, 1.0, 1, False, 2.0, 1.25),
    (1, 0, -2.0, 2, False, -1.75, 1.0625),
    (2, 1, 3.0, 0, True, 3.0, 1.125),
    (1, 1, 0.0, 0, False, 0.625, -0.59375),
]


def policy_iteration_oracle(p, r, gamma):
    """Return q_* of a finite model by policy iteration with exact linear solves."""
    n_states = p.shape[0]
    policy = np.zeros(n_states, dtype=int)
    while True:
        p_pi = p[np.arange(n_states), policy]
        r_pi = r[np.arange(n_states), policy]
        v = np.linalg.solve(np.eye(n_states) - gamma * p_pi, r_pi)
        q = r + gamma * np.einsum("sak,k->sa", p, v)
        better = q.argmax(axis=1)
        if np.all(q[np.arange(n_states), better] <= q[np.arange(n_states), policy] + 1e-12):
            return q
        policy = better


def random_model(rng, n_states=3, n_actions=2, leak=0.2):
    """A random model whose rows of p sum to 1 - leak: the rest goes to a terminal state."""
    p = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)) * (1.0 - leak)
    r = rng.uniform(-1.0, 1.0, size=(n_states, n_actions))
    return p, r


def sample_next(rng, p, s, a):
    """Sample next states for the arrays s and a; the value p.shape[0] stands for the terminal state."""
    cum = np.cumsum(p[s, a], axis=-1)
    return (rng.random(len(s))[:, None] >= cum).sum(axis=-1)


def test_worked_example():
    q1, t1 = q_learning_update(Q0, 0, 0, 1.0, 1, ALPHA, GAMMA)
    assert t1 == pytest.approx(4.6)
    assert q1[0, 0] == pytest.approx(2.8)
    q2, t2 = q_learning_update(q1, 1, 1, 2.0, 0, ALPHA, GAMMA, terminal=True)
    assert t2 == 2.0
    assert q2[1, 1] == pytest.approx(2.5)
    q3, t3 = q_learning_update(q2, 0, 1, 0.0, 1, ALPHA, GAMMA)
    assert t3 == pytest.approx(3.6)
    np.testing.assert_allclose(q3, [[2.8, 2.8], [4.0, 2.5]], rtol=1e-12)
    s1, u1 = sarsa_update(Q0, 0, 0, 1.0, 1, 1, ALPHA, GAMMA)
    assert u1 == pytest.approx(3.7)
    assert s1[0, 0] == pytest.approx(2.35)


def test_shared_fixture():
    q = FIX_Q
    for s, a, r, s_next, terminal, target, value in FIX_STEPS:
        new, t = q_learning_update(q, s, a, r, s_next, 0.25, 0.5, terminal=terminal)
        assert t == pytest.approx(target, abs=1e-12)
        assert new[s, a] == pytest.approx(value, abs=1e-12)
        mask = np.ones_like(q, dtype=bool)
        mask[s, a] = False
        np.testing.assert_array_equal(new[mask], q[mask])
        q = new


def test_terminal_has_no_bootstrap():
    q = np.array([[2.0, 0.0], [100.0, 200.0]])
    new, target = q_learning_update(q, 0, 0, -1.0, 1, 0.5, 0.9, terminal=True)
    assert target == -1.0
    assert new[0, 0] == 0.5


def test_truncation_bootstraps():
    q = np.array([[0.0, 0.0], [3.0, 1.0]])
    cut, target = q_learning_update(q, 0, 0, -1.0, 1, 1.0, 0.9, terminal=False)
    assert target == pytest.approx(1.7)
    assert cut[0, 0] == pytest.approx(1.7)
    ended, _ = q_learning_update(q, 0, 0, -1.0, 1, 1.0, 0.9, terminal=True)
    assert ended[0, 0] == -1.0


def test_max_not_next_action():
    q = np.array([[0.0, 0.0], [4.0, -5.0]])
    new_q, target_q = q_learning_update(q, 0, 0, 1.0, 1, 1.0, 0.5)
    new_s, target_s = sarsa_update(q, 0, 0, 1.0, 1, 1, 1.0, 0.5)
    assert target_q == 3.0
    assert new_q[0, 0] == 3.0
    assert target_s == -1.5
    assert new_s[0, 0] == -1.5


def test_sarsa_terminal_and_discount():
    q = np.array([[1.0, 0.0], [4.0, 8.0]])
    new, target = sarsa_update(q, 0, 0, 2.0, 1, 1, 0.5, 0.25, terminal=True)
    assert target == 2.0
    assert new[0, 0] == 1.5
    new, target = sarsa_update(q, 0, 0, 2.0, 1, 1, 0.5, 0.25)
    assert target == 4.0
    assert new[0, 0] == 2.5


def test_inputs_not_mutated():
    q = Q0.copy()
    q_learning_update(q, 0, 0, 1.0, 1, ALPHA, GAMMA)
    sarsa_update(q, 0, 0, 1.0, 1, 1, ALPHA, GAMMA)
    np.testing.assert_array_equal(q, Q0)


@pytest.mark.parametrize(("alpha", "gamma"), [(0.0, 0.5), (1.01, 0.5), (-0.1, 0.5), (0.5, -0.01), (0.5, 1.01)])
def test_invalid_rates(alpha, gamma):
    with pytest.raises(ValueError):
        q_learning_update(Q0, 0, 0, 1.0, 1, alpha, gamma)
    with pytest.raises(ValueError):
        sarsa_update(Q0, 0, 0, 1.0, 1, 1, alpha, gamma)


@pytest.mark.parametrize(("alpha", "gamma"), [(1.0, 0.0), (1.0, 1.0), (1e-9, 0.5)])
def test_boundary_rates(alpha, gamma):
    new, target = q_learning_update(Q0, 0, 0, 1.0, 1, alpha, gamma)
    assert target == pytest.approx(1.0 + gamma * 4.0)
    assert new[0, 0] == pytest.approx(1.0 + alpha * (target - 1.0))


@pytest.mark.parametrize("epsilon", [-0.01, 1.01])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        epsilon_greedy(Q0, 0, epsilon, np.random.default_rng(0))


def test_epsilon_greedy_extremes():
    q = np.array([[0.0, 5.0, 1.0]])
    rng = np.random.default_rng(3)
    assert {epsilon_greedy(q, 0, 0.0, rng) for _ in range(200)} == {1}
    assert {epsilon_greedy(q, 0, 1.0, rng) for _ in range(600)} == {0, 1, 2}


def test_epsilon_greedy_frequency():
    q = np.array([[0.0, 5.0, 1.0, 2.0]])
    rng = np.random.default_rng(4)
    picks = np.array([epsilon_greedy(q, 0, 0.2, rng) for _ in range(8000)])
    np.testing.assert_allclose(np.bincount(picks, minlength=4) / picks.size, [0.05, 0.85, 0.05, 0.05], atol=0.015)


def test_ties_broken_at_random():
    q = np.array([[3.0, 1.0, 3.0, 3.0]])
    rng = np.random.default_rng(5)
    picks = np.array([greedy(q, 0, rng) for _ in range(6000)])
    np.testing.assert_allclose(np.bincount(picks, minlength=4) / picks.size, [1 / 3, 0, 1 / 3, 1 / 3], atol=0.03)


def test_bellman_contraction():
    rng = np.random.default_rng(6)
    for _ in range(100):
        p, r = random_model(rng, leak=0.0)
        q1, q2 = rng.normal(size=(2, 3, 2)) * 5
        gap = np.max(np.abs(bellman_optimality(q1, p, r, 0.7) - bellman_optimality(q2, p, r, 0.7)))
        assert gap <= 0.7 * np.max(np.abs(q1 - q2)) + 1e-12


def test_fixed_point_matches_oracle():
    rng = np.random.default_rng(7)
    for _ in range(20):
        p, r = random_model(rng)
        q_star = policy_iteration_oracle(p, r, 0.9)
        np.testing.assert_allclose(bellman_optimality(q_star, p, r, 0.9), q_star, atol=1e-10)
    q = np.zeros_like(q_star)
    for _ in range(400):
        q = bellman_optimality(q, p, r, 0.9)
    np.testing.assert_allclose(q, q_star, atol=1e-10)


def test_target_mean_is_operator():
    rng = np.random.default_rng(8)
    p, r = random_model(rng)
    q = rng.normal(size=(3, 2))
    n = 20000
    s_next = sample_next(rng, p, np.ones(n, dtype=int), np.zeros(n, dtype=int))
    rewards = r[1, 0] + rng.uniform(-1.0, 1.0, size=n)
    targets = [q_learning_update(q, 1, 0, rewards[i], min(s_next[i], 2), 1.0, 0.8, s_next[i] == 3)[1] for i in range(n)]
    expected = bellman_optimality(q, p, r, 0.8)[1, 0]
    assert abs(np.mean(targets) - expected) < 4 * np.std(targets) / np.sqrt(n)


def test_converges_to_oracle():
    rng = np.random.default_rng(9)
    p, r = random_model(rng)
    q_star = policy_iteration_oracle(p, r, 0.5)
    n = 20000
    states, actions = rng.integers(3, size=n), rng.integers(2, size=n)
    s_next = sample_next(rng, p, states, actions)
    rewards = r[states, actions] + rng.uniform(-1.0, 1.0, size=n)
    q, visits = np.zeros((3, 2)), np.zeros((3, 2))
    for s, a, x, reward in zip(states, actions, s_next, rewards, strict=True):
        visits[s, a] += 1
        q, _ = q_learning_update(q, s, a, reward, min(x, 2), 1.0 / visits[s, a], 0.5, terminal=x == 3)
    np.testing.assert_allclose(q, q_star, atol=0.05)
