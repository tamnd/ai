# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Tabular Q-learning, the NumPy reference.

The update is eq-q-update of the page, and the operator it samples is eq-bellman-optimality.
sarsa_update is eq-sarsa, the on-policy update that the cliff walking experiment compares with.
A table q has one row per state and one column per action. No function changes its inputs.
"""

import numpy as np


def check_rates(alpha, gamma):
    """Check the step size alpha in (0, 1] and the discount gamma in [0, 1]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")


def td_step(q, s, a, target, alpha):
    """Return a copy of q with q[s, a] moved a fraction alpha of the way to target."""
    new = np.array(q, dtype=np.float64)
    new[s, a] += alpha * (target - new[s, a])
    return new


def q_learning_update(q, s, a, r, s_next, alpha, gamma, terminal=False):
    """Return (new q, target) after the transition (s, a, r, s_next).

    eq: eq-q-update. The target is r + gamma * max_b q[s_next, b], or r when s_next is terminal.
    A time limit that cuts an episode is not terminal: pass terminal=False there.
    """
    check_rates(alpha, gamma)
    target = float(r) if terminal else float(r) + gamma * float(np.max(q[s_next]))
    return td_step(q, s, a, target, alpha), target


def sarsa_update(q, s, a, r, s_next, a_next, alpha, gamma, terminal=False):
    """Return (new q, target) after the transition (s, a, r, s_next, a_next).

    eq: eq-sarsa. The target is r + gamma * q[s_next, a_next], or r when s_next is terminal.
    """
    check_rates(alpha, gamma)
    target = float(r) if terminal else float(r) + gamma * float(q[s_next, a_next])
    return td_step(q, s, a, target, alpha), target


def greedy(q, s, rng):
    """Return an action of largest value in state s, with ties broken uniformly at random."""
    row = np.asarray(q[s], dtype=np.float64)
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best))


def epsilon_greedy(q, s, epsilon, rng):
    """Return a uniformly random action with probability epsilon, and a greedy action otherwise."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return greedy(q, s, rng)


def bellman_optimality(q, p, r, gamma):
    """Apply the Bellman optimality operator to q for a known model.

    eq: eq-bellman-optimality. p[s, a, s2] is the probability of the next state s2, and r[s, a] is the
    expected reward. Probability that is missing from a row of p goes to a terminal state of value 0.
    """
    return np.asarray(r, dtype=np.float64) + gamma * np.asarray(p, dtype=np.float64) @ np.max(q, axis=1)
