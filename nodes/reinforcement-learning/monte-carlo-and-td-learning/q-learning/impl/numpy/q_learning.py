# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Tabular Q-learning for finite states and actions."""

import numpy as np


def update(Q, s, a, r, s_next, alpha, gamma, terminal=False):
    """Apply one observed transition and return its target.

    Q is a two-dimensional NumPy array. The update changes it in place.
    """
    if Q.ndim != 2 or Q.shape[1] == 0:
        raise ValueError("Q must have a nonempty action dimension")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    target = float(r) if terminal else float(r) + gamma * float(np.max(Q[s_next]))
    Q[s, a] += alpha * (target - Q[s, a])
    return target


def epsilon_greedy(Q, s, epsilon, rng):
    """Choose a uniformly random action with probability epsilon."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))
