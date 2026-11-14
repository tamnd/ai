# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Cliff walking (Sutton and Barto 2018, Example 6.6): Q-learning against Sarsa, one arm and one seed per run.

aik exp run gives the arm, the seed and the configuration in job.json (spec 9.1), and this writes metrics.json.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "impl", "numpy"))
from q_learning import epsilon_greedy, greedy, q_learning_update, sarsa_update

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left


def make_step(rows, cols, step_reward, cliff_reward):
    """Return the step function of the grid. The start is the bottom left cell and the goal the bottom right one."""
    start, goal = (rows - 1) * cols, rows * cols - 1

    def step(s, a):
        """Return (next state, reward, terminal, fell). A step into the cliff costs cliff_reward and goes to start."""
        r, c = divmod(s, cols)
        r = min(max(r + MOVES[a][0], 0), rows - 1)
        c = min(max(c + MOVES[a][1], 0), cols - 1)
        if r == rows - 1 and 0 < c < cols - 1:
            return start, cliff_reward, False, True
        s_next = r * cols + c
        return s_next, step_reward, s_next == goal, False

    return step, start, goal


def run(cfg, seed):
    """Train one agent and return its metrics."""
    rng = np.random.default_rng(seed)
    rows, cols = cfg["grid"]["rows"], cfg["grid"]["cols"]
    step, start, goal = make_step(rows, cols, cfg["rewards"]["step"], cfg["rewards"]["cliff"])
    alpha, gamma, eps = cfg["alpha"], cfg["gamma"], cfg["epsilon"]
    q = np.full((rows * cols, len(MOVES)), float(cfg["q_init"]))
    returns, falls = [], []
    for _ in range(cfg["episodes"]):
        s, total, fell, done = start, 0.0, 0, False
        a = epsilon_greedy(q, s, eps, rng)
        while not done:
            s_next, r, done, f = step(s, a)
            total, fell = total + r, fell + f
            a_next = epsilon_greedy(q, s_next, eps, rng)
            if cfg["method"] == "sarsa":
                q, _ = sarsa_update(q, s, a, r, s_next, a_next, alpha, gamma, terminal=done)
            else:
                q, _ = q_learning_update(q, s, a, r, s_next, alpha, gamma, terminal=done)
            s, a = s_next, a_next
        returns.append(total)
        falls.append(fell)
    path, s = [start], start
    while s != goal and len(path) <= 4 * rows * cols:
        s = step(s, greedy(q, s, rng))[0]
        path.append(s)
    edge = {(rows - 2) * cols + c for c in range(cols)}
    skip = cfg["skip_episodes"]
    return {
        "online_return": float(np.mean(returns[skip:])),
        "greedy_path_length": float(len(path) - 1),
        "greedy_edge_path": float(s == goal and edge <= set(path)),
        "falls_per_episode": float(np.mean(falls[skip:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.job) as f:
        job = json.load(f)
    metrics = run(job["config"], int(job["seed"]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"final": metrics, "device": "cpu"}, f, indent=2)


if __name__ == "__main__":
    main()
