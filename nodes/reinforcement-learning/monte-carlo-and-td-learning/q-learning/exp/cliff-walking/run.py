# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/q-learning/ (ai://KDH97E).
"""Compare online cliff-walking returns across independent random seeds.

Run from this directory: python3 run.py
"""

import json
import argparse
import os
import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[2] / "impl" / "numpy"
sys.path.insert(0, str(IMPL))
from q_learning import epsilon_greedy, update  # noqa: E402


HEIGHT, WIDTH = 4, 12
START = (HEIGHT - 1) * WIDTH
GOAL = START + WIDTH - 1
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def step(state, action):
    row, col = divmod(state, WIDTH)
    dr, dc = MOVES[action]
    row = min(max(row + dr, 0), HEIGHT - 1)
    col = min(max(col + dc, 0), WIDTH - 1)
    next_state = row * WIDTH + col
    if row == HEIGHT - 1 and 1 <= col < WIDTH - 1:
        return START, -100.0, False
    return next_state, -1.0, next_state == GOAL


def run_seed(seed, method, episodes=150, max_steps=1000, alpha=0.5, gamma=0.9, epsilon=0.1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((HEIGHT * WIDTH, len(MOVES)))
    returns = []
    for _ in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            next_state, reward, terminal = step(state, action)
            total += reward
            if method == "q_learning":
                update(Q, state, action, reward, next_state, alpha, gamma, terminal)
            else:
                next_action = None if terminal else epsilon_greedy(Q, next_state, epsilon, rng)
                target = reward if terminal else reward + gamma * Q[next_state, next_action]
                Q[state, action] += alpha * (target - Q[state, action])
            if terminal:
                break
            state = next_state
            action = next_action if method == "sarsa" else epsilon_greedy(Q, state, epsilon, rng)
        returns.append(total)
    state = START
    visited = []
    for _ in range(max_steps):
        visited.append(state)
        if state == GOAL:
            break
        state = step(state, int(np.argmax(Q[state])))[0]
    traverses_edge = all((HEIGHT - 2) * WIDTH + col in visited for col in range(1, WIDTH - 1))
    reaches_goal = bool(visited and visited[-1] == GOAL)
    return float(np.mean(returns[-50:])), reaches_goal, traverses_edge


def run_job(job_path, output_dir):
    with open(job_path) as f:
        job = json.load(f)
    config = job["config"]
    seed = int(job["seed"])
    method = job["arm"]
    online_return, reaches_goal, edge_path = run_seed(
        seed, method, episodes=config["episodes"], max_steps=config["max_steps_per_episode"],
        alpha=config["alpha"], gamma=config["gamma"], epsilon=config["epsilon"])
    os.makedirs(output_dir, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump({"final": {"online_return": online_return,
                              "greedy_reaches_goal_fraction": float(reaches_goal),
                              "greedy_edge_path_fraction": float(edge_path)}, "device": "cpu"}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.job:
        run_job(args.job, args.out)
        return
    seeds = range(30)
    result = {
        "task": "four-by-twelve cliff walking",
        "episodes_per_seed": 150,
        "last_episodes_averaged": 50,
        "seeds": list(seeds),
        "gamma": 0.9,
        "alpha": 0.5,
        "epsilon": 0.1,
        "metric": "online return; higher is better",
        "greedy_path": "after training, start at START and follow argmax(Q); edge_path means visiting every cell directly above the cliff",
        "scope": "tabular Q-learning and Sarsa, this grid and these settings only; not a reproduction of Sutton and Barto Figure 6.4",
    }
    for method in ("q_learning", "sarsa"):
        outcomes = [run_seed(seed, method) for seed in seeds]
        values = [outcome[0] for outcome in outcomes]
        result[method] = {
            "mean": float(np.mean(values)),
            "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "greedy_reaches_goal_fraction": float(np.mean([outcome[1] for outcome in outcomes])),
            "greedy_edge_path_fraction": float(np.mean([outcome[2] for outcome in outcomes])),
        }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
