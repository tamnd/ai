# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-hacking/ (ai://NEW-reward-hacking).
"""Train a tabular Q-learning agent in a small gridworld on a shaped proxy reward, and write the final metrics.

The true reward is 10 for reaching the goal cell. The proxy adds a shaping reward for a bonus cell:
in the arm "bonus", 1 each time the agent steps onto it, which is not potential-based;
in the arm "potential", gamma * phi(s') - phi(s) with phi = 1 on the bonus cell, which is potential-based (Ng et al. 1999).
A seed counts as hacked when the greedy policy, run without slips from the start, does not reach the goal.

aik exp run gives the arm, the seed and the configuration in job.json (spec 9.1).
"""

import argparse
import json
import os
import random

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class Grid:
    """A size x size grid. The episode ends at the goal or after horizon steps."""

    def __init__(self, cfg):
        self.size = cfg["size"]
        self.start = tuple(cfg["start"])
        self.goal = tuple(cfg["goal"])
        self.bonus = tuple(cfg["bonus"])
        self.goal_reward = cfg["goal_reward"]
        self.gamma = cfg["gamma"]
        self.slip = cfg["slip"]
        self.horizon = cfg["horizon"]
        self.shaping = cfg["shaping"]

    def phi(self, s):
        return 0.0 if s == self.goal else (1.0 if s == self.bonus else 0.0)

    def step(self, s, a, rng, slip=True):
        if slip and rng.random() < self.slip:
            a = rng.randrange(4)
        dx, dy = MOVES[a]
        n = (min(max(s[0] + dx, 0), self.size - 1), min(max(s[1] + dy, 0), self.size - 1))
        true = self.goal_reward if n == self.goal else 0.0
        if self.shaping == "bonus":
            extra = 1.0 if (n == self.bonus and s != self.bonus) else 0.0
        elif self.shaping == "potential":
            extra = self.gamma * self.phi(n) - self.phi(s)
        else:
            extra = 0.0
        return n, true, true + extra, n == self.goal


def greedy(q, s, rng):
    vals = q[s]
    best = max(vals)
    return rng.choice([a for a in range(4) if vals[a] == best])


def train(env, cfg, rng):
    q = {(x, y): [0.0] * 4 for x in range(env.size) for y in range(env.size)}
    for _ in range(cfg["episodes"]):
        s = env.start
        for _ in range(env.horizon):
            a = rng.randrange(4) if rng.random() < cfg["epsilon"] else greedy(q, s, rng)
            n, _, r, done = env.step(s, a, rng)
            target = r if done else r + env.gamma * max(q[n])
            q[s][a] += cfg["alpha"] * (target - q[s][a])
            s = n
            if done:
                break
    return q


def rollout(env, q, rng, slip):
    s, true, proxy = env.start, 0.0, 0.0
    for _ in range(env.horizon):
        s, t, p, done = env.step(s, greedy(q, s, rng), rng, slip=slip)
        true, proxy = true + t, proxy + p
        if done:
            return True, true, proxy
    return False, true, proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.job) as f:
        job = json.load(f)
    cfg, seed = job["config"], job["seed"]
    rng = random.Random(seed)
    env = Grid(cfg)
    q = train(env, cfg, rng)
    reached, _, _ = rollout(env, q, rng, slip=False)
    runs = [rollout(env, q, rng, slip=True) for _ in range(cfg["eval_rollouts"])]
    final = {
        "hacked": 0.0 if reached else 1.0,
        "goal_rate": sum(r[0] for r in runs) / len(runs),
        "true_return": sum(r[1] for r in runs) / len(runs),
        "proxy_return": sum(r[2] for r in runs) / len(runs),
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "device": "cpu"}, f)


if __name__ == "__main__":
    main()
