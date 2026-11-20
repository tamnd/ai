# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/tree-of-thoughts/ (ai://NEW-tree-of-thoughts).
"""Tree of Thoughts on the Game of 24, with a simulated language model.

A state is the multiset of numbers still to combine, and a thought is one arithmetic step that replaces two of
them by their result. The "language model" proposes k random steps and values a state as solvable or not, wrong
with probability `noise`, so the search algorithms can be studied without an API.
"""

import math
from fractions import Fraction
from functools import cache

import numpy as np

OPS = ("+", "-", "r-", "*", "/", "r/")


def apply(a, b, op):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "r-":
        return b - a
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else None
    return b / a if a != 0 else None


def moves(state):
    """All thoughts from a state: an unordered pair and one of six operations, giving the next state."""
    out = []
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            rest = state[:i] + state[i + 1 : j] + state[j + 1 :]
            for op in OPS:
                r = apply(state[i], state[j], op)
                if r is not None:
                    out.append((f"{state[i]} {op} {state[j]} = {r}", tuple(sorted(rest + (r,)))))
    return out


@cache
def solvable(state, target=Fraction(24)):
    if len(state) == 1:
        return state[0] == target
    return any(solvable(s, target) for _, s in moves(state))


def problem(numbers):
    return tuple(sorted(Fraction(int(x)) for x in numbers))


def random_problems(n, rng, lo=1, hi=13):
    out = []
    while len(out) < n:
        p = problem(rng.integers(lo, hi + 1, size=4))
        if solvable(p):
            out.append(p)
    return out


class SimulatedLM:
    """Proposes k distinct random thoughts; values a state by the true solvability, flipped with probability noise."""

    def __init__(self, rng, noise=0.0, value_samples=3):
        self.rng, self.noise, self.value_samples = rng, noise, value_samples
        self.proposals = self.values = 0

    def propose(self, state, k):
        self.proposals += 1
        ms = moves(state)
        idx = self.rng.choice(len(ms), size=min(k, len(ms)), replace=False)
        return [ms[i][1] for i in idx]

    def value(self, state):
        self.values += 1
        truth = solvable(state)
        flips = self.rng.random(self.value_samples) < self.noise
        return float(np.mean(np.where(flips, not truth, truth)))


def chain(start, rng):
    """Chain of thought: one random step at a time to the end, with no lookahead."""
    s = start
    while len(s) > 1:
        ms = moves(s)
        s = ms[rng.integers(len(ms))][1]
    return s[0] == 24


def bfs(start, lm, k, b):
    """ToT-BFS (Yao et al., Algorithm 1): expand every kept state, value the candidates, keep the best b."""
    frontier = [start]
    while len(frontier[0]) > 1:
        cand = [s2 for s in frontier for s2 in lm.propose(s, k)]
        vals = np.array([lm.value(s) for s in cand])
        order = np.argsort(-vals, kind="stable")[:b]
        frontier = [cand[i] for i in order]
    return any(s[0] == 24 for s in frontier[:1])


def dfs(start, lm, k, threshold=0.5, max_steps=100):
    """ToT-DFS (Algorithm 2): follow the best-valued thought, prune states valued at most the threshold, backtrack."""
    steps = 0

    def go(s):
        nonlocal steps
        if len(s) == 1:
            return s[0] == 24
        cand = lm.propose(s, k)
        vals = [lm.value(c) for c in cand]
        for i in np.argsort(-np.array(vals), kind="stable"):
            steps += 1
            if steps > max_steps:
                return False
            if vals[i] > threshold and go(cand[i]):
                return True
        return False

    return go(start)


def uct(value, visits, parent_visits, w=1.0):
    """UCT(s) = V(s) + w sqrt(ln N(p) / N(s)); an unvisited child is tried first."""
    return math.inf if visits == 0 else value + w * math.sqrt(math.log(parent_visits) / visits)


def mcts(start, rng, iterations=50, w=1.0):
    """Monte Carlo Tree Search with random rollouts; reward 1 when a rollout reaches 24, which ends the search."""
    N, V, kids = {start: 0}, {start: 0.0}, {}
    for _ in range(iterations):
        path, s = [start], start
        while kids.get(s):
            s = max(kids[s], key=lambda c: uct(V[c], N[c], max(N[path[-1]], 1), w))
            path.append(s)
        if len(s) > 1 and s not in kids:
            kids[s] = sorted({m for _, m in moves(s)})
            for c in kids[s]:
                N.setdefault(c, 0)
                V.setdefault(c, 0.0)
        r = float(chain(s, rng)) if len(s) > 1 else float(s[0] == 24)
        if r == 1.0:
            return True
        for n in path:
            N[n] += 1
            V[n] += (r - V[n]) / N[n]
    return False


def chain_success(q, steps, n=1):
    """n independent chains of `steps` steps, each step right with probability q."""
    return 1 - (1 - q**steps) ** n


def tree_success(q, steps, k):
    """Beam of one with k proposals per step and an evaluator that recognises a right step."""
    return (1 - (1 - q) ** k) ** steps


def prune_survival(delta, steps):
    """The right path survives when none of its states is wrongly pruned."""
    return (1 - delta) ** steps


def bfs_calls(steps, b, k, value_samples):
    """Upper bounds on proposal calls, candidate states and value samples of ToT-BFS."""
    proposals = 1 + b * (steps - 1)
    return proposals, k * proposals, value_samples * k * proposals


def self_eval_score(p, c, lam=0.5):
    """Xie et al.: the score of a step is C^lam * P^(1 - lam), confidence C and generation probability P."""
    return c**lam * p ** (1 - lam)
