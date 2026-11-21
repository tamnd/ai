# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-planning/ (ai://NEW-language-model-planning).
"""Blocksworld, a plan validator in the style of VAL, and generate-test loops around an approximate plan source.

A state is a tuple `on` with one entry per block: TABLE if the block is on the table, HAND if it is held, or the
index of the block it sits on. A goal is a dict from block to TABLE or to the block it must sit on; blocks left
out of the goal may end anywhere. Actions are tuples ("pick-up", x), ("put-down", x), ("stack", x, y) and
("unstack", x, y).
"""

import math
from collections import deque

import numpy as np

TABLE, HAND = -1, -2
STANDARD = {"pick-up": "pick-up", "put-down": "put-down", "stack": "stack", "unstack": "unstack"}


def holding(on):
    held = [b for b, v in enumerate(on) if v == HAND]
    return held[0] if held else None


def clear(on, b):
    return on[b] != HAND and all(v != b for v in on)


def preconditions(on, action):
    """The preconditions of an action, each as (text, holds) for the state on."""
    kind, x = action[0], action[1]
    empty = holding(on) is None
    if kind == "pick-up":
        return [(f"clear {x}", clear(on, x)), (f"on-table {x}", on[x] == TABLE), ("hand-empty", empty)]
    if kind == "unstack":
        y = action[2]
        return [(f"clear {x}", clear(on, x)), (f"on {x} {y}", on[x] == y), ("hand-empty", empty)]
    if kind == "put-down":
        return [(f"holding {x}", on[x] == HAND)]
    if kind == "stack":
        y = action[2]
        return [(f"holding {x}", on[x] == HAND), (f"clear {y}", x != y and clear(on, y))]
    raise ValueError(f"unknown action {kind}")


def apply(on, action):
    """The state after a, or None if a precondition fails."""
    if not all(ok for _, ok in preconditions(on, action)):
        return None
    new = list(on)
    kind, x = action[0], action[1]
    new[x] = HAND if kind in ("pick-up", "unstack") else (TABLE if kind == "put-down" else action[2])
    return tuple(new)


def satisfied(on, goal):
    return [b for b, v in goal.items() if on[b] != v]


def validate(init, goal, plan):
    """Check a plan the way VAL does: report the first inexecutable action or the unmet goals."""
    on = init
    for i, action in enumerate(plan):
        unmet = [text for text, ok in preconditions(on, action) if not ok]
        if unmet:
            return {"valid": False, "executable": False, "step": i, "unmet": unmet}
        on = apply(on, action)
    missing = satisfied(on, goal)
    if holding(on) is not None:
        missing = [*missing, "hand-empty"]
    return {"valid": not missing, "executable": True, "step": None, "unmet": missing}


def actions(on):
    n = len(on)
    held = holding(on)
    if held is None:
        for x in range(n):
            if clear(on, x):
                yield ("pick-up", x) if on[x] == TABLE else ("unstack", x, on[x])
    else:
        yield ("put-down", held)
        for y in range(n):
            if y != held and clear(on, y):
                yield ("stack", held, y)


def bfs_plan(init, goal):
    """An optimal plan by breadth-first search, ending with the hand empty."""
    parent = {init: None}
    q = deque([init])
    while q:
        on = q.popleft()
        if holding(on) is None and not satisfied(on, goal):
            plan = []
            while parent[on] is not None:
                on, a = parent[on]
                plan.append(a)
            return plan[::-1]
        for a in actions(on):
            nxt = apply(on, a)
            if nxt not in parent:
                parent[nxt] = (on, a)
                q.append(nxt)
    return None


def reachable(start):
    seen = {start}
    q = deque([start])
    while q:
        on = q.popleft()
        for a in actions(on):
            nxt = apply(on, a)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def hand_empty_states(n):
    """a(n) = (2n - 1) a(n - 1) - (n - 1)(n - 2) a(n - 2), the number of hand-empty states of n blocks."""
    a = [1, 1]
    for k in range(2, n + 1):
        a.append((2 * k - 1) * a[k - 1] - (k - 1) * (k - 2) * a[k - 2])
    return a[n]


def count_states(n):
    return hand_empty_states(n) + n * hand_empty_states(n - 1) if n else 1


def plan_length_bound(init, goal_state):
    """2u + 2g: unstack everything, then build every goal tower from the bottom."""
    u = sum(v >= 0 for v in init)
    g = sum(v >= 0 for v in goal_state)
    return 2 * u + 2 * g


def table_to_stack(n):
    """Every block on the table; the goal is one tower 0 on 1 on ... on n - 1."""
    return tuple([TABLE] * n), {b: b + 1 for b in range(n - 1)} | {n - 1: TABLE}


def random_state(n, rng):
    """A random hand-empty state: shuffle the blocks and cut the order into towers."""
    order = rng.permutation(n)
    on = [TABLE] * n
    for i in range(1, n):
        if rng.random() < 0.5:
            on[order[i]] = int(order[i - 1])
    return tuple(on)


def format_plan(plan, vocab=STANDARD, names=None):
    names = names or [f"b{i}" for i in range(max((max(a[1:]) for a in plan), default=0) + 1)]
    return "\n".join(" ".join([vocab[a[0]], *(names[x] for x in a[1:])]) for a in plan)


def parse_plan(text, vocab=STANDARD, names=None):
    """Read a plan written with the action words of vocab; unknown words raise ValueError."""
    back = {v: k for k, v in vocab.items()}
    plan = []
    for line in text.strip().splitlines():
        word, *args = line.split()
        if word not in back:
            raise ValueError(f"unknown action word {word!r}")
        ids = [names.index(x) if names else int(x[1:]) for x in args]
        plan.append((back[word], *ids))
    return plan


def corrupt(plan, n, eps, rng):
    """Replace each action of plan by a random action with probability eps."""
    out = []
    for a in plan:
        if rng.random() < eps:
            kind = ["pick-up", "put-down", "stack", "unstack"][rng.integers(4)]
            x, y = rng.choice(n, 2, replace=False)
            a = (kind, int(x)) if kind in ("pick-up", "put-down") else (kind, int(x), int(y))
        out.append(a)
    return out


def noisy_proposer(init, goal, eps, rng):
    """An approximate plan source: an optimal plan with each action replaced at random with probability eps."""
    plan = bfs_plan(init, goal)
    return lambda: corrupt(plan, len(init), eps, rng)


def sound_verifier(init, goal):
    return lambda plan: validate(init, goal, plan)["valid"]


def noisy_verifier(init, goal, tpr, fpr, rng):
    """A verifier that accepts valid plans with probability tpr and invalid ones with probability fpr."""

    def verify(plan):
        ok = validate(init, goal, plan)["valid"]
        return bool(rng.random() < (tpr if ok else fpr))

    return verify


def generate_and_test(propose, verify, k):
    """Propose up to k plans and return the first one the verifier accepts, with the round number."""
    for i in range(1, k + 1):
        plan = propose()
        if verify(plan):
            return plan, i
    return None, k


def sound_loop_success(p, k):
    return 1 - (1 - p) ** k


def verifier_precision(p, tpr, fpr):
    return p * tpr / (p * tpr + (1 - p) * fpr)


def unsound_loop_correct(p, tpr, fpr, k):
    r = 1 - p * tpr - (1 - p) * fpr
    return verifier_precision(p, tpr, fpr) * (1 - r**k)


def success_rate(results):
    return float(np.mean(results)) if len(results) else math.nan
