# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-planning/ (ai://NEW-language-model-planning).
"""A learned Blocksworld policy as an approximate plan source, alone and inside a generate-test loop.

The policy is a small network trained by imitation on optimal plans for problems of a few blocks. It plays the role
the language model plays in the planning studies: it proposes plans that are often right on problems like its
training data and degrade on larger ones, and a sound validator decides which proposals to keep.

There are N block slots; a problem uses n of them. A state is a list `on` with one entry per slot: None if the slot
is unused, TABLE, HAND, or the slot of the block below. Actions are 0..N-1 (pick up or unstack block x), N (put the
held block on the table) and N + 1 + y (stack the held block on y).
"""

import itertools
import random
from collections import deque

import torch
from torch import nn

N = 6
TABLE, HAND = -1, -2
A = 2 * N + 1


def held(on):
    return next((b for b, v in enumerate(on) if v == HAND), None)


def clear(on, b):
    return on[b] not in (None, HAND) and all(v != b for v in on)


def step(on, a):
    """The state after action a, or None if a is not executable."""
    h = held(on)
    new = list(on)
    if a < N:
        if h is not None or not clear(on, a):
            return None
        new[a] = HAND
    elif a == N:
        if h is None:
            return None
        new[h] = TABLE
    else:
        y = a - N - 1
        if h is None or y == h or not clear(on, y):
            return None
        new[h] = y
    return new


def done(on, goal):
    return held(on) is None and all(on[b] == v for b, v in enumerate(goal) if v is not None)


def optimal_plan(on, goal):
    start = tuple(on)
    parent = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if done(s, goal):
            plan = []
            while parent[s] is not None:
                s, a = parent[s]
                plan.append(a)
            return plan[::-1]
        for a in range(A):
            t = step(list(s), a)
            if t is not None and tuple(t) not in parent:
                parent[tuple(t)] = (s, a)
                q.append(tuple(t))
    return None


def random_problem(n, rng):
    """n blocks in random slots, a random start and a random full goal."""
    slots = rng.sample(range(N), n)

    def arrangement():
        order = rng.sample(slots, n)
        on = [None] * N
        on[order[0]] = TABLE
        for prev, b in itertools.pairwise(order):
            on[b] = prev if rng.random() < 0.5 else TABLE
        return on

    return arrangement(), arrangement()


def encode(on, goal):
    x = torch.zeros(N, N + 2 + N + 1)
    for b in range(N):
        if on[b] is not None:
            x[b, {TABLE: 0, HAND: 1}.get(on[b], on[b] + 2)] = 1
        if goal[b] is not None:
            x[b, N + 2 + (0 if goal[b] == TABLE else goal[b] + 1)] = 1
    return x.flatten()


class Policy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N * (2 * N + 3), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, A)
        )

    def forward(self, x):
        return self.net(x)


def make_data(sizes, count, rng):
    xs, ys = [], []
    for _ in range(count):
        on, goal = random_problem(rng.choice(sizes), rng)
        for a in optimal_plan(on, goal):
            xs.append(encode(on, goal))
            ys.append(a)
            on = step(on, a)
    return torch.stack(xs), torch.tensor(ys)


def train(sizes=(3, 4), problems=3000, epochs=30, seed=0):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    x, y = make_data(list(sizes), problems, rng)
    model = Policy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            idx = perm[i : i + 256]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def propose(model, on, goal, temperature=0.0, gen=None):
    """Roll the policy out into a plan of at most 4(n - 1) + 4 actions, without checking it."""
    n = sum(v is not None for v in on)
    plan, state = [], list(on)
    for _ in range(4 * (n - 1) + 4):
        if done(state, goal):
            break
        logits = model(encode(state, goal))
        if temperature == 0:
            a = int(logits.argmax())
        else:
            a = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen))
        plan.append(a)
        state = step(state, a)
        if state is None:
            break
    return plan


def valid(on, goal, plan):
    state = list(on)
    for a in plan:
        state = step(state, a)
        if state is None:
            return False
    return done(state, goal)


def generate_and_test(model, on, goal, k, temperature=1.0, gen=None):
    """Sample up to k plans and return the first the validator accepts, or None."""
    for _ in range(k):
        plan = propose(model, on, goal, temperature, gen)
        if valid(on, goal, plan):
            return plan
    return None


def success(model, n, count, k=1, temperature=0.0, seed=1):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    wins = 0
    for _ in range(count):
        on, goal = random_problem(n, rng)
        if k == 1 and temperature == 0:
            wins += valid(on, goal, propose(model, on, goal))
        else:
            wins += generate_and_test(model, on, goal, k, temperature, gen) is not None
    return wins / count
