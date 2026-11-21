# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-mathematical-reasoning/ (ai://NEW-language-model-mathematical-reasoning).
"""Templated word problems in the style of GSM-Symbolic, two toy solvers and the statistics of benchmark accuracy.

A problem is a list of clauses that act on one running quantity. A template fixes the kinds of clause; an instance
draws a name, an item and the numbers. A no-op clause mentions a number but changes nothing, as in GSM-NoOp.
The keyword solver turns every sentence with a number into an operation, the failure GSM-NoOp exposes; the noisy
solver gets each step right with a fixed probability. The statistics functions give the spread of accuracy across
re-instantiated test sets, the standard error of a gap between two benchmarks, majority voting and pass@k.
"""

import math
import re
from collections import Counter

import numpy as np

NAMES = ["Sophie", "Liam", "Ava", "Noah", "Mia", "Omar", "Lena", "Ravi"]
ITEMS = ["apples", "stickers", "marbles", "books", "shells", "cards", "kiwis", "stamps"]
TEXT = {
    "start": "{name} has {v} {item}.",
    "add": "Then {name} gets {v} more {item}.",
    "sub": "{name} gives away {v} {item}.",
    "times": "Later {name} ends up with {v} times as many {item}.",
    "noop": "{v} of the {item} are a bit smaller than average.",
}
OPS = ("add", "sub", "times")


def apply(kind, x, v):
    """The running quantity after one clause."""
    return {"start": v, "add": x + v, "sub": x - v, "times": x * v, "noop": x}[kind]


def make_template(n_ops, rng):
    """A start clause followed by n_ops operations."""
    return ["start"] + [OPS[i] for i in rng.integers(0, len(OPS), n_ops)]


def instantiate(template, rng, base=None, names=True, numbers=True):
    """Draw a name, an item and values for a template; keep those of base where told not to vary them.

    Values are redrawn until every intermediate quantity is a positive integer below 1000.
    """
    who = NAMES[rng.integers(len(NAMES))] if names or base is None else base["name"]
    item = ITEMS[rng.integers(len(ITEMS))] if names or base is None else base["item"]
    while True:
        vals = [int(rng.integers(5, 60) if k in ("start", "add", "sub") else rng.integers(2, 4)) for k in template]
        if not numbers and base is not None:
            vals = [v for _, v in base["clauses"]]
        x, ok = 0, True
        for k, v in zip(template, vals, strict=True):
            x = apply(k, x, v)
            ok &= 0 < x < 1000
        if ok:
            return {"name": who, "item": item, "clauses": list(zip(template, vals, strict=True))}


def add_noop(problem, rng):
    """Insert a no-op clause about part of the items after a random clause."""
    out = dict(problem, clauses=list(problem["clauses"]))
    out["clauses"].insert(int(rng.integers(1, len(out["clauses"]) + 1)), ("noop", int(rng.integers(2, 6))))
    return out


def render(problem):
    s = [TEXT[k].format(name=problem["name"], item=problem["item"], v=v) for k, v in problem["clauses"]]
    return " ".join(s) + f" How many {problem['item']} does {problem['name']} have now?"


def answer(problem):
    x = 0
    for k, v in problem["clauses"]:
        x = apply(k, x, v)
    return x


KEYWORDS = [("times", "times"), ("more", "add"), ("away", "sub"), ("smaller", "sub"), ("has", "start")]


def keyword_solve(text):
    """Map each sentence that holds a number to an operation by its first matching keyword."""
    x = 0
    for sent in re.split(r"(?<=\.)\s", text):
        num = re.search(r"\d+", sent)
        if num is None:
            continue
        kind = next((k for w, k in KEYWORDS if w in sent), "add")
        x = apply(kind, x, int(num.group()))
    return x


def noisy_solve(problem, p, rng):
    """Carry out each operation correctly with probability p; a wrong step is off by one to five."""
    x = 0
    for k, v in problem["clauses"]:
        x = apply(k, x, v)
        if k not in ("start", "noop") and rng.random() > p:
            x += int(rng.choice([-1, 1]) * rng.integers(1, 6))
    return x


def chain_success(p, steps):
    """Probability that steps independent steps are all right."""
    return p**steps


def binomial_sd(p, n):
    """Standard deviation of accuracy on n questions each solved with probability p."""
    return math.sqrt(p * (1 - p) / n)


def set_sd(ps):
    """Standard deviation of accuracy on one fresh instance of each template, template i solved with ps[i]."""
    ps = np.asarray(ps, float)
    return math.sqrt(float(np.sum(ps * (1 - ps)))) / len(ps)


def gap_se(p, n1, n2):
    """Standard error of the difference between accuracies on two independent sets of n1 and n2 questions."""
    return math.sqrt(p * (1 - p) / n1 + p * (1 - p) / n2)


def majority_vote(answers, rng):
    """The most common answer, ties broken at random."""
    c = Counter(answers)
    top = max(c.values())
    tied = sorted(a for a, k in c.items() if k == top)
    return tied[int(rng.integers(len(tied)))]


def vote_accuracy(q, w, k):
    """Exact accuracy of a k-sample majority vote.

    Each sample is right with probability q, gives one fixed wrong answer with probability w, and otherwise gives a
    wrong answer seen in no other sample. Ties are broken uniformly at random.
    """
    r, acc = 1 - q - w, 0.0
    for c in range(1, k + 1):
        for s in range(k - c + 1):
            u = k - c - s
            pr = math.comb(k, c) * math.comb(k - c, s) * q**c * w**s * r**u
            top = max([s] + [1] * u)
            if c > top:
                acc += pr
            elif c == top:
                acc += pr / (1 + (s == top) + u * (top == 1))
    return acc


def pass_at_k(n, c, k):
    """Unbiased estimate of the chance that k of n samples, c of them correct, include a correct one."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
