# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-prompting/ (ai://WFY2QB).
"""Self-consistency over chain-of-thought samples, in pure Python and in NumPy.

The model is not here: the functions take the text or the parsed answers of m sampled chains of thought for one
question. extract_answer reads the final answer the way Kojima et al. and Wang et al. describe, the first number after
an answer trigger. majority_vote is the vote of Wang et al., argmax_a sum_i 1(a_i = a), written with the standard
library; majority_vote_np is the same vote with numpy.unique. Both break ties in favour of the tied answer sampled
first, and skip samples whose answer could not be parsed (None). weighted_vote is the weighted sum and the weighted
average of Table 1 of Wang et al., with path_weight the unnormalized or length-normalized probability of Equation 1.
vote_accuracy is the exact accuracy of the vote under independent samples, from derivations/majority-vote.md, and
simulate_vote_accuracy checks it by drawing the samples.
"""

import math
import re
from fractions import Fraction

import numpy as np

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text, trigger="The answer is"):
    """The first number after the last occurrence of trigger, or after the start if trigger is absent.

    Commas inside numbers are removed and a trailing ".0" is dropped, so "1,000", "1000" and "1000.0" vote together.
    Returns a string, or None when there is no number.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start = text.rfind(trigger) if trigger else -1
    tail = text[start + len(trigger) :] if start >= 0 else text
    m = _NUMBER.search(tail)
    if m is None:
        return None
    s = m.group(0).replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _parsed(answers):
    answers = list(answers)
    if not answers:
        raise ValueError("need at least one sampled answer")
    return answers


def vote_counts(answers):
    """The number of votes of each parsed answer, in the order in which the answers were first sampled."""
    counts = {}
    for a in _parsed(answers):
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    return counts


def majority_vote(answers):
    """The answer with the most votes, ties going to the answer sampled first; None if no answer parsed.

    Pure Python: a dict keeps the order of first appearance, and max returns the first of the maxima.
    """
    counts = vote_counts(answers)
    if not counts:
        return None
    return max(counts, key=counts.get)


def majority_vote_np(answers):
    """The same vote as majority_vote, with numpy.unique; the answers must be hashable and mutually comparable."""
    answers = _parsed(answers)
    kept = [a for a in answers if a is not None]
    if not kept:
        return None
    arr = np.asarray(kept, dtype=object)
    values, first, counts = np.unique(arr, return_index=True, return_counts=True)
    best = counts == counts.max()
    i = np.flatnonzero(best)[np.argmin(first[best])]
    return values[i]


def path_weight(token_logprobs, normalize=True):
    """The weight of one sampled path from the log-probabilities of its K tokens.

    normalize=True gives exp(mean log p), Equation 1 of Wang et al.; normalize=False gives exp(sum log p), the
    unnormalized probability of the whole generation.
    """
    lp = np.asarray(token_logprobs, dtype=float)
    if lp.ndim != 1 or lp.size == 0:
        raise ValueError("token_logprobs must be a non-empty 1-d sequence")
    if np.any(lp > 0):
        raise ValueError("log-probabilities must be <= 0")
    return float(np.exp(lp.mean() if normalize else lp.sum()))


def weighted_vote(answers, weights, average=False):
    """The answer with the largest summed weight, or with the largest mean weight when average=True.

    The weighted sum with length-normalized weights is the normalized weighted sum of Wang et al.; the mean is their
    weighted average, which favours an answer reached by one confident path. Ties go to the answer sampled first.
    """
    answers = _parsed(answers)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(answers),):
        raise ValueError("need one weight per answer")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    total, count = {}, {}
    for a, x in zip(answers, w, strict=True):
        if a is None:
            continue
        total[a] = total.get(a, 0.0) + float(x)
        count[a] = count.get(a, 0) + 1
    if not total:
        return None
    score = {a: total[a] / count[a] for a in total} if average else total
    return max(score, key=score.get)


def vote_accuracy(q, m, correct=0):
    """Exact probability that the vote of m independent samples returns answer `correct`.

    q gives the probability of each answer of a finite set for one sample; ties are broken uniformly at random, which
    for independent samples is the same as the first-seen rule of majority_vote. With Fraction inputs the result is
    an exact Fraction. The sum runs over all count vectors, so keep len(q) and m small.
    """
    q = list(q)
    if len(q) < 2:
        raise ValueError("need at least two answers")
    if any(x < 0 for x in q) or abs(float(sum(q)) - 1.0) > 1e-12:
        raise ValueError("q must be a probability vector")
    if not 0 <= correct < len(q):
        raise ValueError("correct must index q")
    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    exact = all(isinstance(x, (int, Fraction)) for x in q)
    total = Fraction(0) if exact else 0.0
    k = len(q)

    def rec(i, left, n):
        nonlocal total
        if i == k - 1:
            n = n + [left]
            top = max(n)
            if n[correct] != top:
                return
            ties = sum(1 for x in n if x == top)
            coef = math.factorial(m)
            for x in n:
                coef //= math.factorial(x)
            pr = Fraction(coef) if exact else float(coef)
            for qa, x in zip(q, n, strict=True):
                pr *= qa**x
            total += pr / ties
            return
        for x in range(left + 1):
            rec(i + 1, left - x, n + [x])

    rec(0, m, [])
    return total


def binary_vote_accuracy(p, m):
    """Exact accuracy of the vote of m samples between the correct answer (probability p) and one wrong answer."""
    return vote_accuracy([p, 1 - p], m)


def simulate_vote_accuracy(q, m, trials, rng, correct=0):
    """Monte Carlo estimate of vote_accuracy: draw trials x m answers and apply majority_vote_np's tie rule."""
    q = np.asarray(q, dtype=float)
    draws = rng.choice(len(q), size=(trials, m), p=q)
    counts = np.stack([(draws == a).sum(axis=1) for a in range(len(q))], axis=1)
    top = counts.max(axis=1, keepdims=True)
    is_top = counts == top
    # The winner is the tied answer whose first appearance comes first.
    first = np.full((trials, len(q)), m)
    for a in range(len(q)):
        hit = draws == a
        first[:, a] = np.where(hit.any(axis=1), hit.argmax(axis=1), m)
    first = np.where(is_top, first, m + 1)
    winner = first.argmin(axis=1)
    return float(np.mean(winner == correct))
