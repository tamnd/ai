# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/self-consistency/ (ai://NEW-self-consistency).
"""Self-consistency and its relatives, in NumPy.

Self-consistency samples several reasoning paths for one question, reads the final answer off each, and returns
the answer that most paths agree on. The file has:

* voting: the majority vote, the weighted vote, the length-normalized sequence probability used as a weight, and a
  verifier-weighted vote that lets only the top-ranked samples take part;
* exact accuracy of a vote: the binomial formula for two answers, an exact multinomial sum for a few answers, and
  the Hoeffding bound;
* stopping rules: the Beta criterion of Adaptive-Consistency and the window rule of early-stopping self-consistency;
* coverage: 1 - (1 - p)^k and the unbiased pass@k estimator;
* a sampler that runs all of these against any answer distribution.
"""

import math

import numpy as np

# Voting.


def majority_vote(answers):
    """The most common answer, ties broken by first appearance, and the counts of every answer."""
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        raise ValueError("no answers to vote on")
    best = max(counts.values())
    winner = next(a for a in counts if counts[a] == best)
    return winner, counts


def weighted_vote(answers, weights):
    """arg max_a sum_i w_i 1(a_i = a), ties broken by first appearance, and the total weight of every answer."""
    if len(answers) != len(weights):
        raise ValueError("one weight per answer")
    totals = {}
    for a, w in zip(answers, weights):
        if w < 0:
            raise ValueError("weights must be non-negative")
        totals[a] = totals.get(a, 0.0) + float(w)
    if not totals:
        raise ValueError("no answers to vote on")
    best = max(totals.values())
    winner = next(a for a in totals if totals[a] == best)
    return winner, totals


def sequence_weight(token_logprobs, normalize=True):
    """exp of the mean token log-probability (normalized) or of their sum (unnormalized)."""
    lp = np.asarray(token_logprobs, dtype=float)
    if lp.size == 0:
        raise ValueError("empty sequence")
    return float(np.exp(lp.mean() if normalize else lp.sum()))


def consistency(answers):
    """The share of answers that agree with the vote, Wang et al.'s confidence signal."""
    winner, counts = majority_vote(answers)
    return counts[winner] / len(answers)


def verifier_vote(answers, scores, top):
    """Majority vote among the `top` answers with the highest verifier scores (Cobbe et al.)."""
    if not 1 <= top <= len(answers):
        raise ValueError("top must be between 1 and the number of answers")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:top]
    return majority_vote([answers[i] for i in order])[0]


# Exact accuracy.


def binary_vote_accuracy(p, m):
    """P(majority of m independent samples is correct) with two answers; m odd so there are no ties."""
    if m < 1 or m % 2 == 0:
        raise ValueError("m must be a positive odd number")
    return sum(math.comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m // 2 + 1, m + 1))


def plurality_accuracy(probs, m, correct=0):
    """Exact P(correct answer strictly beats every other count) for m samples from `probs`.

    Enumerates every count vector, so it is meant for a handful of answers and modest m. Ties count as errors.
    """
    probs = np.asarray(probs, dtype=float)
    k = len(probs)
    total = 0.0

    def rec(i, left, counts):
        nonlocal total
        if i == k - 1:
            counts = counts + [left]
            others = [c for j, c in enumerate(counts) if j != correct]
            if counts[correct] > max(others, default=-1):
                coef = math.factorial(m)
                for c in counts:
                    coef //= math.factorial(c)
                total += coef * float(np.prod(probs ** np.array(counts)))
            return
        for c in range(left + 1):
            rec(i + 1, left - c, counts + [c])

    rec(0, m, [])
    return total


def hoeffding_bound(delta, m, n_answers):
    """(|A| - 1) exp(-m delta^2 / 2): a bound on P(vote wrong) when the correct answer leads every other by delta."""
    if delta <= 0:
        raise ValueError("the bound needs a positive margin")
    return min(1.0, (n_answers - 1) * math.exp(-m * delta**2 / 2))


def mixture_accuracy(weights, ps, m):
    """Average binary vote accuracy over question types with shares `weights` and per-sample accuracies `ps`."""
    return sum(w * binary_vote_accuracy(p, m) for w, p in zip(weights, ps))


# Stopping rules.


def beta_confidence(v1, v2):
    """P(q > 1/2) for q ~ Beta(v1 + 1, v2 + 1), computed exactly through the binomial identity."""
    n = v1 + v2 + 1
    tail = sum(math.comb(n, j) for j in range(v1 + 1, n + 1))
    return 1.0 - tail / 2**n


def beta_should_stop(counts, threshold=0.95):
    """Adaptive-Consistency's stopping rule on the two largest answer counts."""
    top = sorted(counts.values(), reverse=True) + [0]
    return beta_confidence(top[0], top[1]) > threshold


def window_unanimity(probs, w):
    """P(a window of w samples all agree) and P(they all agree on an answer other than the mode)."""
    probs = np.asarray(probs, dtype=float)
    pw = probs**w
    mode = int(np.argmax(probs))
    return float(pw.sum()), float(pw.sum() - pw[mode])


def window_bound(w):
    """2^-(w - 1), the bound on stopping on a wrong answer in one window when the mode is unique."""
    return 2.0 ** -(w - 1)


# Coverage.


def coverage(p, k):
    """1 - (1 - p)^k, the chance that some sample among k is correct."""
    return 1.0 - (1.0 - p) ** k


def pass_at_k(n, c, k):
    """Unbiased estimate of coverage at k from n samples of which c are correct: 1 - C(n - c, k) / C(n, k)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# Sampling.


def sample_answers(probs, m, rng):
    """m independent answers (indices) drawn from the answer distribution `probs`."""
    return rng.choice(len(probs), size=m, p=probs).tolist()


def adaptive_consistency(probs, rng, threshold=0.95, max_samples=40):
    """Sample one answer at a time until the Beta rule stops; return the vote and the number of samples."""
    answers = []
    while len(answers) < max_samples:
        answers.append(int(rng.choice(len(probs), p=probs)))
        if beta_should_stop(majority_vote(answers)[1], threshold):
            break
    return majority_vote(answers)[0], len(answers)


def early_stopping_sc(probs, rng, w=5, max_samples=40):
    """Sample windows of w answers; stop at the first unanimous window or at max_samples. Vote over all samples."""
    answers = []
    while len(answers) < max_samples:
        win = rng.choice(len(probs), size=min(w, max_samples - len(answers)), p=probs).tolist()
        answers.extend(win)
        if len(win) == w and len(set(win)) == 1:
            break
    return majority_vote(answers)[0], len(answers)
