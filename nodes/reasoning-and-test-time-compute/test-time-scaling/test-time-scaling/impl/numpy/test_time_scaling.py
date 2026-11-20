# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/test-time-scaling/ (ai://NEW-test-time-scaling).
"""Test-time scaling in NumPy: coverage, voting, best-of-n and how to spend a sampling budget.

A problem is described by the distribution q of the final answers a model gives to it, with index 0 the correct
answer, so q[0] is the per-sample success rate p. Repeated sampling draws n answers from q; coverage asks whether any
of them is correct, majority voting returns the most frequent one, and best-of-n returns the one a verifier scores
highest. The functions below compute these curves exactly where a formula exists and by simulation otherwise.
"""

import heapq
import math

import numpy as np


def pass_at_k(n, c, k):
    """Unbiased estimate of coverage at k from n samples with c correct: 1 - C(n - c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    # The product form avoids huge binomials: C(n - c, k) / C(n, k) = prod_{i = n - c + 1}^{n} (1 - k / i).
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def coverage_curve(n, correct, ks):
    """Mean unbiased pass@k over problems, for each k in ks; correct[i] is the count for problem i out of n."""
    return np.array([np.mean([pass_at_k(n, c, k) for c in correct]) for k in ks])


def exact_coverage(p, ks):
    """Coverage 1 - (1 - p)^k averaged over problems with per-sample success rates p."""
    p = np.asarray(p, dtype=float)
    return np.array([np.mean(1.0 - (1.0 - p) ** k) for k in ks])


def beta_mixture_coverage(alpha, beta, k):
    """Coverage at k when the per-sample success rate of a problem is Beta(alpha, beta) distributed."""
    log_fail = math.lgamma(alpha + beta) + math.lgamma(beta + k) - math.lgamma(beta) - math.lgamma(alpha + beta + k)
    return 1.0 - math.exp(log_fail)


def fit_exponentiated_power_law(ks, coverage):
    """Least-squares fit of log(-log c) = log(-a) + b log k, the law c = exp(a k^b) of Brown et al. (2024)."""
    ks, coverage = np.asarray(ks, float), np.asarray(coverage, float)
    keep = (coverage > 0) & (coverage < 1)
    b, intercept = np.polyfit(np.log(ks[keep]), np.log(-np.log(coverage[keep])), 1)
    return -math.exp(intercept), b


def majority_vote(answers):
    """The most frequent answer; ties go to the answer seen first."""
    values, first, counts = np.unique(answers, return_index=True, return_counts=True)
    best = np.flatnonzero(counts == counts.max())
    return values[best[np.argmin(first[best])]]


def weighted_vote(answers, scores):
    """The answer with the largest total verifier score."""
    totals = {}
    for a, s in zip(answers, scores):
        totals[a] = totals.get(a, 0.0) + s
    return max(totals, key=totals.get)


def best_of_n(answers, scores):
    """The answer of the single highest-scored sample."""
    return answers[int(np.argmax(scores))]


def vote_accuracy(q, n, trials, rng):
    """Probability that majority voting over n samples from q returns answer 0, estimated by simulation."""
    draws = rng.choice(len(q), size=(trials, n), p=q)
    counts = np.stack([(draws == j).sum(1) for j in range(len(q))], 1)
    top = counts.max(1)
    unique_top = (counts == top[:, None]).sum(1) == 1
    return float(((counts[:, 0] == top) & unique_top).mean())


def vote_limit(qs):
    """Benchmark accuracy of voting as n grows: the fraction of problems whose correct answer is the unique mode."""
    return float(np.mean([q[0] > np.max(q[1:]) for q in qs]))


def coverage_limit(qs):
    """Benchmark coverage as n grows: the fraction of problems the model can solve at all."""
    return float(np.mean([q[0] > 0 for q in qs]))


def best_of_n_fooled(p, eps, n):
    """Best-of-n accuracy when a fraction eps of samples are wrong answers the verifier prefers to correct ones."""
    return (1.0 - eps) ** n - (1.0 - eps - p) ** n


def best_n_fooled(p, eps):
    """The real n that maximizes best_of_n_fooled: log(log b / log a) / log(a / b) with a = 1 - eps, b = a - p."""
    a, b = 1.0 - eps, 1.0 - eps - p
    return math.log(math.log(b) / math.log(a)) / math.log(a / b)


def allocate_budget(p, budget):
    """Samples per problem that maximize expected coverage; greedy on the marginal gain p (1 - p)^k."""
    k = [0] * len(p)
    heap = [(-pi, i) for i, pi in enumerate(p) if pi > 0]
    heapq.heapify(heap)
    for _ in range(budget):
        if not heap:
            break
        _, i = heapq.heappop(heap)
        k[i] += 1
        gain = p[i] * (1.0 - p[i]) ** k[i]
        if gain > 0:
            heapq.heappush(heap, (-gain, i))
    return k


def flops_matched_multiplier(m, d_pretrain, d_inference):
    """Factor by which a model m times smaller may multiply its inference tokens at equal total FLOPs."""
    return m + 3.0 * (d_pretrain / d_inference) * (m - 1.0)


def random_problems(n_problems, n_answers, concentration, rng):
    """Answer distributions for a synthetic benchmark; problem difficulty varies through a Dirichlet draw."""
    return [rng.dirichlet(np.full(n_answers, concentration)) for _ in range(n_problems)]


def scaling_curves(qs, ns, trials, rng, verifier_noise=1.0):
    """Coverage, majority-vote and best-of-n accuracy on a benchmark, as functions of the number of samples.

    The verifier scores a correct sample 1 and a wrong one 0, plus Gaussian noise of scale verifier_noise.
    """
    nmax = max(ns)
    out = {"coverage": np.zeros(len(ns)), "vote": np.zeros(len(ns)), "best_of_n": np.zeros(len(ns))}
    for q in qs:
        draws = rng.choice(len(q), size=(trials, nmax), p=q)
        scores = (draws == 0) + verifier_noise * rng.normal(size=draws.shape)
        for j, n in enumerate(ns):
            d, s = draws[:, :n], scores[:, :n]
            out["coverage"][j] += (d == 0).any(1).mean()
            out["vote"][j] += np.mean([majority_vote(row) == 0 for row in d])
            out["best_of_n"][j] += (d[np.arange(trials), s.argmax(1)] == 0).mean()
    return {name: v / len(qs) for name, v in out.items()}
