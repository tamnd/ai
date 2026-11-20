# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-model/ (ai://NEW-reward-model).
"""The losses and the arithmetic of a learned reward model in NumPy.

A reward model maps a prompt and a response to one number, and is fitted to human comparisons under the
Bradley-Terry model P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l)). The network that produces the numbers is not
part of this file, except for a linear reward model on features that the tests fit. The file has:

* the pairwise loss -log sigma(r_w - r_l), with Llama 2's margin and Christiano et al.'s label noise, and its gradient;
* InstructGPT's ranked loss over the K(K - 1)/2 pairs of K ranked responses, and its gradient;
* the normalizations: shifting so a reference set scores 0, whitening, and Llama 2's safety-helpfulness combination;
* ensemble combinations: mean, worst case, and mean minus lambda times variance;
* best-of-n selection and its KL divergence log n - (n - 1)/n;
* evaluation: pairwise accuracy, a calibration table, and a weighted section score in the style of RewardBench;
* a check of Skalse et al.'s hackability on deterministic policies that choose subsets of rewarded items.

Rewards are arrays of floats; a comparison is a pair of arrays (r_w, r_l) of the same shape, the first preferred.
"""

import itertools
import math

import numpy as np

# The sigmoid and its logarithm, stable for large |t|.


def sigmoid(t):
    t = np.asarray(t, dtype=float)
    return np.exp(-np.logaddexp(0.0, -t))


def log_sigmoid(t):
    """log sigma(t) = -log(1 + exp(-t)), computed without overflow."""
    return -np.logaddexp(0.0, -np.asarray(t, dtype=float))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


# Pairwise losses.


def preference_prob(r_w, r_l, eps=0.0):
    """Modelled probability that y_w is preferred: eps / 2 + (1 - eps) * sigma(r_w - r_l)."""
    if not 0.0 <= eps < 1.0:
        raise ValueError("eps must be in [0, 1)")
    return eps / 2 + (1 - eps) * sigmoid(np.asarray(r_w) - np.asarray(r_l))


def pairwise_losses(r_w, r_l, margin=0.0, eps=0.0):
    """Per-comparison negative log-likelihood.

    margin (a scalar or an array like r_w) shifts the difference: -log sigma(r_w - r_l - m). eps mixes in a uniformly
    random answer: -log(eps / 2 + (1 - eps) sigma(r_w - r_l - m)). With both 0 this is -log sigma(r_w - r_l)."""
    d = np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float) - margin
    if eps == 0.0:
        return -log_sigmoid(d)
    return -np.log(preference_prob(d, 0.0, eps))


def pairwise_loss(r_w, r_l, margin=0.0, eps=0.0):
    """Mean of pairwise_losses."""
    return float(np.mean(pairwise_losses(r_w, r_l, margin, eps)))


def pairwise_loss_grad(r_w, r_l, margin=0.0, eps=0.0):
    """Gradient of pairwise_loss with respect to r_w and r_l; the two are negatives of each other."""
    d = np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float) - margin
    s = sigmoid(d)
    p = eps / 2 + (1 - eps) * s
    g = -(1 - eps) * s * (1 - s) / p / d.size  # d(loss)/d(d), which is -sigma(-d) / size when eps = 0
    return g, -g


# Ranked responses (InstructGPT).


def ranked_pairs(k):
    """The pairs (i, j), i < j, of k responses listed best first; i is preferred to j."""
    return list(itertools.combinations(range(k), 2))


def ranked_loss(rewards):
    """Mean of -log sigma(r_i - r_j) over the k(k - 1)/2 pairs, for rewards of responses listed best first."""
    r = np.asarray(rewards, dtype=float)
    if r.size < 2:
        raise ValueError("a ranking needs at least two responses")
    i, j = np.array(ranked_pairs(r.size)).T
    return float(np.mean(-log_sigmoid(r[i] - r[j])))


def ranked_loss_grad(rewards):
    r = np.asarray(rewards, dtype=float)
    i, j = np.array(ranked_pairs(r.size)).T
    w = sigmoid(r[j] - r[i]) / len(i)
    g = np.zeros_like(r)
    np.add.at(g, i, -w)
    np.add.at(g, j, w)
    return g


def batch_ranked_loss(rankings):
    """InstructGPT's batch loss: each prompt's ranked loss (already divided by its number of pairs), averaged over
    prompts, so that every prompt counts once however many responses it has."""
    return float(np.mean([ranked_loss(r) for r in rankings]))


def comparisons_and_passes(k):
    """For k ranked responses: number of comparisons, forward passes when scored together, and when pairs are
    separate examples."""
    c = math.comb(k, 2)
    return c, k, 2 * c


# Llama 2's margin.

LLAMA2_MARGINS = {
    "small": {"significantly better": 1.0, "better": 2 / 3, "slightly better": 1 / 3, "negligibly better": 0.0},
    "large": {"significantly better": 3.0, "better": 2.0, "slightly better": 1.0, "negligibly better": 0.0},
}


def llama2_margin(ratings, variant="small"):
    """The margin for each preference rating; "unsure" is treated like "negligibly better"."""
    table = LLAMA2_MARGINS[variant]
    return np.array([table["negligibly better" if r == "unsure" else r] for r in ratings])


# Normalization and combination.


def shift_to_reference(rewards, reference_rewards):
    """Subtract a constant so that the reference set (demonstrations, reference summaries, or the initial policy's
    samples) has mean reward 0. Comparisons do not determine this constant."""
    return np.asarray(rewards, dtype=float) - float(np.mean(reference_rewards))


def whiten(rewards, eps=1e-8):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def llama2_combined_reward(safety, helpfulness, unsafe_prompt, threshold=0.15):
    """Llama 2's RL reward before whitening: the safety score where the prompt is tagged unsafe or the safety score is
    below the threshold, the helpfulness score elsewhere. Scores are sigmoid outputs in (0, 1)."""
    safety, helpfulness = np.asarray(safety, dtype=float), np.asarray(helpfulness, dtype=float)
    use_safety = np.asarray(unsafe_prompt, dtype=bool) | (safety < threshold)
    return np.where(use_safety, safety, helpfulness)


def llama2_rl_reward(safety, helpfulness, unsafe_prompt, threshold=0.15):
    """The combined score mapped back through the logit, then whitened."""
    return whiten(logit(llama2_combined_reward(safety, helpfulness, unsafe_prompt, threshold)))


# Ensembles.


def ensemble(member_rewards, method="mean", lam=0.0):
    """Combine rewards of shape (k, ...) from k members: "mean", "min" (worst case), or "uwo" (mean - lam * var,
    with the population variance across members)."""
    r = np.asarray(member_rewards, dtype=float)
    if method == "mean":
        return r.mean(axis=0)
    if method == "min":
        return r.min(axis=0)
    if method == "uwo":
        if lam < 0:
            raise ValueError("lam must be non-negative")
        return r.mean(axis=0) - lam * r.var(axis=0)
    raise ValueError(f"unknown method {method!r}")


# Best-of-n.


def best_of_n(rewards):
    """Index of the best of the n candidates along the last axis (the first one on ties)."""
    return np.argmax(np.asarray(rewards), axis=-1)


def bon_kl(n):
    """KL(best-of-n || base) in nats when ties have probability zero."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return math.log(n) - (n - 1) / n


def bon_kl_discrete(probs, n):
    """Exact KL(best-of-n || base) for a base policy on finitely many outputs with distinct rewards, the probabilities
    listed in increasing order of reward."""
    q = np.asarray(probs, dtype=float)
    cdf = np.cumsum(q)
    below = np.concatenate([[0.0], cdf[:-1]])
    p = cdf**n - below**n
    keep = p > 0
    return float(np.sum(p[keep] * np.log(p[keep] / q[keep])))


# Evaluation.


def pairwise_accuracy(r_w, r_l):
    """Fraction of comparisons in which the preferred response scores strictly higher."""
    return float(np.mean(np.asarray(r_w) > np.asarray(r_l)))


def calibration_table(r_w, r_l, bins=5):
    """For bins of |r_w - r_l|: mean predicted probability sigma(|diff|) and observed frequency with which the higher
    score was the preferred response. Returns a list of (predicted, observed, count)."""
    d = np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float)
    a = np.abs(d)
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.any():
            rows.append((float(sigmoid(a[m]).mean()), float(np.mean(d[m] > 0)), int(m.sum())))
    return rows


def section_score(section_accuracies, weights=None):
    """Weighted average of section accuracies; by default every section has weight 1 except "prior sets" with 0.5."""
    if weights is None:
        weights = {s: 0.5 if s == "prior sets" else 1.0 for s in section_accuracies}
    total = sum(weights[s] for s in section_accuracies)
    return sum(weights[s] * a for s, a in section_accuracies.items()) / total


# A linear reward model on features.


def fit_linear_reward(x_w, x_l, steps=500, lr=0.5, l2=0.0):
    """Fit r(x) = x @ theta by gradient descent on the mean Bradley-Terry loss plus (l2 / 2) |theta|^2. The rows of
    x_w and x_l are the features of the preferred and rejected response of each comparison."""
    x_w, x_l = np.asarray(x_w, dtype=float), np.asarray(x_l, dtype=float)
    diff = x_w - x_l
    theta = np.zeros(diff.shape[1])
    for _ in range(steps):
        g = -(sigmoid(-(diff @ theta)) @ diff) / len(diff) + l2 * theta
        theta -= lr * g
    return theta


# Hackability on subset policies (Skalse et al.'s cleaning robot).


def hackable(true_reward, proxy_reward):
    """Whether some deterministic policies S, S' (subsets of items, return = sum of their rewards) have a higher proxy
    return for S and a lower true return. Returns a witness (S, S') or None."""
    true_reward, proxy_reward = np.asarray(true_reward, dtype=float), np.asarray(proxy_reward, dtype=float)
    subsets = list(itertools.product([0, 1], repeat=len(true_reward)))
    for s, t in itertools.product(subsets, repeat=2):
        if np.dot(s, proxy_reward) > np.dot(t, proxy_reward) and np.dot(s, true_reward) < np.dot(t, true_reward):
            return s, t
    return None
