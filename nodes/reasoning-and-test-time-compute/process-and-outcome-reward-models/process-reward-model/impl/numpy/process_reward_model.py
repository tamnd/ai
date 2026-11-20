# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/process-reward-model/ (ai://NEW-process-reward-model).
"""Process reward models, in NumPy.

A process reward model (PRM) scores every step of a solution instead of only its final answer. The file has:

* a toy world of step-by-step solutions in which every step is right or wrong, the answer is usually wrong after
  the first wrong step, and a completer can finish any prefix;
* step labels: first-error labels as a human would give them, and soft and hard Monte Carlo labels from rollouts;
* a logistic PRM trained by cross-entropy on step features, and an outcome reward model (ORM) for comparison;
* aggregation of step scores (product, minimum, last) and selection: best-of-N and reward-weighted voting;
* binary search for the first error with rollouts, implicit process rewards from two models' log-probabilities,
  and the F1 of error identification.
"""

import numpy as np

# The toy world.


def sample_solutions(n, steps, p_step, p_lucky, rng):
    """n solutions of `steps` steps. Each step is right with probability p_step until the first error, and every
    later step counts as wrong. The final answer is right if no step is wrong, and otherwise with probability
    p_lucky (a right answer reached with wrong reasoning). Returns step correctness (n, steps) and answers (n,)."""
    ok = rng.random((n, steps)) < p_step
    ok = np.cumprod(ok, axis=1).astype(bool)
    answer = ok[:, -1] | (rng.random(n) < p_lucky)
    return ok, answer


def step_features(ok, noise, rng):
    """One noisy feature per step centred at +1 for right steps and -1 for wrong ones, plus a bias column."""
    z = np.where(ok, 1.0, -1.0) + noise * rng.standard_normal(ok.shape)
    return np.stack([np.ones_like(z), z], axis=-1)


def rollout_value(ok_prefix, remaining, p_step, p_lucky):
    """q_t: the chance that a fresh completion of the prefix reaches the right answer in the toy world."""
    if not ok_prefix:
        return p_lucky
    clean = p_step**remaining
    return clean + (1 - clean) * p_lucky


# Labels.


def first_error_labels(ok):
    """Labels as Lightman et al. collect them: 1 up to the first wrong step, 0 there, and nothing (-1) after."""
    lab = ok.astype(float)
    after = np.cumsum(~ok, axis=1) > 1
    lab[after] = -1.0
    return lab


def mc_labels(q, n_rollouts, rng, kind="soft"):
    """Monte Carlo step labels from n_rollouts completions with success probability q (any shape)."""
    c = rng.binomial(n_rollouts, q)
    if kind == "soft":
        return c / n_rollouts
    if kind == "hard":
        return (c >= 1).astype(float)
    raise ValueError(kind)


# Models.


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic(x, y, lr=0.5, epochs=300):
    """Fit sigmoid(x w) to labels y in [0, 1] by gradient descent on the mean cross-entropy; rows with y < 0 are
    ignored. x is (..., d) and y is (...)."""
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1)
    keep = y >= 0
    x, y = x[keep], y[keep]
    w = np.zeros(x.shape[1])
    for _ in range(epochs):
        p = sigmoid(x @ w)
        w -= lr * x.T @ (p - y) / len(y)
    return w


def bce(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def step_scores(w, feats):
    """PRM probabilities that each step is right, for features of shape (..., steps, d)."""
    return sigmoid(feats @ w)


# Aggregation and selection.


def aggregate(scores, how="prod"):
    """Solution scores from step scores (..., steps): product, minimum or last step."""
    if how == "prod":
        return np.prod(scores, axis=-1)
    if how == "min":
        return np.min(scores, axis=-1)
    if how == "last":
        return scores[..., -1]
    raise ValueError(how)


def best_of_n(solution_scores):
    """Index of the highest-scoring solution along the last axis."""
    return np.argmax(solution_scores, axis=-1)


def weighted_vote(answers, scores):
    """arg max_a sum_i 1(a_i = a) score_i, the reward-weighted vote."""
    totals = {}
    for a, s in zip(answers, scores):
        totals[a] = totals.get(a, 0.0) + float(s)
    return max(totals, key=totals.get)


# Search, implicit rewards and evaluation.


def find_first_error(check, m):
    """Binary search for the first failing prefix length in 1..m, m + 1 if none fails. `check(t)` says whether the
    prefix of t steps is still fine. Returns the step and the number of checks."""
    lo, hi, n = 1, m + 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        n += 1
        if check(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo, n


def rollout_check(ok, p_step, p_lucky, k, rng):
    """A prefix check from k rollouts: the prefix passes if any rollout reaches the right answer."""

    def check(t):
        q = rollout_value(bool(ok[t - 1]), len(ok) - t, p_step, p_lucky)
        return bool((rng.random(k) < q).any())

    return check


def implicit_process_rewards(logp_policy, logp_ref, beta):
    """Per-step rewards beta log pi(y_t | y_<t)/pi_ref(y_t | y_<t) and their partial sums q^t."""
    r = beta * (np.asarray(logp_policy) - np.asarray(logp_ref))
    return r, np.cumsum(r)


def error_f1(pred_first_error, true_first_error):
    """ProcessBench F1: accuracy on erroneous and on correct cases (label -1) and their harmonic mean."""
    pred, true = np.asarray(pred_first_error), np.asarray(true_first_error)
    err, cor = true >= 0, true < 0
    acc_err = float((pred[err] == true[err]).mean()) if err.any() else 0.0
    acc_cor = float((pred[cor] == true[cor]).mean()) if cor.any() else 0.0
    f1 = 0.0 if acc_err + acc_cor == 0 else 2 * acc_err * acc_cor / (acc_err + acc_cor)
    return acc_err, acc_cor, f1


def predict_first_error(scores, threshold=0.5):
    """The first step whose PRM score is below threshold, or -1 if none is."""
    bad = scores < threshold
    return np.where(bad.any(-1), bad.argmax(-1), -1)
