# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/model-merging/ (ai://NEW-model-merging).
"""Model merging in NumPy: combining the weights of several models into one without further training.

Every model here is a flat vector of parameters, or a dict of named arrays for the functions that say so; all models
share one architecture, and the methods that use task vectors assume they were fine-tuned from one pre-trained model.
The file has:

* weight averages: the uniform soup, a weighted average, the greedy soup, WiSE-FT interpolation, and stochastic
  weight averaging as a running mean;
* task arithmetic: task vectors, their sum scaled by lambda, negation and analogies;
* Fisher merging: the per-parameter average weighted by diagonal Fisher information, with a fallback where every
  Fisher value is zero, and the diagonal Fisher of a softmax-regression model computed exactly;
* TIES-merging: trim each task vector to its top-k fraction by magnitude, elect a sign per parameter, and average
  the entries that agree with it;
* DARE: drop each delta parameter with probability p and rescale the rest by 1 / (1 - p);
* the loss barrier on the straight line between two models.
"""

import numpy as np

# Weight averages.


def uniform_soup(models):
    """The average of the models' parameters."""
    return np.mean(np.asarray(models, dtype=float), axis=0)


def weighted_average(models, weights):
    """sum_i w_i theta_i / sum_i w_i for non-negative weights, not all zero."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    return np.tensordot(w / w.sum(), np.asarray(models, dtype=float), axes=1)


def greedy_soup(models, val_score):
    """Wortsman et al.'s greedy soup.

    Sort the models by val_score (higher is better), start the soup with the best, and add each next model if the
    average of the soup's ingredients with it scores at least as well. Returns (soup, indices of the ingredients)."""
    order = sorted(range(len(models)), key=lambda i: val_score(models[i]), reverse=True)
    chosen = [order[0]]
    best = val_score(models[order[0]])
    for i in order[1:]:
        trial = uniform_soup([models[j] for j in chosen + [i]])
        s = val_score(trial)
        if s >= best:
            chosen.append(i)
            best = s
    return uniform_soup([models[j] for j in chosen]), chosen


def interpolate(theta_0, theta_1, alpha):
    """WiSE-FT's weight-space ensemble (1 - alpha) theta_0 + alpha theta_1; alpha = 0 is theta_0, 1 is theta_1."""
    return (1 - alpha) * np.asarray(theta_0, dtype=float) + alpha * np.asarray(theta_1, dtype=float)


class RunningAverage:
    """The SWA update w_swa <- (w_swa n + w) / (n + 1): after n captures it holds their arithmetic mean."""

    def __init__(self):
        self.n = 0
        self.value = None

    def update(self, w):
        w = np.asarray(w, dtype=float)
        self.value = w.copy() if self.n == 0 else (self.value * self.n + w) / (self.n + 1)
        self.n += 1
        return self.value


# Task arithmetic.


def task_vector(theta_ft, theta_pre):
    """tau = theta_ft - theta_pre."""
    return np.asarray(theta_ft, dtype=float) - np.asarray(theta_pre, dtype=float)


def apply_task_vectors(theta_pre, taus, lam):
    """theta_pre + lam * sum_t tau_t. One task vector with lam = 1 gives back the fine-tuned model; a negated task
    vector (-tau) forgets a task; tau_C + (tau_B - tau_A) is an analogy."""
    return np.asarray(theta_pre, dtype=float) + lam * np.sum(np.asarray(taus, dtype=float), axis=0)


# Fisher merging.


def fisher_merge(models, fishers, lambdas=None, target=0, tol=0.0):
    """Matena and Raffel's merge: per parameter, sum_i lambda_i F_i theta_i / sum_i lambda_i F_i.

    fishers holds a diagonal Fisher (non-negative, the shape of a model) per model. Where the denominator is at most
    tol, the parameter takes the value it has in models[target]."""
    th, F = np.asarray(models, dtype=float), np.asarray(fishers, dtype=float)
    if np.any(F < 0):
        raise ValueError("Fisher values are non-negative")
    lam = np.full(len(th), 1 / len(th)) if lambdas is None else np.asarray(lambdas, dtype=float)
    num = np.tensordot(lam, F * th, axes=1)
    den = np.tensordot(lam, F, axes=1)
    out = th[target].copy()
    ok = den > tol
    out[ok] = num[ok] / den[ok]
    return out


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_regression_fisher(W, X):
    """Exact diagonal Fisher of p(y | x) = softmax(W x), averaged over the rows of X, with the expectation over y
    taken under the model: E_x sum_y p(y|x) (d log p(y|x) / dW)^2. The gradient of log p(y|x) in W is
    (e_y - p) x^T."""
    W, X = np.asarray(W, dtype=float), np.asarray(X, dtype=float)
    P = softmax(X @ W.T)  # (N, K)
    k = W.shape[0]
    F = np.zeros_like(W)
    for p, x in zip(P, X, strict=True):
        for y in range(k):
            g = (np.eye(k)[y] - p)[:, None] * x[None, :]
            F += p[y] * g**2
    return F / len(X)


# TIES-merging.


def trim(tau, k):
    """Keep the fraction k of the entries of tau with the largest magnitude and set the rest to 0. Ties at the
    threshold are broken by position, so exactly round(k * size) entries survive."""
    tau = np.asarray(tau, dtype=float)
    keep = round(k * tau.size)
    out = np.zeros_like(tau)
    if keep > 0:
        idx = np.argsort(-np.abs(tau), kind="stable")[:keep]
        out[idx] = tau[idx]
    return out


def elect_sign(trimmed):
    """gamma_m = sgn(sum_t tau_hat_t): the sign with the larger total magnitude (0 where the sum is 0)."""
    return np.sign(np.sum(np.asarray(trimmed, dtype=float), axis=0))


def disjoint_mean(trimmed, sign):
    """Per parameter, the mean of the entries whose sign equals the elected sign; zeros never count, and a
    parameter with no such entry gets 0."""
    T = np.asarray(trimmed, dtype=float)
    agree = (np.sign(T) == sign) & (T != 0)
    count = agree.sum(axis=0)
    total = np.where(agree, T, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def ties_merge(theta_pre, taus, k=0.2, lam=1.0):
    """Trim, elect and disjoint-merge the task vectors; returns (theta_pre + lam * tau_m, tau_m)."""
    trimmed = np.array([trim(t, k) for t in taus])
    tau_m = disjoint_mean(trimmed, elect_sign(trimmed))
    return np.asarray(theta_pre, dtype=float) + lam * tau_m, tau_m


# DARE.


def dare(delta, p, rng):
    """Drop each entry of delta with probability p and multiply the survivors by 1 / (1 - p). The result equals
    delta in expectation."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    delta = np.asarray(delta, dtype=float)
    keep = rng.random(delta.shape) >= p
    return np.where(keep, delta / (1 - p), 0.0)


def dare_merge(theta_pre, finetuned, p, lam, rng):
    """DARE on each model's delta, then task arithmetic: theta_pre + lam * sum_k dare(theta_k - theta_pre)."""
    deltas = [dare(task_vector(t, theta_pre), p, rng) for t in finetuned]
    return apply_task_vectors(theta_pre, deltas, lam)


# Dicts of named arrays, as models are stored in practice.


def merge_state(states, fn):
    """Apply a merge fn (taking a list of flat vectors) to every name shared by the dicts in states; each array is
    flattened for fn and reshaped back."""
    out = {}
    for name in states[0]:
        shape = np.shape(states[0][name])
        out[name] = np.reshape(fn([np.ravel(s[name]) for s in states]), shape)
    return out


# The loss barrier.


def loss_barrier(loss_fn, theta_a, theta_b, points=21):
    """max over lambda in [0, 1] of L((1 - lambda) theta_a + lambda theta_b) - (L(theta_a) + L(theta_b)) / 2,
    on a grid; it is at least |L(theta_a) - L(theta_b)| / 2, so it is never negative."""
    ls = [loss_fn(interpolate(theta_a, theta_b, a)) for a in np.linspace(0, 1, points)]
    return max(ls) - (ls[0] + ls[-1]) / 2
