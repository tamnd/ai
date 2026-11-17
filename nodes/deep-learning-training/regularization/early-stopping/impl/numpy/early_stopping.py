# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://NEW-early-stopping).
"""Early stopping, the NumPy reference.

Two parts. The first is least squares from w = 0, where gradient descent stopped after t steps is a spectral filter:
gd_filter, flow_filter and ridge_filter are the shrinkage of one eigen-direction of X^T X / n (eq-gd-filter,
eq-flow-filter, eq-ridge-filter), and gradient_descent, spectral_fit and ridge compute the weights. The second is the
stopping rules: EarlyStopping is the patience rule with restore-best (eq-patience), and generalization_loss,
training_progress, gl_stop, pq_stop and up_stop are the criteria GL, PQ and UP of Prechelt (eq-gl, eq-progress).
"""

import copy
import math
from typing import Any

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=np.float64)


# eq: eq-gd-filter
def gd_filter(s, eta, t):
    """The filter of t steps of gradient descent with step eta on the eigenvalue s: 1 - (1 - eta s)^t."""
    return 1.0 - (1.0 - eta * _arr(s)) ** t


# eq: eq-flow-filter
def flow_filter(s, tau):
    """The filter of gradient flow at time tau on the eigenvalue s: 1 - exp(-s tau)."""
    return -np.expm1(-_arr(s) * tau)


# eq: eq-ridge-filter
def ridge_filter(s, lam):
    """The filter of ridge regression with penalty lam on the eigenvalue s: s / (s + lam)."""
    s = _arr(s)
    return s / (s + lam)


# eq: eq-matched-lambda
def matched_lambda(eta, t):
    """The ridge penalty whose filter has the slope of the gradient descent filter at s = 0: 1 / (eta t)."""
    if eta <= 0 or t <= 0:
        raise ValueError("eta and t must be positive")
    return 1.0 / (eta * t)


def spectrum(X):
    """The eigenvalues and eigenvectors of X^T X / n, the eigenvalues in ascending order."""
    X = _arr(X)
    s, V = np.linalg.eigh(X.T @ X / X.shape[0])
    return np.clip(s, 0.0, None), V


# eq: eq-gd-step
def gradient_descent(X, y, eta, steps, w0=None):
    """Run steps of w <- w + eta X^T (y - X w) / n, the gradient step on (1 / 2n) |y - X w|^2, from w0 (zero)."""
    X, y = _arr(X), _arr(y)
    n = X.shape[0]
    w = np.zeros(X.shape[1]) if w0 is None else _arr(w0).copy()
    for _ in range(steps):
        w = w + eta * X.T @ (y - X @ w) / n
    return w


def spectral_fit(X, y, filt, tol=1e-12):
    """The weights sum_i filt(s_i) (v_i^T b / s_i) v_i with b = X^T y / n; directions with s_i <= tol get zero."""
    X, y = _arr(X), _arr(y)
    s, V = spectrum(X)
    c = V.T @ (X.T @ y / X.shape[0])
    keep = s > tol * max(1.0, float(s.max()))
    coef = np.zeros_like(s)
    coef[keep] = filt(s[keep]) * c[keep] / s[keep]
    return V @ coef


def ridge(X, y, lam):
    """The minimizer of (1 / 2n) |y - X w|^2 + (lam / 2) |w|^2: (X^T X / n + lam I)^-1 X^T y / n."""
    X, y = _arr(X), _arr(y)
    n, p = X.shape
    return np.linalg.solve(X.T @ X / n + lam * np.eye(p), X.T @ y / n)


# eq: eq-patience
class EarlyStopping:
    """The patience rule. Call step(value, state) after each evaluation; it returns True when training should stop.

    An evaluation improves when value < best - min_delta (mode "min") or value > best + min_delta (mode "max"); a NaN
    never improves. The rule stops when patience evaluations in a row did not improve. best_state is a deep copy of
    the state at the best evaluation, so later changes to the state do not reach it; restore() returns a copy of it.
    """

    def __init__(self, patience, min_delta=0.0, mode="min"):
        if int(patience) != patience or patience < 1:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.best_epoch = None
        self.best_state: Any = None
        self.epoch = 0
        self.wait = 0
        self.stopped = False

    def improves(self, value):
        """True when value is an improvement over the best value so far."""
        if math.isnan(value):
            return False
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value, state=None):
        """Record one evaluation, numbered from 1. Return True when training should stop."""
        if self.stopped:
            raise RuntimeError("step after the rule has stopped")
        value = float(value)
        self.epoch += 1
        if self.improves(value):
            self.best = value
            self.best_epoch = self.epoch
            self.best_state = copy.deepcopy(state)
            self.wait = 0
        else:
            self.wait += 1
        self.stopped = self.wait >= self.patience
        return self.stopped

    def restore(self):
        """A copy of the state of the best evaluation."""
        if self.best_epoch is None:
            raise RuntimeError("no evaluation has improved yet")
        return copy.deepcopy(self.best_state)


# eq: eq-gl
def generalization_loss(val):
    """GL(t) = 100 (E_va(t) / E_opt(t) - 1) for each epoch t, with E_opt(t) the lowest validation error up to t."""
    val = _arr(val)
    if np.any(val <= 0):
        raise ValueError("the validation errors must be positive")
    return 100.0 * (val / np.minimum.accumulate(val) - 1.0)


# eq: eq-progress
def training_progress(train, k):
    """P_k(t) = 1000 (sum of the strip / (k min of the strip) - 1) for the strip of the last k epochs before t.

    The entry for epoch t (index t - 1) is NaN when t < k, because no full strip ends there.
    """
    train = _arr(train)
    if np.any(train <= 0):
        raise ValueError("the training errors must be positive")
    out = np.full(train.shape, np.nan)
    for t in range(k, len(train) + 1):
        strip = train[t - k : t]
        out[t - 1] = 1000.0 * (strip.sum() / (k * strip.min()) - 1.0)
    return out


def gl_stop(val, alpha):
    """The first epoch t (from 1) with GL(t) > alpha, or None: the criterion GL_alpha."""
    hit = np.nonzero(generalization_loss(val) > alpha)[0]
    return int(hit[0]) + 1 if hit.size else None


def pq_stop(val, train, alpha, k=5):
    """The first end of a strip t (a multiple of k) with GL(t) / P_k(t) > alpha, or None: the criterion PQ_alpha."""
    gl = generalization_loss(val)
    pk = training_progress(train, k)
    for t in range(k, len(gl) + 1, k):
        if gl[t - 1] > alpha * pk[t - 1]:
            return t
    return None


def up_stop(val, s, k=5):
    """The first end of a strip t at which the validation error rose in s strips in a row, or None: UP_s."""
    val = _arr(val)
    run = 0
    for t in range(2 * k, len(val) + 1, k):
        run = run + 1 if val[t - 1] > val[t - 1 - k] else 0
        if run >= s:
            return t
    return None
