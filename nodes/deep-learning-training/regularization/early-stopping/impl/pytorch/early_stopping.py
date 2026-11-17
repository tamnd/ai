# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://NEW-early-stopping).
"""Early stopping, the PyTorch reference.

The same functions as the NumPy reference, on tensors: the filters of gradient descent, gradient flow and ridge on
least squares (eq-gd-filter, eq-flow-filter, eq-ridge-filter), and the criteria GL and PQ of Prechelt (eq-gl,
eq-progress). EarlyStopping is the patience rule (eq-patience) for a torch.nn.Module: step(value, model) keeps a
detached clone of model.state_dict() at the best evaluation, and restore(model) loads it back.
"""

import math

import torch


def _t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# eq: eq-gd-filter
def gd_filter(s, eta, t):
    """The filter of t steps of gradient descent with step eta on the eigenvalue s: 1 - (1 - eta s)^t."""
    return 1.0 - (1.0 - eta * _t(s)) ** t


# eq: eq-flow-filter
def flow_filter(s, tau):
    """The filter of gradient flow at time tau on the eigenvalue s: 1 - exp(-s tau)."""
    return -torch.expm1(-_t(s) * tau)


# eq: eq-ridge-filter
def ridge_filter(s, lam):
    """The filter of ridge regression with penalty lam on the eigenvalue s: s / (s + lam)."""
    s = _t(s)
    return s / (s + lam)


# eq: eq-matched-lambda
def matched_lambda(eta, t):
    """The ridge penalty whose filter has the slope of the gradient descent filter at s = 0: 1 / (eta t)."""
    if eta <= 0 or t <= 0:
        raise ValueError("eta and t must be positive")
    return 1.0 / (eta * t)


# eq: eq-gd-step
def gradient_descent(X, y, eta, steps):
    """Run steps of torch.optim.SGD with lr eta on (1 / 2n) |y - X w|^2 from w = 0; return w."""
    X, y = _t(X), _t(y)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([w], lr=eta)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * torch.mean((y - X @ w) ** 2)
        loss.backward()
        opt.step()
    return w.detach()


def ridge(X, y, lam):
    """The minimizer of (1 / 2n) |y - X w|^2 + (lam / 2) |w|^2."""
    X, y = _t(X), _t(y)
    n, p = X.shape
    return torch.linalg.solve(X.T @ X / n + lam * torch.eye(p, dtype=torch.float64), X.T @ y / n)


# eq: eq-patience
class EarlyStopping:
    """The patience rule for a module. step(value, model) returns True when training should stop.

    An evaluation improves when value < best - min_delta (mode "min") or value > best + min_delta (mode "max"); a NaN
    never improves. best_state holds detached clones of the tensors of model.state_dict() at the best evaluation.
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
        self.best_state = None
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

    def step(self, value, model=None):
        """Record one evaluation, numbered from 1. Return True when training should stop."""
        if self.stopped:
            raise RuntimeError("step after the rule has stopped")
        value = float(value)
        self.epoch += 1
        if self.improves(value):
            self.best = value
            self.best_epoch = self.epoch
            if model is not None:
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.wait = 0
        else:
            self.wait += 1
        self.stopped = self.wait >= self.patience
        return self.stopped

    def restore(self, model):
        """Load the state of the best evaluation into model, and return model."""
        if self.best_state is None:
            raise RuntimeError("no state of an improving evaluation was kept")
        model.load_state_dict(self.best_state)
        return model


# eq: eq-gl
def generalization_loss(val):
    """GL(t) = 100 (E_va(t) / E_opt(t) - 1) for each epoch t, with E_opt(t) the lowest validation error up to t."""
    val = _t(val)
    if bool((val <= 0).any()):
        raise ValueError("the validation errors must be positive")
    return 100.0 * (val / torch.cummin(val, dim=0).values - 1.0)


# eq: eq-progress
def training_progress(train, k):
    """P_k(t) = 1000 (sum of the strip / (k min of the strip) - 1); NaN for t < k."""
    train = _t(train)
    if bool((train <= 0).any()):
        raise ValueError("the training errors must be positive")
    out = torch.full_like(train, float("nan"))
    if len(train) >= k:
        strips = train.unfold(0, k, 1)
        out[k - 1 :] = 1000.0 * (strips.sum(dim=1) / (k * strips.min(dim=1).values) - 1.0)
    return out


def gl_stop(val, alpha):
    """The first epoch t (from 1) with GL(t) > alpha, or None: the criterion GL_alpha."""
    hit = torch.nonzero(generalization_loss(val) > alpha)
    return int(hit[0, 0]) + 1 if hit.numel() else None


def pq_stop(val, train, alpha, k=5):
    """The first end of a strip t (a multiple of k) with GL(t) / P_k(t) > alpha, or None: the criterion PQ_alpha."""
    gl = generalization_loss(val)
    pk = training_progress(train, k)
    for t in range(k, len(gl) + 1, k):
        if bool(gl[t - 1] > alpha * pk[t - 1]):
            return t
    return None
