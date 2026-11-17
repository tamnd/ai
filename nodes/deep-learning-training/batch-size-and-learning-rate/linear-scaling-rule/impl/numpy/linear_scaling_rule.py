# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://NEW-linear-scaling-rule).
"""The linear scaling rule and the gradient noise scale in NumPy.

scale_learning_rate applies the linear rule of Goyal et al. (eq-linear-rule), or the square-root rule, to move a learning
rate tuned at one batch size to another; gradual_warmup builds a warmup hook and scaled_lr_schedule combines the scaled
rate, an optional schedule and an optional warmup hook into a function of the step. b_simple estimates the simple noise
scale tr(Sigma) / |G|^2 (eq-b-simple) from per-example gradients; two_batch_estimates is the unbiased estimator of
|G|^2 and tr(Sigma) from the squared norms of a small-batch and a big-batch gradient (McCandlish et al., Appendix A), and
NoiseScaleTracker smooths both with exponential moving averages before taking their ratio. b_noise, max_step,
optimal_step and optimal_improvement are eq-b-noise and eq-eps-opt for a quadratic model; steps_for_batch,
examples_for_batch and fit_critical_batch use the trade-off eq-tradeoff."""

import math

import numpy as np


def scale_learning_rate(base_lr, base_batch, batch, rule="linear"):
    """The learning rate for batch size `batch`, given `base_lr` tuned at `base_batch`.

    rule "linear" multiplies by k = batch / base_batch (eq-linear-rule), "sqrt" by sqrt(k), "none" keeps base_lr."""
    if base_batch <= 0 or batch <= 0:
        raise ValueError("batch sizes must be positive")
    if base_lr < 0:
        raise ValueError("the learning rate must be non-negative")
    k = batch / base_batch
    if rule == "linear":
        return base_lr * k
    if rule == "sqrt":
        return base_lr * math.sqrt(k)
    if rule == "none":
        return float(base_lr)
    raise ValueError(f"unknown rule {rule!r}")


def gradual_warmup(warmup_steps, start_factor):
    """A warmup hook (step, lr) -> lr that ramps linearly from start_factor * lr at step 0 to lr at warmup_steps.

    After warmup_steps the hook returns lr unchanged. start_factor is base_lr / scaled_lr for the gradual warmup of
    Goyal et al., so the ramp starts at the small-batch rate; see the learning-rate-warmup page."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= start_factor <= 1.0:
        raise ValueError("start_factor must be in [0, 1]")

    def hook(step, lr):
        if step >= warmup_steps:
            return lr
        frac = step / warmup_steps
        return lr * (start_factor + (1.0 - start_factor) * frac)

    return hook


def scaled_lr_schedule(base_lr, base_batch, batch, schedule=None, warmup=None, rule="linear"):
    """A function step -> learning rate: the scaled rate times schedule(step), passed through warmup(step, lr).

    schedule is a multiplier such as a step decay (default 1 for every step); warmup is a hook such as gradual_warmup,
    or None for no warmup."""
    peak = scale_learning_rate(base_lr, base_batch, batch, rule)

    def lr_at(step):
        lr = peak * (1.0 if schedule is None else schedule(step))
        return lr if warmup is None else warmup(step, lr)

    return lr_at


def b_simple(per_example_grads):
    """The simple noise scale tr(Sigma) / |G|^2 from an (n, d) array of per-example gradients, n >= 2 (eq-b-simple).

    tr(Sigma) is the sum of the unbiased per-coordinate sample variances; |G|^2 is estimated without bias as
    |mean|^2 - tr(Sigma) / n, since E|mean|^2 = |G|^2 + tr(Sigma) / n. Returns (b, g2, tr_sigma); b is inf when the
    estimate of |G|^2 is not positive."""
    g = np.asarray(per_example_grads, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 2:
        raise ValueError("need an (n, d) array with n >= 2")
    n = g.shape[0]
    mean = g.mean(axis=0)
    tr_sigma = float(np.sum((g - mean) ** 2) / (n - 1))
    g2 = float(mean @ mean) - tr_sigma / n
    b = tr_sigma / g2 if g2 > 0 else math.inf
    return b, g2, tr_sigma


def two_batch_estimates(g_small_sq, g_big_sq, b_small, b_big):
    """Unbiased estimates of |G|^2 and tr(Sigma) from |G_small|^2 and |G_big|^2 at batch sizes b_small < b_big.

    |G|^2 = (b_big |G_big|^2 - b_small |G_small|^2) / (b_big - b_small) and
    tr(Sigma) = (|G_small|^2 - |G_big|^2) / (1 / b_small - 1 / b_big), from E|G_B|^2 = |G|^2 + tr(Sigma) / B."""
    if not 0 < b_small < b_big:
        raise ValueError("need 0 < b_small < b_big")
    g2 = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
    s = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)
    return g2, s


class EMA:
    """An exponential moving average with bias correction, so the first value is returned unchanged."""

    def __init__(self, decay):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.raw = 0.0
        self.weight = 0.0

    def update(self, x):
        self.raw = self.decay * self.raw + (1.0 - self.decay) * x
        self.weight = self.decay * self.weight + (1.0 - self.decay)
        return self.value

    @property
    def value(self):
        if self.weight == 0.0:
            raise ValueError("no value yet")
        return self.raw / self.weight


class NoiseScaleTracker:
    """B_simple from a stream of (|G_small|^2, |G_big|^2) pairs: average |G|^2 and tr(Sigma) first, then divide.

    The ratio of the two unbiased estimates is biased, so McCandlish et al. average numerator and denominator
    separately; in data-parallel training G_small is one worker's gradient and G_big the all-reduced one."""

    def __init__(self, b_small, b_big, decay=0.99):
        self.b_small, self.b_big = b_small, b_big
        self.g2 = EMA(decay)
        self.s = EMA(decay)

    def update(self, g_small_sq, g_big_sq):
        g2, s = two_batch_estimates(g_small_sq, g_big_sq, self.b_small, self.b_big)
        self.g2.update(g2)
        self.s.update(s)
        return self.value

    @property
    def value(self):
        g2 = self.g2.value
        return self.s.value / g2 if g2 > 0 else math.inf


def b_noise(hessian, sigma, grad):
    """The noise scale tr(H Sigma) / (G^T H G) (eq-b-noise); needs G^T H G > 0."""
    h = np.asarray(hessian, dtype=np.float64)
    gHg = float(grad @ h @ grad)
    if gHg <= 0:
        raise ValueError("G^T H G must be positive")
    return float(np.trace(h @ np.asarray(sigma, dtype=np.float64))) / gHg


def max_step(hessian, grad):
    """eps_max = |G|^2 / (G^T H G), the best step with the noiseless gradient on a quadratic."""
    gHg = float(grad @ np.asarray(hessian, dtype=np.float64) @ grad)
    if gHg <= 0:
        raise ValueError("G^T H G must be positive")
    return float(grad @ grad) / gHg


def optimal_step(eps_max, noise_scale, batch):
    """eps_opt(B) = eps_max / (1 + B_noise / B) (eq-eps-opt)."""
    if batch <= 0 or noise_scale < 0:
        raise ValueError("need batch > 0 and noise_scale >= 0")
    return eps_max / (1.0 + noise_scale / batch)


def optimal_improvement(dl_max, noise_scale, batch):
    """Delta L_opt(B) = Delta L_max / (1 + B_noise / B), the expected drop in loss at the optimal step."""
    return optimal_step(dl_max, noise_scale, batch)


def steps_for_batch(batch, s_min, b_crit):
    """S(B) = S_min (1 + B_crit / B), the steps to reach a target loss at batch size B (eq-tradeoff)."""
    return s_min * (1.0 + b_crit / np.asarray(batch, dtype=np.float64))


def examples_for_batch(batch, s_min, b_crit):
    """E(B) = B S(B) = E_min (1 + B / B_crit), with E_min = S_min B_crit."""
    return np.asarray(batch, dtype=np.float64) * steps_for_batch(batch, s_min, b_crit)


def fit_critical_batch(batches, steps):
    """Fit S = S_min + E_min / B to measured (batch, steps) pairs and return (s_min, e_min, b_crit).

    The fit is linear in 1 / B and weights each point by 1 / S^2, so it minimizes relative errors. Needs at least two
    distinct batch sizes."""
    b = np.asarray(batches, dtype=np.float64)
    s = np.asarray(steps, dtype=np.float64)
    if b.shape != s.shape or b.size < 2 or np.unique(b).size < 2:
        raise ValueError("need at least two distinct batch sizes")
    a = np.stack([np.ones_like(b), 1.0 / b], axis=1) / s[:, None]
    (s_min, e_min), *_ = np.linalg.lstsq(a, np.ones_like(s), rcond=None)
    return float(s_min), float(e_min), float(e_min / s_min)
