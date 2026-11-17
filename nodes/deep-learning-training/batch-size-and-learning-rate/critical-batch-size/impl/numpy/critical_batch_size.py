# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://NEW-critical-batch-size).
"""The gradient noise scale and the critical batch size in NumPy.

noise_scale_from_examples estimates B_simple = tr(Sigma) / |G|^2 (eq-b-simple) from the gradients of single examples;
noise_scale_from_two_batches does the same from the squared norms of one small-batch and one big-batch gradient
(eq-two-batch), and RunningNoiseScale keeps moving averages of both estimates and divides them. For a quadratic model
with Hessian H and gradient covariance Sigma, expected_gain is the one-step change of eq-expected-step, and
quadratic_noise_scale, step_scale and optimal_step give B_noise (eq-b-noise) and eps_opt(B) (eq-eps-opt).
steps_needed, examples_needed and fit_tradeoff work with the hyperbola of eq-tradeoff, and kaplan_critical_batch is the
power law of Kaplan et al. (eq-kaplan)."""

import numpy as np


def noise_scale_from_examples(grads):
    """(b_simple, g2, trace) from an (n, d) array of per-example gradients, n >= 2 (eq-b-simple).

    trace is the sum of the per-coordinate sample variances with divisor n - 1, and g2 = |mean|^2 - trace / n; both are
    unbiased. b_simple is inf when g2 is not positive, which happens when the noise swamps the mean."""
    grads = np.asarray(grads, dtype=float)
    if grads.ndim != 2 or grads.shape[0] < 2:
        raise ValueError("need an (n, d) array with n >= 2")
    n = grads.shape[0]
    mean = grads.mean(axis=0)  # (d,)
    trace = np.sum((grads - mean) ** 2) / (n - 1)
    g2 = mean @ mean - trace / n
    return (trace / g2 if g2 > 0 else np.inf), g2, trace


def noise_scale_from_two_batches(sq_small, sq_big, b_small, b_big):
    """Unbiased (g2, trace) from |G_small|^2 at batch size b_small and |G_big|^2 at b_big (eq-two-batch).

    The small batch may be one worker's share of the big batch. Their ratio is a biased estimate of B_simple; average
    each over steps first, as RunningNoiseScale does."""
    if b_small <= 0 or b_big <= 0 or b_small == b_big:
        raise ValueError("batch sizes must be positive and different")
    # eq: two-batch
    g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
    trace = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
    return g2, trace


class RunningNoiseScale:
    """Exponential moving averages of the two-batch estimates; value() is the ratio of the averages.

    The averages start at zero; the usual bias correction divides both by the same factor, so the ratio needs none."""

    def __init__(self, decay=0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.steps = 0
        self.g2 = 0.0
        self.trace = 0.0

    def update(self, sq_small, sq_big, b_small, b_big):
        g2, trace = noise_scale_from_two_batches(sq_small, sq_big, b_small, b_big)
        self.steps += 1
        self.g2 = self.decay * self.g2 + (1.0 - self.decay) * g2
        self.trace = self.decay * self.trace + (1.0 - self.decay) * trace
        return self.value()

    def value(self):
        """The current estimate of B_simple, or inf when the averaged |G|^2 is not positive."""
        if self.steps == 0:
            raise ValueError("no updates yet")
        return self.trace / self.g2 if self.g2 > 0 else np.inf


def expected_gain(eps, batch, G, H, Sigma):
    """The expected fall of a quadratic loss after one SGD step of size eps at batch size `batch` (eq-expected-step).

    G is the (d,) gradient, H the (d, d) Hessian and Sigma the (d, d) per-example gradient covariance."""
    G, H, Sigma = (np.asarray(a, dtype=float) for a in (G, H, Sigma))
    # eq: expected-step
    return eps * (G @ G) - 0.5 * eps**2 * (G @ H @ G + np.trace(H @ Sigma) / batch)


def quadratic_noise_scale(G, H, Sigma):
    """B_noise = tr(H Sigma) / G.H.G (eq-b-noise)."""
    G, H, Sigma = (np.asarray(a, dtype=float) for a in (G, H, Sigma))
    gHg = G @ H @ G
    if gHg <= 0:
        raise ValueError("G.H.G must be positive")
    # eq: b-noise
    return np.trace(H @ Sigma) / gHg


def step_scale(G, H):
    """(eps_max, dl_max) = (|G|^2 / G.H.G, |G|^4 / (2 G.H.G)), the best step and gain with no noise."""
    G, H = np.asarray(G, dtype=float), np.asarray(H, dtype=float)
    gHg = G @ H @ G
    if gHg <= 0:
        raise ValueError("G.H.G must be positive")
    return (G @ G) / gHg, (G @ G) ** 2 / (2.0 * gHg)


def optimal_step(batch, eps_max, b_noise):
    """eps_opt(B) = eps_max / (1 + B_noise / B) (eq-eps-opt); the gain at it is dl_max times the same factor."""
    batch = np.asarray(batch, dtype=float)
    if np.any(batch <= 0) or b_noise < 0:
        raise ValueError("batch must be positive and b_noise non-negative")
    # eq: eps-opt
    return eps_max / (1.0 + b_noise / batch)


def steps_needed(batch, s_min, e_min):
    """S(B) = S_min + E_min / B, the steps of eq-tradeoff solved for S at fixed batch size."""
    batch = np.asarray(batch, dtype=float)
    if np.any(batch <= 0) or s_min <= 0 or e_min <= 0:
        raise ValueError("batch, s_min and e_min must be positive")
    # eq: tradeoff
    return s_min + e_min / batch


def examples_needed(batch, s_min, e_min):
    """E(B) = B S(B) = B S_min + E_min."""
    return np.asarray(batch, dtype=float) * steps_needed(batch, s_min, e_min)


def tradeoff_curve(s_min, e_min, num=50, span=1e3):
    """(B, S, E) on num batch sizes spaced geometrically from B_crit / span to B_crit * span."""
    b_crit = e_min / s_min
    batch = np.geomspace(b_crit / span, b_crit * span, num)
    return (
        batch,
        steps_needed(batch, s_min, e_min),
        examples_needed(batch, s_min, e_min),
    )


def fit_tradeoff(batch, steps):
    """(s_min, e_min, b_crit) from measured steps to a target at several batch sizes.

    Fits S = S_min + E_min / B by least squares on the relative error, so small and large S weigh the same."""
    batch, steps = np.asarray(batch, dtype=float), np.asarray(steps, dtype=float)
    if batch.shape != steps.shape or np.unique(batch).size < 2:
        raise ValueError("need steps at two or more distinct batch sizes")
    if np.any(batch <= 0) or np.any(steps <= 0):
        raise ValueError("batch sizes and steps must be positive")
    design = np.stack([np.ones_like(batch), 1.0 / batch], axis=1) / steps[:, None]  # (k, 2)
    (s_min, e_min), *_ = np.linalg.lstsq(design, np.ones_like(steps), rcond=None)
    if s_min <= 0 or e_min <= 0:
        raise ValueError("the data do not fit a positive hyperbola")
    return s_min, e_min, e_min / s_min


def min_steps_and_examples(batch, steps, b_crit):
    """(S_min, E_min) from one run of `steps` steps at `batch`, given B_crit (Kaplan et al., eqs. 5.4 and 5.5).

    S_min = S / (1 + B_crit / B) and E_min = E / (1 + B / B_crit) with E = B S."""
    if batch <= 0 or steps <= 0 or b_crit <= 0:
        raise ValueError("batch, steps and b_crit must be positive")
    return steps / (1.0 + b_crit / batch), batch * steps / (1.0 + batch / b_crit)


def kaplan_critical_batch(loss, b_star=2.1e8, alpha_b=0.21):
    """B_crit(L) = B_star / L^(1 / alpha_B) in tokens, with the fit of Kaplan et al. for language models (eq-kaplan).

    The defaults are their fitted values, for the loss in nats per token on their WebText2 test set."""
    loss = np.asarray(loss, dtype=float)
    if np.any(loss <= 0) or b_star <= 0 or alpha_b <= 0:
        raise ValueError("loss, b_star and alpha_b must be positive")
    # eq: kaplan
    return b_star / loss ** (1.0 / alpha_b)
