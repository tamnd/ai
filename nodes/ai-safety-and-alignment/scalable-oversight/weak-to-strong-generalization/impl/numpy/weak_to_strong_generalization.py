# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://G0ZGHC).
"""Weak-to-strong generalization in NumPy, for binary tasks. Probabilities are of the positive class, one per example.

pgr is eq-pgr; adaptive_threshold and harden give the hardened predictions of eq-harden, with the threshold set so
that exactly half of a batch is above it, and harden_balanced does the same by rank when values tie; binary_ce is the
mean cross-entropy CE(p, q) between a prediction p and a soft or hard target q; confidence_loss is eq-conf-loss and
bootstrap_target the mixed target of eq-conf-bootstrap; ce_grad_logit is the gradient of eq-conf-bootstrap with
respect to the student's logits; alpha_schedule is the linear warm-up of alpha; agreement is the student-supervisor
agreement."""

import numpy as np

EPS = 1e-12


def pgr(weak, weak_to_strong, ceiling):
    """Performance gap recovered: (weak_to_strong - weak) / (ceiling - weak). Undefined when ceiling == weak."""
    weak, w2s, ceiling = (
        np.asarray(v, dtype=np.float64) for v in (weak, weak_to_strong, ceiling)
    )
    gap = ceiling - weak
    if np.any(gap == 0):
        raise ValueError(
            "PGR is undefined when the strong ceiling equals the weak performance"
        )
    out = (w2s - weak) / gap
    return float(out) if out.ndim == 0 else out


def adaptive_threshold(p):
    """The threshold t for which p > t holds for exactly half of the batch (B // 2 entries) when values do not tie.

    t is the largest value of the lower half, so the upper half is strictly above it."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size < 2:
        raise ValueError(
            "the adaptive threshold needs a batch of at least two examples"
        )
    s = np.sort(p)
    k = p.size // 2
    return float(s[p.size - k - 1])


def harden(p, t):
    """Hardened predictions I[p > t] as 0.0 or 1.0."""
    return (np.asarray(p, dtype=np.float64) > t).astype(np.float64)


def harden_balanced(p):
    """Exactly B // 2 ones, on the largest values; ties broken toward the later index."""
    p = np.asarray(p, dtype=np.float64).ravel()
    k = p.size // 2
    order = np.argsort(p, kind="stable")
    out = np.zeros_like(p)
    out[order[p.size - k :]] = 1.0
    return out


def binary_ce(p, q):
    """Mean over the batch of -[q log p + (1 - q) log(1 - p)]."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    q = np.asarray(q, dtype=np.float64)
    return float(np.mean(-(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))))


def _check_alpha(alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def bootstrap_target(p_weak, hard, alpha):
    """The mixed target (1 - alpha) p_weak + alpha hard."""
    _check_alpha(alpha)
    return (1.0 - alpha) * np.asarray(p_weak, dtype=np.float64) + alpha * np.asarray(
        hard, dtype=np.float64
    )


def confidence_loss(p_strong, p_weak, alpha, t=None):
    """(1 - alpha) CE(p, p_weak) + alpha CE(p, I[p > t]), with t adaptive (half the batch above) when None."""
    _check_alpha(alpha)
    p_strong = np.asarray(p_strong, dtype=np.float64)
    if t is None:
        t = adaptive_threshold(p_strong)
    hard = harden(p_strong, t)
    return (1.0 - alpha) * binary_ce(p_strong, p_weak) + alpha * binary_ce(
        p_strong, hard
    )


def ce_grad_logit(p_strong, target):
    """Gradient of binary_ce(sigmoid(z), target) with respect to the logits z, with the target held fixed."""
    p_strong = np.asarray(p_strong, dtype=np.float64)
    return (p_strong - np.asarray(target, dtype=np.float64)) / p_strong.size


def alpha_schedule(step, total_steps, alpha_max, warmup_frac=0.2):
    """alpha rises linearly from 0 to alpha_max over the first warmup_frac of training, then stays at alpha_max."""
    _check_alpha(alpha_max)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warm = warmup_frac * total_steps
    if warm <= 0 or step >= warm:
        return float(alpha_max)
    return float(alpha_max * max(step, 0) / warm)


def agreement(student_pred, weak_pred):
    """The fraction of examples where the student predicts the same class as the weak supervisor."""
    a = np.asarray(student_pred)
    b = np.asarray(weak_pred)
    if a.shape != b.shape:
        raise ValueError("predictions must have the same shape")
    return float(np.mean(a == b))
