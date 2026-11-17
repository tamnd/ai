# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://NEW-distillation).
"""Knowledge distillation in NumPy.

soft_targets is the softmax at temperature T of eq-softmax-temperature. distillation_loss is eq-kd-loss: alpha times
T^2 times the soft-target term, KL(p_T || q_T) by default or the cross-entropy H(p_T, q_T), plus 1 - alpha times the
cross-entropy with the labels at T = 1. Its gradient in the student logits is distillation_loss_grad, built from
soft_target_grad, the (q - p) / T of eq-soft-gradient. high_temperature_limit_grad is the limit of eq-logit-matching,
logit_matching_loss the regression on logits of Ba and Caruana, hint_loss the hint term of FitNets, ensemble_soft_targets
the arithmetic or geometric mean of an ensemble, and top1_agreement and predictive_kl the fidelity metrics of Stanton
et al. Logits have the classes on the last axis.
"""

import numpy as np


def _check_temperature(t):
    if not t > 0.0:
        raise ValueError(f"temperature must be positive, got {t}")


def _check_alpha(alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def _reduce(per, reduction):
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    if reduction == "none":
        return per
    raise ValueError(f"unknown reduction {reduction!r}")


def log_softmax(z, t=1.0):
    """log softmax(z / t) over the last axis, shifted by the maximum for stability."""
    _check_temperature(t)
    a = np.asarray(z, dtype=float) / t
    m = a.max(axis=-1, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis=-1, keepdims=True))


def soft_targets(logits, t):
    """The class probabilities softmax(logits / t) of eq-softmax-temperature; larger t gives a softer distribution."""
    return np.exp(log_softmax(logits, t))


def soft_cross_entropy(student_logits, teacher_logits, t, reduction="mean"):
    """H(p_t, q_t) = -sum_i p_i log q_i, with p = softmax(teacher / t) and q = softmax(student / t); not scaled by t^2."""
    p = soft_targets(teacher_logits, t)
    per = -(p * log_softmax(student_logits, t)).sum(axis=-1)
    return _reduce(per, reduction)


def soft_kl(student_logits, teacher_logits, t, reduction="mean"):
    """KL(p_t || q_t) = H(p_t, q_t) - H(p_t); not scaled by t^2."""
    lp = log_softmax(teacher_logits, t)
    per = (np.exp(lp) * (lp - log_softmax(student_logits, t))).sum(axis=-1)
    return _reduce(per, reduction)


def soft_target_grad(student_logits, teacher_logits, t):
    """The gradient of H(p_t, q_t), or of KL(p_t || q_t), in the student logits for each example: (q - p) / t."""
    return (soft_targets(student_logits, t) - soft_targets(teacher_logits, t)) / t


def hard_cross_entropy(student_logits, labels, reduction="mean"):
    """-log softmax(student)[label] at temperature 1."""
    lq = log_softmax(student_logits)
    per = -np.take_along_axis(lq, np.asarray(labels)[..., None], axis=-1)[..., 0]
    return _reduce(per, reduction)


def distillation_loss(student_logits, teacher_logits, labels, t, alpha, soft="kl", reduction="mean"):
    """alpha * t^2 * soft term + (1 - alpha) * hard cross-entropy, eq-kd-loss.

    soft is "kl" for KL(p_t || q_t) or "cross_entropy" for H(p_t, q_t); they differ by the entropy of the teacher,
    which does not depend on the student, so they have the same gradient. With alpha = 1 labels may be None.
    log_softmax checks the temperature."""
    _check_alpha(alpha)
    if soft == "kl":
        s = soft_kl(student_logits, teacher_logits, t, reduction="none")
    elif soft == "cross_entropy":
        s = soft_cross_entropy(student_logits, teacher_logits, t, reduction="none")
    else:
        raise ValueError(f"unknown soft term {soft!r}")
    per = alpha * t * t * s
    if alpha < 1.0:
        per = per + (1.0 - alpha) * hard_cross_entropy(student_logits, labels, reduction="none")
    return _reduce(per, reduction)


def distillation_loss_grad(student_logits, teacher_logits, labels, t, alpha):
    """Gradient of distillation_loss with reduction "none" in the student logits: alpha t (q_t - p_t) + (1 - alpha) (q_1 - y)."""
    _check_alpha(alpha)
    g = alpha * t * t * soft_target_grad(student_logits, teacher_logits, t)
    if alpha < 1.0:
        q1 = soft_targets(student_logits, 1.0)
        y = np.eye(q1.shape[-1])[np.asarray(labels)]
        g = g + (1.0 - alpha) * (q1 - y)
    return g


def high_temperature_limit_grad(student_logits, teacher_logits):
    """The limit of t^2 (q_t - p_t) / t as t grows, eq-logit-matching: ((z - mean z) - (v - mean v)) / N."""
    z = np.asarray(student_logits, dtype=float)
    v = np.asarray(teacher_logits, dtype=float)
    n = z.shape[-1]
    zc = z - z.mean(axis=-1, keepdims=True)
    vc = v - v.mean(axis=-1, keepdims=True)
    return (zc - vc) / n


def logit_matching_loss(student_logits, teacher_logits, reduction="mean"):
    """The regression of Ba and Caruana on the logits: 1/2 ||z - v||^2 for each example."""
    d = np.asarray(student_logits, dtype=float) - np.asarray(teacher_logits, dtype=float)
    return _reduce(0.5 * (d * d).sum(axis=-1), reduction)


def hint_loss(teacher_hint, student_guided, regressor, reduction="mean"):
    """The hint term of FitNets with a linear regressor W: 1/2 ||u - W g||^2 for each example.

    teacher_hint has shape (..., H) and student_guided (..., G); regressor has shape (H, G) and maps the guided layer
    of the student to the size of the hint layer of the teacher."""
    u = np.asarray(teacher_hint, dtype=float)
    r = np.asarray(student_guided, dtype=float) @ np.asarray(regressor, dtype=float).T
    d = u - r
    return _reduce(0.5 * (d * d).sum(axis=-1), reduction)


def ensemble_soft_targets(member_logits, t, mean="arithmetic"):
    """Soft targets of an ensemble from the logits of its members on axis 0.

    "arithmetic" averages the members' probabilities at temperature t; "geometric" takes the normalized geometric mean
    of the probabilities, which is the softmax of the averaged logits divided by t."""
    m = np.asarray(member_logits, dtype=float)
    if mean == "arithmetic":
        return soft_targets(m, t).mean(axis=0)
    if mean == "geometric":
        return soft_targets(log_softmax(m, t).mean(axis=0), 1.0)
    raise ValueError(f"unknown mean {mean!r}")


def top1_agreement(student_logits, teacher_logits):
    """The fraction of examples on which the student and the teacher predict the same class."""
    s = np.argmax(np.asarray(student_logits), axis=-1)
    v = np.argmax(np.asarray(teacher_logits), axis=-1)
    return float(np.mean(s == v))


def predictive_kl(student_logits, teacher_logits):
    """The average KL(p_teacher || p_student) at temperature 1."""
    return float(soft_kl(student_logits, teacher_logits, 1.0, reduction="mean"))
