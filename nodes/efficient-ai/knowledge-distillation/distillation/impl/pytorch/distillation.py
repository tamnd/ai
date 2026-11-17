# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://NEW-distillation).
"""Knowledge distillation in PyTorch.

distillation_loss is eq-kd-loss: alpha * T^2 * KL(softmax(teacher / T) || softmax(student / T)), averaged over the
examples, plus (1 - alpha) * F.cross_entropy(student, target) at T = 1. The KL term is torch.nn.functional.kl_div
with the student's log-probabilities as input, the teacher's log-probabilities as target (log_target=True) and
reduction="batchmean"; the teacher logits are detached. DistillationLoss is the module. soft_target_loss gives the
soft term alone, in the KL or the cross-entropy form; hint_loss is the hint term of FitNets; top1_agreement and
predictive_kl are the fidelity metrics of Stanton et al. Logits have the classes on axis 1.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _check(temperature, alpha=0.0):
    if not temperature > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def _per_example(x):
    """Sum over every axis but the first, giving one value per example."""
    return x.reshape(x.shape[0], -1).sum(dim=1)


def soft_target_loss(student_logits, teacher_logits, temperature, form="kl", reduction="batchmean"):
    """T^2 * KL(p_T || q_T) (form "kl") or T^2 * H(p_T, q_T) (form "cross_entropy"), with the teacher detached.

    reduction "batchmean" averages over the examples, "sum" adds them and "none" returns one value per example. The
    two forms differ by T^2 times the entropy of the teacher, which has no gradient in the student."""
    _check(temperature)
    t = temperature
    log_q = F.log_softmax(student_logits / t, dim=1)
    log_p = F.log_softmax(teacher_logits.detach() / t, dim=1)
    if form == "kl":
        per = _per_example(F.kl_div(log_q, log_p, reduction="none", log_target=True))
    elif form == "cross_entropy":
        per = -_per_example(log_p.exp() * log_q)
    else:
        raise ValueError(f"unknown form {form!r}")
    per = per * (t * t)
    if reduction == "batchmean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    if reduction == "none":
        return per
    raise ValueError(f"unknown reduction {reduction!r}")


def distillation_loss(student_logits, teacher_logits, target=None, temperature=4.0, alpha=0.9, form="kl"):
    """alpha * soft_target_loss(..., "batchmean") + (1 - alpha) * F.cross_entropy(student_logits, target).

    target holds class indices and may be None when alpha is 1."""
    _check(temperature, alpha)
    loss = alpha * soft_target_loss(student_logits, teacher_logits, temperature, form)
    if alpha < 1.0:
        if target is None:
            raise ValueError("target is needed when alpha < 1")
        loss = loss + (1.0 - alpha) * F.cross_entropy(student_logits, target)
    return loss


class DistillationLoss(nn.Module):
    """The module of distillation_loss: forward(student_logits, teacher_logits, target=None)."""

    def __init__(self, temperature=4.0, alpha=0.9, form="kl"):
        super().__init__()
        _check(temperature, alpha)
        self.temperature = temperature
        self.alpha = alpha
        self.form = form

    def forward(self, student_logits, teacher_logits, target=None):
        return distillation_loss(
            student_logits,
            teacher_logits,
            target,
            self.temperature,
            self.alpha,
            self.form,
        )

    def extra_repr(self):
        return f"temperature={self.temperature}, alpha={self.alpha}, form={self.form!r}"


def hint_loss(teacher_hint, student_guided, regressor):
    """The hint term of FitNets: 1/2 ||u - r(g)||^2 per example, averaged; the teacher hint is detached."""
    d = teacher_hint.detach() - regressor(student_guided)
    return 0.5 * _per_example(d * d).mean()


@torch.no_grad()
def top1_agreement(student_logits, teacher_logits):
    """The fraction of examples on which student and teacher predict the same class."""
    return (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1)).double().mean().item()


@torch.no_grad()
def predictive_kl(student_logits, teacher_logits):
    """The average KL(p_teacher || p_student) at temperature 1."""
    return soft_target_loss(student_logits, teacher_logits, 1.0).item()
