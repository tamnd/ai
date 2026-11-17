# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://98TJG9).
"""Label smoothing in PyTorch.

smoothed_cross_entropy writes out what torch.nn.functional.cross_entropy(..., label_smoothing=eps) computes for class
index targets: (1 - eps) times the negative log-likelihood plus eps times the cross-entropy with the uniform
distribution, eq-decomposition, with the same class weights, ignore_index and reductions. The uniform part spreads eps
over all K classes, so the label gets 1 - eps + eps / K. smooth_labels gives the target as a distribution, for soft
targets and for other priors; LabelSmoothingCrossEntropy is the module; confidence_penalty_loss is the confidence
penalty of Pereyra et al.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _check_eps(eps):
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")


def smooth_labels(target, num_classes, eps, prior=None):
    """(1 - eps) one_hot(target) + eps * prior, with the class on the last axis; prior defaults to uniform."""
    _check_eps(eps)
    one_hot = F.one_hot(target, num_classes).to(torch.get_default_dtype())
    if prior is None:
        return one_hot * (1.0 - eps) + eps / num_classes
    prior = torch.as_tensor(prior, dtype=one_hot.dtype)
    return one_hot * (1.0 - eps) + eps * prior


def smoothed_cross_entropy(logits, target, eps, weight=None, ignore_index=-100, reduction="mean"):
    """Cross-entropy with label smoothing for class index targets, as F.cross_entropy computes it.

    logits has the classes on axis 1, (N, C) or (N, C, d1, ...), and target the matching shape without axis 1.
    With class weights w the uniform part is -sum_k w_k log p_k / C, and the mean divides both parts by the sum of
    the weights of the labels that are not ignored."""
    _check_eps(eps)
    num_classes = logits.shape[1]
    logp = F.log_softmax(logits, dim=1)
    keep = target != ignore_index
    safe = torch.where(keep, target, torch.zeros_like(target))
    picked = logp.gather(1, safe.unsqueeze(1)).squeeze(1)
    if weight is None:
        w_label = keep.to(logp.dtype)
        uniform = -logp.sum(dim=1)
    else:
        shape = [1, num_classes] + [1] * (logits.dim() - 2)
        w_label = weight[safe] * keep
        uniform = -(logp * weight.view(shape)).sum(dim=1)
    nll = -picked * w_label
    uniform = uniform * keep
    per = (1.0 - eps) * nll + (eps / num_classes) * uniform
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    if reduction == "mean":
        return per.sum() / w_label.sum()
    raise ValueError(f"unknown reduction {reduction!r}")


class LabelSmoothingCrossEntropy(nn.Module):
    """The module form of smoothed_cross_entropy, with the arguments of nn.CrossEntropyLoss."""

    def __init__(self, eps=0.1, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        _check_eps(eps)
        self.eps, self.ignore_index, self.reduction = eps, ignore_index, reduction
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        return smoothed_cross_entropy(logits, target, self.eps, self.weight, self.ignore_index, self.reduction)


def entropy(logits):
    """The entropy H(p) of softmax(logits) over axis 1."""
    logp = F.log_softmax(logits, dim=1)
    return -(logp.exp() * logp).sum(dim=1)


def confidence_penalty_loss(logits, target, beta):
    """The confidence penalty of Pereyra et al.: mean NLL - beta * mean H(p)."""
    return F.cross_entropy(logits, target) - beta * entropy(logits).mean()
