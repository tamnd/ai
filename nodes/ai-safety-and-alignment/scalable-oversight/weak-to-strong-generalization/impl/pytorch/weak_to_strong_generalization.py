# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://G0ZGHC).
"""Weak-to-strong generalization in PyTorch, for binary tasks. The student gives one logit per example; the weak
supervisor gives the probability of the positive class.

pgr is eq-pgr; adaptive_threshold and harden give eq-harden with half of the batch above the threshold;
confidence_loss is eq-conf-loss written as eq-conf-bootstrap, with the hardened predictions detached so that no
gradient flows through the threshold; alpha_schedule is the linear warm-up of alpha; fit_logits trains free logits on
a batch, which shows where each loss leads."""

import torch
import torch.nn.functional as F


def pgr(weak, weak_to_strong, ceiling):
    """Performance gap recovered: (weak_to_strong - weak) / (ceiling - weak). Undefined when ceiling == weak."""
    gap = float(ceiling) - float(weak)
    if gap == 0.0:
        raise ValueError(
            "PGR is undefined when the strong ceiling equals the weak performance"
        )
    return (float(weak_to_strong) - float(weak)) / gap


def adaptive_threshold(p):
    """The threshold t for which p > t holds for exactly B // 2 entries of the batch when values do not tie."""
    p = p.detach().reshape(-1)
    if p.numel() < 2:
        raise ValueError(
            "the adaptive threshold needs a batch of at least two examples"
        )
    s, _ = torch.sort(p)
    k = p.numel() // 2
    return s[p.numel() - k - 1]


def harden(p, t):
    """Hardened predictions I[p > t] as 0.0 or 1.0, without gradient."""
    return (p.detach() > t).to(p.dtype)


def _check_alpha(alpha):
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def confidence_loss(logits, weak_probs, alpha, t=None):
    """CE(f(x), (1 - alpha) f_w(x) + alpha I[f(x) > t]), the mean over the batch, with t adaptive when None."""
    _check_alpha(alpha)
    p = torch.sigmoid(logits)
    if t is None:
        t = adaptive_threshold(p)
    target = (1.0 - alpha) * weak_probs + alpha * harden(p, t)
    return F.binary_cross_entropy_with_logits(logits, target)


def confidence_loss_two_terms(logits, weak_probs, alpha, t=None):
    """The same loss as two cross-entropies, (1 - alpha) CE(f, f_w) + alpha CE(f, I[f > t])."""
    _check_alpha(alpha)
    p = torch.sigmoid(logits)
    if t is None:
        t = adaptive_threshold(p)
    hard = harden(p, t)
    return (1.0 - alpha) * F.binary_cross_entropy_with_logits(
        logits, weak_probs
    ) + alpha * (F.binary_cross_entropy_with_logits(logits, hard))


def alpha_schedule(step, total_steps, alpha_max, warmup_frac=0.2):
    """alpha rises linearly from 0 to alpha_max over the first warmup_frac of training, then stays at alpha_max."""
    _check_alpha(alpha_max)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warm = warmup_frac * total_steps
    if warm <= 0 or step >= warm:
        return float(alpha_max)
    return float(alpha_max) * max(step, 0) / warm


def fit_logits(init_logits, weak_probs, alpha_max, steps=2000, lr=0.5, warmup_frac=0.2):
    """Train one free logit per example with the confidence loss and the warm-up; returns the final probabilities."""
    z = init_logits.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([z], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = confidence_loss(
            z, weak_probs, alpha_schedule(step, steps, alpha_max, warmup_frac)
        )
        loss.backward()
        opt.step()
    return torch.sigmoid(z.detach())
