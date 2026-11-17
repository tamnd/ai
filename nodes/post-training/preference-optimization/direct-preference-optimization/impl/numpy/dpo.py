# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://NEW-direct-preference-optimization).
"""Direct preference optimization in NumPy.

The loss works on per-sequence log-probabilities: for each preference pair, the summed log-probability of the
chosen and of the rejected completion under the policy being trained and under the frozen reference model.
Computing those sums from token logits is the job of the model code; `sequence_logps` shows one way.

Also here: the closed-form optimal policy of the KL-constrained objective, the reward it implies, the
Bradley-Terry probability, the analytic gradient of the loss and the IPO loss of Azar et al.
"""

import numpy as np


def log_sigmoid(x):
    """log(1 / (1 + exp(-x))), computed without overflow for large |x|."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def sigmoid(x):
    """1 / (1 + exp(-x)), computed through log_sigmoid so it does not overflow."""
    return np.exp(log_sigmoid(x))


def implicit_rewards(policy_logps, ref_logps, beta):
    """The implicit reward beta * (log pi(y|x) - log pi_ref(y|x)) of each sequence."""
    return beta * (np.asarray(policy_logps, dtype=float) - np.asarray(ref_logps, dtype=float))


def _reduce(losses, reduction):
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"unknown reduction {reduction!r}")


def dpo_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """The reward margin r_w - r_l of each pair, the argument of the sigmoid in the DPO loss."""
    return implicit_rewards(policy_chosen, ref_chosen, beta) - implicit_rewards(policy_rejected, ref_rejected, beta)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reduction="mean"):
    """-log sigmoid(beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l))), averaged over pairs by default.

    All four inputs are per-sequence log-probabilities of the same shape; beta > 0.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    margins = dpo_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return _reduce(-log_sigmoid(margins), reduction)


def dpo_loss_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reduction="mean"):
    """Gradients of dpo_loss with respect to the policy log-probabilities of the chosen and rejected sequences.

    Per pair the loss has derivative -beta * sigmoid(r_l - r_w) in log pi(y_w) and +beta * sigmoid(r_l - r_w)
    in log pi(y_l); with reduction "mean" both are divided by the number of pairs.
    """
    margins = dpo_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    weight = sigmoid(-margins)
    scale = 1.0 / margins.size if reduction == "mean" else 1.0
    return -beta * weight * scale, beta * weight * scale


def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau=0.1, reduction="mean"):
    """The IPO loss (h - 1/(2 tau))^2 of Azar et al., with h the difference of the policy and reference log-ratios."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    h = dpo_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, 1.0)
    return _reduce((h - 1.0 / (2.0 * tau)) ** 2, reduction)


def preference_accuracy(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """The fraction of pairs whose implicit reward ranks the chosen sequence strictly above the rejected one."""
    margins = dpo_margins(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return float((margins > 0).mean())


def optimal_policy(ref_probs, rewards, beta):
    """pi*(y) = pi_ref(y) exp(r(y)/beta) / Z over a finite set of responses, for one prompt."""
    log_u = np.log(np.asarray(ref_probs, dtype=float)) + np.asarray(rewards, dtype=float) / beta
    log_z = np.logaddexp.reduce(log_u)
    return np.exp(log_u - log_z)


def log_partition(ref_probs, rewards, beta):
    """log Z = log sum_y pi_ref(y) exp(r(y)/beta)."""
    return np.logaddexp.reduce(np.log(np.asarray(ref_probs, dtype=float)) + np.asarray(rewards, dtype=float) / beta)


def kl_regularized_objective(policy_probs, ref_probs, rewards, beta):
    """E_pi[r] - beta KL(pi || pi_ref) for one prompt over a finite set of responses."""
    p = np.asarray(policy_probs, dtype=float)
    q = np.asarray(ref_probs, dtype=float)
    r = np.asarray(rewards, dtype=float)
    nz = p > 0
    return float((p * r).sum() - beta * (p[nz] * (np.log(p[nz]) - np.log(q[nz]))).sum())


def bradley_terry(r1, r2):
    """p(y1 preferred to y2) = exp(r1) / (exp(r1) + exp(r2)) = sigmoid(r1 - r2)."""
    return sigmoid(np.asarray(r1, dtype=float) - np.asarray(r2, dtype=float))


def sequence_logps(logits, labels, mask):
    """Sum of the log-probabilities of `labels` under `logits` over the positions where `mask` is 1.

    logits has shape (batch, length, vocab) and gives the next-token distribution at each position, already aligned
    with labels of shape (batch, length); mask selects the completion tokens and excludes prompt and padding.
    """
    logits = np.asarray(logits, dtype=float)
    m = logits.max(axis=-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    picked = np.take_along_axis(logp, np.asarray(labels)[..., None], axis=-1)[..., 0]
    return (picked * np.asarray(mask, dtype=float)).sum(axis=-1)
