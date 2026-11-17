# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://CZW0BQ).
"""Direct preference optimization in PyTorch.

`dpo_loss` takes the summed log-probabilities of the chosen and rejected completions under the policy and under the
frozen reference model, and returns the loss together with the detached implicit rewards for logging.
`sequence_logps` computes those sums from a causal language model's logits.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _reduce(losses, reduction):
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    if reduction == "none":
        return losses
    raise ValueError(f"unknown reduction {reduction!r}")


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reduction="mean"):
    """-logsigmoid(beta * ((log pi(y_w) - log pi(y_l)) - (log pi_ref(y_w) - log pi_ref(y_l)))).

    Returns (loss, chosen_rewards, rejected_rewards); the rewards beta * (log pi - log pi_ref) are detached.
    The reference log-probabilities are detached too, so a reference that was not computed under no_grad still
    receives no gradient.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    ref_chosen = ref_chosen.detach()
    ref_rejected = ref_rejected.detach()
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen - ref_chosen).detach()
    rejected_rewards = beta * (policy_rejected - ref_rejected).detach()
    return _reduce(losses, reduction), chosen_rewards, rejected_rewards


def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau=0.1, reduction="mean"):
    """The IPO loss (h - 1/(2 tau))^2 of Azar et al., h the gap between policy and reference log-ratios."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    h = (policy_chosen - policy_rejected) - (ref_chosen.detach() - ref_rejected.detach())
    return _reduce((h - 1.0 / (2.0 * tau)) ** 2, reduction)


def sequence_logps(logits, labels, mask):
    """Summed log-probabilities of the completion tokens.

    logits: (batch, length, vocab) from a causal LM over prompt + completion; labels: (batch, length) token ids of the
    same sequence; mask: (batch, length), 1 on completion tokens. The logits at position t predict token t + 1, so
    they are shifted by one before the tokens are gathered.
    """
    labels = labels[:, 1:]
    mask = mask[:, 1:].to(logits.dtype)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


class DPOLoss(nn.Module):
    """Module wrapper of dpo_loss with a fixed beta; returns only the loss."""

    def __init__(self, beta=0.1, reduction="mean"):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = beta
        self.reduction = reduction

    def forward(self, policy_chosen, policy_rejected, ref_chosen, ref_rejected):
        return dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            self.beta,
            self.reduction,
        )[0]
