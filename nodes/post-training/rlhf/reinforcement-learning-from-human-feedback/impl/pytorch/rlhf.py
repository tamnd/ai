# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://6XVY3X).
"""The losses and objectives of RLHF in PyTorch, for training with autograd.

bt_loss and ranked_loss fit a reward model (eq-bt, eq-rm-loss); kl_shaped_rewards, ppo_clip_loss and ptx_loss train
the policy (eq-kl-reward, eq-ppo-clip, eq-ptx); dpo_loss trains it from the preferences directly (eq-implicit-reward,
eq-dpo). The losses are to be minimized, so ppo_clip_loss and ptx_loss are the negatives of the objectives of the page.
sequence_logprob sums the log-probabilities of the response tokens from the logits of a causal language model.
"""

import itertools

import torch
import torch.nn.functional as F


def bt_loss(r_w, r_l):
    """The mean of -log sigma(r_w - r_l) (eq-rm-loss)."""
    return -F.logsigmoid(r_w - r_l).mean()


def ranked_loss(rewards):
    """Eq. 1 of Ouyang et al.: rewards of shape (B, K), each row from the best response to the worst; the mean over
    the K choose 2 pairs of each prompt, then over the prompts."""
    if rewards.dim() != 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must have the shape (B, K) with K >= 2, got {tuple(rewards.shape)}")
    i, j = zip(*itertools.combinations(range(rewards.shape[1]), 2))
    i = torch.tensor(i, device=rewards.device)
    j = torch.tensor(j, device=rewards.device)
    return -F.logsigmoid(rewards[:, i] - rewards[:, j]).mean(dim=1).mean()


def sequence_logprob(logits, tokens, mask):
    """The sum over the response tokens of log pi(token | previous tokens). logits has the shape (B, T, V) and
    predicts tokens (B, T); mask (B, T) is 1 on the response tokens and 0 on the prompt and on padding."""
    logp = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)


def kl_shaped_rewards(score, logp, logp_ref, beta, mask=None):
    """The per-token rewards of eq-kl-reward: -beta (log pi - log pi_ref) at each response token and the score at the
    last one. The reference is not trained, and the reward is a constant for the policy update, so both are
    detached."""
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    m = torch.ones_like(logp) if mask is None else mask.to(logp.dtype)
    if bool((m.sum(dim=1) == 0).any()):
        raise ValueError("every response needs at least one token")
    out = (-beta * (logp - logp_ref) * m).detach().clone()
    last = m.shape[1] - 1 - torch.argmax(m.flip(1), dim=1)
    out[torch.arange(out.shape[0]), last] += score.detach().to(out.dtype)
    return out


def ppo_clip_loss(logp, logp_old, adv, eps=0.2, mask=None):
    """The negative of L^CLIP (eq-ppo-clip), averaged over the tokens. logp_old and adv are detached."""
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    rho = torch.exp(logp - logp_old.detach())
    a = adv.detach()
    obj = torch.minimum(rho * a, torch.clamp(rho, 1.0 - eps, 1.0 + eps) * a)
    m = torch.ones_like(obj) if mask is None else mask.to(obj.dtype)
    return -(obj * m).sum() / m.sum()


def ptx_loss(clip_loss, pretrain_logp, gamma):
    """The negative of the PPO-ptx objective (eq-ptx): the PPO loss less gamma times the mean pretraining
    log-likelihood."""
    return clip_loss - gamma * pretrain_logp.mean()


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    """Eq. 7 of Rafailov et al. (eq-dpo). The reference log-probabilities come from a frozen model and are
    detached."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    margin = beta * ((logp_w - ref_w.detach()) - (logp_l - ref_l.detach()))
    return -F.logsigmoid(margin).mean()


def implicit_reward(logp, logp_ref, beta):
    """beta log(pi / pi_ref), the reward that DPO assigns, detached as Rafailov et al. log it."""
    return (beta * (logp - logp_ref)).detach()
