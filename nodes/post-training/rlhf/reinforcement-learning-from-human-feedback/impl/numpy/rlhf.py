# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
"""The losses and objectives of RLHF on toy tensors, the NumPy reference.

The reward model is fitted with the Bradley-Terry loss (eq-bt, eq-rm-loss), on pairs or on rankings of K responses as
in InstructGPT. The policy is trained on the KL-shaped reward (eq-kl-reward) with the clipped surrogate of PPO
(eq-ppo-clip), and PPO-ptx adds the pretraining term (eq-ptx). On a finite set of responses the KL-regularized optimum
is pi_ref * exp(r / beta) / Z (eq-optimal), and DPO fits it directly from the preferences (eq-implicit-reward, eq-dpo).
Rewards and log-probabilities are arrays; a log-probability of a response is the sum over its tokens.
"""

import itertools

import numpy as np


def log_sigmoid(x):
    """log sigma(x) = -log(1 + exp(-x)), computed without overflow for large |x|."""
    return -np.logaddexp(0.0, -np.asarray(x, dtype=float))


def sigmoid(x):
    """The logistic function, as exp(log_sigmoid(x))."""
    return np.exp(log_sigmoid(x))


def bt_prob(r_w, r_l):
    """The Bradley-Terry probability that y_w is preferred to y_l, sigma(r_w - r_l) (eq-bt)."""
    return sigmoid(np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float))


def bt_loss(r_w, r_l):
    """The reward-model loss, the mean of -log sigma(r_w - r_l) over the pairs (eq-rm-loss)."""
    return float(np.mean(-log_sigmoid(np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float))))


def bt_grad(r_w, r_l):
    """The gradient of bt_loss with respect to r_w and r_l: -sigma(-Delta) / n and +sigma(-Delta) / n
    (derivations/bradley-terry-gradient.md)."""
    d = np.asarray(r_w, dtype=float) - np.asarray(r_l, dtype=float)
    g = sigmoid(-d) / d.size
    return -g, g


def ranked_loss(rewards):
    """The loss of eq. 1 of Ouyang et al. on rankings. rewards has the shape (B, K), with each row ordered from the
    best response to the worst. Each prompt contributes the mean of -log sigma(r_i - r_j) over its K choose 2 pairs
    i < j, so it weighs as one element of the batch whatever K is."""
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError(f"rewards must have the shape (B, K) with K >= 2, got {r.shape}")
    pairs = list(itertools.combinations(range(r.shape[1]), 2))
    i, j = (np.array(p) for p in zip(*pairs))
    per_prompt = np.mean(-log_sigmoid(r[:, i] - r[:, j]), axis=1)
    return float(np.mean(per_prompt))


def kl_shaped_rewards(score, logp, logp_ref, beta, mask=None):
    """The per-token rewards of eq-kl-reward: -beta * (log pi - log pi_ref) at each token of the response, and the
    reward-model score added at its last token. logp and logp_ref have the shape (B, T); mask is 1 on the response
    tokens and 0 on padding. Summed over the tokens, they give r(x, y) - beta log(pi(y|x) / pi_ref(y|x))."""
    logp = np.asarray(logp, dtype=float)
    logp_ref = np.asarray(logp_ref, dtype=float)
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    m = np.ones_like(logp) if mask is None else np.asarray(mask, dtype=float)
    if np.any(m.sum(axis=1) == 0):
        raise ValueError("every response needs at least one token")
    out = -beta * (logp - logp_ref) * m
    last = m.shape[1] - 1 - np.argmax(m[:, ::-1] > 0, axis=1)
    out[np.arange(out.shape[0]), last] += np.asarray(score, dtype=float)
    return out


def ppo_clip_objective(logp, logp_old, adv, eps=0.2, mask=None):
    """The clipped surrogate L^CLIP of Schulman et al. (eq-ppo-clip), the mean over the tokens of
    min(rho * A, clip(rho, 1 - eps, 1 + eps) * A) with rho = exp(logp - logp_old). It is to be maximized."""
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    rho = np.exp(np.asarray(logp, dtype=float) - np.asarray(logp_old, dtype=float))
    a = np.asarray(adv, dtype=float)
    obj = np.minimum(rho * a, np.clip(rho, 1.0 - eps, 1.0 + eps) * a)
    m = np.ones_like(obj) if mask is None else np.asarray(mask, dtype=float)
    return float((obj * m).sum() / m.sum())


def ptx_objective(rl_objective, pretrain_logp, gamma):
    """The objective of PPO-ptx (eq-ptx): the RL objective plus gamma times the mean log-likelihood of pretraining
    text. gamma = 0 gives PPO."""
    return float(rl_objective) + float(gamma) * float(np.mean(pretrain_logp))


def kl_divergence(p, q):
    """KL(p || q) of two distributions over the last axis, with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return t.sum(axis=-1)


def kl_objective(pi, pi_ref, r, beta):
    """The KL-regularized objective of one prompt, E_{y ~ pi}[r(y)] - beta KL(pi || pi_ref), on a finite set of
    responses (eq-objective)."""
    pi = np.asarray(pi, dtype=float)
    return (pi * np.asarray(r, dtype=float)).sum(axis=-1) - beta * kl_divergence(pi, pi_ref)


def kl_optimal_policy(pi_ref, r, beta):
    """The maximizer of kl_objective, pi_ref * exp(r / beta) / Z, and log Z (eq-optimal). The maximum is beta log Z."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    logits = np.log(np.asarray(pi_ref, dtype=float)) + np.asarray(r, dtype=float) / beta
    log_z = np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return np.exp(logits - log_z), log_z[..., 0]


def implicit_reward(logp, logp_ref, beta):
    """The reward that a policy defines under DPO, beta log(pi(y|x) / pi_ref(y|x)) (eq-implicit-reward, without the
    term beta log Z(x), which cancels in every comparison)."""
    return beta * (np.asarray(logp, dtype=float) - np.asarray(logp_ref, dtype=float))


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    """The DPO loss of eq. 7 of Rafailov et al. (eq-dpo): the Bradley-Terry loss of the implicit rewards."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    return bt_loss(implicit_reward(logp_w, ref_w, beta), implicit_reward(logp_l, ref_l, beta))


def dpo_weight(logp_w, logp_l, ref_w, ref_l, beta):
    """The weight of each pair in the gradient of DPO, sigma(r_hat_l - r_hat_w): large when the implicit reward
    orders the pair wrong."""
    return sigmoid(implicit_reward(logp_l, ref_l, beta) - implicit_reward(logp_w, ref_w, beta))
