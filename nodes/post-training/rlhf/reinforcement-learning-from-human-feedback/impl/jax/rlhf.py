# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
"""The losses and objectives of RLHF as pure JAX functions, so they can be jitted and differentiated.

bt_loss and ranked_loss are the reward-model losses (eq-bt, eq-rm-loss); kl_shaped_rewards and ppo_clip_loss train
the policy (eq-kl-reward, eq-ppo-clip); kl_optimal_policy is the KL-regularized optimum on a finite set of responses
(eq-optimal); dpo_loss is eq. 7 of Rafailov et al. (eq-dpo). The tests compare them with the numbers of the NumPy
reference, which the PyTorch tests check against torch.nn.functional.
"""

import itertools

import jax
import jax.numpy as jnp


def bt_loss(r_w, r_l):
    """The mean of -log sigma(r_w - r_l) (eq-rm-loss)."""
    return -jnp.mean(jax.nn.log_sigmoid(r_w - r_l))


def ranked_loss(rewards):
    """Eq. 1 of Ouyang et al. on rankings of shape (B, K), each row from the best response to the worst."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must have the shape (B, K) with K >= 2, got {rewards.shape}")
    i, j = (jnp.array(p) for p in zip(*itertools.combinations(range(rewards.shape[1]), 2)))
    return -jnp.mean(jnp.mean(jax.nn.log_sigmoid(rewards[:, i] - rewards[:, j]), axis=1))


def kl_shaped_rewards(score, logp, logp_ref, beta, mask):
    """The per-token rewards of eq-kl-reward, with the score at the last response token; mask is 1 on the response
    tokens. The rewards are constants for the policy update, so the gradient is stopped."""
    m = mask.astype(logp.dtype)
    out = -beta * (logp - logp_ref) * m
    last = m.shape[1] - 1 - jnp.argmax(m[:, ::-1] > 0, axis=1)
    out = out.at[jnp.arange(out.shape[0]), last].add(score)
    return jax.lax.stop_gradient(out)


def ppo_clip_loss(logp, logp_old, adv, eps=0.2):
    """The negative of L^CLIP (eq-ppo-clip), averaged over the tokens."""
    rho = jnp.exp(logp - jax.lax.stop_gradient(logp_old))
    a = jax.lax.stop_gradient(adv)
    return -jnp.mean(jnp.minimum(rho * a, jnp.clip(rho, 1.0 - eps, 1.0 + eps) * a))


def kl_optimal_policy(logp_ref, r, beta):
    """pi_ref exp(r / beta) / Z and log Z, over the last axis (eq-optimal)."""
    logits = logp_ref + r / beta
    log_z = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.exp(logits - log_z), log_z[..., 0]


def kl_objective(pi, logp_ref, r, beta):
    """E_pi[r] - beta KL(pi || pi_ref) of one prompt on a finite set of responses (eq-objective)."""
    logpi = jnp.log(pi)
    return jnp.sum(pi * r, axis=-1) - beta * jnp.sum(pi * (logpi - logp_ref), axis=-1)


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    """Eq. 7 of Rafailov et al. (eq-dpo), with the reference log-probabilities held fixed."""
    ref_w, ref_l = jax.lax.stop_gradient(ref_w), jax.lax.stop_gradient(ref_l)
    return -jnp.mean(jax.nn.log_sigmoid(beta * ((logp_w - ref_w) - (logp_l - ref_l))))
