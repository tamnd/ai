# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-relative-policy-optimization/ (ai://NEW-group-relative-policy-optimization).
"""Group relative policy optimization (GRPO) with a verifiable reward, in NumPy.

The pieces of GRPO are plain functions: group-relative advantages, the leave-one-out baseline they rescale, the
clipped surrogate with separate lower and upper ranges, the k3 estimate of the KL divergence, and the three ways of
reducing per-token losses. A tabular policy writes an L-token answer to each of Q questions; a checker gives reward 1
when the last token equals the question's answer. `grpo_step` samples a group of G outputs per question from the old
policy, computes advantages, and takes `mu` gradient-ascent steps on the clipped objective minus beta times KL.
"""

import numpy as np


def group_advantages(rewards, normalize=True, eps=1e-6):
    """(r - mean) / (std + eps) along the last axis (the group); without normalize, only centred."""
    r = np.asarray(rewards, dtype=float)
    a = r - r.mean(-1, keepdims=True)
    return a / (r.std(-1, keepdims=True) + eps) if normalize else a


def leave_one_out_advantages(rewards):
    """r_i minus the mean of the other G - 1 rewards (RLOO)."""
    r = np.asarray(rewards, dtype=float)
    G = r.shape[-1]
    return r - (r.sum(-1, keepdims=True) - r) / (G - 1)


def kl_k3(logp, logp_ref):
    """Per-token estimate x - log x - 1 of KL(pi || ref), with x = ref/pi at a token sampled from pi."""
    d = np.asarray(logp_ref) - np.asarray(logp)
    return np.exp(d) - d - 1


def clipped_surrogate(ratio, adv, eps_low=0.2, eps_high=None):
    """min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A) and its derivative with respect to ratio."""
    eps_high = eps_low if eps_high is None else eps_high
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high) * adv
    value = np.minimum(unclipped, clipped)
    inside = (ratio >= 1 - eps_low) & (ratio <= 1 + eps_high)
    grad = np.where((unclipped <= clipped) | inside, adv, 0.0)
    return value, grad


def token_weights(lengths, adv, mode="sequence", max_len=None):
    """Weight of each token of output i in the loss: GRPO 'sequence', DAPO 'token', Dr. GRPO 'constant'."""
    lengths, adv = np.asarray(lengths, dtype=float), np.asarray(adv, dtype=float)
    G = len(lengths)
    if mode == "sequence":
        return adv / (G * lengths)
    if mode == "token":
        return adv / lengths.sum()
    if mode == "constant":
        return adv / (G * max_len)
    raise ValueError(mode)


def informative_probability(q, G):
    """Chance that G independent binary rewards with success q are not all equal."""
    return 1 - q**G - (1 - q) ** G


def dynamic_sampling_mask(rewards):
    """DAPO's filter: keep groups whose rewards are not all equal."""
    r = np.asarray(rewards)
    return r.max(-1) > r.min(-1)


def pass_at_k(p, k):
    return 1 - (1 - np.asarray(p, dtype=float)) ** k


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def sample(logits, q, G, rng):
    """G outputs for each question in q from a policy whose position-t token depends on (question, t)."""
    p = np.exp(log_softmax(logits[q]))  # (B, L, V)
    u = rng.random((len(q), G, p.shape[1], 1))
    return (u > np.cumsum(p, -1)[:, None]).sum(-1)  # (B, G, L)


def token_logp(logits, q, out):
    lp = log_softmax(logits[q])  # (B, L, V)
    B, _, L = out.shape
    return lp[np.arange(B)[:, None, None], np.arange(L), out], lp


def verify(out, answers, q):
    """Verifiable reward: 1 if the last token is the answer, else 0."""
    return (out[..., -1] == answers[q][:, None]).astype(float)


def objective_and_grad(logits, ref, q, out, adv, old_lp, eps_low, eps_high, beta):
    """GRPO objective (sequence-mean reduction, fixed length) and its gradient with respect to the logits."""
    B, G, L = out.shape
    lp, full = token_logp(logits, q, out)
    ratio = np.exp(lp - old_lp)
    A = np.broadcast_to(adv[..., None], lp.shape)
    surr, dsurr = clipped_surrogate(ratio, A, eps_low, eps_high)
    ref_lp, _ = token_logp(ref, q, out)
    x = np.exp(ref_lp - lp)
    w = 1.0 / (B * G * L)
    J = w * (surr - beta * (x - np.log(x) - 1)).sum()
    coef = w * (dsurr * ratio - beta * (1 - x))  # d J / d log pi(token)
    onehot = np.eye(logits.shape[-1])[out]
    g = coef[..., None] * (onehot - np.exp(full)[:, None])  # d log pi / d logits = onehot - pi
    grad = np.zeros_like(logits)
    np.add.at(grad, q, g.sum(1))
    return J, grad


def grpo_step(logits, ref, answers, rng, G=8, batch=8, lr=2.0, mu=2, eps_low=0.2, eps_high=0.2, beta=0.01):
    """One GRPO step: sample groups from the old policy, score, then mu updates. Returns new logits and rewards."""
    q = rng.choice(len(answers), size=batch, replace=False)
    out = sample(logits, q, G, rng)
    r = verify(out, answers, q)
    adv = group_advantages(r)
    old_lp, _ = token_logp(logits, q, out)
    for _ in range(mu):
        _, grad = objective_and_grad(logits, ref, q, out, adv, old_lp, eps_low, eps_high, beta)
        logits = logits + lr * grad * len(answers) / batch
    return logits, r


def accuracy(logits, answers):
    """Exact probability that the last token is right, averaged over questions."""
    p = np.exp(log_softmax(logits[:, -1]))
    return float(p[np.arange(len(answers)), answers].mean())
