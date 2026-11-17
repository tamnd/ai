# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://NEW-sparse-mixture-of-experts).
"""Sparse mixture of experts in NumPy. Tokens are rows, experts columns.

top_k_gates is eq-noisy-top-k and noisy_logits eq-noisy-logits; expert_capacity is eq-capacity and dispatch fills
expert slots first choices first, dropping choices past capacity; switch_balance_loss is eq-switch-balance,
importance_loss and load_loss the losses of Shazeer et al., z_loss eq-z-loss; moe_layer puts them together."""

import math

import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))[..., 0]


def softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def top_k_indices(logits, k):
    """Indices of the k largest logits of each row, largest first; ties go to the lower index."""
    logits = np.asarray(logits, dtype=np.float64)
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k must be in [1, {logits.shape[-1]}], got {k}")
    order = np.argsort(-logits, axis=-1, kind="stable")
    return order[..., :k]


def keep_top_k(v, k):
    """KeepTopK of Shazeer et al.: the top k entries of each row stay, the others become minus infinity."""
    v = np.asarray(v, dtype=np.float64)
    idx = top_k_indices(v, k)
    out = np.full_like(v, -np.inf)
    np.put_along_axis(out, idx, np.take_along_axis(v, idx, axis=-1), axis=-1)
    return out


def top_k_gates(logits, k):
    """Softmax(KeepTopK(logits, k)): a gate vector with k nonzero entries that sum to one."""
    return softmax(keep_top_k(logits, k))


def noisy_logits(x, w_gate, w_noise, noise):
    """H(x) = x W_g + noise * softplus(x W_noise), with the standard normal noise passed in."""
    x = np.asarray(x, dtype=np.float64)
    return x @ w_gate + np.asarray(noise, dtype=np.float64) * softplus(x @ w_noise)


def expert_capacity(tokens, experts, capacity_factor, k=1):
    """Slots per expert: k * tokens / experts * capacity_factor, rounded down."""
    if tokens < 0 or experts < 1:
        raise ValueError("tokens must be >= 0 and experts >= 1")
    if not capacity_factor > 0.0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    return int(math.floor(k * tokens * capacity_factor / experts + 1e-9))


def dispatch(indices, n_experts, capacity):
    """Assign each (token, choice) to a slot of its expert.

    indices has shape (T, k), the chosen experts of each token, best first. Choices are taken rank by rank, and within a
    rank in token order, as in Algorithm 1 of GShard; an expert takes a choice while it has fewer than capacity tokens.
    Returns kept, a (T, k) boolean array, and position, the slot of each kept choice (-1 for a dropped one)."""
    indices = np.asarray(indices)
    t, k = indices.shape
    count = np.zeros(n_experts, dtype=np.int64)
    kept = np.zeros((t, k), dtype=bool)
    position = np.full((t, k), -1, dtype=np.int64)
    for r in range(k):
        for s in range(t):
            e = indices[s, r]
            if count[e] < capacity:
                kept[s, r] = True
                position[s, r] = count[e]
            count[e] += 1
    return kept, position


def tokens_per_expert(indices, n_experts):
    """How many choices each expert receives before any are dropped."""
    return np.bincount(np.asarray(indices).ravel(), minlength=n_experts)


def dropped_choices(indices, n_experts, capacity):
    """Choices that overflow: the sum over experts of max(0, received - capacity)."""
    return int(np.maximum(tokens_per_expert(indices, n_experts) - capacity, 0).sum())


def switch_balance_loss(probs, alpha=1e-2):
    """alpha * N * sum_i f_i * P_i, with f_i the fraction of tokens whose argmax is i and P_i the mean probability."""
    probs = np.asarray(probs, dtype=np.float64)
    t, n = probs.shape
    f = np.bincount(np.argmax(probs, axis=-1), minlength=n) / t
    p = probs.mean(axis=0)
    return alpha * n * float(f @ p)


def switch_balance_grad(probs, alpha=1e-2):
    """Gradient of the balance loss in the router probabilities, with f held constant: alpha * N * f_i / T."""
    probs = np.asarray(probs, dtype=np.float64)
    t, n = probs.shape
    f = np.bincount(np.argmax(probs, axis=-1), minlength=n) / t
    return np.broadcast_to(alpha * n * f / t, probs.shape).copy()


def cv_squared(v):
    """Squared coefficient of variation, population variance over squared mean; 0 for a constant vector."""
    v = np.asarray(v, dtype=np.float64)
    mean = v.mean()
    if mean == 0.0:
        return 0.0
    return float(v.var() / mean**2)


def importance_loss(gates, w_importance=0.1):
    """w_importance * CV(sum over the batch of the gate vectors)^2."""
    return w_importance * cv_squared(np.asarray(gates, dtype=np.float64).sum(axis=0))


def _normal_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def load_probability(clean, noisy, noise_std, k):
    """P(x, i), the chance that expert i is in the top k when only its noise is redrawn: clean is x W_g, noisy is H(x),
    noise_std is softplus(x W_noise); the threshold is the k-th largest entry of H(x) with i left out."""
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    t, n = noisy.shape
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}] for the load estimate, got {k}")
    top = -np.sort(-noisy, axis=-1)[:, : k + 1]
    in_top = noisy >= top[:, k - 1 : k]
    # Leaving i out moves the k-th largest to the (k+1)-th when i is among the top k.
    threshold = np.where(in_top, top[:, k : k + 1], top[:, k - 1 : k])
    return _normal_cdf((clean - threshold) / noise_std)


def load_loss(clean, noisy, noise_std, k, w_load=0.1):
    """w_load * CV(sum over the batch of P(x, i))^2."""
    return w_load * cv_squared(load_probability(clean, noisy, noise_std, k).sum(axis=0))


def z_loss(logits):
    """Mean over tokens of the squared log-sum-exp of the router logits."""
    return float(np.mean(logsumexp(logits) ** 2))


def expert_ffn(x, w_in, w_out):
    """A two-layer ReLU expert, x W_in -> ReLU -> W_out."""
    return np.maximum(x @ w_in, 0.0) @ w_out


def moe_layer(x, w_router, w_in, w_out, k=1, capacity_factor=1.0, renormalize=None):
    """A sparse MoE layer with top-k routing and expert capacity.

    x is (T, d), w_router (d, N), w_in (N, d, h), w_out (N, h, d). With k = 1 the gate is the router probability, as in
    Switch Transformers; with k > 1 the top k probabilities are renormalized (eq-noisy-top-k without noise). A dropped
    choice adds nothing, so a fully dropped token gets zero and passes on through the residual added by the caller."""
    x = np.asarray(x, dtype=np.float64)
    t = x.shape[0]
    n = w_router.shape[1]
    if renormalize is None:
        renormalize = k > 1
    logits = x @ w_router
    probs = softmax(logits)
    idx = top_k_indices(logits, k)
    gate = np.take_along_axis(probs, idx, axis=-1)
    if renormalize:
        gate = gate / gate.sum(axis=-1, keepdims=True)
    cap = expert_capacity(t, n, capacity_factor, k)
    kept, _ = dispatch(idx, n, cap)
    y = np.zeros_like(x)
    for e in range(n):
        rows, ranks = np.nonzero((idx == e) & kept)
        if rows.size:
            y[rows] += gate[rows, ranks][:, None] * expert_ffn(x[rows], w_in[e], w_out[e])
    info = {
        "probs": probs,
        "indices": idx,
        "gates": gate,
        "kept": kept,
        "capacity": cap,
        "dropped": int((~kept).sum()),
        "balance_loss": switch_balance_loss(probs),
        "z_loss": z_loss(logits),
    }
    return y, info
