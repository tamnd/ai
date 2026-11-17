# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://1H67K0).
"""Sparse mixture of experts in PyTorch.

top_k_route picks each token's k experts and their gates: the router probability for k = 1, as in Switch Transformers,
and the top k probabilities renormalized for k > 1, which is the softmax of KeepTopK of eq-noisy-top-k. dispatch_mask
builds the (T, N, C) one-hot dispatch tensor with first choices before second choices and drops choices past the
capacity of eq-capacity; combine weights are the dispatch mask times the gates. switch_balance_loss is eq-switch-balance
with f detached, z_loss is eq-z-loss, and importance_loss is the coefficient-of-variation loss of Shazeer et al.
SparseMoE is a layer of two-layer ReLU experts with noisy or plain top-k routing and the diagnostics in a dict.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def expert_capacity(tokens, experts, capacity_factor, k=1):
    """Slots per expert: k * tokens / experts * capacity_factor, rounded down."""
    if tokens < 0 or experts < 1:
        raise ValueError("tokens must be >= 0 and experts >= 1")
    if not capacity_factor > 0.0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    return int(math.floor(k * tokens * capacity_factor / experts + 1e-9))


def top_k_route(logits, k, renormalize=None):
    """Return (probs, indices, gates) with indices and gates of shape (T, k), best expert first."""
    n = logits.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if renormalize is None:
        renormalize = k > 1
    probs = torch.softmax(logits, dim=-1)
    gates, indices = torch.topk(probs, k, dim=-1)
    if renormalize:
        gates = gates / gates.sum(dim=-1, keepdim=True)
    return probs, indices, gates


def dispatch_mask(indices, n_experts, capacity):
    """A (T, k, N, C) boolean tensor: choice r of token s goes to slot c of expert e.

    Positions are counted rank by rank, and within a rank in token order, so every first choice is placed before any
    second choice; a choice whose position is capacity or more is dropped."""
    t, k = indices.shape
    one_hot = F.one_hot(indices, n_experts)  # (T, k, N)
    flat = one_hot.transpose(0, 1).reshape(k * t, n_experts)  # rank-major order
    position = (torch.cumsum(flat, dim=0) - 1) * flat  # slot of each choice in its expert
    position = position.reshape(k, t, n_experts).transpose(0, 1)  # (T, k, N)
    if capacity == 0:
        return torch.zeros(t, k, n_experts, 0, dtype=torch.bool, device=indices.device)
    keep = (one_hot > 0) & (position < capacity)
    slots = F.one_hot(position.clamp(min=0, max=capacity - 1), capacity) > 0
    return slots & keep.unsqueeze(-1)


def switch_balance_loss(probs, alpha=1e-2):
    """alpha * N * sum_i f_i * P_i; f counts the argmax choices and carries no gradient, P is the mean probability."""
    t, n = probs.shape
    f = torch.bincount(probs.argmax(dim=-1), minlength=n).to(probs.dtype) / t
    p = probs.mean(dim=0)
    return alpha * n * torch.sum(f.detach() * p)


def cv_squared(v):
    """Squared coefficient of variation (population variance); 0 for a vector with zero mean."""
    mean = v.mean()
    if mean.item() == 0.0:
        return v.sum() * 0.0
    return v.var(unbiased=False) / mean**2


def importance_loss(gates, w_importance=0.1):
    """w_importance * CV(sum over the batch of the gate vectors)^2."""
    return w_importance * cv_squared(gates.sum(dim=0))


def z_loss(logits):
    """Mean over tokens of the squared log-sum-exp of the router logits."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


class SparseMoE(nn.Module):
    """Top-k routed mixture of two-layer ReLU experts with capacity and token dropping.

    forward returns (y, info). y is zero for a token whose every choice was dropped; add the residual outside."""

    def __init__(self, d_model, d_hidden, n_experts, k=1, capacity_factor=1.0, noisy=False, renormalize=None):
        super().__init__()
        if not 1 <= k <= n_experts:
            raise ValueError(f"k must be in [1, {n_experts}], got {k}")
        self.k = k
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.noisy = noisy
        self.renormalize = renormalize
        self.w_router = nn.Parameter(torch.zeros(d_model, n_experts))
        self.w_noise = nn.Parameter(torch.zeros(d_model, n_experts)) if noisy else None
        self.w_in = nn.Parameter(torch.randn(n_experts, d_model, d_hidden) / math.sqrt(d_model))
        self.w_out = nn.Parameter(torch.randn(n_experts, d_hidden, d_model) / math.sqrt(d_hidden))

    def router_logits(self, x):
        clean = x @ self.w_router
        if self.noisy and self.training:
            return clean + torch.randn_like(clean) * F.softplus(x @ self.w_noise)
        return clean

    def forward(self, x):
        t = x.shape[0]
        logits = self.router_logits(x)
        probs, indices, gates = top_k_route(logits, self.k, self.renormalize)
        cap = expert_capacity(t, self.n_experts, self.capacity_factor, self.k)
        mask = dispatch_mask(indices, self.n_experts, cap)  # (T, k, N, C)
        dispatch = mask.any(dim=1).to(x.dtype)  # (T, N, C)
        combine = (mask.to(x.dtype) * gates[:, :, None, None]).sum(dim=1)  # (T, N, C)
        expert_in = torch.einsum("tnc,td->ncd", dispatch, x)
        hidden = torch.relu(torch.einsum("ncd,ndh->nch", expert_in, self.w_in))
        expert_out = torch.einsum("nch,nhd->ncd", hidden, self.w_out)
        y = torch.einsum("tnc,ncd->td", combine, expert_out)
        kept = mask.any(dim=(2, 3))
        info = {
            "probs": probs,
            "indices": indices,
            "gates": gates,
            "kept": kept,
            "capacity": cap,
            "dropped": int((~kept).sum()),
            "balance_loss": switch_balance_loss(probs),
            "z_loss": z_loss(logits),
        }
        return y, info
