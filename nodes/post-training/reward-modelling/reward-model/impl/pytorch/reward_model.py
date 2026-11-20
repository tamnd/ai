# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-model/ (ai://NEW-reward-model).
"""A reward model and its training losses in PyTorch.

A reward model is a language model whose next-token head is replaced by a linear head that outputs one number,
read at the last token of the prompt and response. It is trained on comparisons with the Bradley-Terry loss
-log sigma(r_w - r_l). This file has:

* `TinyRewardModel`, a one-block causal transformer with a scalar head read at the last non-padding token, so that
  right padding does not change the reward;
* `pairwise_loss`, with Llama 2's margin and Christiano et al.'s label noise, written with F.logsigmoid;
* `ranked_loss`, InstructGPT's loss over all pairs of K ranked responses to each prompt, batched over prompts with
  different K by a mask, each prompt's loss divided by its number of pairs;
* `ensemble`, which combines member rewards by mean, minimum, or mean minus lambda times variance.
"""

import torch
import torch.nn.functional as F
from torch import nn


def last_token_index(attention_mask):
    """Index of the last position where attention_mask is 1, for right-padded rows of shape (B, T)."""
    lengths = attention_mask.long().sum(dim=-1)
    if (lengths == 0).any():
        raise ValueError("every row needs at least one token")
    return lengths - 1


class TinyRewardModel(nn.Module):
    """One pre-norm causal decoder block with learned positions and a scalar reward head."""

    def __init__(self, vocab, dim=32, heads=4, max_len=64):
        super().__init__()
        self.heads = heads
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)  # replaces the unembedding of a language model

    def hidden(self, inputs):
        b, t = inputs.shape
        x = self.tok(inputs) + self.pos(torch.arange(t, device=inputs.device).expand(b, t))
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        d = q.shape[-1] // self.heads
        q, k, v = (z.view(b, t, self.heads, d).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, -1))
        x = x + self.mlp(self.ln2(x))
        return self.ln_f(x)

    def forward(self, inputs, attention_mask=None):
        """Rewards of shape (B,) for right-padded token ids of shape (B, T)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        h = self.hidden(inputs)
        idx = last_token_index(attention_mask)
        return self.head(h[torch.arange(len(idx), device=inputs.device), idx]).squeeze(-1)


def pairwise_loss(r_w, r_l, margin=0.0, eps=0.0):
    """Mean of -log(eps / 2 + (1 - eps) sigma(r_w - r_l - margin)); with eps = 0, of -log sigma(r_w - r_l - margin)."""
    d = r_w - r_l - margin
    if eps == 0.0:
        return -F.logsigmoid(d).mean()
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in [0, 1)")
    # log(eps/2 + (1 - eps) sigma(d)) = logaddexp(log(eps/2), log(1 - eps) + log sigma(d)), stable for large |d|.
    a = torch.full_like(d, torch.log(torch.tensor(eps / 2)).item())
    b = torch.log(torch.tensor(1 - eps)).item() + F.logsigmoid(d)
    return -torch.logaddexp(a, b).mean()


def ranked_loss(rewards, counts):
    """InstructGPT's loss for a batch of rankings.

    rewards has shape (B, K_max): row b holds the rewards of the counts[b] responses to prompt b, best first, then
    padding. Each prompt's loss is the mean of -log sigma(r_i - r_j) over its counts[b](counts[b] - 1)/2 pairs i < j,
    and the batch loss is the mean over prompts."""
    _, k = rewards.shape
    idx = torch.arange(k, device=rewards.device)
    valid = idx[None, :] < counts[:, None]
    pair = (idx[:, None] < idx[None, :])[None] & valid[:, :, None] & valid[:, None, :]
    diff = rewards[:, :, None] - rewards[:, None, :]
    per_pair = torch.where(pair, -F.logsigmoid(diff), torch.zeros_like(diff))
    npairs = pair.sum(dim=(1, 2))
    if (npairs == 0).any():
        raise ValueError("every prompt needs at least two responses")
    return (per_pair.sum(dim=(1, 2)) / npairs).mean()


def ensemble(member_rewards, method="mean", lam=0.0):
    """Combine rewards of shape (k, ...) from k members: "mean", "min", or "uwo" (mean - lam * population variance)."""
    if method == "mean":
        return member_rewards.mean(dim=0)
    if method == "min":
        return member_rewards.min(dim=0).values
    if method == "uwo":
        return member_rewards.mean(dim=0) - lam * member_rewards.var(dim=0, correction=0)
    raise ValueError(f"unknown method {method!r}")
