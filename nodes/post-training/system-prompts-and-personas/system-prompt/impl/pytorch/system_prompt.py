# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/system-prompt/ (ai://NEW-system-prompt).
"""System prompts on a tiny PyTorch decoder.

* A causal decoder whose attention can apply split-softmax: the attention on the first n_prompt positions is
  rescaled to pi^k at every layer and head, without retraining.
* cfg_logprobs: classifier-free guidance on the system prompt, from two forward passes.
* masked_lm_loss: the cross-entropy over the tokens a mask selects, which is how chat training ignores system and
  user tokens and how Ghost Attention ignores every turn but the last.
* persona_vector and steering through add_at, with projection for monitoring.
"""

import torch
import torch.nn.functional as F
from torch import nn


def split_softmax(attn, n_prompt, k):
    """Rescale attention rows (keys on the last axis) so the first n_prompt keys get pi^k of the mass.

    Rows where pi is 0 or 1 (for example queries inside the prompt, whose causal window holds only prompt keys) are
    left unchanged."""
    pi = attn[..., :n_prompt].sum(-1, keepdim=True)
    ok = (pi > 0) & (pi < 1)
    p = torch.where(ok, pi, torch.full_like(pi, 0.5))
    scale_p = torch.where(ok, p**k / p, torch.ones_like(p))
    scale_r = torch.where(ok, (1 - p**k) / (1 - p), torch.ones_like(p))
    return torch.cat([attn[..., :n_prompt] * scale_p, attn[..., n_prompt:] * scale_r], dim=-1)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.mlp_in = nn.Linear(d_model, d_ff)
        self.mlp_out = nn.Linear(d_ff, d_model)

    def attention(self, x, n_prompt, k, record):
        b, t, d = x.shape
        h = self.n_heads
        q, kk, v = self.qkv(x).view(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)
        scores = q @ kk.transpose(-1, -2) / (d // h) ** 0.5
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        attn = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        if n_prompt and k is not None:
            attn = split_softmax(attn, n_prompt, k)
        if record is not None:
            record.append(attn)
        y = attn @ v
        return self.attn_out(y.transpose(1, 2).reshape(b, t, d))

    def forward(self, x, n_prompt=0, k=None, record=None):
        x = x + self.attention(self.ln1(x), n_prompt, k, record)
        return x + self.mlp_out(F.gelu(self.mlp_in(self.ln2(x))))


class TinyDecoder(nn.Module):
    def __init__(self, vocab, d_model=32, n_heads=4, d_ff=64, n_layers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids, n_prompt=0, k=None, add_at=None, record=None, return_stream=False):
        """Logits. n_prompt and k turn on split-softmax; add_at = (layer, vector) adds the vector to the residual
        stream entering that block at every position; record, a list, collects every layer's attention."""
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        stream = [x]
        for i, block in enumerate(self.blocks):
            if add_at is not None and add_at[0] == i:
                x = x + add_at[1]
            x = block(x, n_prompt, k, record)
            stream.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, stream) if return_stream else logits


def cfg_logprobs(model, system_ids, context_ids, alpha):
    """Next-token log-probabilities after context_ids, guided towards the system prompt:
    log_softmax(log p_u + alpha (log p_c - log p_u)), where p_c sees system_ids + context_ids and p_u context_ids."""
    lc = F.log_softmax(model(torch.cat([system_ids, context_ids], dim=1))[:, -1], dim=-1)
    lu = F.log_softmax(model(context_ids)[:, -1], dim=-1)
    return F.log_softmax(lu + alpha * (lc - lu), dim=-1)


def masked_lm_loss(model, ids, loss_mask):
    """Mean next-token cross-entropy over the positions where loss_mask (aligned with ids) is 1.

    The mask refers to the predicted token: position j contributes -log p(ids[j] | ids[<j]) when loss_mask[j] = 1."""
    logits = model(ids)[:, :-1]
    target = ids[:, 1:]
    m = loss_mask[:, 1:].to(logits.dtype)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return (nll * m.reshape(-1)).sum() / m.sum()


@torch.no_grad()
def persona_vector(model, with_trait, without_trait, layer):
    """Mean residual activation entering block `layer`, averaged over response positions, on responses with the trait
    minus on responses without it. Each item is (ids, response_start)."""

    def mean(items):
        acts = [model(ids, return_stream=True)[1][layer][0, start:].mean(0) for ids, start in items]
        return torch.stack(acts).mean(0)

    return mean(with_trait) - mean(without_trait)


@torch.no_grad()
def trait_projection(model, ids, v, layer):
    """Projection of the last token's activation entering block `layer` onto v / |v|."""
    return model(ids, return_stream=True)[1][layer][:, -1] @ (v / v.norm())
