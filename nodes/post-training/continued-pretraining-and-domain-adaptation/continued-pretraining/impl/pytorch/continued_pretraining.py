# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/continued-pretraining/ (ai://NEW-continued-pretraining).
"""Continued pretraining of a tiny RoPE decoder in PyTorch.

* TinyRoPEDecoder: a pre-norm decoder whose rotary base can be changed after training, as Code Llama and Xiong et
  al. do before long-context continued pretraining.
* rewarm_cosine_lambda: linear warmup into cosine decay, as a LambdaLR multiplier, for re-warming a finished
  checkpoint.
* extend_vocab_: grow the embedding and the output head in place with mean-initialized rows, as when a tokenizer is
  extended for a new language.
* fisher_diagonal and EWC: a penalty that anchors parameters to their old values in proportion to their Fisher
  information.
* continue_pretraining: the loop, with a fraction of every batch replayed from the old data.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_cache(d_head, max_len, base):
    inv = base ** (-torch.arange(0, d_head, 2, dtype=torch.get_default_dtype()) / d_head)
    ang = torch.outer(torch.arange(max_len, dtype=inv.dtype), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, heads, T, d_head); rotates consecutive pairs."""
    t = x.shape[-2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:t], sin[:t]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class TinyRoPEDecoder(nn.Module):
    def __init__(self, vocab, d_model=32, n_heads=4, d_ff=64, n_layers=2, max_len=64, rope_base=10000.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.d_head, self.max_len = d_model // n_heads, max_len
        self.set_rope_base(rope_base)

    def set_rope_base(self, base, max_len=None):
        """Change the rotary base (and optionally the cached length) without touching any weight."""
        self.rope_base = base
        self.max_len = max_len or self.max_len
        self.cos, self.sin = rope_cache(self.d_head, self.max_len, base)

    def hidden(self, ids):
        x = self.tok(ids)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return self.ln_f(x)

    def forward(self, ids):
        return self.head(self.hidden(ids))


def lm_loss(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def rewarm_cosine_lambda(warmup, total, min_ratio):
    """Multiplier of the peak learning rate: linear warmup over `warmup` steps, then cosine decay to min_ratio."""

    def f(step):
        if step < warmup:
            return step / max(warmup, 1)
        s = min((step - warmup) / max(total - warmup, 1), 1.0)
        return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * s)) / 2

    return f


@torch.no_grad()
def extend_vocab_(model, k):
    """Append k tokens: new embedding rows and new output rows are the mean of the old rows of each matrix. The old
    rows are copied unchanged, and before training the new tokens take at most k/(V + k) of the probability."""
    v, d = model.tok.weight.shape
    tok, head = nn.Embedding(v + k, d), nn.Linear(d, v + k, bias=False)
    tok.weight.copy_(torch.cat([model.tok.weight, model.tok.weight.mean(0).expand(k, d)]))
    head.weight.copy_(torch.cat([model.head.weight, model.head.weight.mean(0).expand(k, d)]))
    model.tok, model.head = tok, head
    return model


def fisher_diagonal(model, batches):
    """Empirical diagonal Fisher: mean over sequences of the squared gradient of each sequence's log-likelihood."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for ids in batches:
        for row in ids:
            model.zero_grad()
            lm_loss(model, row[None]).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
    model.zero_grad()
    return {n: f / count for n, f in fisher.items()}


class EWC:
    """(lam/2) sum_i F_i (theta_i - theta_old_i)^2 over the named parameters."""

    def __init__(self, model, fisher, lam):
        self.old = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.fisher, self.lam = fisher, lam

    def __call__(self, model):
        total = 0.0
        for n, p in model.named_parameters():
            total = total + (self.fisher[n] * (p - self.old[n]) ** 2).sum()
        return 0.5 * self.lam * total


def continue_pretraining(
    model,
    new,
    old=None,
    replay_fraction=0.0,
    steps=100,
    batch=16,
    lr=1e-2,
    warmup=10,
    min_ratio=0.1,
    ewc=None,
    generator=None,
):
    """AdamW on next-token loss over `new` (a LongTensor of sequences), re-warmed and re-decayed. A fraction of every
    batch is drawn uniformly from `old`; the rest walks through `new` in order. Returns the losses per step."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, rewarm_cosine_lambda(warmup, steps, min_ratio))
    n_rep = round(replay_fraction * batch) if old is not None else 0
    pos, losses = 0, []
    model.train()
    for _ in range(steps):
        idx = (pos + torch.arange(batch - n_rep)) % len(new)
        pos = (pos + batch - n_rep) % len(new)
        parts = [new[idx]]
        if n_rep:
            parts.append(old[torch.randint(len(old), (n_rep,), generator=generator)])
        loss = lm_loss(model, torch.cat(parts))
        if ewc is not None:
            loss = loss + ewc(model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    model.eval()
    return losses
