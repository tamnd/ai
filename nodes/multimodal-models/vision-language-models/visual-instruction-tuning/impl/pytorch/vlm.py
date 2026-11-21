# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/visual-instruction-tuning/ (ai://NEW-visual-instruction-tuning).
"""A small LLaVA-style model: a frozen image encoder, a trainable connector (linear or two-layer MLP) that makes soft
visual tokens, and a frozen causal Transformer language model. Training follows LLaVA's two stages. Stage 1 trains only
the connector on captions; stage 2 also unfreezes the language model on instruction data. The module also holds
Flamingo's tanh-gated cross-attention layer, a Perceiver Resampler, and the POPE metrics.

The "images" are bags of noisy patch vectors, one prototype per object word. The language model is pretrained on text
before it is frozen, so it has word embeddings the connector can aim at.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class TinyLM(nn.Module):
    """A causal Transformer that reads embeddings, so soft visual tokens can be prepended to word embeddings."""

    def __init__(self, vocab, d=64, heads=4, layers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(max_len, d))
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.out.weight = self.tok.weight

    def forward_embeddings(self, h):
        t = h.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        z = self.blocks(h + self.pos[:t], mask=mask, is_causal=True)
        return self.out(self.norm(z))

    def forward(self, ids):
        return self.forward_embeddings(self.tok(ids))


class Connector(nn.Module):
    """LLaVA's linear projection H_v = W Z_v, or LLaVA-1.5's two-layer MLP."""

    def __init__(self, d_img, d, kind="linear"):
        super().__init__()
        self.net = nn.Linear(d_img, d) if kind == "linear" else nn.Sequential(nn.Linear(d_img, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, z):
        return self.net(z)


class VLM(nn.Module):
    def __init__(self, lm, encoder, connector):
        super().__init__()
        self.lm, self.encoder, self.connector = lm, encoder, connector

    def forward(self, patches, ids):
        """Logits over [visual tokens, text tokens]; the text logits are the last ids.shape[1] positions."""
        hv = self.connector(self.encoder(patches))
        h = torch.cat([hv, self.lm.tok(ids)], dim=1)
        return self.lm.forward_embeddings(h)[:, hv.shape[1] :]


def freeze(m, frozen=True):
    for p in m.parameters():
        p.requires_grad_(not frozen)


def answer_loss(logits, ids, answer_from):
    """Next-token loss on the answer only: positions answer_from-1 .. predict tokens answer_from .. end."""
    pred = logits[:, answer_from - 1 : -1]
    return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), ids[:, answer_from:].reshape(-1))


def train(model, batches, steps, lr, answer_from):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    losses = []
    for s in range(steps):
        patches, ids = batches(s)
        loss = answer_loss(model(patches, ids), ids, answer_from)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class GatedXAttn(nn.Module):
    """y + tanh(a_attn) * xattn(y, x), then y + tanh(a_ffw) * ffw(y), with both gates initialised at 0."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffw = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.alpha_attn = nn.Parameter(torch.zeros(()))
        self.alpha_ffw = nn.Parameter(torch.zeros(()))

    def forward(self, y, x, mask=None):
        a, _ = self.attn(self.norm1(y), x, x, attn_mask=mask, need_weights=False)
        y = y + torch.tanh(self.alpha_attn) * a
        return y + torch.tanh(self.alpha_ffw) * self.ffw(self.norm2(y))


class PerceiverResampler(nn.Module):
    """num_latents learned queries attend to [features, latents]; output size does not depend on the input length."""

    def __init__(self, d, num_latents=8, heads=4, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d) / math.sqrt(d))
        self.attn = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers))
        self.ffw = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        lat = self.latents.expand(x.shape[0], -1, -1)
        for attn, ffw in zip(self.attn, self.ffw):
            kv = torch.cat([x, lat], dim=1)
            lat = lat + attn(lat, kv, kv, need_weights=False)[0]
            lat = lat + ffw(lat)
        return self.norm(lat)


def pope_metrics(pred_yes, label_yes):
    p, t = torch.as_tensor(pred_yes, dtype=torch.bool), torch.as_tensor(label_yes, dtype=torch.bool)
    tp, fp, fn = (p & t).sum().item(), (p & ~t).sum().item(), (~p & t).sum().item()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (p == t).float().mean().item(), "precision": prec, "recall": rec, "f1": f1, "yes": p.float().mean().item()}
