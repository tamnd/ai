# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/instruction-tuning/ (ai://NEW-instruction-tuning).
"""The training side of instruction tuning in PyTorch.

Instruction tuning fine-tunes a pretrained language model by maximum likelihood on instructions paired with
responses. This file has the parts that differ from pretraining:

* `masked_nll`, the cross-entropy over response tokens only, averaged over tokens or over sequences, and
  `ignore_index_labels`, the equivalent form most libraries use, with prompt positions set to -100;
* `collate`, which pads encoded examples into a batch with a loss mask;
* `segment_attention_mask` and `segment_positions` for packed rows, so that examples sharing a row do not attend to
  each other and each starts at position 0;
* `TinyCausalLM`, a one-block decoder used by the tests, which accepts such a mask and such positions.

Encoding an example into inputs, targets and mask is the same as in the NumPy file: the sequence is prompt + response +
eos, position t predicts token t + 1, and the mask is 1 where that token is part of the response or the closing eos.
"""

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100


def encode(prompt_ids, response_ids, eos_id):
    """Inputs, next-token targets and loss mask (as lists) for one example."""
    prompt_ids, response_ids = list(prompt_ids), list(response_ids)
    if not prompt_ids:
        raise ValueError("the prompt must have at least one token")
    seq = prompt_ids + response_ids + [eos_id]
    mask = [0.0] * (len(prompt_ids) - 1) + [1.0] * (len(response_ids) + 1)
    return seq[:-1], seq[1:], mask


def collate(encoded, pad_id):
    """Right-pad a list of (inputs, targets, mask) into tensors of shape (B, T); padding has mask 0."""
    t = max(len(x[0]) for x in encoded)
    b = len(encoded)
    inputs = torch.full((b, t), pad_id, dtype=torch.long)
    targets = torch.full((b, t), pad_id, dtype=torch.long)
    mask = torch.zeros((b, t))
    for i, (x, y, m) in enumerate(encoded):
        n = len(x)
        inputs[i, :n] = torch.tensor(x)
        targets[i, :n] = torch.tensor(y)
        mask[i, :n] = torch.tensor(m, dtype=mask.dtype)
    return inputs, targets, mask


def masked_nll(logits, targets, mask, reduction="token"):
    """Mean cross-entropy over the positions where mask is 1.

    "token" averages over every unmasked token of the batch; "sequence" averages within each sequence, then over the
    sequences that have at least one unmasked token."""
    nll = F.cross_entropy(logits.transpose(1, -1), targets, reduction="none")
    mask = mask.to(nll.dtype)
    if reduction == "token":
        total = mask.sum()
        if total == 0:
            raise ValueError("the mask selects no tokens")
        return (mask * nll).sum() / total
    if reduction == "sequence":
        per = mask.sum(dim=-1)
        keep = per > 0
        if not keep.any():
            raise ValueError("the mask selects no tokens")
        seq = (mask * nll).sum(dim=-1)[keep] / per[keep]
        return seq.mean()
    raise ValueError(f"unknown reduction {reduction!r}")


def ignore_index_labels(targets, mask):
    """Targets with masked positions replaced by -100, for F.cross_entropy(..., ignore_index=-100)."""
    return torch.where(mask > 0, targets, torch.full_like(targets, IGNORE_INDEX))


def segment_attention_mask(segments):
    """Boolean (B, T, T) mask: i may attend to j when j <= i and both are in the same nonzero segment.

    Padding rows (segment 0) get attention to themselves only, so softmax stays defined; their loss is masked."""
    t = segments.shape[-1]
    causal = torch.ones(t, t, dtype=torch.bool, device=segments.device).tril()
    same = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] > 0)
    allowed = causal & same
    eye = torch.eye(t, dtype=torch.bool, device=segments.device)
    return allowed | (eye & (segments[:, :, None] == 0))


def segment_positions(segments):
    """Position ids that restart at 0 at the start of every segment (0 on padding)."""
    b, t = segments.shape
    idx = torch.arange(t, device=segments.device).expand(b, t)
    start = torch.ones_like(segments, dtype=torch.bool)
    start[:, 1:] = segments[:, 1:] != segments[:, :-1]
    first = torch.where(start, idx, torch.zeros_like(idx)).cummax(dim=-1).values
    return torch.where(segments > 0, idx - first, torch.zeros_like(idx))


class TinyCausalLM(nn.Module):
    """One pre-norm decoder block with learned positions; small enough to train in a test."""

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
        self.out = nn.Linear(dim, vocab)

    def forward(self, inputs, attn_mask=None, positions=None):
        b, t = inputs.shape
        if positions is None:
            positions = torch.arange(t, device=inputs.device).expand(b, t)
        if attn_mask is None:
            attn_mask = torch.ones(t, t, dtype=torch.bool, device=inputs.device).tril().expand(b, t, t)
        x = self.tok(inputs) + self.pos(positions)
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        d = q.shape[-1] // self.heads
        q, k, v = (z.view(b, t, self.heads, d).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None])
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, -1))
        x = x + self.mlp(self.ln2(x))
        return self.out(self.ln_f(x))
