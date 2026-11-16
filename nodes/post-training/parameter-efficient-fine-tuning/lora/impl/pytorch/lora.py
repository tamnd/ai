# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""A linear layer with a low-rank adapter (LoRA), the PyTorch reference.

LoRALinear wraps a pretrained torch.nn.Linear, freezes it, and adds s B A to its weight: eq-lora-forward of the page,
with s = alpha / r (eq-lora-scale), or s = alpha / sqrt(r) for the rank-stabilized variant. Autograd gives the
backward pass, and the tests compare it with eq-lora-grads and with the NumPy reference.
"""

import math

import torch


class LoRALinear(torch.nn.Module):
    """y = base(x) + s (x A^T) B^T, with base frozen, A of shape (r, in_features) and B of shape (out_features, r)."""

    def __init__(self, base, r=8, alpha=None, rank_stabilized=False, std=None):
        super().__init__()
        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"base must be a torch.nn.Linear, got {type(base).__name__}")
        d_in, d_out = base.in_features, base.out_features
        r = int(r)
        if not 1 <= r <= min(d_in, d_out):
            raise ValueError(f"r must be in [1, min(d_in, d_out)] = [1, {min(d_in, d_out)}], got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = float(r if alpha is None else alpha)
        self.rank_stabilized = bool(rank_stabilized)
        self.scale = self.alpha / math.sqrt(r) if rank_stabilized else self.alpha / r
        std = 1.0 / math.sqrt(d_in) if std is None else float(std)
        w = base.weight
        # A is Gaussian and B is zero, so the layer starts as the pretrained layer.
        self.lora_A = torch.nn.Parameter(torch.randn(r, d_in, dtype=w.dtype, device=w.device) * std)
        self.lora_B = torch.nn.Parameter(torch.zeros(d_out, r, dtype=w.dtype, device=w.device))

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)

    def delta(self):
        """The update s B A, of the shape of the base weight."""
        return self.scale * (self.lora_B @ self.lora_A)

    def num_trainable(self):
        """The number of trainable entries: r (in_features + out_features), eq-lora-count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def merged(self):
        """Return a new torch.nn.Linear with the weight W0 + s B A, which gives the same outputs (eq-lora-merge)."""
        out = torch.nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                              dtype=self.base.weight.dtype, device=self.base.weight.device)
        out.weight.copy_(self.base.weight + self.delta())
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def add_lora(model, names, r=8, alpha=None, rank_stabilized=False):
    """Freeze every parameter of model, and replace each child torch.nn.Linear whose name is in names by a LoRALinear.

    Return the number of trainable entries that this adds.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    added = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in names and isinstance(child, torch.nn.Linear):
                layer = LoRALinear(child, r=r, alpha=alpha, rank_stabilized=rank_stabilized)
                setattr(parent, name, layer)
                added += layer.num_trainable()
    return added
