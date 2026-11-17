# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
"""Gradient checkpointing on a chain of tanh layers, the PyTorch version.

The chain is eq-layer of the page. With segments = k, each of the k segments runs under torch.utils.checkpoint, which
keeps the input of the segment and recomputes the whole segment in the backward pass (eq-segment-memory).
"""

import torch
from torch.utils.checkpoint import checkpoint


def segment_sizes(n, k):
    """Split n layers into k segments; the longer ones come first, as in the NumPy reference."""
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r)


class Chain(torch.nn.Module):
    """n_layers layers h_i = tanh(W_i h_{i-1} + b_i) of width D; segments=None stores every activation."""

    def __init__(self, n_layers, width, segments=None):
        super().__init__()
        if n_layers < 1 or width < 1:
            raise ValueError("n_layers and width must be at least 1")
        if segments is not None and not 1 <= segments <= n_layers:
            raise ValueError(f"segments must be in [1, {n_layers}], got {segments}")
        self.n_layers, self.segments = n_layers, segments
        self.W = torch.nn.Parameter(torch.zeros(n_layers, width, width))  # (L, D, D)
        self.b = torch.nn.Parameter(torch.zeros(n_layers, width))  # (L, D)
        self.forwards = 0

    @classmethod
    def from_params(cls, params, segments=None):
        W = torch.as_tensor(params["W"])
        model = cls(W.shape[0], W.shape[1], segments).to(W.dtype)
        with torch.no_grad():
            model.W.copy_(W)
            model.b.copy_(torch.as_tensor(params["b"]))
        return model

    def layer(self, i, h):
        self.forwards += 1
        return torch.tanh(h @ self.W[i].T + self.b[i])  # eq: eq-layer  (B, D)

    def run(self, lo, hi, h):
        for i in range(lo, hi):
            h = self.layer(i, h)
        return h

    def forward(self, x):
        if self.segments is None:
            return self.run(0, self.n_layers, x)
        lo, h = 0, x
        for size in segment_sizes(self.n_layers, self.segments):
            h = checkpoint(self.run, lo, lo + size, h, use_reentrant=False)  # eq: eq-segment-memory
            lo += size
        return h


def saved_activations(model, x):
    """Run the forward pass and count the distinct arrays of the shape of x that autograd keeps for the backward."""
    seen = set()

    def pack(t):
        if t.shape == x.shape:
            seen.add(t.untyped_storage().data_ptr())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = model(x)
    return y, len(seen)
