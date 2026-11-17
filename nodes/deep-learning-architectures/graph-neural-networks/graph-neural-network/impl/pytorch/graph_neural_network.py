# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://NEW-graph-neural-network).
"""The message-passing network with sum aggregation and a sum readout, and the GCN layer, in PyTorch.

It computes eq-message, eq-update, eq-readout, eq-renorm and eq-gcn of the page as the NumPy reference does, and
leaves the gradient to autograd. from_params loads the parameter tree of the NumPy reference, stored flat.
"""

import torch
from torch import nn


def config(d_in=3, d_hidden=8, n_layers=2, classes=4):
    """Return the config: node feature width F, hidden width D, number of layers L and number of classes K."""
    if min(d_in, d_hidden, n_layers, classes) < 1:
        raise ValueError("the sizes must be at least 1")
    return {
        "d_in": d_in,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "classes": classes,
    }


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    F, D, L, K = cfg["d_in"], cfg["d_hidden"], cfg["n_layers"], cfg["classes"]
    return (2 * F * D + D) + (L - 1) * (2 * D * D + D) + D * K + K


class MPLayer(nn.Module):
    """h' = relu(h W_self + A h W_nbr + b)."""

    def __init__(self, n_in, n_out, generator=None):
        super().__init__()
        self.W_self = nn.Parameter(torch.randn(n_in, n_out, generator=generator) * n_in**-0.5)
        self.W_nbr = nn.Parameter(torch.randn(n_in, n_out, generator=generator) * n_in**-0.5)
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, h, adj):
        m = adj @ (h @ self.W_nbr)  # (B, N, D); eq: eq-message
        return torch.relu(h @ self.W_self + m + self.b)  # (B, N, D); eq: eq-update


class GNN(nn.Module):
    def __init__(self, cfg, generator=None):
        super().__init__()
        self.cfg = cfg
        F, D, K = cfg["d_in"], cfg["d_hidden"], cfg["classes"]
        self.layers = nn.ModuleList(MPLayer(F if i == 0 else D, D, generator) for i in range(cfg["n_layers"]))
        self.W = nn.Parameter(torch.randn(D, K, generator=generator) * D**-0.5)
        self.b = nn.Parameter(torch.zeros(K))

    @classmethod
    def from_params(cls, flat, cfg):
        """Build the model from the flat parameter tree of the NumPy reference ("layers/0/W_self", "head/W", ...)."""
        model = cls(cfg)
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                for name in ("W_self", "W_nbr", "b"):
                    getattr(layer, name).copy_(torch.as_tensor(flat[f"layers/{i}/{name}"]))
            model.W.copy_(torch.as_tensor(flat["head/W"]))
            model.b.copy_(torch.as_tensor(flat["head/b"]))
        return model

    def forward(self, x, adj):
        """Return (logits, cache) for node features x (B, N, F) and adjacency adj (B, N, N)."""
        h, states = x, []
        for layer in self.layers:
            h = layer(h, adj)  # (B, N, D)
            states.append(h)
        g = h.sum(dim=-2)  # (B, D); eq: eq-readout
        return g @ self.W + self.b, {"h": states, "graph": g}  # (B, K)


def normalized_adjacency(adj):
    """D~^(-1/2) (A + I) D~^(-1/2), eq-renorm."""
    a = adj + torch.eye(adj.shape[-1], dtype=adj.dtype)  # (B, N, N); eq: eq-renorm
    s = a.sum(dim=-1).rsqrt()  # (B, N)
    return s[..., :, None] * a * s[..., None, :]


def gcn_layer(W, h, adj_hat, activation=True):
    """sigma(A_hat H W), eq-gcn."""
    z = adj_hat @ (h @ W)  # (B, N, D); eq: eq-gcn
    return torch.relu(z) if activation else z
