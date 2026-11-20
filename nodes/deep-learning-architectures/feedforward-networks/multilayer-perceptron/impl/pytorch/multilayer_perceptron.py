# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""The multilayer perceptron, the PyTorch reference: eq-layer and eq-output, with autograd for the backward pass.

The weights have the shape (n_out, n_in), as on the page and in the NumPy reference, so from_params loads the
same arrays. The loss is eq-loss and one step of plain SGD is eq-sgd.
"""

import math

import torch
from torch import nn

ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "gelu": lambda z: z * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))),
    "identity": lambda z: z,
}


def config(d_in=2, d_hidden=4, d_out=1, n_hidden=1, activation="relu"):
    if min(d_in, d_hidden, d_out, n_hidden) < 1 or activation not in ACTIVATIONS:
        raise ValueError("the widths and n_hidden must be at least 1, and the activation one of ACTIVATIONS")
    return {"d_in": d_in, "d_hidden": d_hidden, "d_out": d_out, "n_hidden": n_hidden, "activation": activation}


def widths(cfg):
    return [cfg["d_in"]] + [cfg["d_hidden"]] * cfg["n_hidden"] + [cfg["d_out"]]


def param_count(cfg):
    n = widths(cfg)
    return sum(a * b + b for a, b in zip(n[:-1], n[1:], strict=True))


class MLP(nn.Module):
    def __init__(self, cfg, generator=None):
        super().__init__()
        self.cfg = cfg
        n = widths(cfg)
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for n_in, n_out in zip(n[:-1], n[1:], strict=True):
            std = math.sqrt(2.0 / n_in)  # eq: eq-he
            self.W.append(nn.Parameter(torch.randn(n_out, n_in, generator=generator) * std))
            self.b.append(nn.Parameter(torch.zeros(n_out)))

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for i, (W, b) in enumerate(zip(params["W"], params["b"], strict=True)):
                model.W[i].copy_(torch.as_tensor(W))
                model.b[i].copy_(torch.as_tensor(b))
        return model

    def forward(self, x):
        act = ACTIVATIONS[self.cfg["activation"]]
        h = x  # (B, I)
        for W, b in zip(self.W[:-1], self.b[:-1], strict=True):
            h = act(h @ W.T + b)  # (B, H)  eq: eq-layer
        return h @ self.W[-1].T + self.b[-1]  # (B, O)  eq: eq-output


def mse_loss(y, t):
    return 0.5 * ((y - t) ** 2).sum() / y.shape[0]  # eq: eq-loss


def sgd_step(model, lr):
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad  # eq: eq-sgd
            p.grad = None
