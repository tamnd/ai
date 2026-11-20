# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""The autoencoder of the page in PyTorch, with autograd for the backward pass: eq-encoder, eq-decoder and eq-layer,
the losses eq-cross-entropy and eq-squared-error, the corruptions of eq-dae and the penalty of eq-cae.

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. With tied weights the decoder uses the transposes of the encoder's matrices.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = {"sigmoid": torch.sigmoid, "relu": torch.relu, "tanh": torch.tanh, "linear": lambda a: a}


def config(d_in=6, d_hidden=5, n_hidden=1, d_code=2, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    return from_widths([d_in] + [d_hidden] * n_hidden + [d_code], act, code_act, out, tied)


def from_widths(widths, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    widths = [int(w) for w in widths]
    if len(widths) < 2 or min(widths) < 1 or act not in ACTIVATIONS or code_act not in ACTIVATIONS:
        raise ValueError("at least two positive widths, and activations among ACTIVATIONS")
    if out not in ("sigmoid", "linear"):
        raise ValueError("out is sigmoid or linear")
    return {"widths": widths, "act": act, "code_act": code_act, "out": out, "tied": bool(tied)}


def param_count(cfg):
    w = cfg["widths"]
    weights = sum(a * b for a, b in zip(w[:-1], w[1:]))
    return (1 if cfg["tied"] else 2) * weights + sum(w[1:]) + sum(w[:-1])


class Autoencoder(nn.Module):
    def __init__(self, cfg, generator=None, gain=1.0):
        super().__init__()
        self.cfg = cfg
        w = cfg["widths"]
        K = len(w) - 1

        def glorot(n_in, n_out):
            r = gain * math.sqrt(6.0 / (n_in + n_out))
            return nn.Parameter((torch.rand(n_in, n_out, generator=generator) * 2 - 1) * r)

        self.enc_W = nn.ParameterList([glorot(w[i], w[i + 1]) for i in range(K)])
        self.enc_b = nn.ParameterList([nn.Parameter(torch.zeros(w[i + 1])) for i in range(K)])
        self.dec_b = nn.ParameterList([nn.Parameter(torch.zeros(w[K - j - 1])) for j in range(K)])
        if not cfg["tied"]:
            self.dec_W = nn.ParameterList([glorot(w[K - j], w[K - j - 1]) for j in range(K)])

    @classmethod
    def from_params(cls, params, cfg):
        model = cls(cfg)
        with torch.no_grad():
            for name, values in params.items():
                for p, v in zip(getattr(model, name), values):
                    p.copy_(torch.as_tensor(v))
        return model

    def dec_weights(self):
        if self.cfg["tied"]:
            return [W.T for W in reversed(self.enc_W)]
        return list(self.dec_W)

    def encode(self, x):
        K = len(self.enc_W)
        h = x  # (B, D)
        for i, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            name = self.cfg["code_act"] if i == K - 1 else self.cfg["act"]
            h = ACTIVATIONS[name](h @ W + b)  # eq: eq-layer
        return h  # (B, C)  eq: eq-encoder

    def decode_logits(self, y):
        Ws = self.dec_weights()
        h = y
        for j, (W, b) in enumerate(zip(Ws, self.dec_b)):
            a = h @ W + b
            h = ACTIVATIONS[self.cfg["act"]](a) if j < len(Ws) - 1 else a
        return h  # (B, D), the output before its activation

    def forward(self, x):
        y = self.encode(x)
        a = self.decode_logits(y)
        z = torch.sigmoid(a) if self.cfg["out"] == "sigmoid" else a  # eq: eq-decoder
        return {"y": y, "a": a, "z": z}


def loss(cfg, target, a, weight=None):
    """eq-cross-entropy from the logits a, or eq-squared-error; summed over the components, averaged over the batch."""
    if cfg["out"] == "sigmoid":
        e = F.binary_cross_entropy_with_logits(a, target, reduction="none")
    else:
        e = (target - a) ** 2
    if weight is not None:
        e = e * weight
    return e.sum() / target.shape[0]


def corrupt(x, kind, level, generator=None, low=0.0, high=1.0):
    """The Gaussian, masking and salt-and-pepper corruptions of eq-dae; returns x~ and the corrupted components."""
    if kind == "gaussian":
        return x + level * torch.randn(x.shape, generator=generator), torch.ones_like(x, dtype=torch.bool)
    B, D = x.shape
    k = int(round(level * D))
    chosen = torch.rand(B, D, generator=generator).argsort(1)[:, :k]
    mask = torch.zeros(B, D, dtype=torch.bool).scatter_(1, chosen, True)
    if kind == "mask":
        return x.masked_fill(mask, 0.0), mask
    if kind == "salt_pepper":
        coin = torch.rand(B, D, generator=generator) < 0.5
        return torch.where(mask, torch.where(coin, high, low).to(x.dtype), x), mask
    raise ValueError(kind)


def contractive_penalty(model, x):
    """eq-cae-sigmoid: ||J_f(x)||_F^2 of a one-layer sigmoid encoder, averaged over the batch."""
    W, b = model.enc_W[0], model.enc_b[0]
    h = torch.sigmoid(x @ W + b)
    return ((h * (1 - h)) ** 2 * (W**2).sum(0)).sum() / x.shape[0]


def jacobian_penalty(model, x):
    """||J_f(x)||_F^2 of any encoder by autograd, averaged over the batch: the definition behind eq-cae."""
    J = torch.func.vmap(torch.func.jacrev(lambda v: model.encode(v[None])[0]))(x)  # (B, C, D)
    return (J**2).sum() / x.shape[0]


def train(model, X, steps, lr=1e-2, batch=32, noise=None, contraction=0.0, generator=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        idx = torch.randperm(X.shape[0], generator=generator)[:batch]
        x = X[idx]
        xin = corrupt(x, *noise, generator=generator)[0] if noise else x
        value = loss(model.cfg, x, model(xin)["a"])
        if contraction:
            value = value + contraction * contractive_penalty(model, xin)
        opt.zero_grad()
        value.backward()
        opt.step()
        history.append(value.item())
    return history
