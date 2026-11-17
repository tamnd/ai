# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://NEW-maximal-update-parametrization).
"""The maximal update parametrization (muP) of an MLP in PyTorch, with the standard parametrization (SP) for contrast.

layer_hparams is the same map as the NumPy reference (eq-mup-sgd, eq-mup-adam); attention_logits is eq-attn;
MuMLP is the two-hidden-layer tanh network of the reference, initialized by layer_hparams, and param_groups gives
torch.optim one group per layer kind with its own learning rate; feature_change measures the per-coordinate change
of the last hidden preactivations and of the output after one optimizer step."""

import math

import torch
from torch import nn

OPTIMIZERS = ("sgd", "adam")
PARAMS = ("mup", "sp")


def layer_hparams(width, base_width, lr, d_in, sigma=1.0, optimizer="adam", param="mup"):
    """Initialization std and learning rate of each layer kind: input, hidden, output and bias."""
    if width <= 0 or base_width <= 0 or d_in <= 0:
        raise ValueError("widths and the input dimension must be positive")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if param not in PARAMS:
        raise ValueError(f"param must be one of {PARAMS}")
    r = width / base_width
    std = {
        "input": sigma / math.sqrt(d_in),
        "hidden": sigma / math.sqrt(width),
        "output": sigma / math.sqrt(width),
        "bias": 0.0,
    }
    rate = {"input": lr, "hidden": lr, "output": lr, "bias": lr}
    if param == "mup":
        # eq: mup-init
        std["output"] = sigma / math.sqrt(width * r)
        if optimizer == "sgd":
            # eq: mup-sgd
            rate.update(input=lr * r, bias=lr * r, output=lr / r)
        else:
            # eq: mup-adam
            rate.update(hidden=lr / r, output=lr / r)
    return {k: {"std": float(std[k]), "lr": float(rate[k])} for k in std}


def attention_logits(q, k, base_d_head=None, alpha=1.0, param="mup"):
    """alpha * sqrt(d0) / d * q k^T under muP, alpha / sqrt(d) * q k^T under SP; d0 defaults to d."""
    d = q.shape[-1]
    if k.shape[-1] != d:
        raise ValueError("queries and keys must have the same dimension")
    if param == "sp":
        scale = alpha / math.sqrt(d)
    elif param == "mup":
        d0 = d if base_d_head is None else base_d_head
        # eq: attn
        scale = alpha * math.sqrt(d0) / d
    else:
        raise ValueError(f"param must be one of {PARAMS}")
    return scale * (q @ k.transpose(-1, -2))


class MuMLP(nn.Module):
    """f(x) = V tanh(W tanh(U x + b1) + b2) with the initialization and learning rates of layer_hparams."""

    KIND = {"U": "input", "b1": "bias", "W": "hidden", "b2": "bias", "V": "output"}

    def __init__(
        self,
        d_in,
        width,
        d_out,
        base_width,
        lr,
        sigma=1.0,
        optimizer="adam",
        param="mup",
        seed=0,
    ):
        super().__init__()
        self.optimizer_name = optimizer
        self.hp = layer_hparams(width, base_width, lr, d_in, sigma, optimizer, param)
        g = torch.Generator().manual_seed(seed)

        def draw(name, *shape):
            return nn.Parameter(torch.randn(shape, generator=g, dtype=torch.float64) * self.hp[self.KIND[name]]["std"])

        self.U = draw("U", width, d_in)
        self.b1 = draw("b1", width)
        self.W = draw("W", width, width)
        self.b2 = draw("b2", width)
        self.V = draw("V", d_out, width)

    def forward(self, x, return_features=False):
        h1 = x @ self.U.T + self.b1
        h2 = torch.tanh(h1) @ self.W.T + self.b2
        f = torch.tanh(h2) @ self.V.T
        return (f, h2) if return_features else f

    def param_groups(self):
        return [{"params": [getattr(self, n)], "lr": self.hp[k]["lr"], "name": n} for n, k in self.KIND.items()]

    def make_optimizer(self, betas=(0.9, 0.999), eps=1e-12):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.param_groups(), lr=0.0)
        return torch.optim.Adam(self.param_groups(), lr=0.0, betas=betas, eps=eps)


def mse(f, y):
    return 0.5 * ((f - y) ** 2).sum(-1).mean()


def feature_change(net, X, y, opt=None):
    """RMS per-coordinate change of the last hidden preactivations and of the output after one step on (X, y)."""
    opt = net.make_optimizer() if opt is None else opt
    with torch.no_grad():
        f0, h0 = net(X, return_features=True)
    opt.zero_grad()
    mse(net(X), y).backward()
    opt.step()
    with torch.no_grad():
        f1, h1 = net(X, return_features=True)
    return float(((h1 - h0) ** 2).mean().sqrt()), float(((f1 - f0) ** 2).mean().sqrt())
