# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
"""Weight decay in PyTorch.

SGD is torch.optim.SGD with a decoupled switch: with decoupled=False, weight_decay is an L2 term added to the gradient
before the momentum buffer (eq-l2-step), which is what torch.optim.SGD does; with decoupled=True the parameters are
multiplied by 1 - lr * weight_decay beside the step (eq-decay-step), the SGDW of Loshchilov and Hutter written as
PyTorch writes AdamW. AdamW is torch.optim.AdamW without AMSGrad. param_groups puts biases and the parameters of
normalization layers into a group without decay, and l2_penalty is the explicit penalty to add to a loss.
"""

import torch
from torch import nn

_NORMS = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm)


def l2_penalty(params, weight_decay):
    """(weight_decay / 2) times the sum of the squared entries of params; its gradient is weight_decay * p."""
    return 0.5 * weight_decay * sum((p * p).sum() for p in params)


def param_groups(model, weight_decay):
    """Two parameter groups for an optimizer: the weights of linear and convolution layers with weight_decay, and the
    biases and the parameters of normalization layers with 0, as in Goyal et al. and He et al. (no bias decay)."""
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if isinstance(module, _NORMS) or name.endswith("bias") or p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class SGD(torch.optim.Optimizer):
    """torch.optim.SGD (decoupled=False) or SGD with decoupled weight decay (decoupled=True)."""

    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, decoupled=False):
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("nesterov needs a positive momentum and zero dampening")
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                                      nesterov=nesterov, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, damp, wd = group["lr"], group["momentum"], group["dampening"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd and not group["decoupled"]:
                    g = g.add(p, alpha=wd)
                if mu:
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = g.clone()
                    else:
                        state["buf"].mul_(mu).add_(g, alpha=1 - damp)
                    g = g.add(state["buf"], alpha=mu) if group["nesterov"] else state["buf"]
                if wd and group["decoupled"]:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
        return loss


class AdamW(torch.optim.Optimizer):
    """torch.optim.AdamW (decoupled=True, the default) or torch.optim.Adam with an L2 weight_decay (decoupled=False)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, decoupled=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["decoupled"]:
                    p.mul_(1 - lr * wd)
                elif wd:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t = state["t"]
                state["m"].lerp_(g, 1 - b1)
                state["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                # m_hat / (sqrt(v_hat) + eps), computed in the order of torch.optim.AdamW so that rounding agrees.
                denom = state["v"].sqrt() / (1 - b2**t) ** 0.5 + eps
                p.addcdiv_(state["m"], denom, value=-lr / (1 - b1**t))
        return loss
