# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://NEW-adam).
"""Adam and AdamW as PyTorch optimizers, written from tensor operations.

Adam is Algorithm 1 of Kingma and Ba (eq-moments, eq-bias-correction, eq-update) with the denominator
sqrt(v_hat) + eps, as in torch.optim.Adam. weight_decay is an L2 term added to the gradient, unless
decoupled_weight_decay is set; then the parameters are first multiplied by 1 - lr * weight_decay, as in
torch.optim.AdamW (eq-adamw). amsgrad keeps the running maximum of v_t and bias-corrects it, as PyTorch does. The
tests compare both classes with torch.optim.Adam and torch.optim.AdamW.
"""

import torch


class Adam(torch.optim.Optimizer):
    """Adam, with the defaults of torch.optim.Adam and of Algorithm 1 of Kingma and Ba."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                 decoupled_weight_decay=False):
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad,
                    "decoupled_weight_decay": decoupled_weight_decay}
        super().__init__(params, defaults)

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
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["v_max"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                if wd and group["decoupled_weight_decay"]:
                    p.mul_(1.0 - lr * wd)
                elif wd:
                    g = g + wd * p
                state["m"].mul_(b1).add_((1.0 - b1) * g)
                state["v"].mul_(b2).add_((1.0 - b2) * g * g)
                v = state["v"]
                if group["amsgrad"]:
                    torch.maximum(state["v_max"], v, out=state["v_max"])
                    v = state["v_max"]
                m_hat = state["m"] / (1.0 - b1**t)
                v_hat = v / (1.0 - b2**t)
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))
        return loss


class AdamW(Adam):
    """AdamW, Adam with decoupled weight decay and the defaults of torch.optim.AdamW (weight_decay 1e-2)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                         decoupled_weight_decay=True)
