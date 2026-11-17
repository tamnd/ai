# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://8BNWST).
"""Sharpness-aware minimization (SAM) in PyTorch.

sam_perturbation is eq-eps-hat over a list of gradient tensors taken as one vector; SAM is an optimizer that wraps a
base optimizer such as torch.optim.SGD and runs the two passes of Algorithm 1 of Foret et al.: first_step moves each
parameter to w + eps_hat, second_step moves it back to w and lets the base optimizer step with the gradient taken at the
perturbed point; filter_normalized_direction is the random direction of Li et al. with each filter rescaled to the norm
of the matching filter of the weights."""

import math

import torch

EPS = 1e-12


def sam_perturbation(grads, rho, p=2.0):
    """The perturbation rho sign(g) |g|^(q-1) / (||g||_q^q)^(1/p), 1/p + 1/q = 1, over a list of tensors (eq-eps-hat).

    p = 2 gives rho g / ||g||_2 and p = inf gives rho sign(g). A zero gradient gives a zero perturbation."""
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if p <= 1:
        raise ValueError("p must be greater than 1")
    if math.isinf(p):
        return [rho * torch.sign(g) for g in grads]
    scale = max((g.abs().max() for g in grads if g.numel()), default=torch.tensor(0.0))
    if float(scale) == 0.0:
        return [torch.zeros_like(g) for g in grads]
    q = p / (p - 1.0)
    hs = [g / scale for g in grads]
    total = sum((h.abs() ** q).sum() for h in hs)
    denom = total ** (1.0 / p)
    return [rho * torch.sign(h) * h.abs() ** (q - 1.0) / denom for h in hs]


class SAM(torch.optim.Optimizer):
    """SAM around a base optimizer. Usage per batch:

        loss_fn().backward(); opt.first_step(zero_grad=True)
        loss_fn().backward(); opt.second_step(zero_grad=True)

    or opt.step(closure) with a closure that zeroes nothing, runs the forward and backward pass and returns the loss.
    The perturbation is computed from the gradients of all parameter groups as one vector, with each group's rho."""

    def __init__(self, params, base_optimizer, rho=0.05, p=2.0, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        defaults = dict(rho=rho, p=p, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Move each parameter with a gradient from w to w + eps_hat(w) and remember eps_hat."""
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            es = sam_perturbation([p.grad for p in ps], group["rho"], group["p"])
            for p, e in zip(ps, es):
                p.add_(e)
                self.state[p]["e_w"] = e
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Move each parameter back to w, then let the base optimizer step with the gradient taken at w + eps_hat."""
        for group in self.param_groups:
            for p in group["params"]:
                e = self.state[p].pop("e_w", None)
                if e is not None:
                    p.sub_(e)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """Both passes. The closure runs the forward and backward pass and returns the loss; it is called twice."""
        if closure is None:
            raise ValueError("SAM needs a closure that recomputes the loss and its gradient")
        closure = torch.enable_grad()(closure)
        self.zero_grad()
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss


@torch.no_grad()
def filter_normalized_direction(params, generator=None, ignore_1d=True):
    """A random Gaussian direction with each filter rescaled to the norm of the matching filter of params (Li et al.).

    A filter is a slice along the first axis. Tensors with one dimension (biases, batch-normalization parameters) get a
    zero direction when ignore_1d is true; otherwise each entry is a filter. A filter of zero norm gets zero."""
    out = []
    for theta in params:
        d = torch.randn(theta.shape, generator=generator, dtype=theta.dtype, device=theta.device)
        if theta.dim() <= 1:
            if ignore_1d:
                out.append(torch.zeros_like(theta))
                continue
            t, dd = theta.reshape(-1, 1), d.reshape(-1, 1)
        else:
            t, dd = theta.reshape(theta.shape[0], -1), d.reshape(theta.shape[0], -1)
        tn = t.norm(dim=1, keepdim=True)
        dn = dd.norm(dim=1, keepdim=True)
        out.append((dd / (dn + EPS) * tn).reshape(theta.shape))
    return out
