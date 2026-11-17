# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-normalization/ (ai://NEW-weight-normalization).
"""Weight normalization as a torch.nn.Module, the PyTorch reference.

WeightNormLinear keeps a direction v of shape (out, in), a length g of shape (out,) and a bias b, and computes
y = x w^T + b with w_j = g_j v_j / ||v_j|| (eq-wn, eq-neuron). Autograd gives the backward pass, and the tests compare
it with the closed form of eq-grad in weight_grads. data_dependent_init is eq-init. SpectralNormLinear divides its
weight by the estimate of eq-power, kept in a buffer u that is reused from step to step (eq-sn).
"""

import torch


class WeightNormLinear(torch.nn.Module):
    """A linear layer whose weight rows are g_j v_j / ||v_j||."""

    def __init__(self, in_features, out_features, scale=0.05, generator=None):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(f"the sizes must be positive, got in = {in_features} and out = {out_features}")
        self.v = torch.nn.Parameter(torch.randn(out_features, in_features, generator=generator) * scale)
        self.g = torch.nn.Parameter(torch.ones(out_features))
        self.b = torch.nn.Parameter(torch.zeros(out_features))

    def weight(self):
        norms = self.v.pow(2).sum(dim=1).sqrt()
        # eq: eq-wn
        return (self.g / norms).unsqueeze(1) * self.v

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.v.shape[1]:
            raise ValueError(f"x must have the shape (N, {self.v.shape[1]}), got {tuple(x.shape)}")
        # eq: eq-neuron
        return x @ self.weight().T + self.b


@torch.no_grad()
def data_dependent_init(layer, x, eps=0.0):
    """Set g = 1 / sigma[t] and b = -mu[t] / sigma[t] for t = v . x / ||v|| on the batch x, keeping v."""
    if x.dim() != 2 or x.shape[0] < 2:
        raise ValueError(f"the batch must have at least two rows, got shape {tuple(x.shape)}")
    # eq: eq-init
    t = x @ layer.v.T / layer.v.pow(2).sum(dim=1).sqrt()
    mu = t.mean(dim=0)
    sigma = ((t - mu).pow(2).mean(dim=0) + eps).sqrt()
    if bool((sigma == 0).any()):
        raise ValueError("a unit has a constant pre-activation on the batch; use a larger batch or eps > 0")
    layer.g.copy_(1.0 / sigma)
    layer.b.copy_(-mu / sigma)
    return layer


def weight_grads(v, g, grad_w):
    """The closed form of eq-grad, used to check autograd: (grad_v, grad_g) from grad_w, row by row."""
    norms = v.pow(2).sum(dim=1).sqrt()
    # eq: eq-grad
    grad_g = (grad_w * v).sum(dim=1) / norms
    grad_v = (g / norms).unsqueeze(1) * grad_w - (g * grad_g / norms.pow(2)).unsqueeze(1) * v
    return grad_v, grad_g


@torch.no_grad()
def power_iteration(w, u, n_iter=1):
    """Return (sigma, u, v) after n_iter rounds of eq-power from u: v <- W^T u / ||W^T u||, u <- W v / ||W v||."""
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    for _ in range(n_iter):
        # eq: eq-power
        v = w.T @ u
        v = v / v.norm()
        u = w @ v
        u = u / u.norm()
    return float(u @ w @ v), u, v


class SpectralNormLinear(torch.nn.Module):
    """A linear layer whose weight is W / sigma(W), with sigma from n_iter power steps per forward pass in training."""

    def __init__(self, in_features, out_features, n_iter=1, generator=None):
        super().__init__()
        if n_iter < 1:
            raise ValueError(f"n_iter must be at least 1, got {n_iter}")
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, generator=generator) / in_features**0.5)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        u = torch.randn(out_features, generator=generator)
        self.register_buffer("u", u / u.norm())
        v = self.weight.detach().T @ self.u
        self.register_buffer("v", v / v.norm())
        self.n_iter = int(n_iter)

    def normalized_weight(self):
        if self.training:
            _, u, v = power_iteration(self.weight.detach(), self.u, self.n_iter)
            self.u.copy_(u)
            self.v.copy_(v)
        # eq: eq-sn, sigma = u^T W v with u and v held fixed, so the gradient flows through W only
        return self.weight / (self.u @ self.weight @ self.v)

    def forward(self, x):
        return x @ self.normalized_weight().T + self.bias
