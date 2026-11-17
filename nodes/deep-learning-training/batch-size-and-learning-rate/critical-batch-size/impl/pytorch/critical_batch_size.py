# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://NEW-critical-batch-size).
"""The gradient noise scale and the critical batch size in PyTorch.

per_example_grads computes one flattened gradient per example with torch.func, and noise_scale_from_examples turns
them into B_simple = tr(Sigma) / |G|^2 (eq-b-simple). noise_scale_from_two_batches is the estimator of eq-two-batch
from the squared norms of a small-batch and a big-batch gradient, as found in data-parallel training before and after
the all-reduce, and RunningNoiseScale is a torch.nn.Module that keeps moving averages of both in buffers and divides
them. quadratic_noise_scale, step_scale and optimal_step give B_noise (eq-b-noise) and eps_opt(B) (eq-eps-opt) for a
quadratic model; steps_needed and fit_tradeoff work with the hyperbola of eq-tradeoff."""

import torch
from torch.func import functional_call, grad, vmap


def per_example_grads(model, loss_fn, x, y):
    """An (n, d) tensor whose row i is the gradient of loss_fn(model(x_i), y_i) over the trainable parameters.

    loss_fn takes a batch of outputs and targets and returns a scalar; each example is passed as a batch of one."""
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def one(p, xi, yi):
        return loss_fn(functional_call(model, (p, buffers), (xi.unsqueeze(0),)), yi.unsqueeze(0))

    per = vmap(grad(one), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([g.reshape(x.shape[0], -1) for g in per.values()], dim=1)  # (n, d)


def noise_scale_from_examples(grads):
    """(b_simple, g2, trace) from an (n, d) tensor of per-example gradients, n >= 2 (eq-b-simple).

    trace uses the divisor n - 1 and g2 = |mean|^2 - trace / n, so both are unbiased; b_simple is inf if g2 <= 0."""
    if grads.dim() != 2 or grads.shape[0] < 2:
        raise ValueError("need an (n, d) tensor with n >= 2")
    n = grads.shape[0]
    mean = grads.mean(dim=0)  # (d,)
    trace = ((grads - mean) ** 2).sum() / (n - 1)
    g2 = mean @ mean - trace / n
    b = trace / g2 if g2 > 0 else torch.tensor(float("inf"), dtype=grads.dtype)
    return b, g2, trace


def noise_scale_from_two_batches(sq_small, sq_big, b_small, b_big):
    """Unbiased (g2, trace) from |G_small|^2 at batch size b_small and |G_big|^2 at b_big (eq-two-batch)."""
    if b_small <= 0 or b_big <= 0 or b_small == b_big:
        raise ValueError("batch sizes must be positive and different")
    # eq: two-batch
    g2 = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
    trace = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
    return g2, trace


class RunningNoiseScale(torch.nn.Module):
    """Moving averages of the two-batch estimates, kept as buffers so they move with the module and its state_dict.

    forward(sq_small, sq_big, b_small, b_big) updates both averages and returns their ratio, the estimate of B_simple.
    The averages start at zero; a bias correction would scale both by the same factor, so the ratio needs none."""

    g2: torch.Tensor
    trace: torch.Tensor
    steps: torch.Tensor

    def __init__(self, decay=0.99, dtype=torch.float64):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.register_buffer("g2", torch.zeros((), dtype=dtype))
        self.register_buffer("trace", torch.zeros((), dtype=dtype))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def forward(self, sq_small, sq_big, b_small, b_big):
        g2, trace = noise_scale_from_two_batches(sq_small, sq_big, b_small, b_big)
        self.g2.mul_(self.decay).add_((1.0 - self.decay) * g2)
        self.trace.mul_(self.decay).add_((1.0 - self.decay) * trace)
        self.steps.add_(1)
        return self.value()

    def value(self):
        if self.steps.item() == 0:
            raise ValueError("no updates yet")
        return self.trace / self.g2 if self.g2 > 0 else torch.tensor(float("inf"), dtype=self.g2.dtype)


def quadratic_noise_scale(G, H, Sigma):
    """B_noise = tr(H Sigma) / G.H.G (eq-b-noise)."""
    gHg = G @ H @ G
    if gHg <= 0:
        raise ValueError("G.H.G must be positive")
    # eq: b-noise
    return torch.trace(H @ Sigma) / gHg


def step_scale(G, H):
    """(eps_max, dl_max) = (|G|^2 / G.H.G, |G|^4 / (2 G.H.G))."""
    gHg = G @ H @ G
    if gHg <= 0:
        raise ValueError("G.H.G must be positive")
    return (G @ G) / gHg, (G @ G) ** 2 / (2.0 * gHg)


def optimal_step(batch, eps_max, b_noise):
    """eps_opt(B) = eps_max / (1 + B_noise / B) (eq-eps-opt)."""
    batch = torch.as_tensor(batch, dtype=torch.float64)
    if torch.any(batch <= 0) or b_noise < 0:
        raise ValueError("batch must be positive and b_noise non-negative")
    # eq: eps-opt
    return eps_max / (1.0 + b_noise / batch)


def steps_needed(batch, s_min, e_min):
    """S(B) = S_min + E_min / B (eq-tradeoff)."""
    batch = torch.as_tensor(batch, dtype=torch.float64)
    if torch.any(batch <= 0) or s_min <= 0 or e_min <= 0:
        raise ValueError("batch, s_min and e_min must be positive")
    # eq: tradeoff
    return s_min + e_min / batch


def fit_tradeoff(batch, steps):
    """(s_min, e_min, b_crit) from the steps to a target at several batch sizes, by least squares on relative error."""
    batch = torch.as_tensor(batch, dtype=torch.float64)
    steps = torch.as_tensor(steps, dtype=torch.float64)
    if batch.shape != steps.shape or torch.unique(batch).numel() < 2:
        raise ValueError("need steps at two or more distinct batch sizes")
    if torch.any(batch <= 0) or torch.any(steps <= 0):
        raise ValueError("batch sizes and steps must be positive")
    design = torch.stack([torch.ones_like(batch), 1.0 / batch], dim=1) / steps[:, None]  # (k, 2)
    sol = torch.linalg.lstsq(design, torch.ones_like(steps)[:, None]).solution[:, 0]
    s_min, e_min = sol[0], sol[1]
    if s_min <= 0 or e_min <= 0:
        raise ValueError("the data do not fit a positive hyperbola")
    return s_min, e_min, e_min / s_min
