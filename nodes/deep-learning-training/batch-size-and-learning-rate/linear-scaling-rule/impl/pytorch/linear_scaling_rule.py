# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://JDMWH4).
"""The linear scaling rule and the gradient noise scale in PyTorch.

per_example_grads computes one flattened gradient per example with torch.func; b_simple turns them into the simple
noise scale tr(Sigma) / |G|^2 (eq-b-simple); two_batch_estimates and NoiseScaleTracker are the estimator of McCandlish
et al., Appendix A, from the squared norms of a small-batch and a big-batch gradient, as obtained in data-parallel
training from one worker's gradient and the all-reduced one; scaled_lambda_lr sets each parameter group's learning rate
by the linear (eq-linear-rule) or square-root rule and returns a LambdaLR that applies a schedule and a warmup hook."""

import math

import torch
from torch.func import functional_call, grad, vmap


def per_example_grads(model, loss_fn, x, y):
    """An (n, d) tensor whose row i is the gradient of loss_fn(model(x_i), y_i) over all trainable parameters.

    loss_fn takes a batch of outputs and targets and returns a scalar; each example is passed as a batch of one."""
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(p, xi, yi):
        out = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    grads = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([g.reshape(x.shape[0], -1) for g in grads.values()], dim=1)


def b_simple(per_example):
    """(b, |G|^2, tr(Sigma)) from an (n, d) tensor of per-example gradients, n >= 2 (eq-b-simple).

    tr(Sigma) sums the unbiased per-coordinate variances and |G|^2 = |mean|^2 - tr(Sigma) / n is unbiased; b is inf
    when that estimate is not positive."""
    if per_example.dim() != 2 or per_example.shape[0] < 2:
        raise ValueError("need an (n, d) tensor with n >= 2")
    n = per_example.shape[0]
    mean = per_example.mean(dim=0)
    tr_sigma = float(((per_example - mean) ** 2).sum()) / (n - 1)
    g2 = float(mean @ mean) - tr_sigma / n
    return (tr_sigma / g2 if g2 > 0 else math.inf), g2, tr_sigma


def two_batch_estimates(g_small_sq, g_big_sq, b_small, b_big):
    """Unbiased (|G|^2, tr(Sigma)) from |G_small|^2 and |G_big|^2 at batch sizes b_small < b_big."""
    if not 0 < b_small < b_big:
        raise ValueError("need 0 < b_small < b_big")
    g2 = (b_big * float(g_big_sq) - b_small * float(g_small_sq)) / (b_big - b_small)
    s = (float(g_small_sq) - float(g_big_sq)) / (1.0 / b_small - 1.0 / b_big)
    return g2, s


def grad_sq_norm(params):
    """The squared norm of the gradients of an iterable of parameters, taken as one vector."""
    return float(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None))


class NoiseScaleTracker:
    """B_simple from a stream of (|G_small|^2, |G_big|^2): bias-corrected moving averages of |G|^2 and tr(Sigma),
    then their ratio, since the ratio of the unbiased estimates is itself biased."""

    def __init__(self, b_small, b_big, decay=0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.b_small, self.b_big, self.decay = b_small, b_big, decay
        self._g2 = self._s = self._w = 0.0

    def update(self, g_small_sq, g_big_sq):
        g2, s = two_batch_estimates(g_small_sq, g_big_sq, self.b_small, self.b_big)
        d = self.decay
        self._g2 = d * self._g2 + (1.0 - d) * g2
        self._s = d * self._s + (1.0 - d) * s
        self._w = d * self._w + (1.0 - d)
        return self.value

    @property
    def value(self):
        if self._w == 0.0:
            raise ValueError("no value yet")
        g2 = self._g2 / self._w
        return (self._s / self._w) / g2 if g2 > 0 else math.inf


def scale_learning_rate(base_lr, base_batch, batch, rule="linear"):
    """base_lr times k = batch / base_batch ("linear"), times sqrt(k) ("sqrt"), or unchanged ("none")."""
    if base_batch <= 0 or batch <= 0:
        raise ValueError("batch sizes must be positive")
    k = batch / base_batch
    factors = {"linear": k, "sqrt": math.sqrt(k), "none": 1.0}
    if rule not in factors:
        raise ValueError(f"unknown rule {rule!r}")
    return base_lr * factors[rule]


def gradual_warmup(warmup_steps, start_factor):
    """A hook (step, factor) -> factor that ramps linearly from start_factor to 1 over warmup_steps."""
    if warmup_steps < 0 or not 0.0 <= start_factor <= 1.0:
        raise ValueError("need warmup_steps >= 0 and 0 <= start_factor <= 1")

    def hook(step, factor):
        if step >= warmup_steps:
            return factor
        return factor * (start_factor + (1.0 - start_factor) * step / warmup_steps)

    return hook


def scaled_lambda_lr(optimizer, base_batch, batch, schedule=None, warmup=None, rule="linear"):
    """Scale each group's lr (taken as tuned at base_batch) to `batch` and return a LambdaLR for schedule and warmup.

    The LambdaLR multiplies the scaled rate by schedule(step) (default 1) passed through warmup(step, factor)."""
    for group in optimizer.param_groups:
        group["lr"] = scale_learning_rate(group["lr"], base_batch, batch, rule)
        group.pop("initial_lr", None)

    def factor(step):
        f = 1.0 if schedule is None else schedule(step)
        return f if warmup is None else warmup(step, f)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
