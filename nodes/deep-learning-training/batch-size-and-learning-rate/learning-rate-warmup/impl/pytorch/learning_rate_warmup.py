# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/learning-rate-warmup/ (ai://NEW-learning-rate-warmup).
"""Learning rate warmup in PyTorch.

The schedulers are built from torch.optim.lr_scheduler. A scheduler's step() is called once per optimizer update, after
optimizer.step(); its counter starts at 0, so the first update uses the rate for counter 0. gradual_warmup is LinearLR
from start / peak to 1 over warmup_steps updates (eq-gradual), warmup_then_steps puts Goyal et al.'s step decay after
it, and noam is a LambdaLR of eq-noam with the base rate 1 and the first update counted as step 1. radam_rectification
is r_t of eq-rect with the threshold of torch.optim.RAdam (rho_t > 5) by default.
"""

import math

from torch.optim.lr_scheduler import LambdaLR, LinearLR, MultiStepLR, SequentialLR


def gradual_warmup(optimizer, warmup_steps, start):
    """Ramp each group's lr linearly from start (a rate, below the group's lr) to the group's lr over warmup_steps."""
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
    peak = optimizer.param_groups[0]["lr"]
    if not 0.0 < start <= peak:
        raise ValueError(f"start must be in (0, lr], got {start} for lr {peak}")
    return LinearLR(optimizer, start_factor=start / peak, end_factor=1.0, total_iters=warmup_steps)


def warmup_then_steps(optimizer, warmup_steps, start, milestones, gamma=0.1):
    """Gradual warmup for warmup_steps updates, then the lr times gamma at each milestone, counted from update 0."""
    warm = gradual_warmup(optimizer, warmup_steps, start)
    decay = MultiStepLR(optimizer, milestones=[m - warmup_steps for m in milestones], gamma=gamma)
    return SequentialLR(optimizer, schedulers=[warm, decay], milestones=[warmup_steps])


def noam_factor(d_model, warmup_steps):
    """The multiplier of LambdaLR for eq-noam: counter c gives the rate of step c + 1."""

    def f(c):
        s = c + 1
        return d_model**-0.5 * min(s**-0.5, s * warmup_steps**-1.5)

    return f


def noam(optimizer, d_model, warmup_steps):
    """eq-noam as a LambdaLR. Set the optimizer's lr to 1.0, so the factor is the rate."""
    return LambdaLR(optimizer, noam_factor(d_model, warmup_steps))


def radam_rho(t, beta2):
    """rho_t = rho_inf - 2 t beta2^t / (1 - beta2^t), with rho_inf = 2 / (1 - beta2) - 1 (eq-rho)."""
    b = beta2**t
    return 2.0 / (1.0 - beta2) - 1.0 - 2.0 * t * b / (1.0 - b)


def radam_rectification(t, beta2, threshold=5.0):
    """r_t of eq-rect when rho_t > threshold, else None (an un-adapted momentum step)."""
    rho, rho_inf = radam_rho(t, beta2), 2.0 / (1.0 - beta2) - 1.0
    if rho <= threshold:
        return None
    return math.sqrt((rho - 4.0) * (rho - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho))


def lrs(scheduler, optimizer, steps):
    """The lr of the first group at each of the first steps updates, stepping the scheduler after each one."""
    out = []
    for _ in range(steps):
        out.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return out
