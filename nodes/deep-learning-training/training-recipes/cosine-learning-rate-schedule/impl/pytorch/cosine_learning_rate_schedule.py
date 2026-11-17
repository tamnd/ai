# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://FG7K47).
"""Learning-rate schedules in PyTorch.

cosine, linear_decay, step_decay and wsd evaluate eq-cosine, eq-linear, eq-step and eq-wsd on a tensor of steps.
sgdr is eq-cosine with warm restarts, placing each time in its run with the restart epochs of eq-restarts. lr_factor returns
the multiplier of the base rate at each step, with an optional linear warmup, for torch.optim.lr_scheduler.LambdaLR,
and build_scheduler wraps an optimizer with it. Step once per optimizer update and give every length in updates.
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def _steps(t):
    t = torch.as_tensor(t, dtype=torch.get_default_dtype())
    if bool((t < 0).any()):
        raise ValueError("steps must be non-negative")
    return t


def _check_total(total):
    if not total > 0:
        raise ValueError(f"total must be positive, got {total}")


def cosine(t, total, peak, final=0.0):
    """eq-cosine: half a cosine from peak at t = 0 to final at t = total, then final."""
    t = _steps(t)
    _check_total(total)
    frac = torch.clamp(t / total, max=1.0)
    # eq: eq-cosine
    return final + 0.5 * (peak - final) * (1.0 + torch.cos(math.pi * frac))


def linear_decay(t, total, peak, final=0.0):
    """eq-linear: a straight line from peak at t = 0 to final at t = total, then final."""
    t = _steps(t)
    _check_total(total)
    # eq: eq-linear
    return peak - (peak - final) * torch.clamp(t / total, max=1.0)


def step_decay(t, base, milestones, gamma=0.1):
    """eq-step: base * gamma^k, where k is the number of milestones at or before t."""
    t = _steps(t)
    m = torch.as_tensor(list(milestones), dtype=t.dtype)
    # eq: eq-step
    k = (t.unsqueeze(-1) >= m).sum(-1)
    return base * gamma ** k.to(t.dtype)


def wsd(t, total, peak, decay_steps, final=0.0, shape="linear"):
    """eq-wsd after the warmup: peak until total - decay_steps, then a cooldown to final at total."""
    t = _steps(t)
    if not 0 < decay_steps <= total:
        raise ValueError(f"decay_steps must be in (0, total], got {decay_steps} for total {total}")
    # eq: eq-wsd
    u = torch.clamp((t - (total - decay_steps)) / decay_steps, 0.0, 1.0)
    if shape == "linear":
        f = 1.0 - u
    elif shape == "1-sqrt":
        f = 1.0 - torch.sqrt(u)
    elif shape == "cosine":
        f = 0.5 * (1.0 + torch.cos(math.pi * u))
    else:
        raise ValueError(f"unknown cooldown shape {shape!r}")
    return final + (peak - final) * f


def sgdr_position(t, t0, t_mult: float = 1):
    """(T_cur, T_i) for each time in t: the time since the last restart and the length of the current run."""
    t = _steps(t)
    if t0 <= 0 or t_mult < 1:
        raise ValueError(f"need t0 > 0 and t_mult >= 1, got {t0} and {t_mult}")
    # The restarts at or before the last time, by the loop that eq-restarts sums.
    restarts, start, length = [], t0, t0
    while start <= float(t.max()):
        restarts.append(start)
        length *= t_mult
        start += length
    i = (t.unsqueeze(-1) >= torch.tensor(restarts, dtype=t.dtype)).sum(-1).to(t.dtype)
    # eq: eq-restarts
    start = i * t0 if t_mult == 1 else t0 * (t_mult**i - 1) / (t_mult - 1)
    return t - start, t0 * t_mult**i


def sgdr(t, t0, peak, final=0.0, t_mult: float = 1):
    """Cosine annealing with warm restarts: eq-cosine inside the run that contains each time in t."""
    t_cur, t_i = sgdr_position(t, t0, t_mult)
    # eq: eq-cosine
    return final + 0.5 * (peak - final) * (1.0 + torch.cos(math.pi * t_cur / t_i))


def lr_factor(kind, total, warmup=0, final_ratio=0.0, **kw):
    """A function of the step that gives the multiplier of the base rate, for LambdaLR.

    The warmup rises linearly from 0 over the first warmup steps; the schedule then runs over the remaining
    total - warmup steps and ends at final_ratio times the base rate. kind is "constant", "step", "linear", "cosine"
    or "wsd"; "step" takes milestones in steps from 0 and gamma, "wsd" takes decay_steps and shape.
    """
    if not 0 <= warmup < total:
        raise ValueError(f"warmup must be in [0, total), got {warmup} for total {total}")
    if kind not in ("constant", "step", "linear", "cosine", "wsd"):
        raise ValueError(f"unknown schedule kind {kind!r}")
    n = total - warmup

    def factor(step):
        if step < warmup:
            return step / warmup
        s = step - warmup
        if kind == "constant":
            return 1.0
        if kind == "step":
            return float(step_decay(step, 1.0, kw["milestones"], kw.get("gamma", 0.1)))
        if kind == "linear":
            return float(linear_decay(s, n, 1.0, final_ratio))
        if kind == "cosine":
            return float(cosine(s, n, 1.0, final_ratio))
        return float(wsd(s, n, 1.0, kw["decay_steps"], final_ratio, kw.get("shape", "linear")))

    return factor


def build_scheduler(optimizer, kind, total, warmup=0, final_ratio=0.0, **kw):
    """A LambdaLR on optimizer that follows lr_factor; call its step() once per optimizer update."""
    return LambdaLR(optimizer, lr_factor(kind, total, warmup, final_ratio, **kw))
