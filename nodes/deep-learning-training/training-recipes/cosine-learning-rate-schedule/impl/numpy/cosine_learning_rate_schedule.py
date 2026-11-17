# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/cosine-learning-rate-schedule/ (ai://NEW-cosine-learning-rate-schedule).
"""Learning-rate schedules, the NumPy reference.

Each schedule is a pure function of the step t (from 0) and returns one learning rate. step_decay is eq-step,
linear_decay eq-linear, cosine eq-cosine, and wsd eq-wsd with a linear, (1-sqrt) or cosine cooldown. schedule puts
an optional linear warmup in front of any of them. sgdr is eq-cosine with warm restarts, where the run lengths grow
by t_mult (eq-restarts). quadratic_variance runs the exact variance recursion of the noise-floor derivation.
"""

import math

import numpy as np


def _check_t(t):
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")


def step_decay(t, base, milestones, gamma=0.1):
    """base * gamma^k, where k is the number of milestones at or before step t."""
    _check_t(t)
    # eq: eq-step
    k = sum(1 for m in milestones if t >= m)
    return base * gamma**k


def linear_decay(t, total, peak, final=0.0):
    """A straight line from peak at t = 0 to final at t = total, then final."""
    _check_t(t)
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    # eq: eq-linear
    frac = min(t, total) / total
    return peak - (peak - final) * frac


def cosine(t, total, peak, final=0.0):
    """Half a cosine from peak at t = 0 to final at t = total, then final (eq-cosine with T_cur = t, T_i = total)."""
    _check_t(t)
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    # eq: eq-cosine
    frac = min(t, total) / total
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * frac))


def cooldown_factor(u, shape="linear"):
    """The factor f in [0, 1] of the cooldown at the fraction u in [0, 1] of the cooldown: 1 at u = 0, 0 at u = 1."""
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"u must be in [0, 1], got {u}")
    if shape == "linear":
        return 1.0 - u
    if shape == "1-sqrt":
        return 1.0 - math.sqrt(u)
    if shape == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * u))
    raise ValueError(f"unknown cooldown shape {shape!r}")


def wsd(t, total, peak, decay_steps, final=0.0, shape="linear"):
    """Warmup-stable-decay without the warmup: peak until total - decay_steps, then a cooldown to final at total."""
    _check_t(t)
    if not 0 < decay_steps <= total:
        raise ValueError(f"decay_steps must be in (0, total], got {decay_steps} for total {total}")
    start = total - decay_steps
    if t <= start:
        return peak
    # eq: eq-wsd
    u = min(t - start, decay_steps) / decay_steps
    return final + (peak - final) * cooldown_factor(u, shape)


def schedule(t, kind, total, peak, final=0.0, warmup=0, **kw):
    """The rate at step t of a schedule of total steps with a linear warmup from 0 over the first warmup steps.

    kind is "constant", "step", "linear", "cosine" or "wsd". After the warmup the schedule runs over the remaining
    total - warmup steps, with its step counted from the end of the warmup. For "step" the milestones are given in
    steps from 0, as most code gives them.
    """
    _check_t(t)
    if not 0 <= warmup < total:
        raise ValueError(f"warmup must be in [0, total), got {warmup} for total {total}")
    if t < warmup:
        return peak * t / warmup
    s, n = t - warmup, total - warmup
    if kind == "constant":
        return peak
    if kind == "step":
        return step_decay(t, peak, kw["milestones"], kw.get("gamma", 0.1))
    if kind == "linear":
        return linear_decay(s, n, peak, final)
    if kind == "cosine":
        return cosine(s, n, peak, final)
    if kind == "wsd":
        return wsd(s, n, peak, kw["decay_steps"], final, kw.get("shape", "linear"))
    raise ValueError(f"unknown schedule kind {kind!r}")


def restart_epoch(k, t0, t_mult):
    """The epoch of the k-th warm restart (k = 0 is the start): t0 (t_mult^k - 1) / (t_mult - 1), or k t0 if t_mult = 1."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if t0 <= 0 or t_mult < 1:
        raise ValueError(f"need t0 > 0 and t_mult >= 1, got {t0} and {t_mult}")
    if t_mult == 1:
        return k * t0
    # eq: eq-restarts
    if isinstance(t0, int) and isinstance(t_mult, int):
        return t0 * (t_mult**k - 1) // (t_mult - 1)
    return t0 * (t_mult**k - 1) / (t_mult - 1)


def sgdr_position(t, t0, t_mult=1):
    """(i, T_cur, T_i) at time t: the index of the run, the time since its restart and its length, by a loop."""
    if t < 0:
        raise ValueError(f"t must be non-negative, got {t}")
    if t0 <= 0 or t_mult < 1:
        raise ValueError(f"need t0 > 0 and t_mult >= 1, got {t0} and {t_mult}")
    i, start, length = 0, 0, t0
    while t >= start + length:
        start += length
        length *= t_mult
        i += 1
    return i, t - start, length


def sgdr(t, t0, peak, final=0.0, t_mult=1):
    """Cosine annealing with warm restarts: eq-cosine inside the run that contains time t (t may be fractional)."""
    _, t_cur, t_i = sgdr_position(t, t0, t_mult)
    # eq: eq-cosine
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * t_cur / t_i))


def rates(fn, steps, **kw):
    """The array of fn(t, **kw) for t = 0, ..., steps - 1."""
    return np.array([fn(t, **kw) for t in range(steps)])


def quadratic_variance(etas, h, sigma2, v0):
    """E[x_T^2] for SGD on f(x) = h x^2 / 2 with gradient h x + noise of variance sigma2 and rates etas.

    The recursion v_{t+1} = (1 - eta_t h)^2 v_t + eta_t^2 sigma2 is exact for independent zero-mean noise.
    """
    v = v0
    for eta in etas:
        v = (1.0 - eta * h) ** 2 * v + eta**2 * sigma2
    return v


def stationary_variance(eta, h, sigma2):
    """The fixed point eta sigma2 / (h (2 - eta h)) of the recursion with a constant rate, for 0 < eta h < 2."""
    if not 0.0 < eta * h < 2.0:
        raise ValueError(f"need 0 < eta h < 2, got {eta * h}")
    return eta * sigma2 / (h * (2.0 - eta * h))
