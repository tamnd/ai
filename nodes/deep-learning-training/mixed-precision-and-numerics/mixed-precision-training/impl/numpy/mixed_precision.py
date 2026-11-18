# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://EPEFQ9).
"""Mixed precision training in NumPy.

format_limits gives the machine epsilon, smallest normal and subnormal and largest value of a binary floating-point
format, as in the derivation number-formats. round_to_bf16 and truncate_to_bf16 emulate BF16 on float32 by keeping
the top 16 bits, the way Kalamkar et al. emulate it. update_is_lost tells if adding an update to a weight leaves the
weight unchanged, and largest_safe_scale picks a constant loss scale from the largest gradient.

DynamicLossScaler holds a loss scale S: it multiplies the loss by S, unscales the half-precision gradients into
float32, halves S and reports a skipped step when a gradient is inf or NaN, and doubles S after growth_interval steps
in a row without one, with the same rule and defaults as torch.amp.GradScaler. MixedPrecisionSGD keeps the float32
master weights of Micikevicius et al. and hands out a half-precision copy for the forward and backward passes.
"""

import numpy as np

# (exponent bits, mantissa bits) of the IEEE-style formats.
FORMATS = {
    "fp32": (8, 23),
    "fp16": (5, 10),
    "bf16": (8, 7),
    "e5m2": (5, 2),
    "e4m3": (4, 3),
}


def format_limits(exp_bits, man_bits, ieee=True):
    """eps, min_normal, min_subnormal and max of a format with the given field widths and bias 2^(e-1) - 1.

    With ieee=True the top exponent is kept for infinities and NaNs. With ieee=False it holds normal numbers and only
    the all-ones mantissa there is NaN, the E4M3 encoding of Micikevicius et al. (2022)."""
    if exp_bits < 2 or man_bits < 1:
        raise ValueError("need at least 2 exponent bits and 1 mantissa bit")
    bias = 2 ** (exp_bits - 1) - 1
    e_min = 1 - bias
    if ieee:
        top = (2.0 - 2.0**-man_bits) * 2.0 ** (2**exp_bits - 2 - bias)
    else:
        top = (2.0 - 2.0 ** (1 - man_bits)) * 2.0 ** (2**exp_bits - 1 - bias)
    return {
        "eps": 2.0**-man_bits,
        "min_normal": 2.0**e_min,
        "min_subnormal": 2.0 ** (e_min - man_bits),
        "max": top,
    }


def _bits32(x):
    return np.array(x, dtype=np.float32).view(np.uint32)


def round_to_bf16(x):
    """Round float32 values to the nearest BF16 value, ties to even, returned as float32. NaN stays NaN, and values
    that round past the largest BF16 value become infinities, as in a hardware conversion."""
    b = _bits32(x).astype(np.uint64)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    rounded = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    out = rounded.astype(np.uint32).view(np.float32)
    return np.where(nan, np.float32(np.nan), out)


def truncate_to_bf16(x):
    """Cut float32 values to BF16 by zeroing the low 16 bits, rounding toward zero."""
    return (_bits32(x) & np.uint32(0xFFFF0000)).view(np.float32)


def update_is_lost(w, u, dtype: type = np.float16):
    """True where w + u, rounded once to dtype, equals w; w and u are first rounded to dtype.

    For float16 the sum is formed exactly in float64 and rounded once, which NumPy's own float16 addition does not
    promise; float32 addition is already correctly rounded."""
    if dtype not in (np.float16, np.float32):
        raise ValueError("dtype must be np.float16 or np.float32")
    w = np.asarray(w, dtype=dtype)
    u = np.asarray(u, dtype=dtype)
    if dtype is np.float16:
        with np.errstate(over="ignore"):
            total = (w.astype(np.float64) + u.astype(np.float64)).astype(np.float16)
    else:
        with np.errstate(over="ignore"):
            total = w + u
    return total == w


def largest_safe_scale(max_abs_grad, fmt_max=65504.0):
    """The largest power of two S with S * max_abs_grad <= fmt_max, the constant loss scale that cannot overflow."""
    if not max_abs_grad > 0 or not np.isfinite(max_abs_grad):
        raise ValueError("max_abs_grad must be positive and finite")
    return 2.0 ** np.floor(np.log2(fmt_max / max_abs_grad))


class DynamicLossScaler:
    """A loss scale that backs off on overflow and grows after growth_interval good steps in a row."""

    def __init__(
        self,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ):
        if not (init_scale > 0 and np.isfinite(init_scale)):
            raise ValueError("init_scale must be positive and finite")
        if not growth_factor > 1.0:
            raise ValueError("growth_factor must be above 1")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must be in (0, 1)")
        if int(growth_interval) != growth_interval or growth_interval < 1:
            raise ValueError("growth_interval must be a positive integer")
        self.scale = np.float32(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = int(growth_interval)
        self.growth_tracker = 0

    def scale_loss(self, loss):
        """The loss times S; its gradients are S times the gradients of the loss."""
        return loss * self.scale

    def unscale(self, grads):
        """Divide the gradients by S in float32, and tell if any of them is inf or NaN."""
        inv = np.float32(1.0 / np.float64(self.scale))
        out = [np.asarray(g).astype(np.float32) * inv for g in grads]
        found_inf = not all(np.isfinite(g).all() for g in out)
        return out, found_inf

    def update(self, found_inf):
        """The update rule of torch.amp.GradScaler: back off and reset the count on an overflow, otherwise count the
        step and grow after growth_interval of them, unless the grown scale would be infinite."""
        if found_inf:
            self.scale = np.float32(self.scale * self.backoff_factor)
            self.growth_tracker = 0
            return
        self.growth_tracker += 1
        if self.growth_tracker == self.growth_interval:
            with np.errstate(over="ignore"):
                grown = np.float32(self.scale * np.float32(self.growth_factor))
            if np.isfinite(grown):
                self.scale = grown
            self.growth_tracker = 0


class MixedPrecisionSGD:
    """Plain SGD on float32 master weights, with a half-precision copy of them for the passes and loss scaling."""

    def __init__(self, params, lr, half=np.float16, scaler=None):
        self.master = [np.asarray(p, dtype=np.float32).copy() for p in params]
        self.lr = np.float32(lr)
        self.half = half
        self.scaler = scaler if scaler is not None else DynamicLossScaler()

    def half_params(self):
        """The weights rounded to the half type, for the forward and backward passes."""
        return [m.astype(self.half) for m in self.master]

    def step(self, scaled_grads):
        """Apply the gradients of the scaled loss. Return True if the step was skipped because of an overflow."""
        grads, found_inf = self.scaler.unscale(scaled_grads)
        if not found_inf:
            for m, g in zip(self.master, grads, strict=True):
                m -= self.lr * g
        self.scaler.update(found_inf)
        return found_inf
