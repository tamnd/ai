# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixed-precision-training/ (ai://EPEFQ9).
"""Mixed precision training in PyTorch, written out by hand.

DynamicLossScaler keeps a loss scale S as torch.amp.GradScaler does: scale multiplies the loss by S, unscale_ divides
the float32 gradients by S in place and reports an inf or NaN, and update halves S on an overflow and doubles it after
growth_interval good steps in a row. MasterWeights runs a model in float16 or bfloat16 with the float32 master copy
of Micikevicius et al.: backward scales the loss, step copies the half-precision gradients into the master copy,
unscales them, skips the optimizer step on an overflow, and copies the updated weights back to the model.
round_to_bf16_bits emulates the float32-to-bfloat16 conversion with integer operations on the bits.
"""

import math

import torch


class DynamicLossScaler:
    """A loss scale that backs off on overflow and grows after growth_interval good steps in a row."""

    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        if not (init_scale > 0 and math.isfinite(init_scale)):
            raise ValueError("init_scale must be positive and finite")
        if not growth_factor > 1.0:
            raise ValueError("growth_factor must be above 1")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must be in (0, 1)")
        if int(growth_interval) != growth_interval or growth_interval < 1:
            raise ValueError("growth_interval must be a positive integer")
        self.scale = torch.tensor(float(init_scale), dtype=torch.float32)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.growth_tracker = 0

    def get_scale(self):
        return float(self.scale)

    def scale_loss(self, loss):
        """The loss times S; backpropagating it gives S times the gradients."""
        return loss * self.scale.to(loss.dtype)

    @torch.no_grad()
    def unscale_(self, grads):
        """Multiply float32 gradients by 1/S in place, and tell if any of them holds an inf or a NaN."""
        inv = torch.tensor(1.0 / float(self.scale), dtype=torch.float32)
        found_inf = False
        for g in grads:
            if g is None:
                continue
            if g.dtype != torch.float32:
                raise ValueError("unscale float32 gradients; copy half-precision ones to the master weights first")
            g.mul_(inv)
            found_inf = found_inf or not bool(torch.isfinite(g).all())
        return found_inf

    def update(self, found_inf):
        """Back off and reset the count on an overflow; otherwise count the step, and grow after growth_interval of
        them unless the grown scale would be infinite in float32."""
        if found_inf:
            self.scale = self.scale * self.backoff_factor
            self.growth_tracker = 0
            return
        self.growth_tracker += 1
        if self.growth_tracker == self.growth_interval:
            grown = self.scale * self.growth_factor
            if bool(torch.isfinite(grown)):
                self.scale = grown
            self.growth_tracker = 0


class MasterWeights:
    """A model kept in a half type for the passes, with float32 master weights that the optimizer updates."""

    def __init__(self, model, dtype=torch.float16, scaler=None):
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be float16, bfloat16 or float32")
        self.model = model.to(dtype)
        self.dtype = dtype
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.master = [p.detach().to(torch.float32).clone().requires_grad_(True) for p in self.params]
        self.scaler = scaler if scaler is not None else DynamicLossScaler()

    def master_params(self):
        """The float32 parameters to hand to an optimizer."""
        return self.master

    def backward(self, loss):
        """Backpropagate the loss times S through the half-precision model."""
        self.scaler.scale_loss(loss).backward()

    @torch.no_grad()
    def step(self, optimizer):
        """Unscale the gradients into the master weights, step unless one overflowed, and refresh the model.
        Return True if the step was skipped."""
        for m, p in zip(self.master, self.params, strict=True):
            m.grad = None if p.grad is None else p.grad.to(torch.float32)
        found_inf = self.scaler.unscale_([m.grad for m in self.master])
        if not found_inf:
            optimizer.step()
            for m, p in zip(self.master, self.params, strict=True):
                p.copy_(m.to(self.dtype))
        self.scaler.update(found_inf)
        for p in self.params:
            p.grad = None
        return found_inf


def round_to_bf16_bits(x):
    """Round a float32 tensor to bfloat16, ties to even, by adding 0x7FFF plus the lowest kept bit to the bits and
    clearing the low 16 of them; returned as float32. NaNs are passed through."""
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    # The cast to int32 keeps the low 32 bits, two's complement, so negative values get their sign bit back.
    out = b.to(torch.int32).view(torch.float32)
    return torch.where(torch.isnan(x), x, out)
