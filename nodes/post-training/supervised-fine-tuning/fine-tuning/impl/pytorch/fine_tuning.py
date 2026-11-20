# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/fine-tuning/ (ai://NEW-fine-tuning).
"""Fine-tuning a pre-trained network in PyTorch: the pieces that differ from training from scratch.

* `discriminative_param_groups` gives each layer its own learning rate, the one above divided by a factor (ULMFiT);
* `stlr_lambda` is ULMFiT's slanted triangular schedule and `warmup_linear_lambda` BERT's, both for LambdaLR;
* `GradualUnfreezer` trains the top layer first and unfreezes one more layer per epoch;
* `linear_probe_then_fine_tune` trains the new head on frozen features before training everything (LP-FT);
* `reinit_top_layers` re-initializes the top blocks of a pre-trained stack, as Zhang et al. do for BERT;
* `EWC` holds a diagonal Fisher and the old weights and adds the elastic weight consolidation penalty.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def discriminative_param_groups(layers, lr_top, factor=2.6):
    """Optimizer parameter groups for layers listed bottom first; layer l gets lr_top / factor^(L - 1 - l)."""
    n = len(layers)
    return [{"params": list(m.parameters()), "lr": lr_top / factor ** (n - 1 - i)} for i, m in enumerate(layers)]


def stlr_lambda(total, cut_frac=0.1, ratio=32):
    """Multiplier of the peak learning rate at step t for torch.optim.lr_scheduler.LambdaLR."""
    cut = math.floor(total * cut_frac)

    def f(t):
        p = t / cut if t < cut else 1 - (t - cut) / (cut * (1 / cut_frac - 1))
        return (1 + p * (ratio - 1)) / ratio

    return f


def warmup_linear_lambda(total, warmup_frac=0.1):
    w = warmup_frac * total
    return lambda t: t / w if t < w else max(total - t, 0) / (total - w)


class GradualUnfreezer:
    """Freezes all layers (listed bottom first) except the top one; each call to step() unfreezes the next lower."""

    def __init__(self, layers):
        self.layers, self.unfrozen = list(layers), 0
        for m in self.layers:
            set_trainable(m, False)
        self.step()

    def step(self):
        if self.unfrozen < len(self.layers):
            self.unfrozen += 1
            set_trainable(self.layers[-self.unfrozen], True)


def train(params, loss_fn, steps, lr):
    opt = torch.optim.SGD([p for p in params if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()


def linear_probe_then_fine_tune(body, head, loss_fn, probe_steps, tune_steps, lr):
    """Train the head with the body frozen, then train both from there."""
    set_trainable(body, False)
    train(head.parameters(), loss_fn, probe_steps, lr)
    set_trainable(body, True)
    train(list(body.parameters()) + list(head.parameters()), loss_fn, tune_steps, lr)


@torch.no_grad()
def reinit_top_layers(blocks, n, std=0.02):
    """Re-initialize the last n blocks: linear and embedding weights from N(0, std^2), biases to 0, LayerNorm to 1, 0."""
    for block in list(blocks)[len(blocks) - n :]:
        for m in block.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.normal_(0.0, std)
                if getattr(m, "bias", None) is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()


class EWC:
    """Elastic weight consolidation: (lam / 2) sum_i F_i (theta_i - theta_old_i)^2 over the model's parameters, with F
    the diagonal Fisher of a classifier on the old task, the expectation over its own predicted labels taken exactly."""

    def __init__(self, model, inputs, lam):
        self.lam = lam
        self.old = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        params = [p for _, p in model.named_parameters()]
        for x in inputs:
            logp = F.log_softmax(model(x[None]), dim=-1)[0]
            probs = logp.exp().detach()
            for y in range(len(probs)):
                gs = torch.autograd.grad(logp[y], params, retain_graph=True)
                for (n, _), g in zip(model.named_parameters(), gs, strict=True):
                    self.fisher[n] += probs[y] * g * g / len(inputs)

    def penalty(self, model):
        return 0.5 * self.lam * sum((self.fisher[n] * (p - self.old[n]) ** 2).sum() for n, p in model.named_parameters())
