# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://52PVZH).
"""Tests of the PyTorch code: the modular-addition split; the Fourier algorithm, exact at p = 113 with the five key
frequencies and equal to the direct cosine sum; AdamW with zero gradient as pure decay by (1 - lr * wd) per step; the
model's training step against a hand-computed AdamW update; a short run that fits a tiny table; and the worked
example of the page."""

import math

import pytest
import torch
import torch.nn.functional as F

from grokking import (
    KEY_FREQUENCIES,
    ModAddMLP,
    accuracy,
    demo,
    fourier_logits,
    make_optimizer,
    modular_addition_split,
    train_step,
    weight_norm,
)

torch.manual_seed(0)
P = 113


def all_pairs(p):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    return a.reshape(-1), b.reshape(-1)


def test_split_sizes_and_labels():
    g = torch.Generator().manual_seed(0)
    tx, ty, sx, sy = modular_addition_split(P, 0.3, g)
    assert len(tx) == len(ty) == 3831
    assert len(sx) == len(sy) == P * P - 3831
    assert torch.equal(ty, (tx[:, 0] + tx[:, 1]) % P)
    assert torch.equal(sy, (sx[:, 0] + sx[:, 1]) % P)
    seen = {tuple(r.tolist()) for r in torch.cat([tx, sx])}
    assert len(seen) == P * P


def test_split_rejects_bad_arguments():
    for frac in (0.0, 1.0):
        with pytest.raises(ValueError):
            modular_addition_split(7, frac)
    with pytest.raises(ValueError):
        modular_addition_split(1, 0.5)


def test_fourier_algorithm_is_exact_at_113():
    a, b = all_pairs(P)
    logits = fourier_logits(a, b, P)
    assert logits.shape == (P * P, P)
    assert accuracy(logits, (a + b) % P) == 1.0
    y = (a + b) % P
    assert torch.allclose(logits[torch.arange(P * P), y], torch.full((P * P,), 5.0, dtype=torch.float64))


def test_fourier_logits_equal_the_direct_cosine_sum():
    g = torch.Generator().manual_seed(3)
    a = torch.randint(0, P, (40,), generator=g)
    b = torch.randint(0, P, (40,), generator=g)
    alpha = torch.rand(len(KEY_FREQUENCIES), generator=g, dtype=torch.float64) + 0.5
    got = fourier_logits(a, b, P, alpha=alpha)
    c = torch.arange(P, dtype=torch.float64)
    want = torch.zeros(40, P, dtype=torch.float64)
    for k, al in zip(KEY_FREQUENCIES, alpha):
        want += al * torch.cos(2 * math.pi * k / P * (a[:, None] + b[:, None] - c[None, :]))
    assert torch.allclose(got, want, atol=1e-10)


def test_all_frequencies_give_minus_one_half_off_target():
    p = 11
    a, b = all_pairs(p)
    logits = fourier_logits(a, b, p, freqs=tuple(range(1, (p - 1) // 2 + 1)))
    y = (a + b) % p
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(p * p), y] = False
    assert torch.allclose(logits[mask], torch.tensor(-0.5, dtype=torch.float64))
    assert torch.allclose(logits[~mask], torch.tensor((p - 1) / 2, dtype=torch.float64))


def test_adamw_zero_gradient_is_pure_decay():
    w = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.5)
    for _ in range(10):
        opt.zero_grad()
        w.grad = torch.zeros_like(w)
        opt.step()
    assert torch.allclose(w.detach(), torch.tensor([2.0, -3.0], dtype=torch.float64) * 0.995**10)


def test_train_step_matches_a_hand_adamw_update():
    torch.manual_seed(1)
    p = 7
    model = ModAddMLP(p, d=4, hidden=8)
    tx, ty, _, _ = modular_addition_split(p, 0.5, torch.Generator().manual_seed(1))
    before = [q.detach().clone() for q in model.parameters()]
    loss = F.cross_entropy(model(tx), ty)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    opt = make_optimizer(model, lr=0.05, weight_decay=0.3)
    got_loss = train_step(model, opt, tx, ty)
    assert got_loss == pytest.approx(loss.item())
    for q0, g, q in zip(before, grads, model.parameters()):
        # first step: mhat = g and vhat = g^2
        want = q0 * (1 - 0.05 * 0.3) - 0.05 * g / (g.abs() + 1e-8)
        assert torch.allclose(q.detach(), want, atol=1e-6)


def test_make_optimizer_uses_the_original_hyperparameters():
    opt = make_optimizer(ModAddMLP(5))
    group = opt.param_groups[0]
    assert group["lr"] == 1e-3 and group["weight_decay"] == 1.0 and group["betas"] == (0.9, 0.98)
    assert isinstance(opt, torch.optim.AdamW)


def test_model_shapes_and_scale():
    torch.manual_seed(2)
    m = ModAddMLP(7, d=4, hidden=8)
    assert m(torch.tensor([[1, 2], [3, 4]])).shape == (2, 7)
    torch.manual_seed(2)
    big = ModAddMLP(7, d=4, hidden=8, scale=3.0)
    assert weight_norm(big) == pytest.approx(3.0 * weight_norm(m), rel=1e-5)


def test_short_training_fits_the_training_set():
    torch.manual_seed(5)
    p = 7
    model = ModAddMLP(p, d=8, hidden=64)
    tx, ty, _, _ = modular_addition_split(p, 0.6, torch.Generator().manual_seed(5))
    opt = make_optimizer(model, lr=1e-2, weight_decay=0.1)
    first = train_step(model, opt, tx, ty)
    for _ in range(300):
        last = train_step(model, opt, tx, ty)
    assert last < 0.1 * first
    assert accuracy(model(tx).detach(), ty) == 1.0


def test_demo_numbers():
    out = demo()
    assert out["accuracy"] == 1.0
    assert out["correct_logit"] == pytest.approx(5.0)
    assert out["largest_wrong_logit"] == pytest.approx(2.58982, abs=1e-5)
    assert out["decay_1000_steps"] == pytest.approx(0.999**1000, rel=1e-9)
