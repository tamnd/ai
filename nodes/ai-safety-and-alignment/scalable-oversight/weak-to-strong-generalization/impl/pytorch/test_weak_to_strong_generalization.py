# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://NEW-weak-to-strong-generalization).
"""Tests of the PyTorch code: PGR at its end points and at a zero gap; the adaptive threshold; the confidence loss
against its two-term form, at alpha = 0 and alpha = 1, with no gradient through the hardened target and the gradient
(p - target) / B; the warm-up; training free logits, where alpha = 0 imitates the weak labels and alpha = 1 drives the
predictions to the student's own hardened labels; and the worked example of the page."""

import pytest
import torch
import torch.nn.functional as F

from weak_to_strong_generalization import (
    adaptive_threshold,
    alpha_schedule,
    confidence_loss,
    confidence_loss_two_terms,
    fit_logits,
    harden,
    pgr,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def logit(p):
    return torch.log(p) - torch.log1p(-p)


def test_pgr():
    assert pgr(0.6, 0.6, 0.8) == 0.0
    assert pgr(0.6, 0.8, 0.8) == 1.0
    assert pgr(0.6, 0.7, 0.8) == pytest.approx(0.5)
    assert pgr(
        torch.tensor(0.6), torch.tensor(0.55), torch.tensor(0.8)
    ) == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        pgr(0.7, 0.8, 0.7)


def test_adaptive_threshold_hardens_half():
    for b in range(2, 33):
        p = torch.rand(b)
        t = adaptive_threshold(p)
        assert int(harden(p, t).sum()) == b // 2
    with pytest.raises(ValueError):
        adaptive_threshold(torch.tensor([0.5]))
    assert torch.equal(
        harden(torch.tensor([0.2, 0.5, 0.8]), 0.5), torch.tensor([0.0, 0.0, 1.0])
    )


def test_forms_agree_and_reduce_at_the_ends():
    for _ in range(20):
        z = torch.randn(12)
        w = torch.rand(12)
        a = float(torch.rand(()))
        p = torch.sigmoid(z)
        hard = harden(p, adaptive_threshold(p))
        assert confidence_loss(z, w, a).item() == pytest.approx(
            confidence_loss_two_terms(z, w, a).item()
        )
        assert confidence_loss(z, w, 0.0).item() == pytest.approx(
            F.binary_cross_entropy_with_logits(z, w).item()
        )
        assert confidence_loss(z, w, 1.0).item() == pytest.approx(
            F.binary_cross_entropy_with_logits(z, hard).item()
        )
        assert confidence_loss_two_terms(z, w, 1.0).item() == pytest.approx(
            confidence_loss(z, w, 1.0).item()
        )
    with pytest.raises(ValueError):
        confidence_loss(torch.zeros(2), torch.zeros(2), -0.5)
    with pytest.raises(ValueError):
        confidence_loss_two_terms(torch.zeros(2), torch.zeros(2), 1.5)


def test_explicit_threshold():
    z = logit(torch.tensor([0.9, 0.6, 0.4, 0.2]))
    w = torch.tensor([0.7, 0.3, 0.6, 0.1])
    all_ones = F.binary_cross_entropy_with_logits(z, torch.ones(4)).item()
    assert confidence_loss(z, w, 1.0, t=0.1).item() == pytest.approx(all_ones)
    assert confidence_loss_two_terms(z, w, 1.0, t=0.1).item() == pytest.approx(all_ones)


def test_gradient_is_p_minus_target_with_a_detached_threshold():
    z = torch.randn(8, requires_grad=True)
    w = torch.rand(8)
    a = 0.6
    confidence_loss(z, w, a).backward()
    p = torch.sigmoid(z.detach())
    target = (1 - a) * w + a * harden(p, adaptive_threshold(p))
    torch.testing.assert_close(z.grad, (p - target) / 8)
    z2 = z.detach().clone().requires_grad_(True)
    confidence_loss_two_terms(z2, w, a).backward()
    torch.testing.assert_close(z2.grad, (p - target) / 8)


def test_alpha_schedule():
    assert alpha_schedule(0, 100, 0.75) == 0.0
    assert alpha_schedule(10, 100, 0.75) == pytest.approx(0.375)
    assert alpha_schedule(20, 100, 0.75) == 0.75
    assert alpha_schedule(-1, 100, 0.5) == 0.0
    assert alpha_schedule(0, 100, 0.5, warmup_frac=0.0) == 0.5
    with pytest.raises(ValueError):
        alpha_schedule(0, 0, 0.5)
    with pytest.raises(ValueError):
        alpha_schedule(0, 10, 1.5)


def test_alpha_zero_imitates_the_weak_labels():
    w = torch.tensor([0.7, 0.3, 0.6, 0.1, 0.45, 0.9])
    p = fit_logits(torch.zeros(6), w, alpha_max=0.0, steps=3000, lr=2.0)
    torch.testing.assert_close(p, w, atol=1e-3, rtol=0)


def test_alpha_one_follows_the_students_own_labels():
    init = logit(torch.tensor([0.9, 0.6, 0.4, 0.2]))
    w = torch.tensor([0.1, 0.1, 0.9, 0.9])  # the weak labels disagree everywhere
    p = fit_logits(init, w, alpha_max=1.0, steps=3000, lr=2.0, warmup_frac=0.0)
    assert bool((p[:2] > 0.95).all()) and bool((p[2:] < 0.05).all())
    q = fit_logits(init, w, alpha_max=0.0, steps=3000, lr=2.0)
    assert bool((q[:2] < 0.15).all()) and bool((q[2:] > 0.85).all())


def test_mixed_alpha_on_a_fixed_target_reaches_the_mixture():
    init = logit(torch.tensor([0.9, 0.6, 0.4, 0.2]))
    w = torch.tensor([0.7, 0.3, 0.6, 0.1])
    p = fit_logits(init, w, alpha_max=0.5, steps=4000, lr=2.0, warmup_frac=0.0)
    torch.testing.assert_close(
        p, torch.tensor([0.85, 0.65, 0.30, 0.05]), atol=2e-3, rtol=0
    )


def test_worked_example():
    z = logit(torch.tensor([0.9, 0.6, 0.4, 0.2]))
    w = torch.tensor([0.7, 0.3, 0.6, 0.1])
    assert adaptive_threshold(torch.sigmoid(z)).item() == pytest.approx(0.4)
    assert confidence_loss(z, w, 0.0).item() == pytest.approx(0.6688, abs=1e-4)
    assert confidence_loss(z, w, 1.0).item() == pytest.approx(0.3375, abs=1e-4)
    assert confidence_loss(z, w, 0.5).item() == pytest.approx(0.5032, abs=1e-4)


def test_alpha_schedule_short_runs_and_batches_of_any_shape():
    assert alpha_schedule(0, 1, 0.5) == 0.0
    assert alpha_schedule(0, 4, 0.5) == 0.0
    p = torch.rand(2, 3)
    assert int((p > adaptive_threshold(p)).sum()) == 3
