# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://JDMWH4).
"""Tests of the PyTorch code: per-example gradients against a loop of single-example backward passes; B_simple and
the two-batch estimator against their closed forms and the NumPy-style fixtures; the two-batch estimate from a real
model's small- and big-batch gradients; the tracker; and the scaled LambdaLR with step decay and gradual warmup."""

import math

import pytest
import torch

from linear_scaling_rule import (
    NoiseScaleTracker,
    b_simple,
    grad_sq_norm,
    gradual_warmup,
    per_example_grads,
    scale_learning_rate,
    scaled_lambda_lr,
    two_batch_estimates,
)

torch.manual_seed(0)
DT = torch.float64


def _flat_grad(model):
    grads = []
    for p in model.parameters():
        assert p.grad is not None
        grads.append(p.grad.reshape(-1))
    return torch.cat(grads)


def _model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).to(DT)


def test_per_example_grads_match_loop():
    model = _model()
    x = torch.randn(6, 3, dtype=DT)
    y = torch.randint(0, 2, (6,))
    loss_fn = torch.nn.functional.cross_entropy
    pe = per_example_grads(model, loss_fn, x, y)
    n_params = sum(p.numel() for p in model.parameters())
    assert pe.shape == (6, n_params)
    for i in range(6):
        model.zero_grad()
        loss_fn(model(x[i : i + 1]), y[i : i + 1]).backward()
        ref = _flat_grad(model)
        assert torch.allclose(pe[i], ref)
    # the mean of per-example gradients is the batch gradient
    model.zero_grad()
    loss_fn(model(x), y).backward()
    full = _flat_grad(model)
    assert torch.allclose(pe.mean(dim=0), full)


def test_frozen_parameters_skipped():
    model = _model()
    model[0].weight.requires_grad_(False)
    pe = per_example_grads(
        model,
        torch.nn.functional.cross_entropy,
        torch.randn(3, 3, dtype=DT),
        torch.tensor([0, 1, 0]),
    )
    assert pe.shape == (3, 4 + 8 + 2)


def test_b_simple():
    g = torch.tensor([[3.0, 1.0], [1.0, -1.0], [2.0, 2.0], [2.0, -2.0]], dtype=DT)
    b, g2, tr = b_simple(g)
    assert (b, g2, tr) == pytest.approx((4.0 / 3.0, 3.0, 4.0))
    assert b_simple(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DT))[0] == math.inf
    for bad in (torch.ones(1, 3), torch.ones(4)):
        with pytest.raises(ValueError):
            b_simple(bad)


def test_two_batch():
    assert two_batch_estimates(torch.tensor(3.0), 1.25, 32, 256) == pytest.approx((1.0, 64.0))
    for bad in [(0, 4), (4, 4), (8, 4)]:
        with pytest.raises(ValueError):
            two_batch_estimates(1.0, 1.0, *bad)


def test_two_batch_from_model_gradients():
    model = _model()
    loss_fn = torch.nn.functional.cross_entropy
    x = torch.randn(4096, 3, dtype=DT)
    y = (x[:, 0] + 0.5 * torch.randn(4096, dtype=DT) > 0).long()
    pe = per_example_grads(model, loss_fn, x, y)
    true_g2 = float(pe.mean(0) @ pe.mean(0))
    true_tr = float(((pe - pe.mean(0)) ** 2).sum(1).mean())
    bs, bb, est = 8, 128, []
    gen = torch.Generator().manual_seed(5)
    for _ in range(1500):
        idx = torch.randint(0, 4096, (bb,), generator=gen)
        model.zero_grad()
        loss_fn(model(x[idx[:bs]]), y[idx[:bs]]).backward()
        gs = grad_sq_norm(model.parameters())
        model.zero_grad()
        loss_fn(model(x[idx]), y[idx]).backward()
        gb = grad_sq_norm(model.parameters())
        est.append(two_batch_estimates(gs, gb, bs, bb))
    est = torch.tensor(est, dtype=DT)
    assert float(est[:, 1].mean()) == pytest.approx(true_tr, rel=0.06)
    assert float(est[:, 0].mean()) == pytest.approx(true_g2, abs=0.05 * true_tr)


def test_grad_sq_norm_skips_missing():
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    b = torch.nn.Parameter(torch.tensor([1.0]))
    (a.sum() * 1.0).backward()
    assert a.grad is not None
    a.grad.mul_(torch.tensor([3.0, 4.0]))
    assert grad_sq_norm([a, b]) == pytest.approx(25.0)


def test_tracker():
    t = NoiseScaleTracker(32, 256)
    assert t.decay == 0.99
    with pytest.raises(ValueError):
        _ = t.value
    assert t.update(3.0, 1.25) == pytest.approx(64.0)
    assert t.update(3.0, 1.25) == pytest.approx(64.0)
    t2 = NoiseScaleTracker(1, 2, decay=0.5)
    t2.update(3.0, 2.0)  # |G|^2 = 1, tr = 2
    assert t2.update(5.0, 3.0) == pytest.approx((0.25 * 2 + 0.5 * 4) / (0.25 * 1 + 0.5 * 1))
    assert NoiseScaleTracker(1, 2).update(5.0, 2.0) == math.inf
    assert NoiseScaleTracker(1, 2, decay=0.0).update(2.0, 1.0) == math.inf
    for bad in (1.0, -0.5):
        with pytest.raises(ValueError):
            NoiseScaleTracker(1, 2, decay=bad)


def test_scale_and_warmup():
    assert scale_learning_rate(0.1, 256, 8192) == pytest.approx(3.2)
    assert scale_learning_rate(0.1, 256, 8192, "sqrt") == pytest.approx(0.1 * math.sqrt(32))
    assert scale_learning_rate(0.1, 256, 8192, "none") == 0.1
    for bad in [(0.1, 0, 4), (0.1, 4, 0)]:
        with pytest.raises(ValueError):
            scale_learning_rate(*bad)
    with pytest.raises(ValueError):
        scale_learning_rate(0.1, 4, 8, "log")
    h = gradual_warmup(4, 0.25)
    assert [h(s, 1.0) for s in range(6)] == pytest.approx([0.25, 0.4375, 0.625, 0.8125, 1.0, 1.0])
    assert gradual_warmup(0, 0.5)(0, 2.0) == 2.0
    assert gradual_warmup(2, 0.0)(1, 2.0) == pytest.approx(1.0)
    assert gradual_warmup(2, 1.0)(1, 2.0) == pytest.approx(2.0)
    for bad in [(-1, 0.5), (3, -0.1), (3, 1.5)]:
        with pytest.raises(ValueError):
            gradual_warmup(*bad)


def test_scaled_lambda_lr():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)

    def decay(step):
        return 0.1 ** sum(step >= e for e in (30, 60, 80))

    sched = scaled_lambda_lr(opt, 256, 8192, schedule=decay, warmup=gradual_warmup(5, 0.1 / 3.2))
    lrs = []
    for _ in range(90):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[5] == pytest.approx(3.2)
    assert lrs[29] == pytest.approx(3.2)
    assert lrs[30] == pytest.approx(0.32)
    assert lrs[85] == pytest.approx(0.0032)
    opt2 = torch.optim.SGD([p], lr=0.1)
    scaled_lambda_lr(opt2, 256, 1024)
    assert opt2.param_groups[0]["lr"] == pytest.approx(0.4)
    opt3 = torch.optim.SGD([p], lr=0.1)
    scaled_lambda_lr(opt3, 256, 1024, rule="sqrt")
    assert opt3.param_groups[0]["lr"] == pytest.approx(0.2)


def test_scaled_lambda_lr_rescales_existing_scheduler_groups():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)  # sets initial_lr = 0.1
    scaled_lambda_lr(opt, 256, 512)
    assert opt.param_groups[0]["initial_lr"] == pytest.approx(0.2)
