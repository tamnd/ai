# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://98TJG9).
"""Tests of the PyTorch code: the shared fixture, agreement with torch.nn.functional.cross_entropy and
nn.CrossEntropyLoss with label_smoothing (class weights, ignore_index, reductions, extra axes and soft targets), the
convention that spreads eps over all K classes, the pair of losses, the gradient p - q', the logit gap that training
reaches, and the confidence penalty."""

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from label_smoothing import (
    LabelSmoothingCrossEntropy,
    confidence_penalty_loss,
    entropy,
    smooth_labels,
    smoothed_cross_entropy,
)

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
LOGITS = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
LABELS = [0, 2, 3]
FIXTURE = {0.0: 1.0819850661352493, 0.1: 1.1778183994685827, 0.3: 1.3694850661352493}


@pytest.mark.parametrize("eps", sorted(FIXTURE))
def test_fixture(eps):
    z, y = torch.tensor(LOGITS), torch.tensor(LABELS)
    torch.testing.assert_close(
        smoothed_cross_entropy(z, y, eps),
        torch.tensor(FIXTURE[eps]),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        F.cross_entropy(z, y, label_smoothing=eps),
        torch.tensor(FIXTURE[eps]),
        rtol=0,
        atol=1e-12,
    )


def test_worked_example():
    z = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    loss = F.cross_entropy(z, torch.tensor([0]), label_smoothing=0.3)
    assert loss.item() == pytest.approx(0.639545, abs=1e-6)
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[-0.013014, 0.006507, 0.006507]]), rtol=0, atol=1e-6)


def test_default_is_no_smoothing():
    assert nn.CrossEntropyLoss().label_smoothing == 0.0
    z, y = torch.tensor(LOGITS), torch.tensor(LABELS)
    torch.testing.assert_close(F.cross_entropy(z, y), F.cross_entropy(z, y, label_smoothing=0.0))
    torch.testing.assert_close(F.cross_entropy(z, y), -F.log_softmax(z, dim=1)[range(3), y].mean())


CASES = st.fixed_dictionaries(
    {
        "k": st.integers(2, 7),
        "n": st.integers(1, 6),
        "extra": st.sampled_from([(), (3,), (2, 2)]),
        "eps": st.floats(0.0, 1.0),
        "weighted": st.booleans(),
        "ignore": st.booleans(),
        "reduction": st.sampled_from(["mean", "sum", "none"]),
        "seed": st.integers(0, 2**31 - 1),
    }
)


@settings(max_examples=150, deadline=None)
@given(CASES)
def test_matches_torch(c):
    g = torch.Generator().manual_seed(c["seed"])
    k = c["k"]
    z = torch.randn((c["n"], k, *c["extra"]), generator=g) * 3
    y = torch.randint(0, k, (c["n"], *c["extra"]), generator=g)
    kw = {}
    if c["weighted"]:
        kw["weight"] = torch.rand(k, generator=g) + 0.1
    if c["ignore"]:
        y.view(-1)[0] = -100
        if y.numel() == 1:
            return  # every label ignored: the mean is 0/0 in both
    ours = smoothed_cross_entropy(z, y, c["eps"], reduction=c["reduction"], **kw)
    ref = F.cross_entropy(z, y, label_smoothing=c["eps"], reduction=c["reduction"], **kw)
    torch.testing.assert_close(ours, ref, rtol=1e-10, atol=1e-12)
    module = nn.CrossEntropyLoss(label_smoothing=c["eps"], reduction=c["reduction"], **kw)
    torch.testing.assert_close(
        LabelSmoothingCrossEntropy(c["eps"], reduction=c["reduction"], **kw)(z, y),
        module(z, y),
        rtol=1e-10,
        atol=1e-12,
    )


def test_other_ignore_index():
    z, y = torch.tensor(LOGITS), torch.tensor([0, 7, 3])
    torch.testing.assert_close(
        smoothed_cross_entropy(z, y, 0.2, ignore_index=7),
        F.cross_entropy(z, y, label_smoothing=0.2, ignore_index=7),
    )
    keep = torch.tensor([0, 2])
    torch.testing.assert_close(
        smoothed_cross_entropy(z, y, 0.2, ignore_index=7),
        F.cross_entropy(z[keep], y[keep], label_smoothing=0.2),
    )


def test_torch_spreads_eps_over_all_classes():
    """The label gets 1 - eps + eps / K, not 1 - eps: torch equals soft targets from smooth_labels."""
    z, y, eps = torch.tensor(LOGITS), torch.tensor(LABELS), 0.2
    q = smooth_labels(y, 4, eps)
    torch.testing.assert_close(q[0], torch.tensor([0.85, 0.05, 0.05, 0.05]))
    torch.testing.assert_close(F.cross_entropy(z, y, label_smoothing=eps), F.cross_entropy(z, q))
    # Soft targets are smoothed too: a one-hot target with label_smoothing is the same loss.
    torch.testing.assert_close(
        F.cross_entropy(z, F.one_hot(y, 4).double(), label_smoothing=eps),
        F.cross_entropy(z, q),
    )
    # The convention with eps / (K - 1) on the wrong classes is torch's with eps K / (K - 1).
    other = torch.full((3, 4), eps / 3)
    other[range(3), y] = 1 - eps
    assert not torch.allclose(F.cross_entropy(z, other), F.cross_entropy(z, q))
    torch.testing.assert_close(F.cross_entropy(z, other), F.cross_entropy(z, y, label_smoothing=eps * 4 / 3))


def test_prior():
    """A non-uniform prior, given as a list: (1 - eps) one_hot + eps * prior, and the loss with those soft targets."""
    q = smooth_labels(torch.tensor([1]), 3, 0.5, prior=[0.5, 0.25, 0.25])
    torch.testing.assert_close(q, torch.tensor([[0.25, 0.625, 0.125]]))
    q = smooth_labels(torch.tensor([0, 2]), 3, 0.2, prior=[0.6, 0.3, 0.1])
    torch.testing.assert_close(q, torch.tensor([[0.92, 0.06, 0.02], [0.12, 0.06, 0.82]]))
    z = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    logp = F.log_softmax(z, dim=1)
    nll = -logp[range(2), [0, 2]]
    prior_part = -(logp * torch.tensor([0.6, 0.3, 0.1])).sum(1)
    torch.testing.assert_close(F.cross_entropy(z, q), (0.8 * nll + 0.2 * prior_part).mean())


def test_decomposition():
    z, y, eps = torch.tensor(LOGITS), torch.tensor(LABELS), 0.3
    logp = F.log_softmax(z, dim=1)
    nll = F.nll_loss(logp, y)
    h_up = -logp.mean(dim=1).mean()
    kl_up = F.kl_div(logp, torch.full_like(logp, 0.25), reduction="sum") / 3
    loss = F.cross_entropy(z, y, label_smoothing=eps)
    torch.testing.assert_close(loss, (1 - eps) * nll + eps * h_up)
    torch.testing.assert_close(loss, (1 - eps) * nll + eps * kl_up + eps * math.log(4))


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.5, 1.0])
def test_gradient_is_p_minus_q(eps):
    z = torch.tensor(LOGITS, requires_grad=True)
    y = torch.tensor(LABELS)
    F.cross_entropy(z, y, label_smoothing=eps).backward()
    expected = (torch.softmax(z.detach(), dim=1) - smooth_labels(y, 4, eps)) / 3
    grad = z.grad
    assert grad is not None
    torch.testing.assert_close(grad, expected)
    torch.testing.assert_close(grad.sum(dim=1), torch.zeros(3), rtol=0, atol=1e-15)
    assert grad.abs().max() * 3 <= 1


@pytest.mark.parametrize("eps", [-0.1, 1.5])
def test_eps_out_of_range(eps):
    z, y = torch.tensor(LOGITS), torch.tensor(LABELS)
    with pytest.raises(ValueError):
        smoothed_cross_entropy(z, y, eps)
    with pytest.raises(ValueError):
        LabelSmoothingCrossEntropy(eps)
    with pytest.raises(ValueError):
        smooth_labels(y, 4, eps)


def test_torch_range_check():
    """torch rejects label_smoothing above 1, but takes a negative value silently as no smoothing at all."""
    z, y = torch.tensor(LOGITS), torch.tensor(LABELS)
    with pytest.raises(RuntimeError):
        F.cross_entropy(z, y, label_smoothing=1.5)
    for eps in (-0.1, -1.0):
        torch.testing.assert_close(F.cross_entropy(z, y, label_smoothing=eps), F.cross_entropy(z, y))
        torch.testing.assert_close(nn.CrossEntropyLoss(label_smoothing=eps)(z, y), F.cross_entropy(z, y))


def test_bad_reduction():
    with pytest.raises(ValueError):
        smoothed_cross_entropy(torch.tensor(LOGITS), torch.tensor(LABELS), 0.1, reduction="avg")


@pytest.mark.parametrize("k,eps", [(3, 0.3), (10, 0.1), (5, 0.6)])
def test_training_reaches_the_logit_gap(k, eps):
    """SGD on free logits ends at p = q' with every wrong logit log(1 + K (1 - eps) / eps) below the label's."""
    z = torch.zeros(1, k, requires_grad=True)
    opt = torch.optim.SGD([z], lr=2.0)
    y = torch.tensor([1])
    for _ in range(3000):
        opt.zero_grad()
        F.cross_entropy(z, y, label_smoothing=eps).backward()
        opt.step()
    gap = z[0, 1] - torch.cat([z[0, :1], z[0, 2:]])
    torch.testing.assert_close(gap, torch.full((k - 1,), math.log1p(k * (1 - eps) / eps)), rtol=0, atol=1e-8)
    torch.testing.assert_close(torch.softmax(z, 1), smooth_labels(y, k, eps), rtol=0, atol=1e-9)


def test_hard_targets_keep_growing():
    z = torch.zeros(1, 4, requires_grad=True)
    opt = torch.optim.SGD([z], lr=2.0)
    gaps = []
    for _ in range(1000):
        opt.zero_grad()
        F.cross_entropy(z, torch.tensor([0])).backward()
        opt.step()
        gaps.append((z[0, 0] - z[0, 1:].max()).item())
    assert all(gaps[i + 1] > gaps[i] for i in range(len(gaps) - 1))
    assert gaps[-1] > 5


def test_confidence_penalty():
    z = torch.tensor(LOGITS, requires_grad=True)
    y = torch.tensor(LABELS)
    ent = entropy(z)
    logp = F.log_softmax(z, dim=1)
    torch.testing.assert_close(-ent, (logp.exp() * (logp + math.log(4))).sum(1) - math.log(4))
    # The gradient of the entropy in the logits is p_i (-log p_i - H), as Pereyra et al. write it.
    (grad,) = torch.autograd.grad(ent.sum(), z)
    p = logp.exp().detach()
    torch.testing.assert_close(grad, p * (-logp.detach() - ent.detach().unsqueeze(1)))
    loss = confidence_penalty_loss(z, y, 0.5)
    torch.testing.assert_close(loss, F.cross_entropy(z, y) - 0.5 * ent.mean())


def test_smoothing_lowers_confidence_of_a_trained_classifier():
    """On a separable problem a linear classifier trained with smoothing ends less confident than with hard targets,
    and its mean largest probability stays below about 1 - eps + eps / K, the optimum of one example."""
    g = torch.Generator().manual_seed(0)
    centers = ([1.0, 0.0], [0.0, 1.0], [-1.0, -1.0])
    x = torch.cat([torch.randn(60, 2, generator=g) + 4 * torch.tensor(c) for c in centers])
    y = torch.arange(3).repeat_interleave(60)
    conf = {}
    for eps in (0.0, 0.2):
        torch.manual_seed(0)
        model = nn.Linear(2, 3)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        for _ in range(2000):
            opt.zero_grad()
            F.cross_entropy(model(x), y, label_smoothing=eps).backward()
            opt.step()
        conf[eps] = torch.softmax(model(x), 1).max(1).values.mean().item()
    assert conf[0.2] < conf[0.0]
    assert conf[0.0] > 0.97
    assert 0.75 < conf[0.2] < 1 - 0.2 + 0.2 / 3 + 0.01
