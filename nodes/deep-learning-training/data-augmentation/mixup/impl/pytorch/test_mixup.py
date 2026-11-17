# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mixup/ (ai://NEW-mixup).
"""Tests of the PyTorch implementation: the shared fixture, agreement with F.cross_entropy on soft targets, gradients,
the pairing of inputs and labels, the CutMix box and weight, integer inputs, and the module in evaluation mode."""

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from mixup import (
    Mixup,
    beta_variance,
    cutmix_batch,
    cutmix_box,
    cutmix_lambda,
    mixed_cross_entropy,
    mixup_batch,
    mixup_targets,
    sample_cutmix_box,
    sample_lambda,
)

torch.set_default_dtype(torch.float64)

LOGITS = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
Y_A = [0, 2, 3]
Y_B = [1, 2, 0]
FIXTURE = {0.0: 2.3319850661352497, 0.3: 1.9569850661352495, 0.7: 1.4569850661352497, 1.0: 1.0819850661352495}
CUTMIX = {"height": 6, "width": 8, "lam": 0.64, "center": (1.0, 5.0), "box": (0, 3, 3, 6), "adjusted": 0.8125}


@pytest.mark.parametrize("lam", sorted(FIXTURE))
def test_fixture(lam):
    z = torch.tensor(LOGITS)
    ya, yb = torch.tensor(Y_A), torch.tensor(Y_B)
    assert float(mixed_cross_entropy(z, ya, yb, lam)) == pytest.approx(FIXTURE[lam], abs=1e-12)
    q = lam * F.one_hot(ya, 4).double() + (1 - lam) * F.one_hot(yb, 4).double()
    assert float(F.cross_entropy(z, q)) == pytest.approx(FIXTURE[lam], abs=1e-12)


def test_cutmix_fixture():
    box = cutmix_box(CUTMIX["height"], CUTMIX["width"], CUTMIX["lam"], CUTMIX["center"])
    assert box == CUTMIX["box"]
    assert cutmix_lambda(box, CUTMIX["height"], CUTMIX["width"]) == CUTMIX["adjusted"]


def test_worked_example():
    x, _, _ = mixup_batch(torch.tensor([[2.0, 0.0], [0.0, 4.0]]), torch.tensor([0, 2]), 0.7, torch.tensor([1, 0]))
    torch.testing.assert_close(x[0], torch.tensor([1.4, 1.2]), atol=1e-15, rtol=0)
    z = torch.tensor([[1.0, 0.5, -0.5]], requires_grad=True)
    loss = mixed_cross_entropy(z, torch.tensor([0]), torch.tensor([2]), 0.7)
    assert float(loss.detach()) == pytest.approx(1.054131, abs=1e-6)
    loss.backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad[0], torch.tensor([-0.153451, 0.331499, -0.178048]), atol=1e-6, rtol=0)
    assert cutmix_box(32, 32, 0.75, (4.0, 28.0)) == (0, 12, 20, 32)
    assert cutmix_lambda((0, 12, 20, 32), 32, 32) == 0.859375


@settings(max_examples=100, deadline=None)
@given(st.integers(1, 12), st.integers(2, 9), st.floats(0.0, 1.0), st.integers(0, 2**31 - 1))
def test_matches_soft_target_cross_entropy(b, k, lam, seed):
    g = torch.Generator().manual_seed(seed)
    z = (3.0 * torch.randn(b, k, generator=g)).requires_grad_()
    y = torch.randint(k, (b,), generator=g)
    perm = torch.randperm(b, generator=g)
    q = mixup_targets(y, k, lam, perm)
    torch.testing.assert_close(q.sum(-1), torch.ones(b), atol=1e-12, rtol=0)
    soft = F.cross_entropy(z, q)
    two = mixed_cross_entropy(z, y, y[perm], lam)
    assert float(soft.detach()) == pytest.approx(float(two.detach()), abs=1e-10)
    (g_soft,) = torch.autograd.grad(soft, z)
    (g_two,) = torch.autograd.grad(two, z)
    torch.testing.assert_close(g_soft, g_two, atol=1e-12, rtol=0)
    # The gradient of the mean loss is (softmax - target) / B.
    torch.testing.assert_close(g_two, (torch.softmax(z.detach(), -1) - q) / b, atol=1e-12, rtol=0)


def test_labels_follow_inputs():
    g = torch.Generator().manual_seed(0)
    k, b = 5, 16
    y = torch.randint(k, (b,), generator=g)
    x = F.one_hot(y, k).double()
    for _ in range(20):
        lam = sample_lambda(0.4, g)
        perm = torch.randperm(b, generator=g)
        xm, ya, yb = mixup_batch(x, y, lam, perm)
        torch.testing.assert_close(xm, mixup_targets(y, k, lam, perm), atol=1e-15, rtol=0)
        assert torch.equal(ya, y) and torch.equal(yb, y[perm])


def test_endpoints():
    x = torch.arange(12.0).reshape(4, 3)
    perm = torch.tensor([3, 2, 1, 0])
    y = torch.arange(4)
    assert torch.equal(mixup_batch(x, y, 1.0, perm)[0], x)
    assert torch.equal(mixup_batch(x, y, 0.0, perm)[0], x[perm])
    torch.testing.assert_close(mixup_batch(x, y, 0.25, perm)[0], 0.25 * x + 0.75 * x[perm])


@pytest.mark.parametrize("lam", [-0.1, 1.1, float("nan")])
def test_lam_out_of_range(lam):
    with pytest.raises(ValueError):
        mixup_batch(torch.zeros(2, 1), torch.tensor([0, 1]), lam, torch.tensor([1, 0]))
    with pytest.raises(ValueError):
        mixup_targets(torch.tensor([0, 1]), 2, lam, torch.tensor([1, 0]))
    with pytest.raises(ValueError):
        mixed_cross_entropy(torch.zeros(1, 2), torch.tensor([0]), torch.tensor([1]), lam)
    with pytest.raises(ValueError):
        cutmix_box(4, 4, lam, (2.0, 2.0))


@pytest.mark.parametrize("perm", [[0, 0, 1], [0, 1], [0, 1, 3]])
def test_bad_permutation(perm):
    with pytest.raises(ValueError):
        mixup_batch(torch.zeros(3, 2), torch.arange(3), 0.5, perm)
    with pytest.raises(ValueError):
        mixup_targets(torch.arange(3), 3, 0.5, perm)
    with pytest.raises(ValueError):
        cutmix_batch(torch.zeros(3, 1, 4, 4), torch.arange(3), (0, 2, 0, 2), perm)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan")])
def test_alpha_must_be_positive(alpha):
    with pytest.raises(ValueError):
        sample_lambda(alpha)
    with pytest.raises(ValueError):
        Mixup(alpha)


@pytest.mark.parametrize("alpha", [0.1, 0.2, 1.0, 4.0])
def test_beta_moments(alpha):
    torch.manual_seed(1)
    lam = torch.tensor([sample_lambda(alpha) for _ in range(4000)])
    assert bool(((lam >= 0) & (lam <= 1)).all())
    tol = 5 * math.sqrt(beta_variance(alpha) / lam.numel())
    assert abs(float(lam.mean()) - 0.5) < tol
    assert abs(float(lam.var(unbiased=False)) - beta_variance(alpha)) < tol
    # With a generator the draws are repeatable and leave the global RNG alone.
    state = torch.get_rng_state()
    a = [sample_lambda(alpha, torch.Generator().manual_seed(7)) for _ in range(3)]
    assert a[0] == a[1] == a[2] and 0.0 <= a[0] <= 1.0
    assert len({sample_lambda(alpha, torch.Generator().manual_seed(s)) for s in range(8)}) == 8
    assert torch.equal(state, torch.get_rng_state())


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 40), st.integers(1, 40), st.floats(0.0, 1.0), st.integers(0, 2**31 - 1))
def test_cutmix_lambda_is_pixel_fraction(h, w, lam, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, h, w, generator=g)
    box = sample_cutmix_box(h, w, lam, g)
    x1, x2, y1, y2 = box
    assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h
    mixed, ya, yb, adjusted = cutmix_batch(x, torch.tensor([4, 7]), box, torch.tensor([1, 0]))
    assert float((mixed[0, 0] == x[0, 0]).double().mean()) == pytest.approx(adjusted, abs=1e-12)
    assert ya.tolist() == [4, 7] and yb.tolist() == [7, 4]
    assert torch.equal(mixed[1, :, y1:y2, x1:x2], x[0, :, y1:y2, x1:x2])
    # The input batch is not modified.
    assert not torch.equal(x[0], x[1])


def test_unadjusted_lambda_is_wrong_near_edges():
    corner = cutmix_box(32, 32, 0.5, (0.0, 0.0))
    assert corner == (0, 11, 0, 11)
    assert cutmix_lambda(corner, 32, 32) == pytest.approx(1 - 121 / 1024)


def test_integer_inputs_are_mixed_in_float():
    x = torch.tensor([[250, 10], [10, 250]], dtype=torch.uint8)
    mixed, _, _ = mixup_batch(x, torch.tensor([0, 1]), 0.5, torch.tensor([1, 0]))
    assert mixed.dtype == torch.float64
    torch.testing.assert_close(mixed, torch.full((2, 2), 130.0))
    img = torch.arange(32, dtype=torch.uint8).reshape(2, 1, 4, 4)
    out, _, _, _ = cutmix_batch(img, torch.tensor([0, 1]), (0, 2, 0, 2), torch.tensor([1, 0]))
    assert out.dtype == torch.uint8
    assert torch.equal(out[0, 0, :2, :2], img[1, 0, :2, :2])
    assert torch.equal(out[0, 0, 2:], img[0, 0, 2:])
    assert img[0, 0, 0].tolist() == [0, 1, 2, 3]


def test_eval_does_not_mix():
    g = torch.Generator().manual_seed(5)
    m = Mixup(alpha=1.0, generator=g)
    x = torch.randn(8, 3, generator=g)
    y = torch.arange(8)
    m.eval()
    xe, ya, yb, lam = m(x, y)
    assert xe is x and lam == 1.0 and torch.equal(ya, y) and torch.equal(yb, y)
    m.train()
    xt, ya, yb, lam = m(x, y)
    assert 0.0 <= lam <= 1.0 and torch.equal(ya, y)
    assert sorted(yb.tolist()) == y.tolist()
    perm = yb  # the labels are the batch indices, so y[perm] is perm
    torch.testing.assert_close(xt, lam * x + (1 - lam) * x[perm])
    assert not torch.equal(xt, x)


def test_training_step_with_module():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(6)
    model = torch.nn.Linear(4, 3)
    mix = Mixup(alpha=0.4, generator=g)
    x = torch.randn(16, 4, generator=g)
    y = torch.randint(3, (16,), generator=g)
    xm, ya, yb, lam = mix(x, y)
    loss = mixed_cross_entropy(model(xm), ya, yb, lam)
    loss.backward()
    q = lam * F.one_hot(ya, 3).double() + (1 - lam) * F.one_hot(yb, 3).double()
    assert float(loss.detach()) == pytest.approx(float(F.cross_entropy(model(xm), q).detach()), abs=1e-12)
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


@pytest.mark.parametrize("lam", [0.0, 0.19, 0.36, 0.75, 0.91, 1.0])
def test_box_area_before_clipping(lam):
    h, w = 400, 600
    x1, x2, y1, y2 = cutmix_box(h, w, lam, (w / 2, h / 2))
    assert x1 == round(w / 2 - w * math.sqrt(1 - lam) / 2) and y1 == round(h / 2 - h * math.sqrt(1 - lam) / 2)
    assert x2 - x1 == pytest.approx(w * math.sqrt(1 - lam), abs=1)
    assert y2 - y1 == pytest.approx(h * math.sqrt(1 - lam), abs=1)


def test_box_centre_is_uniform():
    # With lam = 1 the box is empty and sits at the rounded centre, which is uniform over the image.
    g = torch.Generator().manual_seed(8)
    h, w = 30, 50
    boxes = [sample_cutmix_box(h, w, 1.0, g) for _ in range(4000)]
    xs = torch.tensor([b[0] for b in boxes], dtype=torch.float64)
    ys = torch.tensor([b[2] for b in boxes], dtype=torch.float64)
    assert all(b[0] == b[1] and b[2] == b[3] for b in boxes)
    assert float(xs.mean()) == pytest.approx(w / 2, abs=1.5) and float(ys.mean()) == pytest.approx(h / 2, abs=1.0)
    assert float(xs.max()) >= w - 2 and float(ys.max()) >= h - 2
    # The two coordinates are drawn independently.
    assert abs(float(torch.corrcoef(torch.stack([xs, ys]))[0, 1])) < 0.1


def test_accepts_lists():
    xm, ya, yb = mixup_batch([[1.0, 2.0], [3.0, 4.0]], [0, 1], 0.5, [1, 0])
    torch.testing.assert_close(xm, torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
    assert ya.tolist() == [0, 1] and yb.tolist() == [1, 0]
    q = mixup_targets([0, 1], 2, 0.25, [1, 0])
    torch.testing.assert_close(q, torch.tensor([[0.25, 0.75], [0.75, 0.25]]))
    out, ya, yb, lam = cutmix_batch([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], [0, 1], (0, 1, 0, 1), [1, 0])
    assert out.tolist() == [[[[3.0, 2.0]]], [[[1.0, 4.0]]]] and yb.tolist() == [1, 0] and lam == 0.5
    m = Mixup(alpha=1.0).eval()
    _, ya, yb, _ = m(torch.zeros(2, 1), [0, 1])
    assert ya.tolist() == [0, 1] and yb.tolist() == [0, 1]
    m.train()
    _, ya, yb, _ = m(torch.zeros(2, 1), [0, 1])
    assert sorted(yb.tolist()) == [0, 1]


def test_float32_inputs_keep_their_dtype():
    x = torch.ones(2, 3, dtype=torch.float32)
    assert mixup_batch(x, [0, 1], 0.3, [1, 0])[0].dtype == torch.float32
