# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://NEW-distillation).
"""Tests of the PyTorch code: the shared fixture, agreement with a hand-built reference from F.kl_div,
F.log_softmax, F.softmax and F.cross_entropy, the gradient (q - p) / T from autograd, the T^2 factor and the
high-temperature limit, the detached teacher, the KL and cross-entropy forms, extra axes, the pitfalls of
F.kl_div (reduction "mean", argument order, probabilities as input), the module, hints and the fidelity metrics."""

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from distillation import (
    DistillationLoss,
    distillation_loss,
    hint_loss,
    predictive_kl,
    soft_target_loss,
    top1_agreement,
)

torch.set_default_dtype(torch.float64)

# The fixture shared with the NumPy tests: T = 4, alpha = 0.9.
STUDENT = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
TEACHER = [[4.0, 1.0, -2.0, 0.0], [0.0, -1.0, 3.0, 1.0], [-2.0, 1.0, 5.0, 0.0]]
LABELS = [0, 2, 3]
FIXTURE = 1.8520924719978928

# The worked example of the page.
V = torch.tensor([[3.0, 1.0, -4.0]])
Z = torch.tensor([[1.0, 0.5, -1.5]])


def reference(s, t, y, temperature, alpha):
    """The usual hand-written distillation loss, built from torch.nn.functional only."""
    soft = F.kl_div(
        F.log_softmax(s / temperature, dim=1),
        F.softmax(t / temperature, dim=1),
        reduction="batchmean",
    )
    loss = alpha * soft * temperature * temperature
    if alpha < 1.0:
        loss = loss + (1.0 - alpha) * F.cross_entropy(s, y)
    return loss


def test_fixture():
    s, t, y = torch.tensor(STUDENT), torch.tensor(TEACHER), torch.tensor(LABELS)
    torch.testing.assert_close(distillation_loss(s, t, y, 4.0, 0.9), torch.tensor(FIXTURE), rtol=0, atol=1e-12)


def test_worked_example():
    y = torch.tensor([0])
    torch.testing.assert_close(distillation_loss(Z, V, y, 4.0, 0.9), torch.tensor(1.038612), rtol=0, atol=1e-6)
    torch.testing.assert_close(soft_target_loss(Z, V, 4.0), torch.tensor(1.095802), rtol=0, atol=1e-6)
    torch.testing.assert_close(soft_target_loss(Z, V, 100.0), torch.tensor(1.732200), rtol=0, atol=1e-6)


@settings(max_examples=60, deadline=None)
@given(
    st.integers(1, 6),
    st.integers(2, 9),
    st.floats(0.3, 25.0),
    st.floats(0.0, 1.0),
    st.integers(0, 2**31 - 1),
)
def test_matches_reference(batch, classes, temperature, alpha, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(batch, classes, generator=g) * 3
    t = torch.randn(batch, classes, generator=g) * 3
    y = torch.randint(classes, (batch,), generator=g)
    want = reference(s, t, y, temperature, alpha)
    torch.testing.assert_close(distillation_loss(s, t, y, temperature, alpha), want, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(DistillationLoss(temperature, alpha)(s, t, y), want, rtol=1e-10, atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    st.integers(1, 5),
    st.integers(2, 8),
    st.floats(0.3, 25.0),
    st.integers(0, 2**31 - 1),
)
def test_gradient_is_q_minus_p_over_t(batch, classes, temperature, seed):
    g = torch.Generator().manual_seed(seed)
    s = (torch.randn(batch, classes, generator=g) * 3).requires_grad_()
    t = torch.randn(batch, classes, generator=g) * 3
    for form in ("kl", "cross_entropy"):
        # The sum of the unscaled soft term over the batch: divide out the T^2 of soft_target_loss.
        loss = soft_target_loss(s, t, temperature, form, reduction="sum") / temperature**2
        (grad,) = torch.autograd.grad(loss, s)
        q = F.softmax(s.detach() / temperature, dim=1)
        p = F.softmax(t / temperature, dim=1)
        torch.testing.assert_close(grad, (q - p) / temperature, rtol=1e-9, atol=1e-12)


def test_full_gradient():
    s = torch.tensor(STUDENT, requires_grad=True)
    t, y = torch.tensor(TEACHER), torch.tensor(LABELS)
    temperature, alpha = 3.0, 0.7
    distillation_loss(s, t, y, temperature, alpha).backward()
    q_t = F.softmax(s.detach() / temperature, dim=1)
    p_t = F.softmax(t / temperature, dim=1)
    q_1 = F.softmax(s.detach(), dim=1)
    onehot = F.one_hot(y, 4).double()
    want = (alpha * temperature * (q_t - p_t) + (1 - alpha) * (q_1 - onehot)) / 3
    torch.testing.assert_close(s.grad, want, rtol=1e-10, atol=1e-12)


def test_t2_factor_and_logit_matching_limit():
    """T^2 times the soft gradient tends to ((z - mean z) - (v - mean v)) / N, the gradient of logit matching."""
    v = V - V.mean()
    limit = ((Z - Z.mean()) - v) / 3
    errors = []
    for temperature in (10.0, 100.0, 1000.0):
        s = Z.clone().requires_grad_()
        loss = soft_target_loss(s, V, temperature)
        (grad,) = torch.autograd.grad(loss, s)
        errors.append((grad - limit).abs().max().item())
        unscaled = grad / temperature**2
        assert unscaled.abs().max().item() < 1.0 / temperature**2
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3
    s = Z.clone().requires_grad_()
    matching = 0.5 * ((s - s.mean()) - v).pow(2).sum() / 3
    (grad,) = torch.autograd.grad(matching, s)
    torch.testing.assert_close(grad, limit)


def test_teacher_receives_no_gradient():
    s = torch.tensor(STUDENT, requires_grad=True)
    t = torch.tensor(TEACHER, requires_grad=True)
    distillation_loss(s, t, torch.tensor(LABELS)).backward()
    assert t.grad is None
    assert s.grad is not None


def test_forms_differ_by_teacher_entropy():
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    p = F.softmax(t / 4.0, dim=1)
    h = -(p * p.log()).sum(dim=1).mean()
    diff = soft_target_loss(s, t, 4.0, "cross_entropy") - soft_target_loss(s, t, 4.0, "kl")
    torch.testing.assert_close(diff, 16 * h)


def test_alpha_endpoints():
    s, t, y = torch.tensor(STUDENT), torch.tensor(TEACHER), torch.tensor(LABELS)
    torch.testing.assert_close(distillation_loss(s, t, y, 3.0, 0.0), F.cross_entropy(s, y))
    torch.testing.assert_close(distillation_loss(s, t, None, 3.0, 1.0), soft_target_loss(s, t, 3.0))
    with pytest.raises(ValueError):
        distillation_loss(s, t, None, 3.0, 0.5)


def test_reductions():
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    per = soft_target_loss(s, t, 2.0, reduction="none")
    assert per.shape == (3,)
    torch.testing.assert_close(soft_target_loss(s, t, 2.0, reduction="sum"), per.sum())
    torch.testing.assert_close(soft_target_loss(s, t, 2.0), per.mean())
    with pytest.raises(ValueError):
        soft_target_loss(s, t, 2.0, reduction="mean")
    with pytest.raises(ValueError):
        soft_target_loss(s, t, 2.0, form="mse")


def test_extra_axes_match_flattened():
    """Logits (N, C, L), as for tokens: the soft term sums over classes and positions, then averages over N."""
    g = torch.Generator().manual_seed(3)
    s = torch.randn(2, 5, 7, generator=g)
    t = torch.randn(2, 5, 7, generator=g)
    flat = lambda x: x.permute(0, 2, 1).reshape(-1, 5)  # noqa: E731
    got = soft_target_loss(s, t, 2.0)
    want = soft_target_loss(flat(s), flat(t), 2.0, reduction="sum") / 2
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("temperature", [0.0, -2.0])
def test_temperature_must_be_positive(temperature):
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    with pytest.raises(ValueError):
        distillation_loss(s, t, torch.tensor(LABELS), temperature, 0.5)
    with pytest.raises(ValueError):
        DistillationLoss(temperature)
    with pytest.raises(ValueError):
        soft_target_loss(s, t, temperature)


@pytest.mark.parametrize("alpha", [-0.5, 1.5])
def test_alpha_must_be_in_unit_interval(alpha):
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    with pytest.raises(ValueError):
        distillation_loss(s, t, torch.tensor(LABELS), 2.0, alpha)
    with pytest.raises(ValueError):
        DistillationLoss(2.0, alpha)


@pytest.mark.filterwarnings("ignore:reduction:UserWarning")
def test_kl_div_mean_reduction_divides_by_classes():
    """Pitfall: reduction="mean" of F.kl_div averages over every element, so it is C times smaller than batchmean."""
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    log_q, p = F.log_softmax(s / 2, dim=1), F.softmax(t / 2, dim=1)
    batchmean = F.kl_div(log_q, p, reduction="batchmean")
    mean = F.kl_div(log_q, p, reduction="mean")
    torch.testing.assert_close(mean * 4, batchmean)
    torch.testing.assert_close(batchmean * 4, soft_target_loss(s, t, 2.0))


def test_kl_div_argument_order():
    """Pitfall: F.kl_div(input, target) is KL(target || input); swapping student and teacher computes KL(q || p)."""
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    right = F.kl_div(F.log_softmax(s, dim=1), F.softmax(t, dim=1), reduction="batchmean")
    swapped = F.kl_div(F.log_softmax(t, dim=1), F.softmax(s, dim=1), reduction="batchmean")
    torch.testing.assert_close(right, soft_target_loss(s, t, 1.0))
    assert abs(right.item() - swapped.item()) > 1e-3


def test_probabilities_as_input_is_wrong():
    """Pitfall: F.kl_div wants log-probabilities as input; passing probabilities gives a different number."""
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    wrong = F.kl_div(F.softmax(s, dim=1), F.softmax(t, dim=1), reduction="batchmean")
    assert abs(wrong.item() - soft_target_loss(s, t, 1.0).item()) > 1e-2


def test_without_t2_the_soft_term_fades():
    """Pitfall: without T^2 the soft gradient shrinks like 1 / T^2 next to the hard term, which does not change."""
    norms = []
    for temperature in (1.0, 4.0, 16.0):
        s = Z.clone().requires_grad_()
        loss = soft_target_loss(s, V, temperature) / temperature**2
        (grad,) = torch.autograd.grad(loss, s)
        norms.append(grad.norm().item())
    assert norms[0] > 5 * norms[1] > 25 * norms[2]


def test_student_matches_teacher_with_free_logits():
    """With alpha = 1, free student logits converge to the teacher's up to a constant shift."""
    torch.manual_seed(0)
    v = torch.randn(4, 6) * 2
    z = torch.zeros(4, 6, requires_grad=True)
    opt = torch.optim.SGD([z], lr=4.0)
    for _ in range(3000):
        opt.zero_grad()
        distillation_loss(z, v, None, 2.0, 1.0).backward()
        opt.step()
    zc = z.detach() - z.detach().mean(dim=1, keepdim=True)
    torch.testing.assert_close(zc, v - v.mean(dim=1, keepdim=True), rtol=0, atol=1e-5)
    assert top1_agreement(z.detach(), v) == 1.0


def test_distillation_trains_a_small_student():
    """A linear student distilled from a fixed two-layer teacher agrees with it on held-out inputs."""
    torch.manual_seed(1)
    teacher = nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 3))
    student = nn.Linear(5, 3)
    x_train, x_test = torch.randn(512, 5), torch.randn(256, 5)
    with torch.no_grad():
        v_train, v_test = teacher(x_train), teacher(x_test)
    y = v_train.argmax(dim=1)
    before = predictive_kl(student(x_test).detach(), v_test)
    opt = torch.optim.Adam(student.parameters(), lr=0.05)
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.9)
    for _ in range(300):
        opt.zero_grad()
        loss_fn(student(x_train), v_train, y).backward()
        opt.step()
    with torch.no_grad():
        z_test = student(x_test)
    assert predictive_kl(z_test, v_test) < 0.5 * before
    assert top1_agreement(z_test, v_test) > 0.8


def test_hint_loss():
    torch.manual_seed(2)
    regressor = nn.Linear(2, 4, bias=False)
    g = torch.randn(3, 2)
    u = regressor(g).detach()
    torch.testing.assert_close(hint_loss(u, g, regressor), torch.tensor(0.0))
    u2 = u + torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(hint_loss(u2, g, regressor), torch.tensor(0.5))
    teacher_hint = u2.clone().requires_grad_()
    hint_loss(teacher_hint, g, regressor).backward()
    assert teacher_hint.grad is None
    assert regressor.weight.grad is not None


def test_fidelity_metrics():
    s, t = torch.tensor(STUDENT), torch.tensor(TEACHER)
    assert top1_agreement(s, t) == pytest.approx(2 / 3)
    assert predictive_kl(t, t) == pytest.approx(0.0, abs=1e-12)
    want = F.kl_div(F.log_softmax(s, dim=1), F.softmax(t, dim=1), reduction="batchmean").item()
    assert predictive_kl(s, t) == pytest.approx(want)


def test_module_repr():
    assert "temperature=3.0" in repr(DistillationLoss(3.0, 0.5))
