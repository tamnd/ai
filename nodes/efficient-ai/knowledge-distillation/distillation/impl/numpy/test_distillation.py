# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/distillation/ (ai://NEW-distillation).
"""Tests of the NumPy reference: the worked example, the gradient (q - p) / T against finite differences, the equal
gradients of the KL and cross-entropy forms, the T^2 factor and the high-temperature limit, the full loss and its
gradient, the logit regression and hint losses, ensemble soft targets, the fidelity metrics, input checks, and the
fixture shared with the PyTorch tests."""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from distillation import (
    distillation_loss,
    distillation_loss_grad,
    ensemble_soft_targets,
    hard_cross_entropy,
    high_temperature_limit_grad,
    hint_loss,
    log_softmax,
    logit_matching_loss,
    predictive_kl,
    soft_cross_entropy,
    soft_kl,
    soft_target_grad,
    soft_targets,
    top1_agreement,
)

# The worked example of the page.
V = np.array([3.0, 1.0, -4.0])
Z = np.array([1.0, 0.5, -1.5])

# The fixture shared with the PyTorch tests: T = 4, alpha = 0.9, KL form, mean over the batch.
STUDENT = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
TEACHER = [[4.0, 1.0, -2.0, 0.0], [0.0, -1.0, 3.0, 1.0], [-2.0, 1.0, 5.0, 0.0]]
LABELS = [0, 2, 3]
FIXTURE = 1.8520924719978928


def numeric_grad(f, z, h=1e-6):
    z = np.asarray(z, dtype=float)
    g = np.zeros_like(z)
    for i in np.ndindex(z.shape):
        e = np.zeros_like(z)
        e[i] = h
        g[i] = (f(z + e) - f(z - e)) / (2 * h)
    return g


def test_worked_example_probabilities():
    np.testing.assert_allclose(soft_targets(V, 1.0), [0.880090, 0.119107, 0.000803], atol=1e-6)
    np.testing.assert_allclose(soft_targets(Z, 1.0), [0.592201, 0.359188, 0.048611], atol=1e-6)
    np.testing.assert_allclose(soft_targets(V, 4.0), [0.561702, 0.340689, 0.097609], atol=1e-6)
    np.testing.assert_allclose(soft_targets(Z, 4.0), [0.413606, 0.365006, 0.221387], atol=1e-6)


def test_worked_example_gradients():
    np.testing.assert_allclose(soft_target_grad(Z, V, 1.0), [-0.287889, 0.240081, 0.047808], atol=1e-6)
    np.testing.assert_allclose(soft_target_grad(Z, V, 4.0), [-0.037024, 0.006079, 0.030945], atol=1e-6)
    np.testing.assert_allclose(16 * soft_target_grad(Z, V, 4.0), [-0.592381, 0.097268, 0.495113], atol=1e-6)
    np.testing.assert_allclose(1e4 * soft_target_grad(Z, V, 100.0), [-0.667294, -0.155354, 0.822648], atol=1e-6)
    np.testing.assert_allclose(high_temperature_limit_grad(Z, V), [-2 / 3, -1 / 6, 5 / 6], atol=1e-12)


def test_worked_example_losses():
    assert soft_kl(Z, V, 1.0) == pytest.approx(0.213906, abs=1e-6)
    assert 16 * soft_kl(Z, V, 4.0) == pytest.approx(1.095802, abs=1e-6)
    assert 1e4 * soft_kl(Z, V, 100.0) == pytest.approx(1.732200, abs=1e-6)
    assert hard_cross_entropy(Z[None], [0]) == pytest.approx(0.523909, abs=1e-6)
    assert distillation_loss(Z[None], V[None], [0], 4.0, 0.9) == pytest.approx(1.038612, abs=1e-6)
    assert distillation_loss(Z[None], V[None], [0], 4.0, 0.5) == pytest.approx(0.809855, abs=1e-6)


def test_fixture():
    assert distillation_loss(STUDENT, TEACHER, LABELS, 4.0, 0.9) == pytest.approx(FIXTURE, abs=1e-12)


def test_softmax_temperature_softens():
    entropies = [-(soft_targets(V, t) * log_softmax(V, t)).sum() for t in (0.5, 1.0, 2.0, 4.0, 20.0)]
    assert all(a < b for a, b in itertools.pairwise(entropies))
    np.testing.assert_allclose(soft_targets(V, 1e9), np.full(3, 1 / 3), atol=1e-8)
    assert soft_targets(V, 1e-3)[0] == pytest.approx(1.0)


def test_soft_targets_are_stable_for_large_logits():
    p = soft_targets([1000.0, 0.0, -1000.0], 1.0)
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


@settings(max_examples=60, deadline=None)
@given(
    st.integers(2, 8),
    st.floats(0.25, 30.0),
    st.integers(0, 2**31 - 1),
)
def test_soft_gradient_matches_finite_differences(n, t, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=n)
    v = rng.normal(scale=3.0, size=n)
    g = soft_target_grad(z, v, t)
    np.testing.assert_allclose(g, numeric_grad(lambda x: soft_cross_entropy(x, v, t), z), atol=1e-7)
    np.testing.assert_allclose(g, numeric_grad(lambda x: soft_kl(x, v, t), z), atol=1e-7)
    assert abs(g.sum()) < 1e-12


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 8), st.floats(0.25, 30.0), st.integers(0, 2**31 - 1))
def test_kl_and_cross_entropy_differ_by_teacher_entropy(n, t, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=n)
    v = rng.normal(scale=3.0, size=n)
    p = soft_targets(v, t)
    h = -(p * log_softmax(v, t)).sum()
    assert soft_cross_entropy(z, v, t) == pytest.approx(soft_kl(z, v, t) + h, abs=1e-10)
    assert soft_kl(z, v, t) >= -1e-12
    assert soft_kl(v, v, t) == pytest.approx(0.0, abs=1e-12)


def test_soft_term_is_shift_invariant():
    for t in (1.0, 3.0):
        assert soft_kl(Z + 7.5, V, t) == pytest.approx(soft_kl(Z, V, t), abs=1e-12)
        assert soft_kl(Z, V - 2.0, t) == pytest.approx(soft_kl(Z, V, t), abs=1e-12)


def test_t2_keeps_the_gradient_scale():
    """Without the factor the soft gradient falls like 1 / T^2; with it, it tends to the limit."""
    ts = np.array([10.0, 100.0, 1000.0])
    raw = np.array([np.abs(soft_target_grad(Z, V, t)).max() for t in ts])
    np.testing.assert_allclose(raw * ts**2, np.abs(high_temperature_limit_grad(Z, V)).max(), rtol=0.2)
    assert raw[2] / raw[1] == pytest.approx(0.01, rel=0.02)
    # The error of the scaled gradient falls like 1 / T: T times the error stays near 1.13 here.
    err = np.array([np.abs(t * t * soft_target_grad(Z, V, t) - high_temperature_limit_grad(Z, V)).max() for t in ts])
    np.testing.assert_allclose(err * ts, 1.13, rtol=0.12)


def test_high_temperature_limit_works_row_by_row():
    # Two examples of five classes: the means are taken over the classes of each row, not over the batch.
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 5)) + [[0.0], [4.0]]
    v = rng.normal(size=(2, 5)) - [[1.0], [2.0]]
    lim = high_temperature_limit_grad(z, v)
    for i in range(2):
        np.testing.assert_allclose(lim[i], high_temperature_limit_grad(z[i], v[i]), atol=1e-15)
    np.testing.assert_allclose(lim.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(1e8 * soft_target_grad(z, v, 1e4), lim, atol=1e-3)


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 10), st.integers(0, 2**31 - 1))
def test_high_temperature_limit_is_logit_matching(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    t = 1e4
    lim = high_temperature_limit_grad(z, v)
    np.testing.assert_allclose(t * t * soft_target_grad(z, v, t), lim, atol=1e-3)
    zc, vc = z - z.mean(), v - v.mean()
    # With zero-mean logits the limit is the gradient of (1 / 2N) sum (z - v)^2.
    np.testing.assert_allclose(lim, (zc - vc) / n, atol=1e-12)
    np.testing.assert_allclose(
        lim,
        numeric_grad(lambda x: logit_matching_loss(x - x.mean(), vc) / n, zc),
        atol=1e-7,
    )
    assert t * t * soft_kl(z, v, t) == pytest.approx(0.5 * ((zc - vc) ** 2).sum() / n, rel=1e-3, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    st.integers(2, 6),
    st.floats(0.5, 10.0),
    st.floats(0.0, 1.0),
    st.integers(0, 2**31 - 1),
)
def test_full_gradient_matches_finite_differences(n, t, alpha, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=(3, n))
    v = rng.normal(scale=2.0, size=(3, n))
    y = rng.integers(n, size=3)
    g = distillation_loss_grad(z, v, y, t, alpha)
    num = numeric_grad(lambda x: distillation_loss(x, v, y, t, alpha, reduction="sum"), z)
    np.testing.assert_allclose(g, num, atol=1e-6)
    np.testing.assert_allclose(
        g,
        numeric_grad(lambda x: distillation_loss(x, v, y, t, alpha, "cross_entropy", "sum"), z),
        atol=1e-6,
    )


def test_alpha_endpoints():
    z, v = np.array(STUDENT), np.array(TEACHER)
    assert distillation_loss(z, v, LABELS, 3.0, 0.0) == pytest.approx(hard_cross_entropy(z, LABELS), abs=1e-12)
    assert distillation_loss(z, v, None, 3.0, 1.0) == pytest.approx(9 * soft_kl(z, v, 3.0), abs=1e-12)
    assert distillation_loss(z, v, LABELS, 1.0, 1.0, soft="cross_entropy") == pytest.approx(
        soft_cross_entropy(z, v, 1.0), abs=1e-12
    )


def test_cross_entropy_form_adds_teacher_entropy():
    z, v = np.array(STUDENT), np.array(TEACHER)
    p = soft_targets(v, 4.0)
    h = -(p * log_softmax(v, 4.0)).sum(axis=-1).mean()
    kl = distillation_loss(z, v, LABELS, 4.0, 0.9)
    ce = distillation_loss(z, v, LABELS, 4.0, 0.9, soft="cross_entropy")
    assert ce - kl == pytest.approx(0.9 * 16 * h, abs=1e-12)


def test_hard_label_gradient_at_temperature_one():
    z = np.array([[1.0, 0.5, -1.5]])
    np.testing.assert_allclose(
        distillation_loss_grad(z, z, [0], 1.0, 0.0),
        soft_targets(z, 1.0) - [1.0, 0.0, 0.0],
        atol=1e-12,
    )
    onehot_teacher = np.array([[60.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        soft_target_grad(z, onehot_teacher, 1.0),
        soft_targets(z, 1.0) - [1.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_reductions():
    z, v = np.array(STUDENT), np.array(TEACHER)
    per = distillation_loss(z, v, LABELS, 2.0, 0.7, reduction="none")
    assert per.shape == (3,)
    assert distillation_loss(z, v, LABELS, 2.0, 0.7, reduction="sum") == pytest.approx(per.sum())
    assert distillation_loss(z, v, LABELS, 2.0, 0.7) == pytest.approx(per.mean())
    with pytest.raises(ValueError):
        distillation_loss(z, v, LABELS, 2.0, 0.7, reduction="batchmean")


@pytest.mark.parametrize("t", [0.0, -1.0])
def test_temperature_must_be_positive(t):
    with pytest.raises(ValueError):
        distillation_loss(STUDENT, TEACHER, LABELS, t, 0.5)
    with pytest.raises(ValueError):
        soft_targets(V, t)
    with pytest.raises(ValueError):
        distillation_loss_grad(STUDENT, TEACHER, LABELS, t, 0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_must_be_in_unit_interval(alpha):
    with pytest.raises(ValueError):
        distillation_loss(STUDENT, TEACHER, LABELS, 2.0, alpha)
    with pytest.raises(ValueError):
        distillation_loss_grad(STUDENT, TEACHER, LABELS, 2.0, alpha)


def test_unknown_soft_term():
    with pytest.raises(ValueError):
        distillation_loss(STUDENT, TEACHER, LABELS, 2.0, 0.5, soft="mse")


def test_free_logits_converge_to_teacher_up_to_a_shift():
    """Gradient descent on free student logits with alpha = 1 recovers the teacher's logits up to a constant."""
    rng = np.random.default_rng(0)
    v = rng.normal(scale=2.0, size=5)
    z = np.zeros(5)
    for _ in range(3000):
        z = z - 2.0 * distillation_loss_grad(z[None], v[None], None, 2.0, 1.0)[0]
    np.testing.assert_allclose(z - z.mean(), v - v.mean(), atol=1e-6)


def test_logit_matching_loss():
    assert logit_matching_loss(Z, V) == pytest.approx(0.5 * (4 + 0.25 + 6.25))
    assert logit_matching_loss([Z, V], [V, V], reduction="sum") == pytest.approx(5.25)
    np.testing.assert_allclose(logit_matching_loss([Z, V], [V, V], reduction="none"), [5.25, 0.0])


def test_hint_loss():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 2))
    g = rng.normal(size=(3, 2))
    assert hint_loss(g @ w.T, g, w) == pytest.approx(0.0, abs=1e-12)
    u = g @ w.T + np.array([1.0, 0.0, 0.0, 0.0])
    assert hint_loss(u, g, w) == pytest.approx(0.5)
    num = numeric_grad(lambda x: hint_loss(u, g, x, reduction="sum"), w)
    np.testing.assert_allclose(num, -((u - g @ w.T).T @ g), atol=1e-6)


def test_ensemble_soft_targets():
    m = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, -1.0, 3.0]])
    a = ensemble_soft_targets(m, 2.0)
    np.testing.assert_allclose(a, soft_targets(m, 2.0).mean(axis=0), atol=1e-12)
    g = ensemble_soft_targets(m, 2.0, mean="geometric")
    np.testing.assert_allclose(g, soft_targets(m.mean(axis=0), 2.0), atol=1e-12)
    gm = np.exp(np.log(soft_targets(m, 2.0)).mean(axis=0))
    np.testing.assert_allclose(g, gm / gm.sum(), atol=1e-12)
    assert a.sum() == pytest.approx(1.0) and g.sum() == pytest.approx(1.0)
    assert not np.allclose(a, g)
    with pytest.raises(ValueError):
        ensemble_soft_targets(m, 2.0, mean="median")


def test_fidelity_metrics():
    z, v = np.array(STUDENT), np.array(TEACHER)
    assert top1_agreement(z, v) == pytest.approx(2 / 3)
    assert top1_agreement(v, v) == 1.0
    assert predictive_kl(v, v) == pytest.approx(0.0, abs=1e-12)
    assert predictive_kl(z, v) == pytest.approx(soft_kl(z, v, 1.0))
    assert predictive_kl(z, v) > 0
