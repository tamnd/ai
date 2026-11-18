# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/label-smoothing/ (ai://98TJG9).
"""Tests of the NumPy reference: the worked example, the smoothed targets, the pair of losses and the KL form, the
gradient against finite differences, the confidence penalty, the optimal logit gap, label dropout, and the fixture
shared with the PyTorch tests."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from label_smoothing import (
    confidence_penalty_loss,
    decomposed_loss,
    entropy,
    eps_from_wrong_class_mass,
    kl_form,
    kl_from_uniform,
    label_dropout,
    log_softmax,
    optimal_logit_gap,
    smooth_labels,
    smoothed_cross_entropy,
    smoothed_cross_entropy_grad,
)

# The shared fixture. The PyTorch tests check these numbers against torch.nn.functional.cross_entropy.
LOGITS = [[2.0, 0.0, 0.0, -1.0], [0.5, -0.5, 1.5, 0.0], [-1.0, 3.0, 0.25, 0.75]]
LABELS = [0, 2, 3]
FIXTURE = {0.0: 1.0819850661352493, 0.1: 1.1778183994685827, 0.3: 1.3694850661352493}


def test_worked_example():
    z, y, eps = np.array([[2.0, 0.0, 0.0]]), [0], 0.3
    np.testing.assert_allclose(smooth_labels(y, 3, eps), [[0.8, 0.1, 0.1]], atol=1e-15)
    p = np.exp(log_softmax(z))[0]
    np.testing.assert_allclose(p, [0.786986, 0.106507, 0.106507], atol=1e-6)
    assert smoothed_cross_entropy(z, y, eps) == pytest.approx(0.639545, abs=1e-6)
    assert smoothed_cross_entropy(z, y, 0.0) == pytest.approx(0.239545, abs=1e-6)
    np.testing.assert_allclose(
        smoothed_cross_entropy_grad(z, y, eps),
        [[-0.013014, 0.006507, 0.006507]],
        atol=1e-6,
    )
    assert kl_from_uniform([[2.0, 0.0, 0.0]])[0] == pytest.approx(0.474266, abs=1e-6)
    assert kl_from_uniform([[1.0, 1.0, 1.0, 1.0]])[0] == pytest.approx(0.0, abs=1e-15)
    assert kl_form([[2.0, 0.0, 0.0]], [0], eps) == pytest.approx(0.639545, abs=1e-6)
    # The gap 2 is below the optimal gap log 8, so descent raises the correct logit.
    assert optimal_logit_gap(3, eps) == pytest.approx(np.log(8.0), abs=1e-12)
    assert smoothed_cross_entropy_grad(z, y, eps)[0, 0] < 0


@pytest.mark.parametrize("eps", sorted(FIXTURE))
def test_fixture(eps):
    assert smoothed_cross_entropy(LOGITS, LABELS, eps) == pytest.approx(FIXTURE[eps], abs=1e-10)


def test_smooth_labels():
    q = smooth_labels([0, 2], 4, 0.2)
    np.testing.assert_allclose(q, [[0.85, 0.05, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]], atol=1e-15)
    np.testing.assert_allclose(q.sum(axis=-1), 1.0)
    np.testing.assert_allclose(smooth_labels([1], 3, 0.0), [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(smooth_labels([1], 3, 1.0), [[1 / 3, 1 / 3, 1 / 3]])
    np.testing.assert_allclose(smooth_labels([1], 3, 0.5, prior=[0.5, 0.25, 0.25]), [[0.25, 0.625, 0.125]])
    # A prior may give some classes no mass.
    np.testing.assert_allclose(smooth_labels([1], 3, 0.5, prior=[1.0, 0.0, 0.0]), [[0.5, 0.5, 0.0]])


@pytest.mark.parametrize("eps", [-0.1, 1.1, float("nan")])
def test_eps_out_of_range(eps):
    with pytest.raises(ValueError):
        smooth_labels([0], 3, eps)
    with pytest.raises(ValueError):
        optimal_logit_gap(3, eps)


def test_bad_prior():
    with pytest.raises(ValueError):
        smooth_labels([0], 3, 0.1, prior=[0.5, 0.5])
    with pytest.raises(ValueError):
        smooth_labels([0], 3, 0.1, prior=[0.5, 0.6, -0.1])
    with pytest.raises(ValueError):
        smooth_labels([0], 3, 0.1, prior=[0.5, 0.25, 0.5])


def test_bad_reduction():
    with pytest.raises(ValueError):
        smoothed_cross_entropy(LOGITS, LABELS, 0.1, reduction="avg")
    with pytest.raises(ValueError):
        smoothed_cross_entropy_grad(LOGITS, LABELS, 0.1, reduction="avg")


def test_reductions():
    per = np.asarray(smoothed_cross_entropy(LOGITS, LABELS, 0.1, reduction="none"))
    assert per.shape == (3,)
    assert smoothed_cross_entropy(LOGITS, LABELS, 0.1, reduction="sum") == pytest.approx(per.sum(), abs=1e-12)
    assert smoothed_cross_entropy(LOGITS, LABELS, 0.1) == pytest.approx(per.mean(), abs=1e-12)


logit_rows = st.integers(2, 12).flatmap(
    lambda k: st.tuples(
        st.lists(st.lists(st.floats(-20, 20), min_size=k, max_size=k), min_size=1, max_size=5),
        st.lists(st.integers(0, k - 1), min_size=5, max_size=5),
    )
)


@settings(max_examples=200, deadline=None)
@given(logit_rows, st.floats(0.0, 1.0))
def test_decomposition(rows, eps):
    z, y = np.array(rows[0]), rows[1][: len(rows[0])]
    a = smoothed_cross_entropy(z, y, eps, reduction="none")
    np.testing.assert_allclose(decomposed_loss(z, y, eps, reduction="none"), a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kl_form(z, y, eps, reduction="none"), a, rtol=1e-10, atol=1e-10)


@settings(max_examples=100, deadline=None)
@given(logit_rows, st.floats(0.0, 1.0))
def test_gradient(rows, eps):
    z, y = np.array(rows[0]), rows[1][: len(rows[0])]
    g = smoothed_cross_entropy_grad(z, y, eps, reduction="sum")
    assert np.all(np.abs(g) <= 1.0)
    np.testing.assert_allclose(g.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(g, np.exp(log_softmax(z)) - smooth_labels(y, z.shape[-1], eps), atol=1e-15)


def test_gradient_finite_differences():
    z, y, h = np.array(LOGITS), LABELS, 1e-6
    for eps in (0.0, 0.1, 0.7):
        g = smoothed_cross_entropy_grad(z, y, eps)
        fd = np.zeros_like(z)
        for i in np.ndindex(z.shape):
            e = np.zeros_like(z)
            e[i] = h
            fd[i] = (smoothed_cross_entropy(z + e, y, eps) - smoothed_cross_entropy(z - e, y, eps)) / (2 * h)
        np.testing.assert_allclose(g, fd, atol=1e-8)
        np.testing.assert_allclose(smoothed_cross_entropy_grad(z, y, eps, reduction="sum"), 3 * g, atol=1e-15)


def test_prior_decomposition():
    prior = [0.4, 0.3, 0.2, 0.1]
    a = smoothed_cross_entropy(LOGITS, LABELS, 0.25, prior=prior)
    assert decomposed_loss(LOGITS, LABELS, 0.25, prior=prior) == pytest.approx(a, abs=1e-12)
    assert a != pytest.approx(smoothed_cross_entropy(LOGITS, LABELS, 0.25), abs=1e-3)


def test_confidence_penalty_is_reverse_kl():
    z = np.array(LOGITS)
    lp = log_softmax(z)
    p, k = np.exp(lp), z.shape[-1]
    kl_pu = (p * (lp + np.log(k))).sum(axis=-1)
    np.testing.assert_allclose(-entropy(z), kl_pu - np.log(k), atol=1e-12)
    np.testing.assert_allclose(entropy(LOGITS), entropy(z), atol=0)
    nll = smoothed_cross_entropy(z, LABELS, 0.0, reduction="none")
    np.testing.assert_allclose(
        confidence_penalty_loss(z, LABELS, 0.5, reduction="none"),
        nll - 0.5 * entropy(z),
        atol=1e-12,
    )
    # The two directions differ: label smoothing is unbounded as a probability goes to 0, the penalty is not.
    far = np.array([[60.0, 0.0, 0.0, -60.0]])
    assert kl_form(far, [0], 1.0) > 25.0
    assert -entropy(far)[0] + np.log(4) <= np.log(4) + 1e-12


def test_optimal_logit_gap():
    assert optimal_logit_gap(1000, 0.1) == pytest.approx(np.log(9001), abs=1e-12)
    assert optimal_logit_gap(1000, 0.1) == pytest.approx(9.105, abs=1e-3)
    assert optimal_logit_gap(5, 1.0) == 0.0
    assert optimal_logit_gap(5, 0.0) == np.inf
    # Gradient descent on free logits stops at the gap, with p = q'.
    k, y, eps = 6, 2, 0.2
    z = np.zeros((1, k))
    for _ in range(5000):
        z -= 2.0 * smoothed_cross_entropy_grad(z, [y], eps)
    gaps = z[0, y] - np.delete(z[0], y)
    np.testing.assert_allclose(gaps, optimal_logit_gap(k, eps), atol=1e-8)
    np.testing.assert_allclose(np.exp(log_softmax(z)), smooth_labels([y], k, eps), atol=1e-9)


def test_hard_targets_gap_grows():
    z = np.zeros((1, 3))
    prev = 0.0
    for _ in range(2000):
        z -= 2.0 * smoothed_cross_entropy_grad(z, [0], 0.0)
        gap = z[0, 0] - z[0, 1:].max()
        assert gap > prev
        prev = gap
    assert prev > 6.0


def test_wrong_class_convention():
    k, eps = 5, 0.1
    other = np.full(k, eps / (k - 1))
    other[1] = 1 - eps
    np.testing.assert_allclose(smooth_labels([1], k, eps_from_wrong_class_mass(eps, k))[0], other, atol=1e-15)
    assert not np.allclose(smooth_labels([1], k, eps)[0], other)


def test_label_dropout_marginal():
    rng = np.random.default_rng(20261115)
    k, eps, n = 4, 0.3, 200_000
    noisy = label_dropout(np.full(n, 1), k, eps, rng)
    freq = np.bincount(noisy, minlength=k) / n
    np.testing.assert_allclose(freq, smooth_labels([1], k, eps)[0], atol=5e-3)
    z = np.array([[0.5, 2.0, -1.0, 0.0]])
    mc = -log_softmax(z)[0, noisy].mean()
    assert mc == pytest.approx(smoothed_cross_entropy(z, [1], eps), abs=1e-2)
    assert np.array_equal(label_dropout(np.arange(4), k, 0.0, rng), np.arange(4))
    assert np.array_equal(label_dropout([0, 1, 2, 3], k, 0.0, rng), np.arange(4))
    with pytest.raises(ValueError):
        label_dropout([0], k, 1.5, rng)
