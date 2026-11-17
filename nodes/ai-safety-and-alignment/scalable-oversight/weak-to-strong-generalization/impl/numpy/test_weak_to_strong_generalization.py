# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weak-to-strong-generalization/ (ai://G0ZGHC).
"""Tests of the NumPy reference: PGR at its end points, undefined at a zero gap, invariant under affine changes of the
metric and able to leave [0, 1]; the adaptive threshold that hardens exactly half a batch; the confidence loss at
alpha = 0 and alpha = 1, its equality with the bootstrapped form, its minimizer and its gradient; the warm-up of alpha;
student-supervisor agreement; and the worked example of the page."""

import numpy as np
import pytest

from weak_to_strong_generalization import (
    adaptive_threshold,
    agreement,
    alpha_schedule,
    binary_ce,
    bootstrap_target,
    ce_grad_logit,
    confidence_loss,
    harden,
    harden_balanced,
    pgr,
)

rng = np.random.default_rng(7)


def test_pgr_end_points():
    assert pgr(0.6, 0.6, 0.8) == 0.0
    assert pgr(0.6, 0.8, 0.8) == 1.0
    assert pgr(0.6, 0.7, 0.8) == pytest.approx(0.5)
    assert pgr(0.5, 0.75, 0.6) == pytest.approx(2.5)


def test_pgr_undefined_at_zero_gap():
    with pytest.raises(ValueError):
        pgr(0.7, 0.75, 0.7)
    with pytest.raises(ValueError):
        pgr(np.array([0.6, 0.7]), np.array([0.7, 0.7]), np.array([0.8, 0.7]))


def test_pgr_leaves_the_unit_interval():
    assert pgr(0.6, 0.55, 0.8) == pytest.approx(-0.25)
    assert pgr(0.6, 0.85, 0.8) == pytest.approx(1.25)


def test_pgr_is_invariant_under_affine_maps():
    for _ in range(100):
        w, s, c = rng.uniform(0, 1, size=3)
        if abs(c - w) < 1e-3:
            continue
        a = rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 10)
        b = rng.normal()
        assert pgr(a * w + b, a * s + b, a * c + b) == pytest.approx(
            pgr(w, s, c), rel=1e-9, abs=1e-9
        )
    # accuracy and error rate give the same PGR
    assert pgr(1 - 0.6, 1 - 0.7, 1 - 0.8) == pytest.approx(pgr(0.6, 0.7, 0.8))


def test_pgr_is_vectorized():
    out = pgr(np.array([0.5, 0.6]), np.array([0.55, 0.8]), np.array([0.7, 0.8]))
    np.testing.assert_allclose(out, [0.25, 1.0])


def test_adaptive_threshold_hardens_exactly_half():
    for b in range(2, 40):
        p = rng.uniform(size=b)
        t = adaptive_threshold(p)
        assert int(np.sum(p > t)) == b // 2
        assert np.sum(harden(p, t)) == b // 2
        np.testing.assert_array_equal(harden(p, t), harden_balanced(p))


def test_adaptive_threshold_needs_two_examples():
    with pytest.raises(ValueError):
        adaptive_threshold([0.3])


def test_harden_is_strict():
    np.testing.assert_array_equal(harden([0.2, 0.5, 0.8], 0.5), [0.0, 0.0, 1.0])


def test_ties_at_the_threshold_and_the_balanced_version():
    p = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.9])
    t = adaptive_threshold(p)
    assert t == 0.5 and np.sum(harden(p, t)) == 1  # ties fall below the threshold
    h = harden_balanced(p)
    assert np.sum(h) == 3 and h[5] == 1.0 and h[0] == 0.0
    np.testing.assert_array_equal(h, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


def test_binary_ce_values():
    assert binary_ce([0.5, 0.5], [1.0, 0.0]) == pytest.approx(np.log(2))
    assert binary_ce([0.8], [1.0]) == pytest.approx(-np.log(0.8))
    assert binary_ce([0.8], [0.0]) == pytest.approx(-np.log(0.2))
    assert np.isfinite(binary_ce([0.0, 1.0], [1.0, 0.0]))


def test_alpha_zero_is_naive_finetuning_and_alpha_one_is_self_training():
    for _ in range(50):
        p = rng.uniform(0.01, 0.99, size=8)
        w = rng.uniform(size=8)
        hard = harden(p, adaptive_threshold(p))
        assert confidence_loss(p, w, 0.0) == pytest.approx(binary_ce(p, w))
        assert confidence_loss(p, w, 1.0) == pytest.approx(binary_ce(p, hard))


def test_the_two_forms_agree_and_the_loss_is_linear_in_alpha():
    for _ in range(50):
        p = rng.uniform(0.01, 0.99, size=10)
        w = rng.uniform(size=10)
        a = float(rng.uniform())
        hard = harden(p, adaptive_threshold(p))
        l_a = confidence_loss(p, w, a)
        assert l_a == pytest.approx(binary_ce(p, bootstrap_target(w, hard, a)))
        assert l_a == pytest.approx(
            (1 - a) * confidence_loss(p, w, 0.0) + a * confidence_loss(p, w, 1.0)
        )


def test_explicit_threshold_is_used():
    p = np.array([0.9, 0.6, 0.4, 0.2])
    w = np.array([0.7, 0.3, 0.6, 0.1])
    assert confidence_loss(p, w, 1.0, t=0.5) == pytest.approx(
        binary_ce(p, [1.0, 1.0, 0.0, 0.0])
    )
    assert confidence_loss(p, w, 1.0, t=0.1) == pytest.approx(
        binary_ce(p, [1.0, 1.0, 1.0, 1.0])
    )


def test_alpha_out_of_range():
    with pytest.raises(ValueError):
        confidence_loss([0.4, 0.6], [0.5, 0.5], 1.5)
    with pytest.raises(ValueError):
        bootstrap_target([0.5], [1.0], -0.1)
    with pytest.raises(ValueError):
        alpha_schedule(0, 10, 2.0)


def test_minimizer_for_a_fixed_target_is_the_target():
    grid = np.linspace(1e-4, 1 - 1e-4, 200001)
    for q in rng.uniform(size=20):
        losses = -(q * np.log(grid) + (1 - q) * np.log(1 - grid))
        assert abs(grid[np.argmin(losses)] - q) < 1e-4


def test_gradient_with_respect_to_logits():
    z = rng.normal(size=6)
    q = rng.uniform(size=6)

    def f(zz):
        return binary_ce(1 / (1 + np.exp(-zz)), q)

    p = 1 / (1 + np.exp(-z))
    g = ce_grad_logit(p, q)
    h = 1e-6
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        assert (f(z + e) - f(z - e)) / (2 * h) == pytest.approx(g[i], abs=1e-7)


def test_alpha_schedule():
    assert alpha_schedule(0, 1000, 0.75) == 0.0
    assert alpha_schedule(100, 1000, 0.75) == pytest.approx(0.375)
    assert alpha_schedule(200, 1000, 0.75) == 0.75
    assert alpha_schedule(999, 1000, 0.75) == 0.75
    assert alpha_schedule(-5, 1000, 0.5) == 0.0
    assert alpha_schedule(3, 10, 0.5, warmup_frac=0.0) == 0.5
    with pytest.raises(ValueError):
        alpha_schedule(0, 0, 0.5)


def test_full_agreement_means_zero_pgr():
    y = rng.integers(0, 2, size=50)
    weak = rng.integers(0, 2, size=50)
    student = weak.copy()
    assert agreement(student, weak) == 1.0
    acc_weak = np.mean(weak == y)
    acc_student = np.mean(student == y)
    assert pgr(acc_weak, acc_student, 1.0) == 0.0
    assert agreement([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    with pytest.raises(ValueError):
        agreement([1, 0], [1, 0, 1])


def test_worked_example():
    p = np.array([0.9, 0.6, 0.4, 0.2])
    w = np.array([0.7, 0.3, 0.6, 0.1])
    t = adaptive_threshold(p)
    assert t == 0.4
    hard = harden(p, t)
    np.testing.assert_array_equal(hard, [1.0, 1.0, 0.0, 0.0])
    target = bootstrap_target(w, hard, 0.5)
    np.testing.assert_allclose(target, [0.85, 0.65, 0.30, 0.05])
    assert binary_ce(p, w) == pytest.approx(0.6688, abs=1e-4)
    assert binary_ce(p, hard) == pytest.approx(0.3375, abs=1e-4)
    assert confidence_loss(p, w, 0.5) == pytest.approx(0.5032, abs=1e-4)
    np.testing.assert_allclose(
        4 * ce_grad_logit(p, target), [0.05, -0.05, 0.10, 0.15], atol=1e-12
    )
    assert pgr(0.6, 0.7, 0.8) == pytest.approx(0.5)
    assert pgr(0.4, 0.3, 0.2) == pytest.approx(0.5)


def test_alpha_schedule_short_runs_and_batches_of_any_shape():
    assert alpha_schedule(0, 1, 0.5) == 0.0
    assert alpha_schedule(0, 4, 0.5) == 0.0
    p = rng.uniform(size=(2, 3))
    assert int(np.sum(p > adaptive_threshold(p))) == 3
