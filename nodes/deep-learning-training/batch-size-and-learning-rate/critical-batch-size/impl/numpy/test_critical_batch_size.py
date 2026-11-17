# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://X29NFT).
"""Tests of the NumPy reference: the per-example and two-batch noise-scale estimators against known moments and for
bias by Monte Carlo; the moving-average ratio; the one-step gain of a quadratic, its optimum and its regimes; the
trade-off hyperbola, its fit and the factor of two at B_crit; one-run conversion to S_min and E_min; and the Kaplan
power law, with its doubling for every 13% fall in the loss."""

import numpy as np
import pytest

from critical_batch_size import (
    RunningNoiseScale,
    examples_needed,
    expected_gain,
    fit_tradeoff,
    kaplan_critical_batch,
    min_steps_and_examples,
    noise_scale_from_examples,
    noise_scale_from_two_batches,
    optimal_step,
    quadratic_noise_scale,
    step_scale,
    steps_needed,
    tradeoff_curve,
)

rng = np.random.default_rng(5)


def test_examples_estimator_by_hand():
    grads = np.array([[1.0, 0.0], [3.0, 2.0]])  # mean (2, 1); deviations (+-1, +-1)
    b, g2, trace = noise_scale_from_examples(grads)
    assert trace == pytest.approx(4.0)  # (1 + 1 + 1 + 1) / (2 - 1)
    assert g2 == pytest.approx(5.0 - 2.0)
    assert b == pytest.approx(4.0 / 3.0)


def test_examples_estimator_unbiased():
    pop = rng.normal(size=(30, 4)) + np.array([0.5, -0.2, 0.0, 0.3])
    G = pop.mean(axis=0)
    trace_true = np.sum((pop - G) ** 2) / 30
    est = [noise_scale_from_examples(pop[rng.integers(0, 30, size=6)])[1:] for _ in range(40_000)]
    g2, trace = np.mean(est, axis=0)
    assert g2 == pytest.approx(G @ G, rel=0.03)
    assert trace == pytest.approx(trace_true, rel=0.01)


def test_examples_estimator_noise_dominated_and_errors():
    b, g2, _ = noise_scale_from_examples(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    assert g2 < 0 and b == np.inf
    for bad in [np.ones(3), np.ones((1, 3))]:
        with pytest.raises(ValueError):
            noise_scale_from_examples(bad)


def test_two_batch_matches_bessel():
    x = rng.normal(size=(9, 3)) + 1.0
    mean = x.mean(axis=0)
    g2, trace = noise_scale_from_two_batches(np.mean(np.sum(x**2, axis=1)), mean @ mean, 1, 9)
    _, g2_ex, trace_ex = noise_scale_from_examples(x)
    assert trace == pytest.approx(trace_ex) and g2 == pytest.approx(g2_ex)


def test_two_batch_unbiased_nested():
    pop = rng.exponential(size=(25, 3)) - 0.7
    G = pop.mean(axis=0)
    trace_true = np.sum((pop - G) ** 2) / 25
    per = pop[rng.integers(0, 25, size=(100_000, 16))]  # (M, 16, 3)
    sq_big = np.sum(per.mean(axis=1) ** 2, axis=1)
    sq_small = np.sum(per[:, :4].mean(axis=1) ** 2, axis=1)  # one worker of four
    g2, trace = noise_scale_from_two_batches(sq_small, sq_big, 4, 16)
    assert g2.mean() == pytest.approx(G @ G, rel=0.03)
    assert trace.mean() == pytest.approx(trace_true, rel=0.02)
    for args in [(1.0, 1.0, 4, 4), (1.0, 1.0, 0, 4), (1.0, 1.0, 4, -1)]:
        with pytest.raises(ValueError):
            noise_scale_from_two_batches(*args)


def test_running_noise_scale():
    tracker = RunningNoiseScale(decay=0.5)
    with pytest.raises(ValueError):
        tracker.value()
    # |G|^2 = 1 and tr(Sigma) = 8 give |G_B|^2 = 1 + 8 / B exactly
    for _ in range(3):
        assert tracker.update(1 + 8 / 2, 1 + 8 / 16, 2, 16) == pytest.approx(8.0)
    tracker = RunningNoiseScale(decay=0.5)
    tracker.update(1 + 8 / 2, 1 + 8 / 16, 2, 16)  # g2 = 1, trace = 8
    tracker.update(3 + 4 / 2, 3 + 4 / 16, 2, 16)  # g2 = 3, trace = 4
    assert tracker.g2 == pytest.approx(0.25 * 1 + 0.5 * 3)
    assert tracker.value() == pytest.approx((0.25 * 8 + 0.5 * 4) / (0.25 + 1.5))
    tracker.update(-20.0 + 1.0, -20.0, 2, 4)  # a large negative g2 estimate
    assert tracker.value() == np.inf
    for bad in [1.0, -0.1]:
        with pytest.raises(ValueError):
            RunningNoiseScale(bad)
    assert RunningNoiseScale(0.0).update(1 + 8 / 2, 1 + 8 / 16, 2, 16) == pytest.approx(8.0)


def quadratic(d=6):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    H = Q @ np.diag(np.geomspace(0.2, 4.0, d)) @ Q.T
    A = rng.normal(size=(d, d))
    return rng.normal(size=d), H, A @ A.T / d


def test_quadratic_optimum():
    G, H, Sigma = quadratic()
    eps_max, dl_max = step_scale(G, H)
    b_noise = quadratic_noise_scale(G, H, Sigma)
    assert eps_max == pytest.approx(G @ G / (G @ H @ G))
    assert dl_max == pytest.approx(expected_gain(eps_max, np.inf, G, H, Sigma))
    for B in [1, 7, 50, 1000]:
        eps = optimal_step(B, eps_max, b_noise)
        best = expected_gain(eps, B, G, H, Sigma)
        assert best == pytest.approx(dl_max / (1 + b_noise / B))
        assert best > expected_gain(eps * 1.01, B, G, H, Sigma)
        assert best > expected_gain(eps * 0.99, B, G, H, Sigma)
        # the gain is zero at twice the optimal step and negative beyond
        assert expected_gain(2 * eps, B, G, H, Sigma) == pytest.approx(0.0, abs=1e-12 * float(dl_max))
        assert expected_gain(2.1 * eps, B, G, H, Sigma) < 0


def test_isotropic_and_regimes():
    G, _, Sigma = quadratic()
    H = 2.5 * np.eye(len(G))
    assert quadratic_noise_scale(G, H, Sigma) == pytest.approx(np.trace(Sigma) / (G @ G))
    assert optimal_step(40.0, 1.0, 40.0) == pytest.approx(0.5)
    assert optimal_step(0.01, 1.0, 40.0) * 2 == pytest.approx(optimal_step(0.02, 1.0, 40.0), rel=1e-3)
    assert optimal_step(1e9, 1.0, 40.0) == pytest.approx(1.0, rel=1e-6)
    assert np.allclose(optimal_step(np.array([1.0, 3.0]), 2.0, 0.0), 2.0)
    with pytest.raises(ValueError):
        optimal_step(0.0, 1.0, 40.0)
    with pytest.raises(ValueError):
        optimal_step(1.0, 1.0, -1.0)
    for f in (quadratic_noise_scale, lambda g, h, s: step_scale(g, h)):
        with pytest.raises(ValueError):
            f(np.zeros(3), np.eye(3), np.eye(3))


def test_tradeoff():
    s_min, e_min = 1000.0, 3.0e5
    b_crit = e_min / s_min
    B = np.array([3.0, 30.0, 300.0, 3000.0, 30000.0])
    S, E = steps_needed(B, s_min, e_min), examples_needed(B, s_min, e_min)
    assert np.allclose((S / s_min - 1) * (E / e_min - 1), 1.0)
    assert steps_needed(b_crit, s_min, e_min) == pytest.approx(2 * s_min)
    assert examples_needed(b_crit, s_min, e_min) == pytest.approx(2 * e_min)
    batch, S, E = tradeoff_curve(s_min, e_min, num=7, span=100.0)
    assert batch[0] == pytest.approx(3.0) and batch[-1] == pytest.approx(30000.0) and batch[3] == pytest.approx(300.0)
    assert np.all(np.diff(S) < 0) and np.all(np.diff(E) > 0)
    for bad in [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]:
        with pytest.raises(ValueError):
            steps_needed(*bad)


def test_fit_tradeoff():
    B = np.geomspace(4, 65536, 9)
    S = (250.0 + 1.0e5 / B) * (1 + 0.01 * rng.normal(size=B.size))
    s_min, e_min, b_crit = fit_tradeoff(B, S)
    assert s_min == pytest.approx(250.0, rel=0.03) and e_min == pytest.approx(1.0e5, rel=0.03)
    assert b_crit == pytest.approx(400.0, rel=0.05)
    exact = fit_tradeoff([10.0, 1000.0], [12.0, 2.1])
    assert exact[0] == pytest.approx(2.0) and exact[1] == pytest.approx(100.0)
    for bad in [
        ([1.0, 1.0], [2.0, 2.0]),
        ([1.0], [2.0]),
        ([1.0, 2.0], [3.0]),
        ([-1.0, 2.0], [3.0, 1.0]),
        ([1.0, 2.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 1.0]),
    ]:
        with pytest.raises(ValueError):
            fit_tradeoff(*bad)


def test_one_run_conversion():
    s_min, e_min = 500.0, 2.0e5
    b_crit = e_min / s_min
    for B in [10.0, 400.0, 8000.0]:
        S = s_min + e_min / B
        assert min_steps_and_examples(B, S, b_crit) == pytest.approx((s_min, e_min))
    with pytest.raises(ValueError):
        min_steps_and_examples(0.0, 1.0, 1.0)


def test_kaplan():
    assert kaplan_critical_batch(1.0) == pytest.approx(2.1e8)
    # 1 / alpha_B = 4.76, so a 13% fall in the loss nearly doubles B_crit
    ratio = kaplan_critical_batch(0.87 * 3.0) / kaplan_critical_batch(3.0)
    assert 1.9 < ratio < 2.0
    assert kaplan_critical_batch(2.0, b_star=1.0, alpha_b=0.5) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        kaplan_critical_batch(0.0)


def test_inputs_as_lists_and_boundaries():
    # plain lists are accepted wherever an array is
    b, g2, trace = noise_scale_from_examples([[0.5, 0.0], [1.0, 0.0]])  # g2 = 0.5 * 1.0, trace = 0.125
    assert (b, g2, trace) == pytest.approx((0.25, 0.5, 0.125))
    assert noise_scale_from_examples([[0.0, 0.0], [2.0, 0.0]])[0] == np.inf  # g2 = 0 exactly
    G, H, Sigma = [0.1, 0.0], [[1.0, 0.0], [0.0, 2.0]], [[0.5, 0.0], [0.0, 0.5]]  # G.H.G = 0.01
    assert quadratic_noise_scale(G, H, Sigma) == pytest.approx(150.0)
    assert step_scale(G, H) == pytest.approx((1.0, 0.005))
    assert expected_gain(1.0, 2, G, H, Sigma) == pytest.approx(0.01 - 0.5 * (0.01 + 1.5 / 2))
    assert optimal_step([0.5, 1.0], 1.0, 1.0).tolist() == pytest.approx([1 / 3, 0.5])
    assert steps_needed([0.5, 2.0], 0.5, 0.5).tolist() == pytest.approx([1.5, 0.75])
    assert examples_needed([0.5], 0.5, 0.5).tolist() == pytest.approx([0.75])
    assert kaplan_critical_batch([0.5, 1.0], b_star=0.5, alpha_b=1.0).tolist() == pytest.approx([1.0, 0.5])
    for bad in [{"b_star": 0.0}, {"alpha_b": 0.0}]:
        with pytest.raises(ValueError):
            kaplan_critical_batch(1.0, **bad)
    # either order of the two batch sizes, and sizes below 1 for averaged or fractional batches
    assert noise_scale_from_two_batches(1 + 8 / 4, 1 + 8 / 2, 4, 2) == pytest.approx((1.0, 8.0))
    assert noise_scale_from_two_batches(1 + 8 / 0.5, 1 + 8, 0.5, 1) == pytest.approx((1.0, 8.0))
    with pytest.raises(ValueError):
        noise_scale_from_two_batches(1.0, 1.0, 4, 0)
    assert len(tradeoff_curve(1.0, 10.0)[0]) == 50


def test_tracker_counts_and_fit_boundaries():
    tracker = RunningNoiseScale(0.9)
    for k in range(1, 4):
        tracker.update(3.0, 2.0, 1, 2)
        assert tracker.steps == k
    s_min, e_min, _ = fit_tradeoff([1.0, 4.0], [2.5, 1.0])  # S = 0.5 + 2 / B, with B = 1 and S = 1
    assert (s_min, e_min) == pytest.approx((0.5, 2.0))
    s_min, e_min, _ = fit_tradeoff([1.0, 4.0], [3.5, 3.125])  # S = 3 + 0.5 / B
    assert (s_min, e_min) == pytest.approx((3.0, 0.5))
    assert min_steps_and_examples(0.5, 0.5, 0.5) == pytest.approx((0.25, 0.125))
    for bad in [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]:
        with pytest.raises(ValueError):
            min_steps_and_examples(*bad)
