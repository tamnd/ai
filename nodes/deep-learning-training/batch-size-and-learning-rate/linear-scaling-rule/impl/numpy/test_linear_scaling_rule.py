# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-scaling-rule/ (ai://JDMWH4).
"""Tests of the NumPy reference: the linear and square-root rules, the warmup hook and the combined schedule; k small
SGD steps against one scaled step on least squares; B_simple from per-example gradients and the two-batch estimator,
both checked for bias by Monte Carlo; the moving-average tracker; B_noise and the optimal step against the expected
loss of a quadratic; the steps-examples trade-off and its fit; and the worked example of the page."""

import math

import numpy as np
import pytest

from linear_scaling_rule import (
    EMA,
    NoiseScaleTracker,
    b_noise,
    b_simple,
    examples_for_batch,
    fit_critical_batch,
    gradual_warmup,
    max_step,
    optimal_improvement,
    optimal_step,
    scale_learning_rate,
    scaled_lr_schedule,
    steps_for_batch,
    two_batch_estimates,
)

rng = np.random.default_rng(3)


def test_rules():
    assert scale_learning_rate(0.1, 256, 8192) == pytest.approx(3.2)
    assert scale_learning_rate(0.1, 256, 8192, "sqrt") == pytest.approx(0.1 * math.sqrt(32))
    assert scale_learning_rate(0.1, 256, 8192, "none") == 0.1
    assert scale_learning_rate(0.4, 1024, 256) == pytest.approx(0.1)
    assert scale_learning_rate(0.0, 256, 512) == 0.0
    for bad in [(0.1, 0, 256), (0.1, 256, 0), (-0.1, 256, 512)]:
        with pytest.raises(ValueError):
            scale_learning_rate(*bad)
    with pytest.raises(ValueError):
        scale_learning_rate(0.1, 256, 512, "cubic")


def test_warmup_hook():
    hook = gradual_warmup(10, 0.25)
    assert hook(0, 4.0) == pytest.approx(1.0)
    assert hook(5, 4.0) == pytest.approx(2.5)
    assert hook(9, 4.0) == pytest.approx(4.0 * (0.25 + 0.75 * 0.9))
    assert hook(10, 4.0) == 4.0
    assert hook(50, 4.0) == 4.0
    assert gradual_warmup(0, 0.5)(0, 3.0) == 3.0
    assert gradual_warmup(4, 0.0)(1, 8.0) == pytest.approx(2.0)
    assert gradual_warmup(4, 1.0)(1, 8.0) == pytest.approx(8.0)
    for bad in [(-1, 0.5), (5, -0.1), (5, 1.1)]:
        with pytest.raises(ValueError):
            gradual_warmup(*bad)


def test_goyal_schedule():
    # ResNet-50: 0.1 per 256 images, minibatch 8k, warmup from 0.1 over 5 epochs, decay by 10 at epochs 30, 60, 80
    def decay(epoch):
        return 0.1 ** sum(epoch >= e for e in (30, 60, 80))

    lr = scaled_lr_schedule(0.1, 256, 8192, schedule=decay, warmup=gradual_warmup(5, 0.1 / 3.2))
    assert lr(0) == pytest.approx(0.1)
    assert lr(2.5) == pytest.approx(0.1 + (3.2 - 0.1) / 2)
    assert lr(5) == pytest.approx(3.2)
    assert lr(29) == pytest.approx(3.2)
    assert lr(30) == pytest.approx(0.32)
    assert lr(85) == pytest.approx(0.0032)
    assert scaled_lr_schedule(0.1, 256, 1024)(7) == pytest.approx(0.4)
    assert scaled_lr_schedule(0.1, 256, 1024, rule="sqrt")(0) == pytest.approx(0.2)


def _lsq(n_total, d, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n_total, d))
    y = x @ r.normal(size=d) + 0.5 * r.normal(size=n_total)
    return x, y


def _grad(x, y, w):
    return x.T @ (x @ w - y) / len(y)


def test_k_small_steps_vs_one_scaled_step():
    x, y = _lsq(256, 5, 0)
    k, n = 8, 32
    w0 = np.zeros(5)
    gaps = {}
    for rule in ("linear", "sqrt", "none"):
        for eta in (0.01, 0.02):
            w = w0.copy()
            for j in range(k):
                w = w - eta * _grad(x[j * n : (j + 1) * n], y[j * n : (j + 1) * n], w)
            big = w0 - scale_learning_rate(eta, n, k * n, rule) * _grad(x, y, w0)
            gaps[rule, eta] = np.linalg.norm(w - big) / np.linalg.norm(w - w0)
    assert gaps["linear", 0.01] < 0.05
    # the gap of the linear rule is second order: halving eta halves the relative gap
    assert gaps["linear", 0.02] / gaps["linear", 0.01] == pytest.approx(2.0, rel=0.15)
    assert gaps["sqrt", 0.01] > 10 * gaps["linear", 0.01]
    assert gaps["none", 0.01] > 10 * gaps["linear", 0.01]


def test_b_simple_small_case():
    g = np.array([[3.0, 1.0], [1.0, -1.0], [2.0, 2.0], [2.0, -2.0]])
    b, g2, tr = b_simple(g)
    assert tr == pytest.approx(4.0)
    assert g2 == pytest.approx(3.0)
    assert b == pytest.approx(4.0 / 3.0)
    # orthogonal pair: the unbiased |G|^2 is exactly zero
    assert b_simple([[1.0, 0.0], [0.0, 1.0]])[0] == math.inf
    assert b_simple([[1.0, 0.0], [-1.0, 0.0]])[0] == math.inf
    for bad in ([[1.0, 2.0]], [1.0, 2.0, 3.0]):
        with pytest.raises(ValueError):
            b_simple(bad)


def _population(d=4):
    a = rng.normal(size=(d, d))
    sigma = a @ a.T
    return rng.normal(size=d), sigma


def test_b_simple_unbiased_parts():
    grad, sigma = _population()
    chol = np.linalg.cholesky(sigma)
    n, reps = 8, 40000
    z = rng.normal(size=(reps, n, 4)) @ chol.T + grad
    g2s, trs = [], []
    for batch in z:
        _, g2, tr = b_simple(batch)
        g2s.append(g2)
        trs.append(tr)
    assert np.mean(trs) == pytest.approx(np.trace(sigma), rel=0.02)
    assert np.mean(g2s) == pytest.approx(grad @ grad, abs=0.03 * np.trace(sigma))


def test_two_batch_exact_and_errors():
    g2, s = two_batch_estimates(3.0, 1.25, 32, 256)
    assert g2 == pytest.approx(1.0)
    assert s == pytest.approx(64.0)
    for bad in [(0, 4), (4, 4), (8, 4), (-1, 4)]:
        with pytest.raises(ValueError):
            two_batch_estimates(1.0, 1.0, *bad)


def test_two_batch_unbiased():
    grad, sigma = _population()
    chol = np.linalg.cholesky(sigma)
    bs, bb, reps = 4, 32, 20000
    est = []
    for _ in range(reps):
        x = rng.normal(size=(bb, 4)) @ chol.T + grad
        gs = x[:bs].mean(axis=0)
        gb = x.mean(axis=0)
        est.append(two_batch_estimates(gs @ gs, gb @ gb, bs, bb))
    est = np.array(est)
    assert est[:, 1].mean() == pytest.approx(np.trace(sigma), rel=0.03)
    assert est[:, 0].mean() == pytest.approx(grad @ grad, abs=0.03 * np.trace(sigma))


def test_ema():
    e = EMA(0.9)
    with pytest.raises(ValueError):
        _ = e.value
    assert e.update(5.0) == pytest.approx(5.0)
    assert e.update(5.0) == pytest.approx(5.0)
    e2 = EMA(0.5)
    e2.update(2.0)
    assert e2.update(4.0) == pytest.approx((0.25 * 2.0 + 0.5 * 4.0) / 0.75)
    assert EMA(0.0).update(7.0) == 7.0
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError):
            EMA(bad)


def test_tracker():
    t = NoiseScaleTracker(32, 256)
    assert t.g2.decay == 0.99
    assert t.update(3.0, 1.25) == pytest.approx(64.0)
    assert t.update(3.0, 1.25) == pytest.approx(64.0)
    assert NoiseScaleTracker(1, 2).update(2.0, 1.0) == math.inf
    assert NoiseScaleTracker(1, 2).update(5.0, 2.0) == math.inf
    # noisy pairs: ratio of averages recovers tr(Sigma) / |G|^2
    grad, sigma = _population()
    chol = np.linalg.cholesky(sigma)
    tr = NoiseScaleTracker(4, 64, decay=0.999)
    for _ in range(6000):
        x = rng.normal(size=(64, 4)) @ chol.T + grad
        gs, gb = x[:4].mean(axis=0), x.mean(axis=0)
        tr.update(gs @ gs, gb @ gb)
    assert tr.value == pytest.approx(np.trace(sigma) / (grad @ grad), rel=0.25)


def _quad(d=5):
    a = rng.normal(size=(d, d))
    h = a @ a.T + np.eye(d)
    c = rng.normal(size=(d, d))
    return h, c @ c.T, rng.normal(size=d)


def test_b_noise_and_optimal_step():
    h, sigma, g = _quad()
    bn = b_noise(h, sigma, g)
    assert bn == pytest.approx(np.trace(h @ sigma) / (g @ h @ g))
    assert b_noise(2.5 * np.eye(5), sigma, g) == pytest.approx(np.trace(sigma) / (g @ g))
    emax = max_step(h, g)
    assert emax == pytest.approx((g @ g) / (g @ h @ g))
    for batch in (1, 10, 1000):
        eps = np.linspace(1e-4, 2 * emax, 20001)
        exp_loss = -eps * (g @ g) + 0.5 * eps**2 * (g @ h @ g + np.trace(h @ sigma) / batch)
        best = eps[np.argmin(exp_loss)]
        assert optimal_step(emax, bn, batch) == pytest.approx(best, rel=1e-3)
        dl_max = 0.5 * (g @ g) ** 2 / (g @ h @ g)
        assert optimal_improvement(dl_max, bn, batch) == pytest.approx(-exp_loss.min(), rel=1e-6)
    assert optimal_step(2.0, 0.0, 7) == 2.0
    assert optimal_step(2.0, 8.0, 8) == pytest.approx(1.0)
    for bad in [(1.0, 1.0, 0), (1.0, -1.0, 4)]:
        with pytest.raises(ValueError):
            optimal_step(*bad)
    with pytest.raises(ValueError):
        b_noise(h, sigma, np.zeros(5))
    with pytest.raises(ValueError):
        max_step(h, np.zeros(5))
    with pytest.raises(ValueError):
        max_step(-np.eye(5), g)


def test_tradeoff():
    b = np.array([1.0, 16.0, 64.0, 1024.0, 1e6])
    s = steps_for_batch(b, 1000.0, 64.0)
    e = examples_for_batch(b, 1000.0, 64.0)
    assert np.allclose(e, b * s)
    assert np.allclose((s / 1000.0 - 1) * (e / 64000.0 - 1), 1.0)
    assert steps_for_batch(64.0, 1000.0, 64.0) == pytest.approx(2000.0)
    assert examples_for_batch(64.0, 1000.0, 64.0) == pytest.approx(128000.0)
    assert steps_for_batch(1024.0, 1000.0, 64.0) == pytest.approx(1062.5)


def test_fit_critical_batch():
    b = np.array([8.0, 32.0, 128.0, 512.0, 2048.0])
    s_min, e_min, b_crit = fit_critical_batch(b, steps_for_batch(b, 500.0, 200.0))
    assert (s_min, e_min, b_crit) == pytest.approx((500.0, 100000.0, 200.0))
    noisy = steps_for_batch(b, 500.0, 200.0) * np.exp(0.02 * rng.normal(size=5))
    assert fit_critical_batch(b, noisy)[2] == pytest.approx(200.0, rel=0.15)
    for bad in [([8.0], [1.0]), ([8.0, 8.0], [1.0, 2.0]), ([8.0, 16.0], [1.0])]:
        with pytest.raises(ValueError):
            fit_critical_batch(*bad)


def test_worked_example():
    # the numbers of the Worked example section
    assert scale_learning_rate(0.1, 256, 8192) == pytest.approx(3.2)
    assert scale_learning_rate(0.1, 256, 8192, "sqrt") == pytest.approx(0.5657, abs=1e-4)
    g2, s = two_batch_estimates(3.0, 1.25, 32, 256)
    assert s / g2 == pytest.approx(64.0)
    assert [optimal_step(1.0, 64.0, b) for b in (16, 64, 256)] == pytest.approx([0.2, 0.5, 0.8])
    assert optimal_step(1.0, 64.0, 64) / optimal_step(1.0, 64.0, 16) == pytest.approx(2.5)
    assert steps_for_batch(64.0, 1000.0, 64.0) == pytest.approx(2000.0)
    assert examples_for_batch(1024.0, 1000.0, 64.0) == pytest.approx(1088000.0)
