# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/fine-tuning/ (ai://NEW-fine-tuning).
import math

import numpy as np
import pytest
from fine_tuning import (
    adam,
    auxiliary_objective,
    bias_correction_ratio,
    discriminative_lrs,
    ewc_grad,
    ewc_penalty,
    ewc_quadratic_optimum,
    fine_tune,
    grads,
    gradual_unfreezing,
    linear_probe,
    loss,
    lp_ft,
    predict,
    stlr,
    warmup_linear,
)

PHI = (1 + math.sqrt(5)) / 2
# The worked example: perfect features B0 = (1, 1), a zero head, one training input along the first axis.
B0, X, Y, X_OOD = np.array([[1.0, 1.0]]), np.array([[1.0, 0.0]]), np.array([1.0]), np.array([[0.0, 1.0]])


def test_full_fine_tuning_distorts_features():
    v, B = fine_tune(np.array([0.0]), B0, X, Y, lr=1e-3, steps=1)
    assert np.all(B == B0) and v[0] > 0  # a zero head gives the features no gradient on the first step
    v, B = fine_tune(np.array([0.0]), B0, X, Y, lr=1e-3, steps=60_000)
    assert loss(v, B, X, Y) < 1e-12
    assert B[0, 1] == 1.0  # the feature of the unseen direction never moves
    assert abs(B[0, 0] - math.sqrt(PHI)) < 1e-3 and abs(v[0] - 1 / math.sqrt(PHI)) < 1e-3  # gradient flow, up to step size
    assert abs(predict(v, B, X_OOD)[0] - 0.786151) < 1e-3  # the truth is 1


def test_linear_probe_and_lp_ft_keep_features():
    v = linear_probe(B0, X, Y)
    assert np.allclose(predict(v, B0, X_OOD), 1.0)
    v2, B2 = lp_ft(B0, X, Y, steps=100)
    assert np.allclose(B2, B0) and np.allclose(v2, v)  # zero residual, zero gradient


def test_feature_updates_stay_in_the_span_of_training_inputs():
    rng = np.random.default_rng(0)
    d, k, n = 6, 2, 3
    Xs, Ys = rng.normal(size=(n, d)), rng.normal(size=n)
    B, v = rng.normal(size=(k, d)), rng.normal(size=k)
    _, B2 = fine_tune(v, B, Xs, Ys, lr=1e-2, steps=3000)
    null = np.linalg.svd(Xs)[2][n:]  # orthonormal basis of the directions orthogonal to every training input
    assert np.allclose((B2 - B) @ null.T, 0, atol=1e-10)
    assert not np.allclose(B2, B)


def test_balancedness_is_nearly_conserved():
    rng = np.random.default_rng(1)
    Xs, Ys = rng.normal(size=(5, 4)), rng.normal(size=5)
    B, v = rng.normal(size=(2, 4)) * 0.5, rng.normal(size=2) * 0.5
    before = B @ B.T - np.outer(v, v)
    v2, B2 = fine_tune(v, B, Xs, Ys, lr=1e-4, steps=20_000)
    after = B2 @ B2.T - np.outer(v2, v2)
    assert np.abs(after - before).max() < 1e-3


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    Xs, Ys = rng.normal(size=(4, 3)), rng.normal(size=4)
    B, v = rng.normal(size=(2, 3)), rng.normal(size=2)
    gv, gB = grads(v, B, Xs, Ys)
    h = 1e-6
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        assert abs((loss(v + e, B, Xs, Ys) - loss(v - e, B, Xs, Ys)) / (2 * h) - gv[i]) < 1e-7
    for i in range(2):
        for j in range(3):
            E = np.zeros((2, 3))
            E[i, j] = h
            assert abs((loss(v, B + E, Xs, Ys) - loss(v, B - E, Xs, Ys)) / (2 * h) - gB[i, j]) < 1e-7


def test_stlr():
    T = 1000
    rates = [stlr(t, T) for t in range(T + 1)]
    assert rates[0] == pytest.approx(0.01 / 32) and rates[100] == pytest.approx(0.01)
    assert rates[T] == pytest.approx(0.01 / 32)
    assert np.all(np.diff(rates[:101]) > 0) and np.all(np.diff(rates[100:]) < 0)


def test_discriminative_lrs_and_unfreezing():
    lrs = discriminative_lrs(0.01, 4)
    assert lrs[-1] == 0.01 and lrs[-1] / lrs[0] == pytest.approx(2.6**3) and lrs[-1] / lrs[0] == pytest.approx(17.576)
    assert gradual_unfreezing(4, 5) == [[3], [2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]


def test_warmup_linear():
    assert warmup_linear(0, 1000, 2e-5) == 0 and warmup_linear(100, 1000, 2e-5) == pytest.approx(2e-5)
    assert warmup_linear(550, 1000, 2e-5) == pytest.approx(1e-5) and warmup_linear(1000, 1000, 2e-5) == 0


def test_adam_bias_correction():
    g = np.array([0.3, -2.0])
    corrected = adam(lambda th: g, np.zeros(2), 1e-3, 30)
    uncorrected = adam(lambda th: g, np.zeros(2), 1e-3, 30, bias_correction=False)
    for t in (1, 12, 30):
        step_c = np.abs(corrected[t] - corrected[t - 1])
        step_u = np.abs(uncorrected[t] - uncorrected[t - 1])
        assert np.allclose(step_c, 1e-3, rtol=1e-4)
        assert np.allclose(step_u / step_c, bias_correction_ratio(t), rtol=1e-4)
    assert bias_correction_ratio(1) == pytest.approx(math.sqrt(10))
    assert max(range(1, 500), key=bias_correction_ratio) == 12
    assert bias_correction_ratio(12) == pytest.approx(6.5685, abs=1e-4)


def test_ewc():
    rng = np.random.default_rng(3)
    old, fisher = rng.normal(size=5), rng.random(5)
    a, b = rng.random(5) + 0.1, rng.normal(size=5)
    opt = ewc_quadratic_optimum(a, b, old, fisher, lam=4.0)
    total_grad = a * (opt - b) + ewc_grad(opt, old, fisher, 4.0)
    assert np.allclose(total_grad, 0)
    assert np.allclose(ewc_quadratic_optimum(a, b, old, fisher, 0.0), b)
    assert np.allclose(ewc_quadratic_optimum(a, b, old, fisher, 1e12), old)
    assert ewc_penalty(old, old, fisher, 4.0) == 0
    assert ewc_penalty(old + 1, old, fisher, 2.0) == pytest.approx(float(fisher.sum()))


def test_auxiliary_objective():
    assert auxiliary_objective(1.2, 3.0) == pytest.approx(2.7)
