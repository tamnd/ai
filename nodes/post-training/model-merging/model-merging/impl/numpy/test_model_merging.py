# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/model-merging/ (ai://NEW-model-merging).
"""Tests for model_merging.py. The task vectors TAUS are shared with the PyTorch tests."""

import numpy as np
import pytest
from model_merging import (
    RunningAverage,
    apply_task_vectors,
    dare,
    disjoint_mean,
    elect_sign,
    fisher_merge,
    greedy_soup,
    interpolate,
    loss_barrier,
    merge_state,
    softmax,
    softmax_regression_fisher,
    task_vector,
    ties_merge,
    trim,
    uniform_soup,
    weighted_average,
)

PRE = np.zeros(5)
TAUS = np.array(
    [
        [0.5, -0.2, 0.05, 0.8, -0.6],
        [-0.45, -0.4, 0.02, 0.6, 0.1],
        [0.4, 0.1, -0.01, -0.05, -0.7],
    ]
)
TIES_TAU = np.array([0.45, -0.4, 0.0, 0.7, -0.65])  # k = 0.6
MEAN_TAU = np.array([0.15, -0.5 / 3, 0.02, 1.35 / 3, -0.4])


def test_averages():
    models = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 1.0]])
    assert np.allclose(uniform_soup(models), [3.0, 3.0])
    assert np.allclose(weighted_average(models, [1, 1, 1]), uniform_soup(models))
    assert np.allclose(weighted_average(models, [0, 2, 0]), models[1])
    with pytest.raises(ValueError):
        weighted_average(models, [0, 0, 0])


def test_greedy_soup_never_worse_on_validation_and_skips_a_far_model():
    rng = np.random.default_rng(0)
    opt = np.ones(10)
    near = [opt + 0.3 * rng.normal(size=10) for _ in range(5)]
    far = opt + 5.0 + rng.normal(size=10)  # fine-tuned "with too high a learning rate"

    def score(theta):
        return -float(np.sum((theta - opt) ** 2))

    models = near + [far]
    soup, chosen = greedy_soup(models, score)
    assert score(soup) >= max(score(t) for t in models)
    assert len(models) - 1 not in chosen and len(chosen) >= 2
    assert score(soup) > score(uniform_soup(models))


def test_wise_ft_is_one_task_vector():
    rng = np.random.default_rng(1)
    zero, ft = rng.normal(size=6), rng.normal(size=6)
    assert np.allclose(interpolate(zero, ft, 0.0), zero) and np.allclose(interpolate(zero, ft, 1.0), ft)
    for a in (0.25, 0.5, 0.8):
        assert np.allclose(interpolate(zero, ft, a), apply_task_vectors(zero, [task_vector(ft, zero)], a))


def test_running_average_is_the_mean():
    rng = np.random.default_rng(2)
    ws = rng.normal(size=(7, 4))
    avg = RunningAverage()
    for w in ws:
        avg.update(w)
    assert avg.n == 7 and np.allclose(avg.value, ws.mean(axis=0))


def test_task_arithmetic():
    rng = np.random.default_rng(3)
    pre = rng.normal(size=5)
    fts = [pre + t for t in TAUS]
    taus = [task_vector(f, pre) for f in fts]
    assert np.allclose(taus, TAUS)
    assert np.allclose(apply_task_vectors(pre, taus[:1], 1.0), fts[0])
    # lambda = 1/n is the uniform soup of the fine-tuned models.
    assert np.allclose(apply_task_vectors(pre, taus, 1 / 3), uniform_soup(fts))
    assert np.allclose(apply_task_vectors(pre, taus, 1 / 3) - pre, MEAN_TAU)
    # Negation and an analogy are sums of task vectors with signs.
    assert np.allclose(apply_task_vectors(fts[0], [-taus[0]], 1.0), pre)
    assert np.allclose(apply_task_vectors(pre, [taus[2], taus[1], -taus[0]], 1.0), pre + TAUS[2] + TAUS[1] - TAUS[0])


def test_fisher_merge():
    models = [[1.0, 0.0, 2.0], [0.0, 1.0, 4.0]]
    fishers = [[3.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    # Parameter 3 has zero Fisher in both models and falls back to the target model's value.
    assert np.allclose(fisher_merge(models, fishers), [0.75, 0.5, 2.0])
    assert np.allclose(fisher_merge(models, fishers, target=1), [0.75, 0.5, 4.0])
    # Equal Fishers reduce to the (lambda-weighted) average: isotropic merging.
    rng = np.random.default_rng(4)
    th = rng.normal(size=(4, 6))
    assert np.allclose(fisher_merge(th, np.ones((4, 6))), th.mean(axis=0))
    lam = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(fisher_merge(th, np.full((4, 6), 2.5), lam), weighted_average(th, lam))
    # The merge maximizes sum_i lambda_i log N(theta; theta_i, diag(F_i)^-1): the gradient vanishes there.
    F = rng.random((4, 6)) + 0.1
    merged = fisher_merge(th, F, lam)
    assert np.allclose(np.tensordot(lam, F * (th - merged), axes=1), 0.0)


def test_softmax_regression_fisher_closed_form():
    rng = np.random.default_rng(5)
    W, X = rng.normal(size=(3, 4)), rng.normal(size=(50, 4))
    P = softmax(X @ W.T)
    # sum_y p_y (delta_yk - p_k)^2 = p_k (1 - p_k), so F[k, j] = mean_x p_k (1 - p_k) x_j^2.
    want = (P * (1 - P)).T @ X**2 / len(X)
    assert np.allclose(softmax_regression_fisher(W, X), want)


def test_ties_fixture():
    trimmed = np.array([trim(t, 0.6) for t in TAUS])
    assert np.allclose(
        trimmed,
        [[0.5, 0.0, 0.0, 0.8, -0.6], [-0.45, -0.4, 0.0, 0.6, 0.0], [0.4, 0.1, 0.0, 0.0, -0.7]],
    )
    assert np.array_equal(elect_sign(trimmed), [1, -1, 0, 1, -1])
    merged, tau_m = ties_merge(PRE, TAUS, k=0.6, lam=1.0)
    assert np.allclose(tau_m, TIES_TAU) and np.allclose(merged, TIES_TAU)
    # The plain mean shrinks the parameters where signs conflict; the disjoint mean does not.
    assert np.all(np.abs(TIES_TAU[[0, 1, 3, 4]]) > np.abs(MEAN_TAU[[0, 1, 3, 4]]))


def test_trim_and_disjoint_mean():
    rng = np.random.default_rng(6)
    t = rng.normal(size=100)
    for k in (0.0, 0.2, 0.5, 1.0):
        out = trim(t, k)
        assert np.count_nonzero(out) == round(100 * k)
        if 0 < k < 1:
            assert np.abs(out[out != 0]).min() >= np.abs(t[out == 0]).max()
    # With no sign conflict and nothing trimmed, the disjoint mean is the plain mean.
    pos = np.abs(rng.normal(size=(3, 8))) + 0.1
    assert np.allclose(disjoint_mean(pos, elect_sign(pos)), pos.mean(axis=0))


def test_dare_is_unbiased_with_variance_p_over_1_minus_p():
    rng = np.random.default_rng(7)
    delta = np.array([0.002, -0.001, 0.0005, 0.003])
    assert np.allclose(dare(delta, 0.0, rng), delta)
    p, n = 0.9, 200_000
    big = dare(np.tile(delta, (n, 1)), p, rng)
    assert abs(np.mean(big == 0) - p) < 0.005
    assert np.allclose(big.mean(axis=0), delta, rtol=0.05)
    assert np.allclose(big.var(axis=0), delta**2 * p / (1 - p), rtol=0.05)
    # A linear layer's output delta is preserved in expectation; without rescaling it shrinks by (1 - p).
    x = rng.normal(size=4)
    out = big @ x
    se = np.sqrt(p / (1 - p) * np.sum(delta**2 * x**2) / n)
    assert abs(out.mean() - delta @ x) < 5 * se
    dropped_only = np.where(big != 0, delta, 0.0) @ x
    assert abs(dropped_only.mean() - (1 - p) * (delta @ x)) < 5 * (1 - p) * se
    with pytest.raises(ValueError):
        dare(delta, 1.0, rng)


def test_merge_state_shapes():
    rng = np.random.default_rng(8)
    states = [{"w": rng.normal(size=(2, 3)), "b": rng.normal(size=3)} for _ in range(3)]
    out = merge_state(states, uniform_soup)
    assert out["w"].shape == (2, 3) and np.allclose(out["b"], np.mean([s["b"] for s in states], axis=0))


def test_loss_barrier():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert abs(loss_barrier(lambda t: float(t @ t), a, b)) < 1e-12  # convex: the maximum is at an end
    # A loss with two wells along the line has a positive barrier.
    assert loss_barrier(lambda t: float(np.sin(np.pi * t[1]) ** 2), a, b) > 0.99
    assert loss_barrier(lambda t: float(3 * t[0]), a, b) == pytest.approx(1.5)
