# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-model/ (ai://NEW-reward-model).
"""Tests of the NumPy code: the shared fixture, the Bradley-Terry identity, gradients against finite differences, the
noisy loss's bound and vanishing pull, the margin, the ranked loss and its worked example, shift invariance, Llama 2's
combination, ensembles, the best-of-n KL by simulation and exactly, accuracy and calibration on simulated labels, a
linear reward model that recovers the direction of the true reward, and the cleaning-robot cases."""

import math

import numpy as np
import pytest
from reward_model import (
    batch_ranked_loss,
    best_of_n,
    bon_kl,
    bon_kl_discrete,
    calibration_table,
    comparisons_and_passes,
    ensemble,
    fit_linear_reward,
    hackable,
    llama2_combined_reward,
    llama2_margin,
    llama2_rl_reward,
    log_sigmoid,
    pairwise_accuracy,
    pairwise_loss,
    pairwise_loss_grad,
    pairwise_losses,
    preference_prob,
    ranked_loss,
    ranked_loss_grad,
    section_score,
    shift_to_reference,
    sigmoid,
    whiten,
)

# The fixture shared with the PyTorch tests: four comparisons, the first reward preferred.
RW = np.array([2.0, 0.5, -1.0, 0.3])
RL = np.array([0.5, 1.0, -1.0, -2.0])
MARGIN = np.array([1.0, 2 / 3, 0.0, 1 / 3])
FIXTURE_PLAIN = 0.4910457268301918
FIXTURE_MARGIN = 0.6839872064402405
FIXTURE_NOISY = 0.5044775494077371
FIXTURE_BATCH_RANKED = 0.303617303750964
RANKINGS = [[2.0, 0.5, 0.0], [1.0, -1.0], [0.3, 0.2, 0.1, -0.5]]


def test_fixture():
    assert abs(pairwise_loss(RW, RL) - FIXTURE_PLAIN) < 1e-12
    assert abs(pairwise_loss(RW, RL, margin=MARGIN) - FIXTURE_MARGIN) < 1e-12
    assert abs(pairwise_loss(RW, RL, eps=0.1) - FIXTURE_NOISY) < 1e-12
    assert abs(batch_ranked_loss(RANKINGS) - FIXTURE_BATCH_RANKED) < 1e-12


def test_bradley_terry_identity_and_stability():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=100) * 3, rng.normal(size=100) * 3
    assert np.allclose(np.exp(a) / (np.exp(a) + np.exp(b)), preference_prob(a, b))
    assert np.allclose(pairwise_losses(a, b), np.log1p(np.exp(b - a)))
    big = log_sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(big)) and abs(big[0] + 1000) < 1e-9 and abs(big[1]) < 1e-12
    assert abs(sigmoid(0.0) - 0.5) < 1e-15


@pytest.mark.parametrize(("margin", "eps"), [(0.0, 0.0), (MARGIN, 0.0), (0.0, 0.1), (MARGIN, 0.25)])
def test_pairwise_gradient(margin, eps):
    gw, gl = pairwise_loss_grad(RW, RL, margin, eps)
    assert np.allclose(gw, -gl)
    h = 1e-6
    for k in range(len(RW)):
        e = np.zeros_like(RW)
        e[k] = h
        num_w = (pairwise_loss(RW + e, RL, margin, eps) - pairwise_loss(RW - e, RL, margin, eps)) / (2 * h)
        num_l = (pairwise_loss(RW, RL + e, margin, eps) - pairwise_loss(RW, RL - e, margin, eps)) / (2 * h)
        assert abs(num_w - gw[k]) < 1e-8 and abs(num_l - gl[k]) < 1e-8
    if eps == 0.0 and np.isscalar(margin):
        assert np.allclose(gw, -sigmoid(RL - RW) / len(RW))


def test_noise_bounds_the_loss_and_its_pull():
    d = np.array([-3.0])
    assert abs(pairwise_loss(d, 0.0) - 3.048587) < 1e-6
    assert abs(pairwise_loss(d, 0.0, eps=0.1) - 2.378567) < 1e-6
    assert abs(abs(pairwise_loss_grad(d, 0.0)[0][0]) - 0.952574) < 1e-6
    assert abs(abs(pairwise_loss_grad(d, 0.0, eps=0.1)[0][0]) - 0.438687) < 1e-6
    grid = np.linspace(-20, 20, 401)
    assert np.all(pairwise_losses(grid, 0.0, eps=0.1) < -math.log(0.05))
    assert abs(pairwise_loss_grad(np.array([-30.0]), 0.0, eps=0.1)[0][0]) < 1e-11
    assert abs(pairwise_loss_grad(np.array([-30.0]), 0.0)[0][0] + 1) < 1e-12
    with pytest.raises(ValueError):
        preference_prob(0.0, 0.0, eps=1.0)


def test_margin():
    assert abs(pairwise_loss([1.0], [0.0], margin=1.0) - math.log(2)) < 1e-12
    assert abs(pairwise_loss([1.0], [0.0]) - 0.313262) < 1e-6
    d = np.linspace(-5, 5, 101)
    assert np.all(np.diff(pairwise_losses(d, 0.0, margin=1.0)) < 0)
    assert np.allclose(
        llama2_margin(["significantly better", "better", "slightly better", "unsure"]), [1, 2 / 3, 1 / 3, 0]
    )
    assert list(llama2_margin(["significantly better", "negligibly better"], "large")) == [3.0, 0.0]


def test_ranked_loss_worked_example():
    r = [2.0, 0.5, 0.0]
    assert abs(ranked_loss(r) - 0.267473) < 1e-6
    assert np.allclose(ranked_loss_grad(r), [-0.100543, -0.065038, 0.165581], atol=1e-6)
    assert abs(ranked_loss(r) - pairwise_loss([2.0, 2.0, 0.5], [0.5, 0.0, 0.0])) < 1e-12
    with pytest.raises(ValueError):
        ranked_loss([1.0])


def test_ranked_gradient_and_counts():
    rng = np.random.default_rng(1)
    r = rng.normal(size=7)
    g = ranked_loss_grad(r)
    assert abs(g.sum()) < 1e-12
    h = 1e-6
    num = [(ranked_loss(r + h * e) - ranked_loss(r - h * e)) / (2 * h) for e in np.eye(7)]
    assert np.allclose(num, g, atol=1e-8)
    assert comparisons_and_passes(9) == (36, 9, 72)
    assert comparisons_and_passes(4) == (6, 4, 12)
    assert 64 * comparisons_and_passes(9)[0] == 2304
    # Every prompt counts once in the batch loss, however many responses it has.
    assert abs(batch_ranked_loss(RANKINGS) - np.mean([ranked_loss(x) for x in RANKINGS])) < 1e-15


def test_shift_invariance_and_normalization():
    rng = np.random.default_rng(2)
    c = rng.normal()
    assert abs(pairwise_loss(RW + c, RL + c) - pairwise_loss(RW, RL)) < 1e-12
    assert abs(pairwise_loss(2 * RW, 2 * RL) - pairwise_loss(RW, RL)) > 1e-3
    ref = rng.normal(loc=3.0, size=50)
    shifted = shift_to_reference(ref, ref)
    assert abs(shifted.mean()) < 1e-12
    w = whiten(rng.normal(loc=5, scale=3, size=1000))
    assert abs(w.mean()) < 1e-12 and abs(w.std() - 1) < 1e-6


def test_llama2_combination():
    safety = np.array([0.9, 0.1, 0.9, 0.5])
    helpful = np.array([0.2, 0.8, 0.7, 0.6])
    unsafe = np.array([True, False, False, False])
    assert list(llama2_combined_reward(safety, helpful, unsafe)) == [0.9, 0.1, 0.7, 0.6]
    out = llama2_rl_reward(safety, helpful, unsafe)
    assert abs(out.mean()) < 1e-12
    assert np.argsort(out).tolist() == np.argsort([0.9, 0.1, 0.7, 0.6]).tolist()


def test_ensembles():
    r = np.array([[1.0], [1.2], [3.0]])
    assert abs(ensemble(r)[0] - 1.733333) < 1e-6
    assert ensemble(r, "min")[0] == 1.0
    assert abs(ensemble(r, "uwo", lam=0.5)[0] - 1.328889) < 1e-6
    rng = np.random.default_rng(3)
    rr = rng.normal(size=(5, 100))
    assert np.all(ensemble(rr, "min") <= ensemble(rr)) and np.all(ensemble(rr, "uwo", 0.3) <= ensemble(rr))
    with pytest.raises(ValueError):
        ensemble(rr, "median")
    with pytest.raises(ValueError):
        ensemble(rr, "uwo", lam=-1)


def test_best_of_n_kl():
    assert bon_kl(1) == 0.0
    for n, want in [(4, 0.636294), (16, 1.835089), (64, 3.174508)]:
        assert abs(bon_kl(n) - want) < 1e-6
    rng = np.random.default_rng(4)
    # On a fine discrete policy with distinct rewards the exact KL approaches the continuous value from below.
    q = rng.dirichlet(np.ones(50_000))
    for n in (2, 4, 16):
        assert bon_kl(n) - 0.01 < bon_kl_discrete(q, n) <= bon_kl(n) + 1e-12
    # With few outputs (ties likely) the KL is well below.
    assert bon_kl_discrete([0.5, 0.5], 16) < 0.5 * bon_kl(16)
    # Selection: the best-of-n output's empirical law matches the exact best-of-n probabilities.
    q = np.array([0.5, 0.3, 0.2])
    draws = rng.choice(3, p=q, size=(100_000, 4))
    picked = np.take_along_axis(draws, best_of_n(draws)[:, None], axis=1)[:, 0]
    freq = np.bincount(picked, minlength=3) / len(picked)
    cdf = np.cumsum(q)
    exact = cdf**4 - np.concatenate([[0.0], cdf[:-1]]) ** 4
    assert np.allclose(freq, exact, atol=0.01)


def test_accuracy_ceiling_and_calibration():
    rng = np.random.default_rng(5)
    n = 200_000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    first = rng.random(n) < sigmoid(a - b)  # Bradley-Terry labels from the true reward
    r_w, r_l = np.where(first, a, b), np.where(first, b, a)
    acc = pairwise_accuracy(r_w, r_l)
    expected = float(np.mean(sigmoid(np.abs(a - b))))
    assert abs(acc - expected) < 0.005
    # The true reward is calibrated: predicted and observed agree in every bin.
    for pred, obs, count in calibration_table(r_w, r_l, bins=5):
        assert abs(pred - obs) < 4 * math.sqrt(0.25 / count)
    # A reward scaled by 3 has the same accuracy but is overconfident.
    rows = calibration_table(3 * r_w, 3 * r_l, bins=5)
    assert pairwise_accuracy(3 * r_w, 3 * r_l) == acc
    assert all(pred > obs for pred, obs, _ in rows[1:])


def test_section_score():
    acc = {"chat": 0.9, "chat hard": 0.5, "safety": 0.8, "reasoning": 0.7, "prior sets": 0.6}
    assert abs(section_score(acc) - (0.9 + 0.5 + 0.8 + 0.7 + 0.5 * 0.6) / 4.5) < 1e-12
    assert abs(section_score(acc, dict.fromkeys(acc, 1.0)) - 0.7) < 1e-12


def test_linear_reward_recovers_direction():
    rng = np.random.default_rng(6)
    theta_true = np.array([2.0, -1.0, 0.5])
    xa, xb = rng.normal(size=(4000, 3)), rng.normal(size=(4000, 3))
    first = rng.random(4000) < sigmoid((xa - xb) @ theta_true)
    xw, xl = np.where(first[:, None], xa, xb), np.where(first[:, None], xb, xa)
    theta = fit_linear_reward(xw, xl, steps=2000, lr=1.0)
    cos = theta @ theta_true / np.linalg.norm(theta) / np.linalg.norm(theta_true)
    assert cos > 0.99
    assert np.linalg.norm(theta - theta_true) < 0.3
    # Held-out accuracy approaches the ceiling set by the true reward.
    ya, yb = rng.normal(size=(20000, 3)), rng.normal(size=(20000, 3))
    f = rng.random(20000) < sigmoid((ya - yb) @ theta_true)
    yw, yl = np.where(f[:, None], ya, yb), np.where(f[:, None], yb, ya)
    assert pairwise_accuracy(yw @ theta, yl @ theta) > pairwise_accuracy(yw @ theta_true, yl @ theta_true) - 0.01
    # Regularization shrinks theta.
    assert np.linalg.norm(fit_linear_reward(xw, xl, steps=2000, lr=1.0, l2=0.5)) < np.linalg.norm(theta)


def test_cleaning_robot():
    assert hackable([1, 1, 1], [1, 0, 0]) is not None
    assert hackable([1, 1, 1], [1, 1, 0]) is None
    assert hackable([1, 1.5, 2], [1, 1, 1]) is None
    s, t = hackable([1, 1.5, 3], [1, 1, 1])
    assert np.dot(s, [1, 1, 1]) > np.dot(t, [1, 1, 1]) and np.dot(s, [1, 1.5, 3]) < np.dot(t, [1, 1.5, 3])
    assert hackable([1, 1.5, 3], [1, 1, 1]) is not None
    # Any positive multiple of the true reward is unhackable.
    assert hackable([1, 1.5, 3], [2, 3, 6]) is None
