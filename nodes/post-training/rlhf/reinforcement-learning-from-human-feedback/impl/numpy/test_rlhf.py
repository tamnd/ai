# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
"""Tests of the NumPy reference: the worked example, the fixture shared with the PyTorch and JAX tests, the gradient
and the shift invariance of the Bradley-Terry loss, rankings of K responses, the KL-shaped reward and its masking,
the clipped surrogate of PPO, PPO-ptx, the KL-regularized optimum, and DPO."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlhf import (bt_grad, bt_loss, bt_prob, dpo_loss, dpo_weight, implicit_reward, kl_divergence, kl_objective,
                  kl_optimal_policy, kl_shaped_rewards, log_sigmoid, ppo_clip_objective, ptx_objective, ranked_loss)

# The shared fixture. The PyTorch and JAX tests hold the same numbers, and the PyTorch tests check them against
# torch.nn.functional.
RW = [1.0, -0.5, 2.0, 0.3]
RL = [0.0, 0.5, -1.0, 0.3]
BT_LOSS = 0.5920644767925333
BT_GRAD_W = [-0.06723535534249878, -0.18276464465750122, -0.011856468294391695, -0.125]
RANKED = [[2.0, 1.0, 0.0, -1.0], [0.5, 1.5, -0.5, 0.0]]
RANKED_LOSS = 0.38710392238639496
LOGP = [[-1.0, -2.0, -0.5], [-0.3, -0.7, 0.0]]
LOGP_REF = [[-1.2, -1.5, -0.5], [-0.4, -0.2, 0.0]]
MASK = [[1, 1, 1], [1, 1, 0]]
SCORE = [1.0, -0.5]
KL_REWARDS = [[-0.02, 0.05, 1.0], [-0.01, -0.45, 0.0]]
PPO_LOGP = [0.0, 0.5, -0.5, 0.1]
PPO_OLD = [0.0, 0.0, 0.0, 0.0]
PPO_ADV = [1.0, 1.0, -1.0, -2.0]
PPO_OBJ = -0.20258545903782382
DPO_ARGS = ([-1.0, -2.0], [-1.5, -1.0], [-1.2, -1.8], [-1.2, -1.4], 0.1)
DPO_LOSS = 0.6960283805447136


def test_fixture():
    assert bt_loss(RW, RL) == pytest.approx(BT_LOSS, rel=1e-12)
    gw, gl = bt_grad(RW, RL)
    np.testing.assert_allclose(gw, BT_GRAD_W, rtol=1e-12)
    np.testing.assert_allclose(gl, -np.array(BT_GRAD_W), rtol=1e-12)
    assert ranked_loss(RANKED) == pytest.approx(RANKED_LOSS, rel=1e-12)
    np.testing.assert_allclose(kl_shaped_rewards(SCORE, LOGP, LOGP_REF, 0.1, MASK), KL_REWARDS, atol=1e-12)
    assert ppo_clip_objective(PPO_LOGP, PPO_OLD, PPO_ADV) == pytest.approx(PPO_OBJ, rel=1e-12)
    assert dpo_loss(*DPO_ARGS) == pytest.approx(DPO_LOSS, rel=1e-12)


def test_worked_example():
    # One prompt, three responses A, B, C, beta = 1, pi_ref = (1/2, 1/4, 1/4) and r = (0, log 2, log 4).
    ref = np.array([0.5, 0.25, 0.25])
    r = np.array([0.0, math.log(2), math.log(4)])
    pi, log_z = kl_optimal_policy(ref, r, 1.0)
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.5], rtol=1e-12)
    assert log_z == pytest.approx(math.log(2), rel=1e-12)
    assert kl_divergence(pi, ref) == pytest.approx(math.log(2) / 4, rel=1e-12)
    assert (pi * r).sum() == pytest.approx(5 * math.log(2) / 4, rel=1e-12)
    assert kl_objective(pi, ref, r, 1.0) == pytest.approx(math.log(2), rel=1e-12)
    # The reward model on the pair C preferred to A: probability 4/5, loss log(5/4), gradient -1/5 on r_C.
    assert bt_prob(r[2], r[0]) == pytest.approx(0.8, rel=1e-12)
    assert bt_loss([r[2]], [r[0]]) == pytest.approx(math.log(1.25), rel=1e-12)
    assert bt_grad([r[2]], [r[0]])[0][0] == pytest.approx(-0.2, rel=1e-12)
    # DPO on the same pair: log 2 at pi = pi_ref with weight 1/2, and log(5/4) at pi = pi_star.
    lr, lp = np.log(ref), np.log(pi)
    assert dpo_loss([lr[2]], [lr[0]], [lr[2]], [lr[0]], 1.0) == pytest.approx(math.log(2), rel=1e-12)
    assert dpo_weight([lr[2]], [lr[0]], [lr[2]], [lr[0]], 1.0)[0] == pytest.approx(0.5, rel=1e-12)
    assert dpo_loss([lp[2]], [lp[0]], [lr[2]], [lr[0]], 1.0) == pytest.approx(math.log(1.25), rel=1e-12)
    np.testing.assert_allclose(implicit_reward(lp, lr, 1.0), [-math.log(2), 0.0, math.log(2)], atol=1e-12)
    # The worked example of PPO: eps = 0.2.
    assert ppo_clip_objective([math.log(1.5)], [0.0], [1.0]) == pytest.approx(1.2, rel=1e-12)
    assert ppo_clip_objective([math.log(0.5)], [0.0], [1.0]) == pytest.approx(0.5, rel=1e-12)
    assert ppo_clip_objective([math.log(0.5)], [0.0], [-1.0]) == pytest.approx(-0.8, rel=1e-12)


def test_log_sigmoid_stable():
    # log(sigmoid(x)) underflows to -inf at x = -800; log_sigmoid gives -800.
    with np.errstate(over="ignore", divide="ignore"):
        naive = np.log(1.0 / (1.0 + np.exp(800.0)))
    assert naive == -np.inf
    assert log_sigmoid(-800.0) == pytest.approx(-800.0, rel=1e-12)
    assert log_sigmoid(800.0) == 0.0
    assert math.isfinite(bt_loss([-800.0], [0.0]))


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.floats(-20, 20), st.floats(-20, 20)), min_size=1, max_size=8))
def test_bt_gradient_matches_finite_differences(pairs):
    rw = np.array([p[0] for p in pairs])
    rl = np.array([p[1] for p in pairs])
    gw, gl = bt_grad(rw, rl)
    h = 1e-6
    for k in range(rw.size):
        e = np.zeros_like(rw)
        e[k] = h
        fw = (bt_loss(rw + e, rl) - bt_loss(rw - e, rl)) / (2 * h)
        fl = (bt_loss(rw, rl + e) - bt_loss(rw, rl - e)) / (2 * h)
        assert fw == pytest.approx(gw[k], abs=1e-7)
        assert fl == pytest.approx(gl[k], abs=1e-7)
    # The gradient is -(1 - sigma(Delta)) / n, is below 1 / n in size, and pushes r_w up and r_l down.
    assert np.all(gw <= 0) and np.all(gl >= 0) and np.all(np.abs(gw) <= 1.0 / rw.size)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.floats(-10, 10), st.floats(-10, 10)), min_size=1, max_size=8), st.floats(-50, 50))
def test_shift_invariance(pairs, c):
    rw = np.array([p[0] for p in pairs])
    rl = np.array([p[1] for p in pairs])
    assert bt_loss(rw + c, rl + c) == pytest.approx(bt_loss(rw, rl), rel=1e-9, abs=1e-12)


def test_bt_is_logistic_regression():
    # The loss at Delta = 0 is log 2, and the loss is the logistic loss log(1 + exp(-Delta)).
    assert bt_loss([0.3], [0.3]) == pytest.approx(math.log(2), rel=1e-12)
    for d in [-3.0, -0.5, 0.0, 1.0, 4.0]:
        assert bt_loss([d], [0.0]) == pytest.approx(math.log1p(math.exp(-d)), rel=1e-12)


def test_softmax_of_two_is_bradley_terry():
    # The four-way softmax loss of Ziegler et al. with two options is the Bradley-Terry loss.
    for rw, rl in [(1.0, 0.0), (-2.0, 0.5), (3.0, 3.0)]:
        softmax = -(rw - np.logaddexp(rw, rl))
        assert softmax == pytest.approx(bt_loss([rw], [rl]), rel=1e-12)


def test_ranked_loss_one_element():
    # Each prompt weighs as one element whatever K is: a ranking of K = 2 is one pair, a ranking of K = 9 is the mean
    # over its 36 pairs, and a batch is the mean over its prompts.
    assert ranked_loss([[1.0, 0.0]]) == pytest.approx(bt_loss([1.0], [0.0]), rel=1e-12)
    r9 = np.linspace(2.0, -2.0, 9)
    pairs = [(r9[i], r9[j]) for i in range(9) for j in range(i + 1, 9)]
    assert len(pairs) == math.comb(9, 2) == 36
    l9 = bt_loss([p[0] for p in pairs], [p[1] for p in pairs])
    assert ranked_loss([r9]) == pytest.approx(l9, rel=1e-12)
    other = r9[::-1]
    assert ranked_loss([r9, other]) == pytest.approx((l9 + ranked_loss([other])) / 2, rel=1e-12)
    with pytest.raises(ValueError):
        ranked_loss([[1.0]])


def test_kl_penalty_sign():
    # The penalty is -beta log(pi / pi_ref): negative where the policy puts more mass than the reference.
    out = kl_shaped_rewards([0.0], [[-0.5]], [[-1.0]], 0.2)
    assert out[0, 0] == pytest.approx(-0.1, rel=1e-12)
    out = kl_shaped_rewards([0.0], [[-1.0]], [[-0.5]], 0.2)
    assert out[0, 0] == pytest.approx(0.1, rel=1e-12)


def test_kl_shaped_sum_is_sequence_reward():
    rng = np.random.default_rng(20261115)
    logp = -rng.exponential(size=(3, 5))
    ref = -rng.exponential(size=(3, 5))
    score = rng.normal(size=3)
    out = kl_shaped_rewards(score, logp, ref, 0.05)
    np.testing.assert_allclose(out.sum(axis=1), score - 0.05 * (logp.sum(axis=1) - ref.sum(axis=1)), rtol=1e-12)


def test_mask_excludes_padding():
    # The padded token gets no penalty and no score, and the score goes to the last response token.
    out = kl_shaped_rewards([2.0], [[-1.0, -1.0, -9.0]], [[-1.0, -1.0, 0.0]], 1.0, [[1, 1, 0]])
    np.testing.assert_allclose(out, [[0.0, 2.0, 0.0]], atol=1e-12)
    with pytest.raises(ValueError):
        kl_shaped_rewards([0.0], [[0.0]], [[0.0]], 0.1, [[0]])
    with pytest.raises(ValueError):
        kl_shaped_rewards([0.0], [[0.0]], [[0.0]], -0.1)


def test_kl_penalty_estimates_kl():
    # Averaged over samples from pi, the KL-shaped reward is E[r] - beta KL(pi || pi_ref).
    rng = np.random.default_rng(20261114)
    ref = np.array([0.5, 0.3, 0.2])
    pi = np.array([0.2, 0.3, 0.5])
    r = np.array([0.0, 1.0, 2.0])
    y = rng.choice(3, size=400_000, p=pi)
    shaped = kl_shaped_rewards(r[y], np.log(pi[y])[:, None], np.log(ref[y])[:, None], 0.5)[:, 0]
    assert shaped.mean() == pytest.approx(kl_objective(pi, ref, r, 0.5), abs=5e-3)


def test_clip_flat_outside_interval():
    # For A > 0 the objective is constant once rho passes 1 + eps, and for A < 0 once it falls below 1 - eps, so the
    # gradient that would move rho further is zero.
    h = 1e-6
    for logp, adv in [(math.log(1.5), 1.0), (math.log(0.5), -1.0)]:
        d = (ppo_clip_objective([logp + h], [0.0], [adv]) - ppo_clip_objective([logp - h], [0.0], [adv])) / (2 * h)
        assert d == pytest.approx(0.0, abs=1e-9)
    # Inside the interval, and on the side that lowers the objective, the gradient is rho * A.
    for logp, adv in [(math.log(1.1), 1.0), (math.log(0.5), 1.0), (math.log(1.5), -1.0)]:
        d = (ppo_clip_objective([logp + h], [0.0], [adv]) - ppo_clip_objective([logp - h], [0.0], [adv])) / (2 * h)
        assert d == pytest.approx(math.exp(logp) * adv, rel=1e-6)


@settings(max_examples=200, deadline=None)
@given(st.floats(-3, 3), st.floats(-5, 5), st.floats(0.0, 0.5))
def test_clip_is_lower_bound(logp, adv, eps):
    rho = math.exp(logp)
    clipped = ppo_clip_objective([logp], [0.0], [adv], eps)
    assert clipped <= rho * adv + 1e-12
    if 1 - eps <= rho <= 1 + eps:
        assert clipped == pytest.approx(rho * adv, rel=1e-12, abs=1e-12)


def test_ptx():
    assert ptx_objective(1.5, [-2.0, -4.0], 0.0) == 1.5
    assert ptx_objective(1.5, [-2.0, -4.0], 27.8) == pytest.approx(1.5 - 27.8 * 3.0, rel=1e-12)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-3, 3), min_size=2, max_size=6), st.floats(0.05, 5.0), st.integers(0, 2**31))
def test_kl_optimum(r, beta, seed):
    rng = np.random.default_rng(seed)
    r = np.array(r)
    ref = rng.dirichlet(np.ones(r.size))
    pi, log_z = kl_optimal_policy(ref, r, beta)
    best = kl_objective(pi, ref, r, beta)
    assert best == pytest.approx(beta * log_z, rel=1e-9, abs=1e-9)
    # The gap of any other policy is beta KL(pi || pi_star).
    for _ in range(5):
        q = rng.dirichlet(np.ones(r.size))
        gap = best - kl_objective(q, ref, r, beta)
        assert gap == pytest.approx(beta * kl_divergence(q, pi), rel=1e-7, abs=1e-9)
        assert gap >= -1e-12
    # The reward is beta log(pi_star / pi_ref) + beta log Z.
    np.testing.assert_allclose(beta * (np.log(pi) - np.log(ref)) + beta * log_z, r, atol=1e-9)


def test_dpo_partition_cancels():
    # With pi = pi_star, the DPO loss is the Bradley-Terry loss of the true rewards, whatever Z is.
    rng = np.random.default_rng(7)
    ref = rng.dirichlet(np.ones(5))
    r = rng.normal(size=5)
    for beta in [0.1, 0.5, 2.0]:
        pi, _ = kl_optimal_policy(ref, r + 3.0, beta)
        lp, lr = np.log(pi), np.log(ref)
        w, lo = [0, 1, 2], [3, 4, 0]
        assert dpo_loss(lp[w], lp[lo], lr[w], lr[lo], beta) == pytest.approx(bt_loss(r[w], r[lo]), rel=1e-9)


def test_dpo_gradient_weight():
    # The derivative of the DPO loss of one pair with respect to log pi(y_w) is -beta sigma(r_hat_l - r_hat_w).
    args = [-1.0], [-1.5], [-1.2], [-1.2]
    beta, h = 0.1, 1e-6
    d = (dpo_loss([-1.0 + h], *args[1:], beta) - dpo_loss([-1.0 - h], *args[1:], beta)) / (2 * h)
    assert d == pytest.approx(-beta * dpo_weight(*args, beta)[0], rel=1e-6)
    assert dpo_weight([0.0], [0.0], [0.0], [0.0], beta)[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        dpo_loss([0.0], [0.0], [0.0], [0.0], 0.0)
