# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-relative-policy-optimization/ (ai://NEW-group-relative-policy-optimization).
import math

import numpy as np
from grpo import (
    accuracy,
    clipped_surrogate,
    dynamic_sampling_mask,
    group_advantages,
    grpo_step,
    informative_probability,
    kl_k3,
    leave_one_out_advantages,
    objective_and_grad,
    pass_at_k,
    sample,
    token_logp,
    token_weights,
    verify,
)


def test_group_advantages_sum_to_zero_and_binary_closed_form():
    r = np.array([[1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=float)
    a = group_advantages(r, eps=0.0 + 1e-12)
    assert np.allclose(a.sum(-1), 0)
    p = 0.25
    assert np.allclose(a[0, r[0] == 1], math.sqrt((1 - p) / p), atol=1e-6)
    assert np.allclose(a[0, r[0] == 0], -math.sqrt(p / (1 - p)), atol=1e-6)
    assert np.all(a[1] == 0)
    assert dynamic_sampling_mask(r).tolist() == [True, False]


def test_centred_advantage_is_scaled_leave_one_out():
    r = np.random.default_rng(0).normal(size=(5, 6))
    G = r.shape[-1]
    assert np.allclose(group_advantages(r, normalize=False), (G - 1) / G * leave_one_out_advantages(r))


def test_clipped_surrogate_zeroes_gradient_outside_the_trust_region():
    ratio = np.array([0.5, 0.9, 1.0, 1.1, 1.5])
    _, g_pos = clipped_surrogate(ratio, np.ones(5), 0.2, 0.28)
    _, g_neg = clipped_surrogate(ratio, -np.ones(5), 0.2, 0.28)
    assert g_pos.tolist() == [1, 1, 1, 1, 0]  # a good token stops gaining above 1 + eps_high
    assert g_neg.tolist() == [0, -1, -1, -1, -1]  # a bad token stops losing below 1 - eps_low
    v, _ = clipped_surrogate(np.array([1.5]), np.array([1.0]), 0.2, 0.28)
    assert math.isclose(v[0], 1.28)
    assert math.isclose(0.01 * 1.2, 0.012) and math.isclose(0.9 * 1.2, 1.08)


def test_k3_is_nonnegative_and_unbiased():
    rng = np.random.default_rng(1)
    p, ref = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
    s = rng.choice(6, p=p, size=300000)
    k3 = kl_k3(np.log(p[s]), np.log(ref[s]))
    assert k3.min() >= 0
    assert abs(k3.mean() - (p * np.log(p / ref)).sum()) < 5e-3


def test_loss_reductions_weight_long_outputs_differently():
    lengths, adv = [100, 1000], [-1.0, -1.0]
    seq = token_weights(lengths, adv, "sequence")
    assert math.isclose(seq[1] / seq[0], 0.1)
    for mode in ("token", "constant"):
        w = token_weights(lengths, adv, mode, max_len=2048)
        assert math.isclose(w[0], w[1])


def test_informative_groups_and_pass_at_k():
    assert round(informative_probability(0.9, 16), 3) == 0.815
    assert round(informative_probability(0.95, 8), 3) == 0.337
    base, sharp = pass_at_k([0.2, 0.05], 64).mean(), pass_at_k([0.6, 0.0], 64).mean()
    assert base > 0.98 and abs(sharp - 0.5) < 1e-9


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    Q, L, V, G = 4, 3, 5, 6
    logits, ref = rng.normal(size=(Q, L, V)), rng.normal(size=(Q, L, V))
    answers = rng.integers(V, size=Q)
    q = np.arange(Q)
    out = sample(logits, q, G, rng)
    adv = group_advantages(verify(out, answers, q) + 0.1 * rng.normal(size=(Q, G)))
    old_lp, _ = token_logp(logits + 0.05 * rng.normal(size=logits.shape), q, out)
    args = (q, out, adv, old_lp, 0.2, 0.28, 0.1)
    _, grad = objective_and_grad(logits, ref, *args)
    h, num = 1e-6, np.zeros_like(logits)
    for idx in np.ndindex(logits.shape):
        d = np.zeros_like(logits)
        d[idx] = h
        num[idx] = (objective_and_grad(logits + d, ref, *args)[0] - objective_and_grad(logits - d, ref, *args)[0]) / (
            2 * h
        )
    assert np.allclose(grad, num, atol=1e-7)


def test_first_inner_step_is_reinforce_with_group_baseline():
    rng = np.random.default_rng(3)
    Q, L, V, G = 3, 2, 4, 5
    logits = rng.normal(size=(Q, L, V))
    q = np.arange(Q)
    out = sample(logits, q, G, rng)
    adv = rng.normal(size=(Q, G))
    old_lp, full = token_logp(logits, q, out)
    _, grad = objective_and_grad(logits, logits, q, out, adv, old_lp, 0.2, 0.2, 0.5)
    onehot = np.eye(V)[out]
    expected = (adv[..., None, None] * (onehot - np.exp(full)[:, None])).sum(1) / (Q * G * L)
    assert np.allclose(grad, expected)  # ratio 1 and KL gradient 0 at pi = ref


def test_grpo_learns_a_verifiable_task():
    rng = np.random.default_rng(4)
    Q, L, V = 16, 3, 6
    answers = rng.integers(V, size=Q)
    logits = np.zeros((Q, L, V))
    ref = logits.copy()
    start = accuracy(logits, answers)
    rewards = []
    for _ in range(150):
        logits, r = grpo_step(logits, ref, answers, rng)
        rewards.append(r.mean())
    assert math.isclose(start, 1 / V)
    assert accuracy(logits, answers) > 0.9
    assert np.mean(rewards[-20:]) > np.mean(rewards[:20]) + 0.5
