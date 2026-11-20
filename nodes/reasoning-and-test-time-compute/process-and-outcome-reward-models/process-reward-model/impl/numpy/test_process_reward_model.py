# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/process-reward-model/ (ai://NEW-process-reward-model).
import math

import numpy as np
from process_reward_model import (
    aggregate,
    bce,
    best_of_n,
    error_f1,
    find_first_error,
    first_error_labels,
    implicit_process_rewards,
    mc_labels,
    predict_first_error,
    rollout_check,
    rollout_value,
    sample_solutions,
    sigmoid,
    step_features,
    step_scores,
    train_logistic,
    weighted_vote,
)

P_STEP, P_LUCKY, STEPS = 0.9, 0.15, 6


def test_first_error_labels():
    ok = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool)
    assert first_error_labels(ok).tolist() == [[1, 1, 0, -1], [1, 1, 1, 1]]


def test_mc_label_means():
    rng = np.random.default_rng(0)
    q = np.full(200000, 0.1)
    assert abs(mc_labels(q, 8, rng).mean() - 0.1) < 0.003
    assert abs(mc_labels(q, 8, rng, "hard").mean() - (1 - 0.9**8)) < 0.005


def test_cross_entropy_fits_label_mean():
    rng = np.random.default_rng(1)
    x = np.ones((50000, 1))
    y = mc_labels(np.full(50000, 0.3), 8, rng)
    w = train_logistic(x, y, lr=1.0, epochs=500)
    assert abs(sigmoid(w[0]) - 0.3) < 0.01
    assert bce(np.full(10, 0.3), np.full(10, 0.3)) < bce(np.full(10, 0.5), np.full(10, 0.3))


def test_prm_finds_wrong_steps_and_beats_outcome_selection():
    rng = np.random.default_rng(2)
    ok, ans = sample_solutions(4000, STEPS, P_STEP, P_LUCKY, rng)
    feats = step_features(ok, 0.8, rng)
    w = train_logistic(feats, first_error_labels(ok))
    s = step_scores(w, feats)
    # step scores separate right from wrong steps
    assert s[ok].mean() > 0.85 and s[~ok].mean() < 0.5
    # best-of-16 on fresh problems: the PRM picks answer-right solutions more often than a single sample, and with
    # fewer process errors than picking any answer-right solution (what an oracle on final answers would do)
    picks_prm, clean_prm, clean_out = [], [], []
    for _ in range(400):
        ok, ans = sample_solutions(16, STEPS, P_STEP, P_LUCKY, rng)
        sc = aggregate(step_scores(w, step_features(ok, 0.8, rng)), "min")
        i = best_of_n(sc)
        picks_prm.append(ans[i])
        clean_prm.append(ok[i, -1])
        right = np.flatnonzero(ans)
        if len(right):
            clean_out.append(ok[rng.choice(right), -1])
    assert np.mean(picks_prm) > (P_STEP**STEPS + (1 - P_STEP**STEPS) * P_LUCKY) + 0.2
    assert np.mean(clean_prm) > np.mean(clean_out)


def test_aggregation_order():
    s = np.array([0.95] * 10)
    assert aggregate(s, "prod") <= aggregate(s, "min") <= aggregate(s, "last")
    assert round(float(aggregate(s, "prod")), 3) == 0.599


def test_weighted_vote():
    assert weighted_vote([3, 5, 5], [0.9, 0.2, 0.3]) == 3
    assert weighted_vote([3, 5, 5], [0.4, 0.3, 0.3]) == 5


def test_binary_search_with_rollouts():
    rng = np.random.default_rng(3)
    for m in (1, 7, 20):
        for f in range(1, m + 2):
            got, n = find_first_error(lambda t, f=f: t < f, m)
            assert got == f and n <= math.ceil(math.log2(m + 1))
    # noisy checks from rollouts: with enough rollouts the search still finds the first error
    hits = 0
    for _ in range(300):
        ok, _ = sample_solutions(1, 12, 0.85, 0.0, rng)
        ok = ok[0]
        true = int(np.argmin(ok)) + 1 if not ok.all() else 13
        got, _ = find_first_error(rollout_check(ok, 0.85, 0.0, 16, rng), 12)
        hits += got == true
    assert hits / 300 > 0.9
    assert math.isclose(rollout_value(True, 2, 0.9, 0.0), 0.81)


def test_implicit_rewards_telescope():
    rng = np.random.default_rng(4)
    lp, lr = np.log(rng.random(7)), np.log(rng.random(7))
    r, q = implicit_process_rewards(lp, lr, 0.1)
    assert math.isclose(q[-1], 0.1 * (lp.sum() - lr.sum())) and math.isclose(r.sum(), q[-1])


def test_error_f1_and_prediction():
    assert error_f1([0, 1, -1, -1], [0, 2, -1, 3]) == (1 / 3, 1.0, 0.5)
    assert error_f1([0, 0], [-1, 1])[2] == 0.0
    s = np.array([[0.9, 0.2, 0.8], [0.9, 0.9, 0.9]])
    assert predict_first_error(s).tolist() == [1, -1]
