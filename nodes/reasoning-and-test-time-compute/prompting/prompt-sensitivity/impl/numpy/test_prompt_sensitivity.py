# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/prompt-sensitivity/ (ai://NEW-prompt-sensitivity).
"""Tests for the toy biased classifier, contextual calibration, GlobalE and the sensitivity statistics."""

import itertools
import math

import numpy as np
from prompt_sensitivity import (
    all_formats,
    binary_calibrated,
    calibrate,
    content_free_prob,
    expected_range,
    flip_probability,
    global_entropy,
    icl_prob,
    kendall_tau_b,
    kendall_w,
    local_entropy,
    order_accuracies,
    prompt_bias,
    ranks,
    render_format,
    selection_optimism,
)

RNG = np.random.default_rng(0)
W = np.array([1.0, -0.5, 0.25])
X = RNG.normal(size=(4000, 3))
Y = np.sign(X @ W)


def test_formats_render_differently():
    fields = [("Passage", "a film"), ("Answer", "positive")]
    texts = {render_format(f, fields) for f in all_formats()}
    assert len(texts) == len(all_formats()) == 60


def test_calibration_is_a_threshold():
    p = RNG.uniform(size=10_000)
    for a in [0.1, 0.5, 0.62]:
        assert np.array_equal(binary_calibrated(p, a), p > a)
    q = RNG.dirichlet(np.ones(4), size=500)
    cf = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.array_equal(calibrate(q, cf), np.argmax(q / cf, 1))


def test_bias_moves_with_order_and_balance():
    assert prompt_bias([1, 1, -1, -1]) < 0 < prompt_bias([-1, -1, 1, 1])
    assert prompt_bias([1, 1, 1, -1]) > prompt_bias([1, 1, -1, -1])
    p = content_free_prob(W, [1, 1, 1, 1])
    assert p > 0.9 and abs(content_free_prob(W, [1, -1, 1, -1], rec=0.0) - 0.5) < 1e-12


def test_calibration_removes_order_spread():
    raw = order_accuracies([1, 1, -1, -1], X, Y, W, fmt=0.8)
    cal = order_accuracies([1, 1, -1, -1], X, Y, W, fmt=0.8, calibrated=True)
    assert len(raw) == 6 == math.comb(4, 2)
    assert max(raw.values()) - min(raw.values()) > 0.1
    assert max(cal.values()) - min(cal.values()) < 1e-12
    assert min(cal.values()) == 1.0


def test_global_entropy_picks_balanced_orders():
    assert abs(global_entropy([0, 1, 0, 1]) - math.log(2)) < 1e-12 and global_entropy([1, 1, 1]) == 0
    probe = RNG.normal(size=(300, 3))
    orders = sorted(set(itertools.permutations([1, 1, -1, -1])))
    ent = [global_entropy((icl_prob(probe, W, o, fmt=0.8) > 0.5).astype(int)) for o in orders]
    acc = order_accuracies([1, 1, -1, -1], X, Y, W, fmt=0.8)
    best = orders[int(np.argmax(ent))]
    assert acc[best] >= np.mean(list(acc.values()))
    sure, unsure = np.array([[0.99, 0.01]] * 5), np.array([[0.5, 0.5]] * 5)
    assert local_entropy(sure) < local_entropy(unsure)


def test_spread_flip_and_optimism_match_simulation():
    z = RNG.normal(size=(20_000, 10))
    assert abs((z.max(1) - z.min(1)).mean() - expected_range(10)) < 0.03
    d = 2 + 5 * RNG.normal(size=200_000) - 5 * RNG.normal(size=200_000)
    assert abs((d < 0).mean() - flip_probability(2, 5)) < 0.005
    best = RNG.binomial(100, 0.7, size=(20_000, 24)).max(1) / 100
    assert abs(best.mean() - 0.7 - selection_optimism(0.7, 100, 24)) < 0.004


def test_kendall_statistics():
    r = ranks([0.9, 0.5, 0.7, 0.1])
    assert list(r) == [1, 3, 2, 4]
    assert kendall_w(np.tile(r, (5, 1))) == 1.0
    m, n = 6, 8
    ws = [kendall_w(np.array([RNG.permutation(n) + 1 for _ in range(m)])) for _ in range(4000)]
    assert abs(np.mean(ws) - 1 / m) < 0.005
    a = [1, 2, 3, 4, 5]
    assert kendall_tau_b(a, a) == 1 and kendall_tau_b(a, a[::-1]) == -1
    assert abs(kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) - 4 / 6) < 1e-12
