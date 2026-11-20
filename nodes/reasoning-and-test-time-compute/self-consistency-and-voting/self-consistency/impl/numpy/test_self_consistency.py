# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/self-consistency/ (ai://NEW-self-consistency).
import math

import numpy as np
import pytest
from self_consistency import (
    adaptive_consistency,
    beta_confidence,
    beta_should_stop,
    binary_vote_accuracy,
    consistency,
    coverage,
    early_stopping_sc,
    hoeffding_bound,
    majority_vote,
    mixture_accuracy,
    pass_at_k,
    plurality_accuracy,
    sample_answers,
    sequence_weight,
    verifier_vote,
    weighted_vote,
    window_bound,
    window_unanimity,
)


def test_majority_vote_counts_and_ties():
    w, c = majority_vote(["18", "26", "18"])
    assert w == "18" and c == {"18": 2, "26": 1}
    assert majority_vote(["a", "b"])[0] == "a"
    with pytest.raises(ValueError):
        majority_vote([])


def test_weighted_vote_reduces_to_majority_and_can_flip_it():
    ans = ["x", "y", "y", "x", "y"]
    assert weighted_vote(ans, [1] * 5)[0] == majority_vote(ans)[0] == "y"
    assert weighted_vote(ans, [5, 1, 1, 5, 1])[0] == "x"


def test_sequence_weight_length_bias():
    short = [math.log(0.6)] * 5
    long_ = [math.log(0.9)] * 40
    assert sequence_weight(short, normalize=False) > sequence_weight(long_, normalize=False)
    assert sequence_weight(short) < sequence_weight(long_)
    assert math.isclose(sequence_weight(long_), 0.9)


def test_consistency_and_verifier_vote():
    assert consistency(["a", "a", "b", "a"]) == 0.75
    ans = ["a", "b", "b", "b", "a", "a", "a"]
    scores = [0.9, 0.95, 0.92, 0.91, 0.1, 0.2, 0.05]
    assert majority_vote(ans)[0] == "a"
    assert verifier_vote(ans, scores, top=4) == "b"
    assert verifier_vote(ans, scores, top=7) == "a"


def test_binary_vote_accuracy_values_and_symmetry():
    assert [round(binary_vote_accuracy(0.6, m), 3) for m in (1, 5, 11, 41)] == [0.6, 0.683, 0.753, 0.903]
    for m in (1, 3, 21):
        assert math.isclose(binary_vote_accuracy(0.4, m), 1 - binary_vote_accuracy(0.6, m))
    with pytest.raises(ValueError):
        binary_vote_accuracy(0.6, 4)


def test_plurality_matches_binary_and_simulation():
    for m in (1, 5, 9):
        assert math.isclose(plurality_accuracy([0.6, 0.4], m), binary_vote_accuracy(0.6, m))
    rng = np.random.default_rng(0)
    p = [0.45, 0.35, 0.2]
    s = rng.multinomial(12, p, size=200000)
    sim = (s[:, 0] > s[:, 1:].max(1)).mean()
    assert abs(plurality_accuracy(p, 12) - sim) < 0.005


def test_hoeffding_bound_holds():
    for m in (5, 20, 60):
        exact_wrong = 1 - plurality_accuracy([0.5, 0.3, 0.2], m)
        assert exact_wrong <= hoeffding_bound(0.2, m, 3)
    assert hoeffding_bound(0.3, 100, 2) < 0.012


def test_mixture_rises_then_falls():
    acc = [mixture_accuracy([0.6, 0.4], [0.6, 0.45], m) for m in range(1, 402, 2)]
    peak = int(np.argmax(acc))
    assert 2 * peak + 1 == 71 and acc[0] < acc[peak] and acc[-1] < acc[peak]


def test_beta_rule():
    assert math.isclose(beta_confidence(9, 1), 1 - 12 / 2048)
    assert beta_should_stop({"a": 4}) and not beta_should_stop({"a": 3})
    assert not beta_should_stop({"a": 6, "b": 2})


def test_windows():
    q, wrong = window_unanimity([0.9, 0.08, 0.02], 5)
    assert round(q, 3) == 0.590 and wrong < window_bound(5)
    rng = np.random.default_rng(1)
    for _ in range(500):
        p = rng.dirichlet(np.ones(4))
        if np.sort(p)[-1] - np.sort(p)[-2] > 1e-9:
            assert window_unanimity(p, 8)[1] <= window_bound(8)


def test_pass_at_k_is_unbiased():
    p, n, k = 0.2, 10, 3
    rng = np.random.default_rng(2)
    est = np.mean([pass_at_k(n, int((rng.random(n) < p).sum()), k) for _ in range(40000)])
    assert abs(est - coverage(p, k)) < 0.006
    assert pass_at_k(10, 8, 3) == 1.0 and pass_at_k(10, 0, 3) == 0.0


def test_vote_converges_on_mode_while_coverage_goes_to_one():
    rng = np.random.default_rng(3)
    probs = [0.2, 0.3, 0.25, 0.25]  # answer 0 is correct, 1 is the mode
    ans = sample_answers(probs, 5000, rng)
    assert majority_vote(ans)[0] == 1
    assert coverage(0.2, 100) > 1 - 1e-9


def test_stopping_saves_samples_on_easy_questions():
    rng = np.random.default_rng(4)
    easy = [0.9, 0.08, 0.02]
    runs = [adaptive_consistency(easy, rng) for _ in range(300)]
    assert np.mean([n for _, n in runs]) < 10 and np.mean([a == 0 for a, _ in runs]) > 0.97
    runs = [early_stopping_sc(easy, rng) for _ in range(300)]
    assert np.mean([n for _, n in runs]) < 12 and np.mean([a == 0 for a, _ in runs]) > 0.97
    hard = [0.4, 0.35, 0.25]
    assert np.mean([adaptive_consistency(hard, rng)[1] for _ in range(100)]) > 30
