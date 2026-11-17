# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-prompting/ (ai://WFY2QB).
"""Tests of the reference: answer extraction, the pure-Python vote against the NumPy vote, the tie rule, the weighted
votes of Wang et al., the exact vote accuracy against enumeration and simulation, the step identity and the limits of
derivations/majority-vote.md, the benchmark where voting lowers accuracy, and the worked example of the page."""

import itertools
from fractions import Fraction as F
from math import comb, exp

import numpy as np
import pytest

from self_consistency import (
    binary_vote_accuracy,
    extract_answer,
    majority_vote,
    majority_vote_np,
    path_weight,
    simulate_vote_accuracy,
    vote_accuracy,
    vote_counts,
    weighted_vote,
)

# The worked example of the page: five sampled chains of thought for the cafeteria question of Wei et al.
SAMPLES = [
    "They had 23 apples. They used 20, so 23 - 20 = 3. They bought 6 more, so 3 + 6 = 9. The answer is 9.",
    "23 - 20 = 3 apples are left, and 3 + 6 = 9. The answer is 9.",
    "The cafeteria had 23 apples and bought 6 more, so 23 + 6 = 29, then 29 - 20 = 9. The answer is 9.0.",
    "They used 20 to make lunch and bought 6 more. 20 + 6 = 26 and 26 + 1 = 27. The answer is 27.",
    "23 - 20 = 3. The answer is 3.",
]


def test_worked_example():
    answers = [extract_answer(s) for s in SAMPLES]
    assert answers == ["9", "9", "9", "27", "3"]
    assert vote_counts(answers) == {"9": 3, "27": 1, "3": 1}
    assert majority_vote(answers) == majority_vote_np(answers) == "9"
    # Greedy decoding keeps only one path; if that path was the fourth, the answer would be 27.
    assert majority_vote(answers[3:4]) == "27"


def test_extract_answer():
    assert extract_answer("The answer is 1,000.") == "1000"
    assert extract_answer("so 5 + 6 = 11. The answer is 11.") == "11"
    assert extract_answer("The answer is 2.50 dollars") == "2.5"
    assert extract_answer("The answer is -3.") == "-3"
    assert extract_answer("The answer is -0.0") == "0"
    assert extract_answer("no number here") is None
    # Without the trigger the first number of the text is taken, as in the answer cleansing of Kojima et al.
    assert extract_answer("probably 375 and 376") == "375"
    # With the trigger, the number after its last occurrence is taken.
    assert extract_answer("The answer is 4? No. The answer is 5.") == "5"
    with pytest.raises(TypeError):
        extract_answer(5)


def test_formatting_variants_vote_together():
    raw = [
        "The answer is 1,000.",
        "The answer is 1000",
        "The answer is 1000.0",
        "The answer is 999.",
    ]
    assert majority_vote(raw) == raw[0]  # the raw strings are all different, so the first one wins a 1-1-1-1 tie
    assert majority_vote([extract_answer(s) for s in raw]) == "1000"


def test_ties_go_to_first_sampled():
    assert majority_vote(["b", "a", "a", "b"]) == "b"
    assert majority_vote_np(["b", "a", "a", "b"]) == "b"
    assert majority_vote(["c", "a", "b"]) == majority_vote_np(["c", "a", "b"]) == "c"


def test_unparsed_answers_are_skipped():
    assert majority_vote([None, "4", None, None]) == "4"
    assert majority_vote([None, None]) is None
    assert majority_vote_np([None, None]) is None
    with pytest.raises(ValueError):
        majority_vote([])
    with pytest.raises(ValueError):
        majority_vote_np([])


def test_pure_python_matches_numpy():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(1, 41))
        answers = [None if x == 6 else int(x) for x in rng.integers(0, 7, size=n)]
        assert majority_vote(answers) == majority_vote_np(answers)
        winner = majority_vote(answers)
        if winner is not None:
            counts = vote_counts(answers)
            assert counts[winner] == max(counts.values())


def test_vote_is_a_function_of_counts_up_to_ties():
    rng = np.random.default_rng(2)
    for _ in range(300):
        answers = [int(x) for x in rng.integers(0, 5, size=int(rng.integers(1, 31)))]
        shuffled = [answers[i] for i in rng.permutation(len(answers))]
        counts = vote_counts(answers)
        assert counts[majority_vote(shuffled)] == max(counts.values())


def test_path_weight():
    lp = [-0.1, -0.2, -0.3]
    assert path_weight(lp) == pytest.approx(np.exp(-0.2))
    assert path_weight(lp, normalize=False) == pytest.approx(np.exp(-0.6))
    with pytest.raises(ValueError):
        path_weight([])
    with pytest.raises(ValueError):
        path_weight([0.1])


def test_weighted_sum_with_equal_weights_is_the_vote():
    answers = ["9", "27", "9", "3", "9", "27"]
    assert weighted_vote(answers, [0.5] * 6) == majority_vote(answers) == "9"


def test_weighted_average_favours_a_rare_confident_answer():
    # One confident path to "27" outweighs three ordinary paths to "9" under the mean, not under the sum.
    answers = ["9", "9", "9", "27"]
    weights = [0.30, 0.31, 0.29, 0.40]
    assert weighted_vote(answers, weights) == "9"
    assert weighted_vote(answers, weights, average=True) == "27"


def test_unnormalized_weights_favour_short_paths():
    # Three long paths to "9" and one short path to "3", each token with probability 0.8.
    lps = [[np.log(0.8)] * 40] * 3 + [[np.log(0.8)] * 5]
    answers = ["9", "9", "9", "3"]
    norm = [path_weight(x) for x in lps]
    raw = [path_weight(x, normalize=False) for x in lps]
    assert weighted_vote(answers, norm) == "9"
    assert weighted_vote(answers, raw) == "3"


def test_weighted_vote_rejects_bad_weights():
    with pytest.raises(ValueError):
        weighted_vote(["a", "b"], [1.0])
    with pytest.raises(ValueError):
        weighted_vote(["a", "b"], [1.0, -1.0])
    assert weighted_vote([None, None], [1.0, 1.0]) is None


def brute_vote_accuracy(q, m, correct=0):
    """Sum over all len(q)^m ordered outcomes with the first-seen tie rule of majority_vote."""
    total = F(0)
    for seq in itertools.product(range(len(q)), repeat=m):
        pr = F(1)
        for a in seq:
            pr *= q[a]
        if majority_vote(list(seq)) == correct:
            total += pr
    return total


@pytest.mark.parametrize(
    "q",
    [
        (F(1, 2), F(1, 2)),
        (F(3, 5), F(2, 5)),
        (F(2, 5), F(7, 20), F(1, 4)),
        (F(1, 3), F(1, 2), F(1, 6)),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_exact_accuracy_matches_enumeration_with_first_seen_ties(q, m):
    for c in range(len(q)):
        assert vote_accuracy(q, m, correct=c) == brute_vote_accuracy(q, m, correct=c)
    assert sum(vote_accuracy(q, m, correct=c) for c in range(len(q))) == 1


def test_binary_formula_and_numbers():
    for p in [F(1, 10), F(2, 5), F(3, 5), F(9, 10)]:
        for m in range(1, 30, 2):
            formula = sum(comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m // 2 + 1, m + 1))
            assert binary_vote_accuracy(p, m) == formula
    assert binary_vote_accuracy(F(3, 5), 3) == F(648, 1000)
    assert binary_vote_accuracy(F(3, 5), 5) == F(68256, 100000)
    assert binary_vote_accuracy(F(2, 5), 3) == F(352, 1000)
    assert round(float(binary_vote_accuracy(F(3, 5), 21)), 4) == 0.8256


def test_step_identity_and_monotonicity():
    for i in range(1, 10):
        p = F(i, 10)
        for m in range(1, 26, 2):
            h = (m + 1) // 2
            step = binary_vote_accuracy(p, m + 2) - binary_vote_accuracy(p, m)
            assert step == comb(m, h) * p**h * (1 - p) ** h * (2 * p - 1)


def test_even_count_adds_nothing():
    for p in [F(1, 5), F(3, 5), F(7, 10)]:
        for k in range(1, 10):
            assert binary_vote_accuracy(p, 2 * k) == binary_vote_accuracy(p, 2 * k - 1)


def test_hoeffding_bound():
    for p in [0.55, 0.7, 0.9]:
        for m in [1, 11, 41, 101]:
            assert 1 - binary_vote_accuracy(p, m) <= exp(-2 * m * (p - 0.5) ** 2) + 1e-12


def test_plurality_wins_below_one_half():
    q = (F(2, 5), F(7, 20), F(1, 4))
    accs = [vote_accuracy(q, m) for m in (1, 9, 25, 49)]
    assert accs[0] == F(2, 5)
    assert all(b > a for a, b in itertools.pairwise(accs))
    assert accs[-1] > F(3, 5)  # 0.631 with 49 samples, from 0.4 with one


def test_vote_can_lower_benchmark_accuracy():
    # Question 1: correct with probability 0.9. Question 2: correct 0.3, a wrong answer 0.5, another 0.2.
    q1, q2 = (F(9, 10), F(1, 10)), (F(3, 10), F(1, 2), F(1, 5))
    single = (q1[0] + q2[0]) / 2
    voted = (vote_accuracy(q1, 41) + vote_accuracy(q2, 41)) / 2
    assert single == F(3, 5)
    assert voted < single
    assert abs(float(voted) - 0.5) < 0.05


def test_simulation_matches_exact():
    rng = np.random.default_rng(20261115)
    q = [0.4, 0.35, 0.25]
    for m in (1, 4, 15):
        exact = float(vote_accuracy([F(2, 5), F(7, 20), F(1, 4)], m))
        sim = simulate_vote_accuracy(q, m, 20000, rng)
        assert abs(sim - exact) < 4 * np.sqrt(exact * (1 - exact) / 20000)


def test_vote_accuracy_validates():
    with pytest.raises(ValueError):
        vote_accuracy([1.0], 3)
    with pytest.raises(ValueError):
        vote_accuracy([0.5, 0.6], 3)
    with pytest.raises(ValueError):
        vote_accuracy([0.5, 0.5], 0)
    with pytest.raises(ValueError):
        vote_accuracy([0.5, 0.5], 3, correct=2)
    assert vote_accuracy([0.5, 0.5], 3) == pytest.approx(0.5)
