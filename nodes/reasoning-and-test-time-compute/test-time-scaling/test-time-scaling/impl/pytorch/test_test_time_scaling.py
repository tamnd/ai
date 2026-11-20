# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/test-time-scaling/ (ai://NEW-test-time-scaling).
import itertools
import math

import pytest
import torch

from test_time_scaling import (
    Solver,
    best_of_n,
    majority_vote,
    pass_at_k,
    per_sample_success,
    sample_answers,
    scaling_curves,
    teacher_task,
    train,
)

M = 10


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    tr, te = teacher_task(200, 500, 20, M, g)
    model = train(Solver(20, M), tr, steps=300, generator=g)
    return model, te


def test_pass_at_k_matches_the_binomial_formula():
    for n, c, k in ((10, 3, 4), (100, 1, 50), (100, 0, 7), (5, 5, 2), (8, 6, 3)):
        want = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert abs(pass_at_k(n, torch.tensor([c]), k).item() - want) < 1e-9


def test_vote_and_best_of_n():
    answers = torch.tensor([[3, 1, 1, 3, 3], [0, 2, 2, 4, 2]])
    assert majority_vote(answers, 5).tolist() == [3, 2]
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.0, 0.0], [0.5, 0.1, 0.1, 0.7, 0.1]])
    assert best_of_n(answers, scores).tolist() == [1, 4]


def test_coverage_grows_and_voting_saturates(setup):
    model, (pairs, labels) = setup
    g = torch.Generator().manual_seed(1)
    answers = sample_answers(model, pairs, 256, 1.0, g)
    ks = [1, 4, 16, 64, 256]
    c = scaling_curves(answers, labels, ks, M, verifier_noise=0.5, generator=g)
    cov, vote, bon = c["coverage"], c["vote"], c["best_of_n"]
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(cov))
    assert cov[-1] > cov[0] + 0.2
    # Voting converges to the fraction of problems whose correct answer is the mode. Temperature keeps the argmax,
    # so for this one-step model the mode is the greedy answer and voting tends to greedy accuracy.
    p = per_sample_success(answers, labels)
    greedy = (model(pairs).argmax(1) == labels).double().mean().item()
    assert abs(vote[-1] - vote[-2]) < 0.05
    assert abs(vote[-1] - greedy) < 0.03
    assert cov[-1] - vote[-1] > 0.2
    # A good verifier gets much closer to coverage than voting does.
    assert bon[-1] > vote[-1] + 0.05
    assert (p > 0).double().mean().item() >= cov[-1] - 1e-9


def test_temperature_trades_pass_at_1_for_coverage(setup):
    model, (pairs, labels) = setup
    g = torch.Generator().manual_seed(2)
    greedy = sample_answers(model, pairs, 64, 0.0, g)
    hot = sample_answers(model, pairs, 64, 1.5, g)
    acc_greedy = (greedy[:, 0] == labels).double().mean().item()
    cg = scaling_curves(greedy, labels, [1, 64], M, 0.5, g)["coverage"]
    ch = scaling_curves(hot, labels, [1, 64], M, 0.5, g)["coverage"]
    assert abs(cg[0] - acc_greedy) < 1e-9 and abs(cg[1] - acc_greedy) < 1e-9  # greedy never gains from repeats
    assert ch[0] < cg[0]  # sampling hot lowers single-sample accuracy
    assert ch[1] > cg[1]  # and raises coverage at 64 samples
