# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/process-reward-model/ (ai://NEW-process-reward-model).
import functools
import itertools
import math

import torch
from process_reward_model import (
    PRM,
    TinyLM,
    aggregate,
    implicit_step_rewards,
    make_solutions,
    prm_loss,
    train_prm,
)

V = 7


@functools.cache
def trained():
    torch.manual_seed(0)
    return train_prm(PRM(V), V)


def test_labels_stop_after_first_error():
    g = torch.Generator().manual_seed(0)
    tokens, ok, labels = make_solutions(500, 5, V, 0.7, g)
    assert tokens.shape == (500, 15) and (tokens[:, 2::3] == V).all()
    a, b = tokens[:, 0::3], tokens[:, 1::3]
    right = b == (a + 1) % V
    assert (ok == torch.cumprod(right.long(), 1).bool()).all()
    first = (~ok).long().cumsum(1) == 1
    assert ((labels == 0) == (first & ~ok)).all() and (labels[ok] == 1).all()


def test_prm_learns_step_correctness_in_one_pass():
    model = trained()
    g = torch.Generator().manual_seed(1)
    tokens, _ok, labels = make_solutions(2000, 6, V, 0.85, g)
    with torch.no_grad():
        p = torch.sigmoid(model(tokens))
    mask = labels >= 0
    acc = ((p[mask] > 0.5) == labels[mask].bool()).float().mean()
    assert acc > 0.97
    assert prm_loss(model(tokens), labels) < 0.15


def test_best_of_n_with_prm_prefers_clean_solutions():
    model = trained()
    g = torch.Generator().manual_seed(2)
    clean = []
    for _ in range(100):
        tokens, ok, _ = make_solutions(8, 6, V, 0.8, g)
        with torch.no_grad():
            s = aggregate(torch.sigmoid(model(tokens)), "min")
        clean.append(bool(ok[s.argmax(), -1]))
    # a single sample is fully right with probability 0.8^6 ~ 0.26; best-of-8 under the PRM does far better
    assert sum(clean) / 100 > 0.75


def test_aggregations_are_ordered():
    p = torch.rand(100, 9) * 0.99 + 0.01
    assert (aggregate(p, "prod") <= aggregate(p, "min") + 1e-7).all()
    assert (aggregate(p, "min") <= aggregate(p, "last")).all()


def test_implicit_rewards_sum_to_outcome_and_match_proposition():
    pol, ref, beta = TinyLM(3, seed=1), TinyLM(3, seed=2), 0.5
    ys = torch.tensor(list(itertools.product(range(3), repeat=4)))
    with torch.no_grad():
        lp, lr = pol.token_logprobs(ys), ref.token_logprobs(ys)
    step_ids = torch.tensor([0, 0, 1, 1])  # two steps of two tokens
    r, q = implicit_step_rewards(lp, lr, step_ids, beta)
    outcome = beta * (lp.sum(1) - lr.sum(1))
    assert torch.allclose(r.sum(1), outcome, atol=1e-5)
    # q after step 1 equals beta log E_{pi_ref(rest | first step)} exp(r / beta)
    for i in range(0, len(ys), 5):
        pre = ys[i, :2]
        rows = (ys[:, :2] == pre).all(1)
        ref_rest = (lr[rows, 2:].sum(1)).exp()
        e = (ref_rest * (outcome[rows] / beta).exp()).sum()
        assert math.isclose(float(q[i, 0]), float(beta * e.log()), abs_tol=1e-4)
