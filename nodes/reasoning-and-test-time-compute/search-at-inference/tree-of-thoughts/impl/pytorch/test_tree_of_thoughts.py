# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/tree-of-thoughts/ (ai://NEW-tree-of-thoughts).
import pytest
import torch

from tree_of_thoughts import (
    Evaluator,
    beam_search,
    features,
    learned_score,
    moves,
    problem,
    random_score,
    random_states,
    solvable,
    train,
)


@pytest.fixture(scope="module")
def trained():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    x, y = random_states(20000, g)
    model = train(Evaluator(256), x, y, steps=3000, generator=g)
    probs = []
    while len(probs) < 50:
        p = problem(torch.randint(1, 14, (4,), generator=g))
        if solvable(p):
            probs.append(p)
    return model, probs, g


def test_game_and_features():
    p = problem([4, 9, 10, 13])
    assert solvable(p) and len(moves(p)) == 36
    assert len(features(p)) == 28 and len(features(p[:2])) == 28


def test_evaluator_ranks_solvable_states_higher(trained):
    model, _, _ = trained
    g = torch.Generator().manual_seed(1)
    x, y = random_states(3000, g)
    with torch.no_grad():
        score = model(x)
    for n in (2, 3):  # states with two or three numbers left; few of them can still make 24
        sel = x[:, -4 + n - 1] == 1
        pos, neg = score[sel][y[sel] == 1], score[sel][y[sel] == 0]
        auc = (pos[:, None] > neg[None, :]).float().mean().item()  # chance that a solvable state ranks higher
        assert auc > 0.65  # 0.86 and 0.72 with this seed; 0.5 is a coin flip


def test_learned_evaluator_guides_the_search(trained):
    model, probs, g = trained
    learned = {b: sum(beam_search(p, learned_score(model), b) for p in probs) / len(probs) for b in (1, 5)}
    rand = sum(beam_search(p, random_score(g), 5) for p in probs) / len(probs)
    assert learned[5] > rand + 0.4
    assert learned[5] >= learned[1]
