# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/chain-of-thought-faithfulness/ (ai://NEW-chain-of-thought-faithfulness).
import math

import pytest
import torch
from chain_of_thought_faithfulness import (
    Answerer,
    content_margin,
    hint_test,
    make_data,
    normalized_faithfulness,
    train,
)

DIM, N = 8, 4


def trained(hint_accuracy, seed):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(DIM, N, generator=g)
    torch.manual_seed(seed)
    x, hint, y = make_data(6000, DIM, N, 0.5, hint_accuracy, teacher, g)
    model = train(Answerer(DIM, N), x, hint, y)
    xt, _, _ = make_data(20000, DIM, N, 0.0, 0.0, teacher, g)
    return model, xt, g


@pytest.fixture(scope="module")
def informative():
    return trained(0.9, 0)


@pytest.fixture(scope="module")
def uninformative():
    return trained(1 / N, 0)


def test_model_follows_informative_hints(informative):
    model, xt, g = informative
    p, _, alpha, _ = hint_test(model, xt, N, g)
    assert p > 0.4
    assert alpha > 0.8


def test_model_ignores_uninformative_hints(informative, uninformative):
    p_inf = hint_test(*informative[:2], N, informative[2])[0]
    p_uni, _, alpha_uni, _ = hint_test(*uninformative[:2], N, uninformative[2])
    assert p_inf > 3 * p_uni
    # Without real hint use, moves to the hint are about as common as moves to any other option.
    assert alpha_uni < 0.5


def test_hints_win_where_content_is_unsure(informative):
    model, xt, g = informative
    _, _, _, moved = hint_test(model, xt, N, g)
    margin = content_margin(model, xt, N)
    low = margin < margin.median()
    assert moved[low].float().mean() > moved[~low].float().mean() + 0.05


def test_normalization_matches_the_formula():
    assert math.isclose(normalized_faithfulness(0.20, 0.30, 0.04, 4), 0.2143, abs_tol=5e-5)
    assert normalized_faithfulness(0.1, 0.05, 0.10, 4) is None
    assert normalized_faithfulness(0.95, 0.30, 0.04, 4) == 1.0
