# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/language-model-mathematical-reasoning/ (ai://NEW-language-model-mathematical-reasoning).
"""Tests for the multiplication experiment."""

import torch
from language_model_mathematical_reasoning import digit_accuracy, digits, low_digits_rule, seen_fraction, train


def test_low_digits_depend_only_on_low_digits():
    gen = torch.Generator().manual_seed(0)
    a = torch.randint(0, 10**6, (5000,), generator=gen)
    b = torch.randint(0, 10**6, (5000,), generator=gen)
    for j in (1, 2, 3):
        assert torch.equal(low_digits_rule(a, b, j), (a * b) % 10**j)
    assert torch.equal(digits(torch.tensor([4321]), 4), torch.tensor([[1, 2, 3, 4]]))


def test_memorized_table_is_near_perfect():
    model, pairs = train(2)
    assert seen_fraction(pairs, 2) > 0.99
    _, whole = digit_accuracy(model)
    assert whole > 0.9


def test_unseen_pairs_get_the_outer_digits_but_not_the_middle():
    model, pairs = train(3)
    assert seen_fraction(pairs, 3) < 0.25
    per_digit, whole = digit_accuracy(model)
    assert per_digit[0] > 0.95 and per_digit[1] > 0.9
    assert per_digit[4] > 0.75 and per_digit[5] > 0.95
    assert per_digit[2] < 0.2 and per_digit[3] < 0.2
    assert whole < 0.05
