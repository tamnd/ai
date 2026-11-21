# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/prompt-sensitivity/ (ai://NEW-prompt-sensitivity).
"""Tests that order sensitivity needs positional information and that both models learn in context."""

import torch
from prompt_sensitivity import (
    InContextClassifier,
    accuracy,
    label_balance_shift,
    make_batch,
    order_flips,
    tokens,
    train,
)


def test_without_positions_order_cannot_matter():
    torch.manual_seed(0)
    model = InContextClassifier(positional=False).eval()
    x, y = make_batch(64, gen=torch.Generator().manual_seed(0))
    idx = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4, 8])
    with torch.no_grad():
        a, b = model(tokens(x, y)), model(tokens(x[:, idx], y[:, idx]))
    assert torch.allclose(a, b, atol=1e-5)


def test_positions_make_trained_model_order_sensitive():
    pos, nopos = train(positional=True), train(positional=False)
    assert accuracy(pos) > 0.8 and accuracy(nopos) > 0.8
    assert order_flips(nopos) == 0.0
    assert order_flips(pos) > 0.01
    assert label_balance_shift(pos) > 1.0 and label_balance_shift(nopos) > 1.0
