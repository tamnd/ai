# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/instruction-tuning/ (ai://NEW-instruction-tuning).
"""Tests of the PyTorch code: the shared fixture, agreement with the ignore-index form, token versus sequence
averaging, autograd gradients that vanish on the prompt and match the analytic form, collation, packed rows that give
the same logits as separate examples, and a training run in which the response-only loss teaches the response but,
unlike the full-sequence loss, leaves the model unable to predict the prompt."""

import math

import pytest
import torch
import torch.nn.functional as F
from instruction_tuning import (
    TinyCausalLM,
    collate,
    encode,
    ignore_index_labels,
    masked_nll,
    segment_attention_mask,
    segment_positions,
)

torch.set_default_dtype(torch.float64)

# The fixture shared with the NumPy tests.
Z = [
    [
        [2.0, 0.5, -1.0, 0.0],
        [0.1, 1.5, 0.3, -0.5],
        [-0.2, 0.4, 2.2, 0.1],
        [1.0, 1.0, 0.0, 0.5],
    ],
    [
        [0.3, -0.7, 1.1, 0.2],
        [0.0, 0.0, 0.0, 0.0],
        [1.2, 0.1, -0.4, 0.9],
        [0.5, 2.0, 0.1, -1.0],
    ],
]
Y = [[0, 1, 2, 3], [2, 0, 0, 1]]
M = [[0, 1, 1, 1], [0, 0, 1, 1]]
FIXTURE_TOKEN = 0.7212637615150113
FIXTURE_SEQUENCE = 0.6989400897373603


def fixture(grad=False):
    return (
        torch.tensor(Z, requires_grad=grad),
        torch.tensor(Y),
        torch.tensor(M, dtype=torch.float64),
    )


def test_fixture():
    z, y, m = fixture()
    assert abs(masked_nll(z, y, m).item() - FIXTURE_TOKEN) < 1e-12
    assert abs(masked_nll(z, y, m, "sequence").item() - FIXTURE_SEQUENCE) < 1e-12


def test_matches_ignore_index_form():
    z, y, m = fixture()
    labels = ignore_index_labels(y, m)
    ref = F.cross_entropy(z.transpose(1, 2), labels, ignore_index=-100)
    assert abs(masked_nll(z, y, m).item() - ref.item()) < 1e-12
    assert (labels[m == 0] == -100).all() and (labels[m == 1] == y[m == 1]).all()


def test_token_versus_sequence():
    torch.manual_seed(0)
    z = torch.randn(2, 90, 5)
    y = torch.randint(0, 5, (2, 90))
    m = torch.zeros(2, 90)
    m[0, :10] = 1
    m[1] = 1
    nll = F.cross_entropy(z.transpose(1, 2), y, reduction="none")
    a, b = nll[0, :10].mean(), nll[1].mean()
    assert torch.isclose(masked_nll(z, y, m), 0.1 * a + 0.9 * b)
    assert torch.isclose(masked_nll(z, y, m, "sequence"), 0.5 * a + 0.5 * b)
    with pytest.raises(ValueError):
        masked_nll(z, y, torch.zeros_like(m))
    with pytest.raises(ValueError):
        masked_nll(z, y, m, "batch")


@pytest.mark.parametrize("reduction", ["token", "sequence"])
def test_gradient_vanishes_on_prompt_and_matches_formula(reduction):
    z, y, m = fixture(grad=True)
    masked_nll(z, y, m, reduction).backward()
    g = z.grad
    assert (g[m == 0] == 0).all()
    p = torch.softmax(z.detach(), dim=-1)
    onehot = F.one_hot(y, 4).to(p.dtype)
    if reduction == "token":
        w = m / m.sum()
    else:
        w = m / m.sum(dim=-1, keepdim=True) / 2
    assert torch.allclose(g, w[..., None] * (p - onehot))


def test_encode_and_collate():
    ex = [encode([5, 6, 7], [8, 9], eos_id=1), encode([4], [3], eos_id=1)]
    assert ex[0] == ([5, 6, 7, 8, 9], [6, 7, 8, 9, 1], [0.0, 0.0, 1.0, 1.0, 1.0])
    inputs, targets, mask = collate(ex, pad_id=0)
    assert inputs.shape == (2, 5)
    assert inputs[1].tolist() == [4, 3, 0, 0, 0]
    assert targets[1].tolist() == [3, 1, 0, 0, 0]
    assert mask[1].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        encode([], [1], eos_id=1)


def test_segment_mask_and_positions():
    seg = torch.tensor([[1, 1, 2, 2, 2, 0]])
    a = segment_attention_mask(seg)[0]
    assert a[1, 0] and not a[0, 1]
    assert not a[2, 1] and a[4, 2]
    assert a[5].tolist() == [False] * 5 + [True]
    assert segment_positions(seg)[0].tolist() == [0, 1, 0, 1, 2, 0]


def test_packed_rows_equal_separate_examples():
    torch.manual_seed(1)
    model = TinyCausalLM(vocab=10, dim=16, heads=2, max_len=16).eval()
    exs = [encode([2, 3], [4], 1), encode([5, 6, 2], [7, 8], 1)]
    seq = [torch.tensor(x) for x, _, _ in exs]
    row = torch.cat(seq + [torch.zeros(2, dtype=torch.long)])[None]
    seg = torch.tensor([[1] * len(seq[0]) + [2] * len(seq[1]) + [0, 0]])
    packed = model(row, segment_attention_mask(seg), segment_positions(seg))
    alone = [model(s[None])[0] for s in seq]
    n0 = len(seq[0])
    assert torch.allclose(packed[0, :n0], alone[0], atol=1e-10)
    assert torch.allclose(packed[0, n0 : n0 + len(seq[1])], alone[1], atol=1e-10)
    # Without the segment mask the second example sees the first, and its logits change.
    leaky = model(row)
    assert not torch.allclose(leaky[0, n0 : n0 + len(seq[1])], alone[1], atol=1e-4)


def _toy_data():
    """Prompts [task, a, (a + 3) % 8]; the response copies a for task 8 and (a + 3) % 8 for task 9; eos is 10."""
    exs = []
    for task in (8, 9):
        for a in range(8):
            b = (a + 3) % 8
            exs.append(encode([task, a, b], [a if task == 8 else b], eos_id=10))
    return collate(exs, pad_id=11)


def _train(full_sequence, steps=400):
    torch.manual_seed(2)
    model = TinyCausalLM(vocab=12, dim=32, heads=4, max_len=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    inputs, targets, mask = _toy_data()
    train_mask = torch.ones_like(mask) if full_sequence else mask
    for _ in range(steps):
        opt.zero_grad()
        masked_nll(model(inputs), targets, train_mask).backward()
        opt.step()
    with torch.no_grad():
        logits = model(inputs)
        response = masked_nll(logits, targets, mask).item()
        # Position 1 holds a and predicts (a + 3) % 8, a prompt token that a full-sequence model can learn.
        prompt_mask = torch.zeros_like(mask)
        prompt_mask[:, 1] = 1
        prompt = masked_nll(logits, targets, prompt_mask).item()
    return response, prompt


def test_response_only_training_does_not_learn_the_prompt():
    resp_only, prompt_only = _train(full_sequence=False)
    resp_full, prompt_full = _train(full_sequence=True)
    assert resp_only < 0.05 and resp_full < 0.05
    assert prompt_full < 0.05
    assert prompt_only > 0.5 * math.log(12)
