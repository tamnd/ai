# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/nucleus-sampling/ (ai://NEW-nucleus-sampling).
"""Tests of the PyTorch code: the shared fixture, the worked example of the node, temperature before truncation,
the flat and peaked cases, min-p, the crossing token, ties, agreement with a direct reference on random batches, and
sampling frequencies with a seeded torch.Generator."""

import math

import pytest
import torch
from nucleus_sampling import min_p_filter, sample_next, top_k_filter, top_p_filter

torch.set_default_dtype(torch.float64)

P = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])

# The shared fixture of the NumPy tests.
FIXTURE_LOGITS = [
    [2.0, 1.0, 0.5, 0.0, -1.0, -3.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 0.0, -1.0, -1.0, -2.0, -4.0],
]
FIXTURE_TOP_P = 0.9
FIXTURE_KEPT = [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0]]


def kept(filtered):
    return torch.isfinite(filtered).nonzero().flatten().tolist()


def test_fixture():
    out = top_p_filter(torch.tensor(FIXTURE_LOGITS), FIXTURE_TOP_P)
    for row, want in zip(out, FIXTURE_KEPT):
        assert kept(row) == want


def test_worked_example():
    out = torch.softmax(top_p_filter(P.log(), 0.8), dim=-1)
    assert out.tolist() == pytest.approx([10 / 17, 4 / 17, 3 / 17, 0, 0], abs=1e-12)
    assert 0.5 * (P - out).abs().sum().item() == pytest.approx(0.15)
    top2 = torch.softmax(top_k_filter(P.log(), 2), dim=-1)
    assert top2.tolist() == pytest.approx([5 / 7, 2 / 7, 0, 0, 0], abs=1e-12)


def test_temperature_shrinks_nucleus():
    cold = P.log() / 0.5
    assert torch.softmax(cold, -1).tolist() == pytest.approx(
        [0.769, 0.123, 0.069, 0.031, 0.008], abs=6e-4
    )
    assert len(kept(top_p_filter(P.log(), 0.8))) == 3
    assert len(kept(top_p_filter(cold, 0.8))) == 2
    # tempering after the nucleus is chosen keeps three tokens and only reweights them
    after = top_p_filter(P.log(), 0.8) / 0.5
    assert len(kept(after)) == 3


def test_flat_and_peaked():
    assert len(kept(top_p_filter(torch.zeros(50), 0.95))) == 48
    assert kept(top_p_filter(torch.tensor([0.96, 0.02, 0.01, 0.01]).log(), 0.95)) == [0]


def test_min_p_worked_example():
    assert kept(min_p_filter(P.log(), 0.25)) == [0, 1, 2]


def test_includes_crossing_token():
    assert kept(top_p_filter(torch.tensor([0.96, 0.02, 0.01, 0.01]).log(), 0.9)) == [0]
    assert len(kept(top_p_filter(P.log(), 0.8))) == 3


def test_ties_are_deterministic():
    logits = torch.tensor([0.1, 0.3, 0.3, 0.3]).log()
    assert kept(top_p_filter(logits, 0.5)) == [1, 2]
    assert kept(top_k_filter(logits, 1)) == [1]


def test_batch_matches_reference():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 30, generator=g) * 2.0
    for p in (0.1, 0.5, 0.9, 0.99, 1.0):
        out = top_p_filter(logits, p)
        for row, filtered in zip(logits, out):
            probs = torch.softmax(row, -1)
            order = sorted(range(len(probs)), key=lambda i: (-probs[i].item(), i))
            total, want = 0.0, []
            for i in order:
                want.append(i)
                total += probs[i].item()
                if total >= p:
                    break
            assert kept(filtered) == sorted(want)
            assert math.isclose(torch.softmax(filtered, -1).sum().item(), 1.0)


def test_extra_axes_and_validation():
    logits = torch.randn(2, 3, 7, generator=torch.Generator().manual_seed(1))
    assert top_p_filter(logits, 0.7).shape == logits.shape
    with pytest.raises(ValueError):
        top_p_filter(logits, 0.0)
    with pytest.raises(ValueError):
        sample_next(logits, torch.Generator(), temperature=0.0)


def test_sampling_frequencies():
    g = torch.Generator().manual_seed(2)
    logits = P.log().expand(20000, 5)
    tokens, _ = sample_next(logits, g, top_p=0.8)
    counts = torch.bincount(tokens, minlength=5).double() / tokens.numel()
    assert counts[3].item() == 0 and counts[4].item() == 0
    assert counts.tolist() == pytest.approx([10 / 17, 4 / 17, 3 / 17, 0, 0], abs=0.015)
