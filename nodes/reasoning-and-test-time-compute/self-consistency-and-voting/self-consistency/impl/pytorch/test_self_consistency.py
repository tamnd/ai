# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/self-consistency/ (ai://NEW-self-consistency).
import math

import torch
from self_consistency import (
    TinyLM,
    adaptive_self_consistency,
    exact_answer_marginal,
    greedy_path,
    sample_paths,
    self_consistency,
    sequence_logprob,
    vote,
)


def parse(tokens):
    """The 'answer' is the last token modulo 3, the way a parser reads the number after 'The answer is'."""
    return int(tokens[-1]) % 3


def test_low_temperature_collapses_to_greedy():
    model = TinyLM(seed=1)
    g = torch.Generator().manual_seed(0)
    tokens, _ = sample_paths(model, 4, 50, temperature=1e-4, generator=g)
    assert (tokens == greedy_path(model, 4)).all()
    tokens, _ = sample_paths(model, 4, 50, temperature=1.0, top_k=1, generator=g)
    assert (tokens == greedy_path(model, 4)).all()


def test_logprobs_add_up():
    model = TinyLM(seed=2)
    g = torch.Generator().manual_seed(1)
    tokens, lps = sample_paths(model, 5, 20, temperature=0.7, generator=g)
    assert torch.allclose(lps.sum(1), sequence_logprob(model, tokens), atol=1e-5)


def test_exact_marginal_sums_to_one():
    model = TinyLM(seed=3)
    for t, k in [(1.0, None), (0.5, None), (0.7, 2)]:
        marg = exact_answer_marginal(model, parse, 3, temperature=t, top_k=k)
        assert math.isclose(sum(marg.values()), 1.0, abs_tol=1e-5)


def test_vote_frequencies_match_the_marginal():
    model = TinyLM(seed=4)
    marg = exact_answer_marginal(model, parse, 3, temperature=0.7, top_k=3)
    g = torch.Generator().manual_seed(2)
    _, totals, _ = self_consistency(model, parse, 3, 20000, temperature=0.7, top_k=3, generator=g)
    for a, p in marg.items():
        assert abs(totals.get(a, 0.0) / 20000 - p) < 0.015
    # seed 4 is a near tie (0.346 against 0.345), so check the mode on a model with a clear margin
    model = TinyLM(seed=8)
    marg = exact_answer_marginal(model, parse, 3, temperature=0.7, top_k=3)
    best, _, _ = self_consistency(model, parse, 3, 400, temperature=0.7, top_k=3, generator=g)
    assert best == max(marg, key=marg.get)


def test_marginal_mode_can_differ_from_greedy_answer():
    # Over a few seeds at least one model has a greedy path whose answer is not the mode of the marginal:
    # the vote then disagrees with greedy chain of thought.
    differ = 0
    for seed in range(30):
        model = TinyLM(seed=seed)
        marg = exact_answer_marginal(model, parse, 3)
        differ += parse(greedy_path(model, 3)) != max(marg, key=marg.get)
    assert differ > 0


def test_weightings():
    assert vote([1, 2, 2])[0] == 2
    assert vote([1, 2, 2], [0.9, 0.3, 0.3])[0] == 1
    model = TinyLM(seed=5)
    g = torch.Generator().manual_seed(3)
    for w in ("none", "normalized", "unnormalized"):
        best, totals, tokens = self_consistency(model, parse, 3, 64, weighting=w, generator=g)
        assert tokens.shape == (64, 3) and best in totals


def test_adaptive_stops_early_when_unanimous():
    model = TinyLM(seed=6)
    g = torch.Generator().manual_seed(4)
    best, n = adaptive_self_consistency(model, parse, 3, temperature=1e-4, generator=g)
    assert n == 4 and best == parse(greedy_path(model, 3))
