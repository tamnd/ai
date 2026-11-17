# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/speculative-decoding/ (ai://JKWQF4).
"""Tests of the PyTorch reference: input checks; the acceptance rule, the residual and the acceptance rate on the worked
example of the page; that the exact law of one verify step is p for random, equal and disjoint p and q; that sampled
outputs of one step and of whole speculative steps follow the target distribution; greedy decoding with a one-hot
target; the lengths of the accept loop and its bonus token; and the expected token count, its capped geometric law,
the walltime and the arithmetic factors against the tables of Leviathan et al."""

import numpy as np
import pytest
import torch

from speculative_decoding import (
    acceptance_probability,
    acceptance_rate,
    expected_tokens,
    operations_factor,
    output_distribution,
    residual,
    sample,
    speculative_step,
    token_count_pmf,
    tv_distance,
    verify,
    verify_token,
    walltime_improvement,
)

def gen(seed):
    return torch.Generator().manual_seed(seed)


P = np.array([1 / 2, 1 / 3, 1 / 6])
Q = np.array([1 / 4, 1 / 2, 1 / 4])


def random_pair(rng, v):
    d = torch.distributions.Dirichlet(torch.ones(v, dtype=torch.float64))
    torch.manual_seed(int(torch.randint(0, 2**31, (), generator=rng)))
    return d.sample().numpy(), d.sample().numpy()


def test_input_checks():
    rng = gen(0)
    with pytest.raises(ValueError):
        sample(np.ones((2, 2)) / 4, rng)
    with pytest.raises(ValueError):
        sample(np.array([]), rng)
    with pytest.raises(ValueError):
        sample(np.array([1.2, -0.2]), rng)
    with pytest.raises(ValueError):
        sample(np.array([0.5, 0.5 + 1e-6]), rng)
    with pytest.raises(ValueError):
        acceptance_rate(P, np.array([0.5, 0.5]))
    assert sample(np.array([0.0, 1.0, 0.0]), rng) == 1
    assert sample(np.array([0.5, 0.5 + 1e-12]), rng) in (0, 1)


def test_sample_frequencies():
    rng = gen(1)
    draws = np.array([sample(P, rng) for _ in range(5000)])
    assert np.allclose(np.bincount(draws, minlength=3) / draws.size, P, atol=0.03)


def test_acceptance_probability_worked_example():
    assert [acceptance_probability(P, Q, x) for x in range(3)] == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert acceptance_probability(P, P, 2) == 1.0
    with pytest.raises(ValueError):
        acceptance_probability(P, np.array([0.5, 0.5, 0.0]), 2)


def test_residual():
    assert residual(P, Q) == pytest.approx([1.0, 0.0, 0.0])
    rng = gen(2)
    for _ in range(50):
        p, q = random_pair(rng, 5)
        r = residual(p, q).numpy()
        assert r.sum() == pytest.approx(1.0)
        assert np.all(r >= 0)
        assert np.all(r[q >= p] == 0)
        assert np.allclose(r * np.maximum(p - q, 0).sum(), np.maximum(p - q, 0))
    with pytest.raises(ValueError):
        residual(P, P)


def test_acceptance_rate_is_one_minus_tv():
    assert acceptance_rate(P, Q) == pytest.approx(0.75)
    assert tv_distance(P, Q) == pytest.approx(0.25)
    assert acceptance_rate(P, P) == pytest.approx(1.0)
    assert tv_distance(P, P) == 0.0
    a, b = np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.3, 0.7])
    assert acceptance_rate(a, b) == 0.0
    assert tv_distance(a, b) == pytest.approx(1.0)
    rng = gen(3)
    for _ in range(100):
        p, q = random_pair(rng, 6)
        assert acceptance_rate(p, q) == pytest.approx(1.0 - tv_distance(p, q))
        assert acceptance_rate(p, q) == pytest.approx(acceptance_rate(q, p))


def test_output_distribution_is_exactly_p():
    assert output_distribution(P, Q) == pytest.approx(P)
    assert output_distribution(P, P) == pytest.approx(P)
    a, b = np.array([0.5, 0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.3, 0.7])
    assert output_distribution(a, b) == pytest.approx(a)
    rng = gen(4)
    for v in (2, 3, 7):
        for _ in range(100):
            p, q = random_pair(rng, v)
            q[int(torch.randint(0, v, (), generator=rng))] = 0.0
            q /= q.sum()
            assert np.allclose(output_distribution(p, q), p, atol=1e-12)


def test_verify_token_distribution_matches_p():
    rng = gen(5)
    n = 8000
    tokens = np.empty(n, dtype=int)
    accepted = 0
    for i in range(n):
        x = sample(Q, rng)
        ok, tokens[i] = verify_token(P, Q, x, rng)
        accepted += ok
    assert np.allclose(np.bincount(tokens, minlength=3) / n, P, atol=0.025)
    assert accepted / n == pytest.approx(acceptance_rate(P, Q), abs=0.025)


def test_verify_token_rejects_into_the_residual():
    rng = gen(6)
    for _ in range(300):
        ok, t = verify_token(P, Q, 1, rng)
        assert ok and t == 1 or not ok and t == 0
    for _ in range(50):
        assert verify_token(P, Q, 0, rng) == (True, 0)


def test_greedy_one_hot_target():
    p = np.array([0.0, 0.0, 1.0, 0.0])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    rng = gen(7)
    for x in range(4):
        for _ in range(20):
            ok, t = verify_token(p, q, x, rng)
            assert t == 2
            assert ok == (x == 2)


def test_verify_lengths_and_bonus_token():
    rng = gen(8)
    gamma = 4
    rows = np.tile(P, (gamma, 1))
    bonus = np.array([0.0, 0.0, 1.0])
    target = np.vstack([rows, bonus])
    for _ in range(50):
        xs = [sample(P, rng) for _ in range(gamma)]
        out = verify(target, rows, xs, rng)
        assert out == xs + [2]
    disjoint_q = np.tile(np.array([0.0, 0.0, 1.0]), (gamma, 1))
    disjoint_p = np.vstack([np.tile(np.array([0.5, 0.5, 0.0]), (gamma, 1)), bonus])
    for _ in range(50):
        out = verify(disjoint_p, disjoint_q, [2] * gamma, rng)
        assert len(out) == 1 and out[0] in (0, 1)
    with pytest.raises(ValueError):
        verify(rows, rows, [0] * gamma, rng)
    with pytest.raises(ValueError):
        verify(target, rows[:, :2], [0] * gamma, rng)
    with pytest.raises(ValueError):
        verify(P, rows, [0] * gamma, rng)


def test_mean_length_matches_expected_tokens():
    rng = gen(9)
    gamma = 3
    rows = np.tile(Q, (gamma, 1))
    target = np.tile(P, (gamma + 1, 1))
    lengths = []
    for _ in range(5000):
        xs = [sample(Q, rng) for _ in range(gamma)]
        lengths.append(len(verify(target, rows, xs, rng)))
    lengths = np.array(lengths)
    assert lengths.min() == 1 and lengths.max() == gamma + 1
    assert lengths.mean() == pytest.approx(expected_tokens(0.75, gamma), abs=0.06)
    freq = np.bincount(lengths, minlength=gamma + 2)[1:] / lengths.size
    assert np.allclose(freq, token_count_pmf(0.75, gamma).numpy(), atol=0.03)


# A toy bigram target and a different bigram draft over three tokens.
TARGET = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.1, 0.7, 0.2]])
DRAFT = np.array([[0.3, 0.4, 0.3], [0.5, 0.1, 0.4], [0.2, 0.5, 0.3]])


def test_speculative_step_follows_the_target():
    rng = gen(10)
    counts = np.zeros((3, 3))
    runs = 3000
    for _ in range(runs):
        seq = [0]
        while len(seq) < 3:
            seq = speculative_step(seq, lambda s: TARGET[s[-1]], lambda s: DRAFT[s[-1]], 2, rng)
        counts[seq[1], seq[2]] += 1
    joint = TARGET[0][:, None] * TARGET
    assert np.allclose(counts / runs, joint, atol=0.03)


def test_speculative_step_lengths_and_checks():
    rng = gen(11)
    for gamma in (1, 3, 5):
        for _ in range(30):
            out = speculative_step([1, 2], lambda s: TARGET[s[-1]], lambda s: TARGET[s[-1]], gamma, rng)
            assert out[:2] == [1, 2]
            assert len(out) == 2 + gamma + 1
    with pytest.raises(ValueError):
        speculative_step([0], lambda s: TARGET[s[-1]], lambda s: DRAFT[s[-1]], 0, rng)


def test_expected_tokens():
    assert expected_tokens(0.75, 3) == pytest.approx(175 / 64)
    assert expected_tokens(1.0, 4) == 5.0
    assert expected_tokens(0.0, 4) == 1.0
    assert expected_tokens(0.6, 0) == pytest.approx(1.0)
    for a in (0.1, 0.5, 0.9):
        for g in (1, 2, 7):
            pmf = token_count_pmf(a, g).numpy()
            assert pmf.sum() == pytest.approx(1.0)
            assert np.dot(np.arange(1, g + 2), pmf) == pytest.approx(expected_tokens(a, g))
            assert 1.0 <= expected_tokens(a, g) <= min(g + 1, 1 / (1 - a)) + 1e-12
    assert token_count_pmf(0.5, 2) == pytest.approx([0.5, 0.25, 0.25])
    assert token_count_pmf(1.0, 2) == pytest.approx([0.0, 0.0, 1.0])
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            expected_tokens(bad, 3)
        with pytest.raises(ValueError):
            token_count_pmf(bad, 3)
    with pytest.raises(ValueError):
        expected_tokens(0.5, -1)


def test_walltime_and_operations_reproduce_table_1():
    rows = [(0.6, 2, 1.53, 1.96), (0.7, 3, 1.58, 2.53), (0.8, 2, 1.23, 2.44),
            (0.8, 5, 1.63, 3.69), (0.9, 2, 1.11, 2.71), (0.9, 10, 1.60, 6.86)]
    for a, g, ops, speed in rows:
        assert round(walltime_improvement(a, g, 0.0), 2) == speed
        assert round(operations_factor(a, g, 0.0), 2) == ops


def test_walltime_reproduces_table_4_and_corollary():
    assert round(walltime_improvement(0.75, 7, 0.02), 1) == 3.2
    assert round(walltime_improvement(0.65, 5, 0.02), 1) == 2.4
    assert walltime_improvement(0.5, 1, 0.1) == pytest.approx(1.5 / 1.1)
    assert walltime_improvement(0.3, 1, 0.3) == pytest.approx(1.0)
    assert walltime_improvement(0.75, 3, 0.05) == pytest.approx((175 / 64) / 1.15)
    assert operations_factor(0.75, 3, 0.05) == pytest.approx(4.15 / (175 / 64))
    with pytest.raises(ValueError):
        walltime_improvement(0.5, 2, -0.01)
    with pytest.raises(ValueError):
        operations_factor(0.5, 2, -0.01)
