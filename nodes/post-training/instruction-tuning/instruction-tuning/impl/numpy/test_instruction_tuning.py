# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/instruction-tuning/ (ai://NEW-instruction-tuning).
"""Tests of the NumPy code: templates and input inversion, the worked example of capped mixing, rates, passes and
sampling, T0's size rule, encoding with the response-only mask, packing with segment masks and positions, the masked
loss against a hand computation and the shared fixture, token versus sequence averaging, the gradient against finite
differences, and the chain-rule identity on a bigram model."""

import itertools

import numpy as np
import pytest
from instruction_tuning import (
    encode,
    expected_passes,
    invert,
    log_softmax,
    masked_nll,
    masked_nll_grad,
    mixing_rates,
    pack,
    render,
    sample_mixture,
    segment_attention_mask,
    segment_positions,
    t0_sizes,
    token_nll,
)

# The fixture shared with the PyTorch tests: two sequences of four positions over a vocabulary of four.
Z = np.array(
    [
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
)
Y = np.array([[0, 1, 2, 3], [2, 0, 0, 1]])
M = np.array([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=float)
FIXTURE_TOKEN = 0.7212637615150113
FIXTURE_SEQUENCE = 0.6989400897373603


def test_render_and_invert():
    ex = {"question": "What is 2 + 3?", "answer": "5"}
    assert render(("Q: {question}\nA:", " {answer}"), ex) == (
        "Q: What is 2 + 3?\nA:",
        " 5",
    )
    assert invert(("Write a question whose answer is {answer}.", "{question}"), ex) == (
        "Write a question whose answer is 5.",
        "What is 2 + 3?",
    )


def test_worked_example_rates_and_passes():
    sizes = [1000, 3000, 100000]
    r = mixing_rates(sizes, cap=3000)
    assert np.allclose(r, [1 / 7, 3 / 7, 3 / 7])
    assert np.allclose(mixing_rates(sizes), np.array(sizes) / 104000)
    assert np.allclose(expected_passes(sizes, r, 70000), [10, 10, 0.3])


def test_rates_properties():
    rng = np.random.default_rng(1)
    for _ in range(100):
        e = rng.integers(1, 10**6, size=int(rng.integers(2, 10)))
        cap = int(rng.integers(10, 10**5))
        r = mixing_rates(e, cap=cap)
        assert abs(r.sum() - 1) < 1e-12
        big = e >= cap
        if big.sum() > 1:
            assert np.allclose(r[big], r[big][0])
        passes = expected_passes(e, r, 10**6)
        small = e <= cap
        if small.sum() > 1:
            assert np.allclose(passes[small], passes[small][0])
        if small.any() and (~small).any():
            assert passes[~small].max() < passes[small].min()


def test_temperature_moves_toward_uniform():
    e = [10, 100, 1000]
    r1 = mixing_rates(e)
    r2 = mixing_rates(e, temperature=2.0)
    r_inf = mixing_rates(e, temperature=1e9)
    assert r2[0] > r1[0] and r2[-1] < r1[-1]
    assert np.allclose(r_inf, 1 / 3, atol=1e-6)
    with pytest.raises(ValueError):
        mixing_rates([0, 1])
    with pytest.raises(ValueError):
        mixing_rates([1, 2], temperature=0)


def test_sampling_matches_rates():
    rng = np.random.default_rng(2)
    sizes = np.array([1000, 3000, 100000])
    r = mixing_rates(sizes, cap=3000)
    which, index = sample_mixture(rng, sizes, r, 100_000)
    freq = np.bincount(which, minlength=3) / len(which)
    assert np.all(np.abs(freq - r) < 4 * np.sqrt(r * (1 - r) / len(which)))
    assert np.all(index >= 0) and np.all(index < sizes[which])


def test_t0_rule_is_not_monotone():
    eff = t0_sizes([400_000, 500_000, 500_001, 2_000_000], templates=10)
    assert list(eff) == [400_000, 500_000, 50_000, 50_000]
    assert list(t0_sizes([600_000, 600_000], templates=[2, 5])) == [250_000, 100_000]


def test_encode_masks_only_the_response():
    inputs, targets, mask = encode([5, 6, 7], [8, 9], eos_id=1)
    assert list(inputs) == [5, 6, 7, 8, 9]
    assert list(targets) == [6, 7, 8, 9, 1]
    assert list(mask) == [0, 0, 1, 1, 1]
    inputs, targets, mask = encode([5], [], eos_id=1)
    assert list(targets) == [1] and list(mask) == [1]
    with pytest.raises(ValueError):
        encode([], [3], eos_id=1)


def test_pack_rows_segments_and_masks():
    exs = [encode([2, 3], [4], 1), encode([5], [6, 7], 1), encode([2, 2, 2], [3, 3], 1)]
    inputs, _, mask, seg = pack(exs, length=7)
    assert inputs.shape == (2, 7)
    assert list(seg[0]) == [1, 1, 1, 2, 2, 2, 0]
    assert list(inputs[0, :6]) == [2, 3, 4, 5, 6, 7]
    assert list(mask[0]) == [0, 1, 1, 1, 1, 1, 0]
    assert list(seg[1]) == [1, 1, 1, 1, 1, 0, 0]
    assert mask.sum() == sum(m.sum() for _, _, m in exs)
    with pytest.raises(ValueError):
        pack([encode([1] * 9, [2], 1)], length=7)


def test_segment_attention_and_positions():
    seg = np.array([[1, 1, 2, 2, 2, 0]])
    a = segment_attention_mask(seg)[0]
    assert a[1, 0] and not a[0, 1]
    assert not a[2, 1] and a[4, 2] and a[4, 4]
    assert not a[5].any()
    assert list(segment_positions(seg)[0]) == [0, 1, 0, 1, 2, 0]


def test_masked_loss_fixture_and_hand_computation():
    nll = token_nll(Z, Y)
    hand = -np.log(np.exp(Z[0, 1, 1]) / np.exp(Z[0, 1]).sum())
    assert abs(nll[0, 1] - hand) < 1e-12
    assert abs(nll[1, 1] - np.log(4)) < 1e-12
    assert abs(masked_nll(Z, Y, M) - FIXTURE_TOKEN) < 1e-12
    assert abs(masked_nll(Z, Y, M, "sequence") - FIXTURE_SEQUENCE) < 1e-12
    assert abs(masked_nll(Z, Y, M) - nll[M == 1].mean()) < 1e-12


def test_token_versus_sequence_weights():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2, 90, 5))
    y = rng.integers(0, 5, size=(2, 90))
    m = np.zeros((2, 90))
    m[0, :10] = 1
    m[1, :] = 1
    nll = token_nll(z, y)
    a, b = nll[0, :10].mean(), nll[1].mean()
    assert abs(masked_nll(z, y, m) - (0.1 * a + 0.9 * b)) < 1e-12
    assert abs(masked_nll(z, y, m, "sequence") - (0.5 * a + 0.5 * b)) < 1e-12
    with pytest.raises(ValueError):
        masked_nll(z, y, np.zeros_like(m))
    with pytest.raises(ValueError):
        masked_nll(z, y, m, "batch")


@pytest.mark.parametrize("reduction", ["token", "sequence"])
def test_gradient_matches_finite_differences(reduction):
    g = masked_nll_grad(Z, Y, M, reduction)
    num = np.zeros_like(Z)
    h = 1e-6
    for idx in np.ndindex(Z.shape):
        zp, zm = Z.copy(), Z.copy()
        zp[idx] += h
        zm[idx] -= h
        num[idx] = (masked_nll(zp, Y, M, reduction) - masked_nll(zm, Y, M, reduction)) / (2 * h)
    assert np.allclose(g, num, atol=1e-8)
    assert np.all(g[M == 0] == 0)
    assert np.allclose(g.sum(axis=-1), 0)


def test_log_softmax_is_stable():
    z = np.array([[1000.0, 0.0, -1000.0]])
    lp = log_softmax(z)
    assert np.all(np.isfinite(lp)) and abs(lp[0, 0]) < 1e-12


def test_response_loss_is_conditional_log_likelihood():
    """On a bigram model, the masked loss summed over the response is -log p(response, eos | prompt)."""
    rng = np.random.default_rng(4)
    V = 4
    start = rng.dirichlet(np.ones(V))
    trans = rng.dirichlet(np.ones(V), size=V)
    logits = np.log(trans)  # the bigram model's logits at a position depend only on the input token there
    prompt = [2, 0]
    total = 0.0
    for resp in itertools.product(range(V), repeat=2):
        inputs, targets, mask = encode(prompt, list(resp), eos_id=3)
        nll = token_nll(logits[inputs], targets)
        cond = np.exp(-(mask * nll).sum())
        joint = start[prompt[0]] * np.prod([trans[a, b] for a, b in zip(inputs, targets)])
        p_prompt = start[prompt[0]] * trans[prompt[0], prompt[1]]
        assert abs(cond - joint / p_prompt) < 1e-12
        total += cond
    # Summing over every response of length 2 leaves the probability of eos after it, which is not 1.
    assert 0 < total < 1
