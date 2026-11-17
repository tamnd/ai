# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/grokking/ (ai://NEW-grokking).
"""Tests of the NumPy reference: the modular-addition split; the Fourier algorithm, exact at p = 113 with the five
key frequencies, equal to the direct cosine sum, peaked at a + b mod p, and at -1/2 on every wrong answer when all
frequencies are used; a composite modulus that breaks it; pure weight decay; the hand-written gradient against finite
differences; the AdamW step against its formula; a short training run; and the worked example of the page."""

import numpy as np
import pytest

from grokking import (
    KEY_FREQUENCIES,
    accuracy,
    adamw_init,
    adamw_step,
    decay_only,
    demo,
    forward,
    fourier_logits,
    init_mlp,
    loss_and_grads,
    modular_addition_split,
    weight_norm,
)

P = 113


def all_pairs(p):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return a.ravel(), b.ravel()


def test_split_sizes_and_labels():
    rng = np.random.default_rng(0)
    tx, ty, sx, sy = modular_addition_split(P, 0.3, rng)
    assert len(tx) == len(ty) == round(0.3 * P * P) == 3831
    assert len(sx) == len(sy) == P * P - 3831
    assert np.array_equal(ty, (tx[:, 0] + tx[:, 1]) % P)
    assert np.array_equal(sy, (sx[:, 0] + sx[:, 1]) % P)
    rows = {tuple(r) for r in tx} | {tuple(r) for r in sx}
    assert len(rows) == P * P
    assert not {tuple(r) for r in tx} & {tuple(r) for r in sx}


def test_split_is_random_and_seeded():
    x1 = modular_addition_split(11, 0.5, np.random.default_rng(1))[0]
    x2 = modular_addition_split(11, 0.5, np.random.default_rng(1))[0]
    x3 = modular_addition_split(11, 0.5, np.random.default_rng(2))[0]
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, x3)
    assert len(x1) == round(0.5 * 121) == 60


def test_split_rejects_bad_arguments():
    rng = np.random.default_rng(0)
    for frac in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            modular_addition_split(7, frac, rng)
    with pytest.raises(ValueError):
        modular_addition_split(1, 0.5, rng)
    assert len(modular_addition_split(2, 0.5, rng)[0]) == 2


def test_fourier_algorithm_is_exact_at_113():
    a, b = all_pairs(P)
    logits = fourier_logits(a, b, P)
    assert logits.shape == (P * P, P)
    assert accuracy(logits, (a + b) % P) == 1.0


def test_fourier_logits_equal_the_direct_cosine_sum():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, P, 50), rng.integers(0, P, 50)
    alpha = rng.uniform(0.5, 2.0, len(KEY_FREQUENCIES))
    got = fourier_logits(a, b, P, alpha=alpha)
    c = np.arange(P)
    want = np.zeros((50, P))
    for k, al in zip(KEY_FREQUENCIES, alpha):
        want += al * np.cos(2 * np.pi * k / P * (a[:, None] + b[:, None] - c[None, :]))
    assert np.allclose(got, want, atol=1e-10)
    assert accuracy(got, (a + b) % P) == 1.0


def test_fourier_peak_value_and_margin():
    a, b = all_pairs(P)
    logits = fourier_logits(a, b, P)
    y = (a + b) % P
    right = logits[np.arange(P * P), y]
    assert np.allclose(right, len(KEY_FREQUENCIES))
    logits[np.arange(P * P), y] = -np.inf
    assert logits.max() == pytest.approx(2.5898235660602444, abs=1e-9)


def test_all_frequencies_give_minus_one_half_off_target():
    p = 13
    a, b = all_pairs(p)
    logits = fourier_logits(a, b, p, freqs=range(1, (p - 1) // 2 + 1))
    y = (a + b) % p
    assert np.allclose(logits[np.arange(p * p), y], (p - 1) / 2)
    mask = np.ones_like(logits, dtype=bool)
    mask[np.arange(p * p), y] = False
    assert np.allclose(logits[mask], -0.5)


def test_one_frequency_with_a_shared_factor_ties():
    p = 12
    a, b = all_pairs(p)
    logits = fourier_logits(a, b, p, freqs=(3,))
    # d = 4 gives cos(2 pi 3 * 4 / 12) = 1, the same logit as the right answer
    y = (a + b) % p
    assert np.allclose(logits[np.arange(p * p), (y - 4) % p], logits[np.arange(p * p), y])
    assert accuracy(logits, y) < 1.0


def test_one_frequency_is_enough_for_a_prime():
    p = 13
    a, b = all_pairs(p)
    for k in range(1, p):
        assert accuracy(fourier_logits(a, b, p, freqs=(k,)), (a + b) % p) == 1.0


def test_accuracy_counts_argmax_hits():
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert accuracy(logits, [1, 0, 0]) == pytest.approx(2 / 3)


def test_decay_only_is_geometric():
    theta = np.array([2.0, -3.0, 0.5])
    out = decay_only(theta, 1e-3, 1.0, 1000)
    assert np.allclose(out, theta * 0.999**1000)
    assert 0.999**1000 == pytest.approx(np.exp(-1.0), rel=6e-4)
    assert np.array_equal(decay_only(theta, 1e-3, 1.0, 0), theta)
    assert np.allclose(decay_only(theta, 0.01, 0.5, 3), theta * 0.995**3)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    params = init_mlp(rng, 7, d=4, hidden=8)
    tx, ty, _, _ = modular_addition_split(7, 0.5, rng)
    loss, grads = loss_and_grads(params, tx, ty)
    h = 1e-6
    for k in params:
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in params[k].shape)
            up = {kk: vv.copy() for kk, vv in params.items()}
            dn = {kk: vv.copy() for kk, vv in params.items()}
            up[k][idx] += h
            dn[k][idx] -= h
            num = (loss_and_grads(up, tx, ty)[0] - loss_and_grads(dn, tx, ty)[0]) / (2 * h)
            assert grads[k][idx] == pytest.approx(num, abs=1e-6)
    logits = forward(params, tx)
    lse = np.log(np.exp(logits).sum(axis=1))
    assert loss == pytest.approx(float(np.mean(lse - logits[np.arange(len(ty)), ty])))


def test_adamw_step_matches_its_formula():
    params = {"w": np.array([1.0, -2.0])}
    grads = {"w": np.array([0.5, 0.1])}
    st = adamw_init(params)
    new, st2 = adamw_step(params, grads, st, lr=0.1, wd=0.5, betas=(0.9, 0.98), eps=1e-8)
    # first step: mhat = g, vhat = g^2, so the Adam part is lr * sign(g) up to eps
    want = params["w"] * (1 - 0.05) - 0.1 * grads["w"] / (np.abs(grads["w"]) + 1e-8)
    assert np.allclose(new["w"], want)
    assert st2["t"] == 1
    assert np.allclose(st2["m"]["w"], 0.1 * grads["w"])
    assert np.allclose(st2["v"]["w"], 0.02 * grads["w"] ** 2)
    new2, st3 = adamw_step(new, grads, st2, lr=0.1, wd=0.5)
    m = 0.9 * 0.1 * grads["w"] + 0.1 * grads["w"]
    v = 0.98 * 0.02 * grads["w"] ** 2 + 0.02 * grads["w"] ** 2
    want2 = new["w"] * 0.95 - 0.1 * (m / (1 - 0.81)) / (np.sqrt(v / (1 - 0.98**2)) + 1e-8)
    assert np.allclose(new2["w"], want2)
    assert st3["t"] == 2


def test_adamw_with_zero_gradient_is_pure_decay():
    params = {"w": np.array([1.0, -2.0, 3.0])}
    st = adamw_init(params)
    for _ in range(5):
        params, st = adamw_step(params, {"w": np.zeros(3)}, st, lr=1e-2, wd=2.0)
    assert np.allclose(params["w"], np.array([1.0, -2.0, 3.0]) * 0.98**5)


def test_short_training_fits_the_training_set():
    rng = np.random.default_rng(5)
    p = 7
    params = init_mlp(rng, p, d=8, hidden=64)
    tx, ty, _, _ = modular_addition_split(p, 0.6, rng)
    st = adamw_init(params)
    first, _ = loss_and_grads(params, tx, ty)
    for _ in range(300):
        loss, grads = loss_and_grads(params, tx, ty)
        params, st = adamw_step(params, grads, st, lr=1e-2, wd=0.1)
    assert loss < 0.1 * first
    assert accuracy(forward(params, tx), ty) == 1.0


def test_weight_decay_shrinks_the_norm_after_fitting():
    rng = np.random.default_rng(6)
    p = 5
    params = init_mlp(rng, p, d=8, hidden=32, scale=3.0)
    tx, ty, _, _ = modular_addition_split(p, 0.8, rng)
    st = adamw_init(params)
    n0 = weight_norm(params)
    for _ in range(200):
        _, grads = loss_and_grads(params, tx, ty)
        params, st = adamw_step(params, grads, st, lr=1e-2, wd=1.0)
    assert weight_norm(params) < 0.5 * n0
    assert weight_norm({"a": np.array([3.0, 4.0])}) == pytest.approx(5.0)


def test_demo_numbers():
    out = demo(np.random.default_rng(0))
    assert out["accuracy"] == 1.0
    assert out["pairs"] == 12769
    assert out["train"] == 3831 and out["test"] == 8938
    assert out["correct_logit"] == pytest.approx(5.0)
    assert out["largest_wrong_logit"] == pytest.approx(2.58982, abs=1e-5)
    assert out["all_freq_wrong_logit"] == pytest.approx(-0.5, abs=1e-9)
    assert out["decay_1000_steps"] == pytest.approx(0.367695, abs=1e-6)
