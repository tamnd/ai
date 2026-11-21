# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/visual-instruction-tuning/ (ai://NEW-visual-instruction-tuning).
"""Tests for the linear connector, the gated cross-attention block, the resampler and the POPE metrics."""

import math

import numpy as np
from vlm import (
    FrozenBagLM,
    gated_xattn,
    gated_xattn_grads,
    image_causal_mask,
    make_images,
    perceiver_resampler,
    pope_metrics,
    pope_questions,
    soft_prompt_loss,
    train_projection,
    visual_tokens,
)

RNG = np.random.default_rng(0)
VOCAB, D, D_IMG = 40, 24, 16
WORDS = list(range(10, 20))  # the ten "object" words
PROMPT = [1, 2, 3]  # "what is this"


def test_projection_gradient_matches_finite_differences():
    lm = FrozenBagLM(VOCAB, D)
    Z, y, _ = make_images(5, WORDS, D_IMG)
    W = 0.1 * RNG.standard_normal((D_IMG, D))
    _, dW, _ = soft_prompt_loss(W, lm, Z, PROMPT, y)
    eps = 1e-6
    for idx in [(0, 0), (3, 7), (15, 23)]:
        E = np.zeros_like(W)
        E[idx] = eps
        num = (soft_prompt_loss(W + E, lm, Z, PROMPT, y)[0] - soft_prompt_loss(W - E, lm, Z, PROMPT, y)[0]) / (2 * eps)
        assert math.isclose(dW[idx], num, rel_tol=1e-5, abs_tol=1e-8)


def test_stage_one_learns_soft_tokens_while_the_language_model_stays_frozen():
    lm = FrozenBagLM(VOCAB, D)
    E_before = lm.E.copy()
    Z, y, protos = make_images(800, WORDS, D_IMG)
    Zt, yt, _ = make_images(300, WORDS, D_IMG, seed=1, protos=protos)
    W0 = 0.01 * np.random.default_rng(0).standard_normal((D_IMG, D))
    acc0 = np.mean(soft_prompt_loss(W0, lm, Zt, PROMPT, yt)[2].argmax(1) == yt)
    W, losses = train_projection(lm, Z, y, PROMPT, D_IMG)
    acc = np.mean(soft_prompt_loss(W, lm, Zt, PROMPT, yt)[2].argmax(1) == yt)
    assert acc0 < 0.3 and acc > 0.95
    assert losses[-1] < 0.5 * losses[0]
    assert np.array_equal(lm.E, E_before)  # only W changed


def test_gate_is_identity_at_init_and_only_alpha_gets_gradient():
    T, M, d = 4, 6, 8
    y, x = RNG.standard_normal((T, d)), RNG.standard_normal((M, d))
    Wq, Wk, Wv = (RNG.standard_normal((d, d)) for _ in range(3))
    out, f, _ = gated_xattn(y, x, 0.0, Wq, Wk, Wv)
    assert np.allclose(out, y)
    g = RNG.standard_normal((T, d))
    da, dWv = gated_xattn_grads(y, x, 0.0, Wq, Wk, Wv, g)
    assert math.isclose(da, float(np.sum(g * f)))
    assert np.allclose(dWv, 0.0)
    eps = 1e-6
    num = (np.sum(g * gated_xattn(y, x, eps, Wq, Wk, Wv)[0]) - np.sum(g * gated_xattn(y, x, -eps, Wq, Wk, Wv)[0])) / (2 * eps)
    assert math.isclose(da, num, rel_tol=1e-6)


def test_gate_gradient_in_values_once_open():
    T, M, d = 3, 5, 6
    y, x = RNG.standard_normal((T, d)), RNG.standard_normal((M, d))
    Wq, Wk, Wv = (RNG.standard_normal((d, d)) for _ in range(3))
    g = RNG.standard_normal((T, d))
    _, dWv = gated_xattn_grads(y, x, 0.7, Wq, Wk, Wv, g)
    eps = 1e-6
    E = np.zeros_like(Wv)
    E[1, 2] = eps
    num = (np.sum(g * gated_xattn(y, x, 0.7, Wq, Wk, Wv + E)[0]) - np.sum(g * gated_xattn(y, x, 0.7, Wq, Wk, Wv - E)[0])) / (2 * eps)
    assert math.isclose(dWv[1, 2], num, rel_tol=1e-5)


def test_each_text_token_sees_only_the_last_image():
    # text tokens: two before any image, three after image 0, two after image 1
    text_img = [-1, -1, 0, 0, 0, 1, 1]
    vis_img = [0, 0, 0, 1, 1, 1]
    mask = image_causal_mask(text_img, vis_img)
    d = 4
    y, x = RNG.standard_normal((7, d)), RNG.standard_normal((6, d))
    Wq, Wk, Wv = (RNG.standard_normal((d, d)) for _ in range(3))
    out, _, P = gated_xattn(y, x, 1.0, Wq, Wk, Wv, mask)
    assert np.allclose(P[:2], 0) and np.allclose(out[:2], y[:2])
    assert np.allclose(P[2:5, 3:], 0) and np.allclose(P[5:, :3], 0)
    assert np.allclose(P[2:].sum(1), 1)


def test_resampler_output_size_is_fixed_and_order_free():
    d, m = 8, 5
    lat = RNG.standard_normal((m, d))
    Wq, Wk, Wv = (0.3 * RNG.standard_normal((d, d)) for _ in range(3))
    for n in (3, 49, 257):
        X = RNG.standard_normal((n, d))
        out = perceiver_resampler(X, lat, Wq, Wk, Wv, layers=2)
        assert out.shape == (m, d)
        perm = RNG.permutation(n)
        assert np.allclose(out, perceiver_resampler(X[perm], lat, Wq, Wk, Wv, layers=2))


def test_visual_token_counts():
    assert visual_tokens(224, 14) == 256 and visual_tokens(224, 14, cls_token=True) == 257
    assert visual_tokens(336, 14) == 576
    assert 5 * visual_tokens(336, 14) == 2880


def test_pope_always_yes_baseline():
    labels = [True] * 1500 + [False] * 1500
    r = pope_metrics([True] * 3000, labels)
    assert r["accuracy"] == 0.5 and r["precision"] == 0.5 and r["recall"] == 1.0
    assert round(100 * r["f1"], 2) == 66.67 and r["yes"] == 1.0
    perfect = pope_metrics(labels, labels)
    assert perfect["f1"] == 1.0 and perfect["yes"] == 0.5


def test_pope_sampling_strategies():
    freq = {"person": 50, "chair": 30, "table": 25, "car": 20, "knife": 12, "fork": 10, "surfboard": 3, "giraffe": 2}
    cooc = {("person", "surfboard"): 40, ("chair", "table"): 22, ("person", "car"): 5, ("chair", "fork"): 9}
    img = {"person", "chair", "giraffe"}
    for strat in ("random", "popular", "adversarial"):
        qs = pope_questions(img, freq, cooc, l=4, strategy=strat)
        ans = [a for _, a, _ in qs]
        assert ans.count("yes") == ans.count("no") == 2
        assert all((o in img) == (a == "yes") for _, a, o in qs)
    pop = [o for _, a, o in pope_questions(img, freq, cooc, l=4, strategy="popular") if a == "no"]
    adv = [o for _, a, o in pope_questions(img, freq, cooc, l=4, strategy="adversarial") if a == "no"]
    assert pop == ["table", "car"]
    assert adv == ["surfboard", "table"]
