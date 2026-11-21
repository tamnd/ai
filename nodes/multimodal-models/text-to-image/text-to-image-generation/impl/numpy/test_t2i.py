# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/text-to-image-generation/ (ai://NEW-text-to-image-generation).
"""Tests for guidance with thresholding, contrastive reranking, unCLIP's PCA tokens and the pooled text embedding."""

import numpy as np
from t2i import (
    best_of_n,
    cosine_schedule,
    ddim_guided,
    dynamic_threshold,
    fit_pca,
    gaussian_posterior_mean,
    guided_x0,
    pca_detokenize,
    pca_tokens,
    pooled_text,
    rerank,
    saturation,
    static_threshold,
)

D = 4096
RNG = np.random.default_rng(0)
MU_C = RNG.uniform(-0.6, 0.6, D)  # the "image" the caption describes
S = 0.1
MU_U, S_U = np.zeros(D), np.sqrt(S**2 + 0.12)  # one Gaussian fit to all images


def noisy_guided(w, t=0.5, seed=1):
    rng = np.random.default_rng(seed)
    x = MU_C + S * rng.standard_normal(D)
    a, sg = cosine_schedule(t)
    z = (a * x + sg * rng.standard_normal(D))[None]
    xc = gaussian_posterior_mean(z, a, sg, MU_C, S)
    xu = gaussian_posterior_mean(z, a, sg, MU_U, S_U)
    return guided_x0(xc, xu, w)


def test_guidance_weight_one_is_the_conditional_model_and_zero_the_unconditional():
    a, b = RNG.standard_normal(10), RNG.standard_normal(10)
    assert np.allclose(guided_x0(a, b, 1.0), a)
    assert np.allclose(guided_x0(a, b, 0.0), b)


def test_guided_x_prediction_leaves_the_training_range_as_w_grows():
    peaks = [ddim_guided(np.random.default_rng(2).standard_normal((1, D)), MU_C, S, MU_U, S_U, w)[1] for w in (1, 2, 4, 8)]
    assert peaks[0] < 1.0  # the conditional model stays inside [-1, 1]
    assert all(p2 > p1 for p1, p2 in zip(peaks, peaks[1:]))
    assert peaks[-1] > 4.0


def test_thresholds_do_nothing_inside_the_range():
    x = RNG.uniform(-0.9, 0.9, (3, 50))
    assert np.array_equal(static_threshold(x), x)
    assert np.allclose(dynamic_threshold(x), x)


def test_static_threshold_saturates_and_dynamic_threshold_rescales():
    for w, min_sat in ((4, 0.4), (8, 0.7)):
        x0 = noisy_guided(w)
        st, dy = static_threshold(x0), dynamic_threshold(x0, 99.5)
        assert np.abs(st).max() <= 1.0 and np.abs(dy).max() <= 1.0 + 1e-12
        assert saturation(st) > min_sat  # most pixels pinned at -1 or +1
        assert saturation(dy) < 0.01  # only the top 0.5% by magnitude
        # Inside the percentile, dynamic thresholding is one division, so it keeps every contrast ratio.
        s = np.percentile(np.abs(x0), 99.5)
        inside = np.abs(x0) <= s
        assert np.allclose(dy[inside], x0[inside] / s)
        assert np.corrcoef(dy[0], x0[0])[0, 1] > np.corrcoef(st[0], x0[0])[0, 1]


def test_thresholded_sampler_stays_in_range():
    z = np.random.default_rng(3).standard_normal((2, D))
    for th in (static_threshold, dynamic_threshold):
        x, _ = ddim_guided(z, MU_C, S, MU_U, S_U, 8.0, threshold=th)
        assert np.abs(x).max() <= 1.0 + 1e-9
        assert np.corrcoef(x[0], MU_C)[0, 1] > 0.9


def test_rerank_keeps_the_candidates_closest_to_the_caption():
    rng = np.random.default_rng(4)
    text = rng.standard_normal(16)
    good = text + 0.3 * rng.standard_normal((3, 16))
    bad = rng.standard_normal((13, 16))
    cands = np.concatenate([bad[:5], good, bad[5:]])
    top, scores = rerank(cands, text, 3)
    assert sorted(top.tolist()) == [5, 6, 7]
    assert scores[top].min() > np.delete(scores, top).max()


def test_best_of_n_has_diminishing_returns():
    p = 0.1
    gains = [best_of_n(p, 2 * n) - best_of_n(p, n) for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    # Each doubling of n gains more until n is near ln 2 / p (about 7 here), then less every time.
    assert all(g2 > g1 for g1, g2 in zip(gains[:3], gains[1:4]))
    assert all(g2 < g1 for g1, g2 in zip(gains[3:], gains[4:]))
    assert best_of_n(p, 32) > 0.96 and best_of_n(p, 512) - best_of_n(p, 32) < 0.035
    # Monte Carlo agrees with the formula.
    rng = np.random.default_rng(5)
    hits = (rng.random((20000, 16)) < p).any(axis=1).mean()
    assert abs(hits - best_of_n(p, 16)) < 0.01


def test_pca_tokens_reconstruct_low_rank_embeddings():
    rng = np.random.default_rng(6)
    d, r, n = 64, 20, 2000
    Z = rng.standard_normal((n, r)) @ rng.standard_normal((r, d)) + 0.01 * rng.standard_normal((n, d))
    mean, comps, sv = fit_pca(Z, r)
    assert np.all(np.diff(sv) <= 1e-9)  # ordered by decreasing singular value
    c = (Z - mean) @ comps.T
    lo, hi = c.min(), c.max()
    tok = pca_tokens(Z, mean, comps, lo, hi, 1024)
    assert tok.shape == (n, r) and tok.min() >= 0 and tok.max() <= 1023
    rec = pca_detokenize(tok, mean, comps, lo, hi, 1024)
    rel = np.mean((rec - Z) ** 2) / np.mean((Z - Z.mean(0)) ** 2)
    assert rel < 0.01  # under 1% of the variance lost, with r of d coordinates kept


def test_pooled_text_cannot_bind_attributes():
    E = np.random.default_rng(7).standard_normal((10, 8))
    RED, BLUE, LEFT, RIGHT = 1, 2, 3, 4
    a = pooled_text([RED, LEFT, BLUE, RIGHT], E)
    b = pooled_text([BLUE, LEFT, RED, RIGHT], E)
    assert np.allclose(a, b)  # opposite scenes, one embedding
