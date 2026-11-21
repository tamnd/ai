# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/contrastive-language-image-pretraining/ (ai://NEW-contrastive-language-image-pretraining).
"""Tests for the contrastive and sigmoid losses, their gradients, training and zero-shot classification."""

import math

import numpy as np
from clip import (
    DualEncoder,
    clip_loss,
    loss_floor,
    make_pairs,
    modality_gap,
    normalize,
    retrieval_accuracy,
    sigmoid_loss,
    sigmoid_loss_chunked,
    train,
    zero_shot_classify,
)

RNG = np.random.default_rng(0)


def unit(n, d):
    return normalize(RNG.standard_normal((n, d)))[0]


def numeric_grad(f, a, eps=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old = a[idx]
        a[idx] = old + eps
        hi = f()
        a[idx] = old - eps
        lo = f()
        a[idx] = old
        g[idx] = (hi - lo) / (2 * eps)
    return g


def test_clip_gradients_match_finite_differences():
    u, v, ls = unit(6, 5), unit(6, 5), np.array(1.3)
    _, du, dv, dls = clip_loss(u, v, float(ls))
    assert np.allclose(du, numeric_grad(lambda: clip_loss(u, v, float(ls))[0], u), atol=1e-6)
    assert np.allclose(dv, numeric_grad(lambda: clip_loss(u, v, float(ls))[0], v), atol=1e-6)
    assert np.isclose(dls, numeric_grad(lambda: clip_loss(u, v, float(ls))[0], ls), atol=1e-6)


def test_sigmoid_gradients_match_finite_differences():
    u, v, lt, b = unit(5, 4), unit(5, 4), np.array(0.7), np.array(-1.5)
    _, du, dv, dlt, db = sigmoid_loss(u, v, float(lt), float(b))

    def f():
        return sigmoid_loss(u, v, float(lt), float(b))[0]

    assert np.allclose(du, numeric_grad(f, u), atol=1e-6)
    assert np.allclose(dv, numeric_grad(f, v), atol=1e-6)
    assert np.isclose(dlt, numeric_grad(f, lt), atol=1e-6)
    assert np.isclose(db, numeric_grad(f, b), atol=1e-6)


def test_encoder_gradient_through_normalization():
    x, y, *_ = make_pairs(8, np.random.default_rng(2), dx=6, dy=5)
    m = DualEncoder(6, 5, d=4)
    wi0, wt0, ls0 = m.wi.copy(), m.wt.copy(), m.log_s

    def f():
        return clip_loss(normalize(x @ m.wi)[0], normalize(y @ m.wt)[0], m.log_s)[0]

    g = numeric_grad(f, m.wi)
    m.step(x, y, lr=1e-3)
    assert np.allclose((wi0 - m.wi) / 1e-3, g, atol=1e-5)
    assert not np.allclose(wt0, m.wt) and m.log_s != ls0


def test_uninformative_loss_is_log_n_and_floor():
    u, v = unit(1024, 512), unit(1024, 512)
    assert abs(clip_loss(u, v, 0.0)[0] - math.log(1024)) < 0.01
    assert abs(loss_floor(1.0, 32768, 0.0) - 9.3973) < 1e-4
    assert abs(loss_floor(1 / 0.07, 32768, 0.0) - 0.020268) < 1e-6


def test_chunked_sigmoid_equals_full():
    u, v = unit(37, 8), unit(37, 8)
    full = sigmoid_loss(u, v, math.log(10), -10.0)[0]
    assert np.isclose(full, sigmoid_loss_chunked(u, v, math.log(10), -10.0, chunks=4))


def test_sigmoid_initial_loss_with_bias():
    n = 4096
    u = np.zeros((n, 2))
    u[:, 0] = 1.0
    v = np.zeros((n, 2))
    v[:, 1] = 1.0  # every cosine is 0
    loss = sigmoid_loss(u, v, math.log(10), -10.0)[0]
    assert np.isclose(loss, math.log1p(math.exp(10)) + (n - 1) * math.log1p(math.exp(-10)))
    assert np.isclose(sigmoid_loss(u, v, math.log(10), 0.0)[0], n * math.log(2))


def test_training_aligns_the_views():
    rng = np.random.default_rng(3)
    x, y, _, a, b = make_pairs(4000, rng)
    xt, yt, *_ = make_pairs(256, rng, a=a, b=b)
    for kind in ("clip", "sigmoid"):
        m = DualEncoder(x.shape[1], y.shape[1], loss=kind)
        before = retrieval_accuracy(m, xt, yt)
        losses = train(m, x, y, steps=400)
        after = retrieval_accuracy(m, xt, yt)
        assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:5]), kind
        assert before < 0.05 and after > 0.3, (kind, before, after)


def test_zero_shot_classification_from_captions():
    rng = np.random.default_rng(4)
    x, y, _, a, b = make_pairs(4000, rng)
    m = DualEncoder(x.shape[1], y.shape[1])
    train(m, x, y, steps=400)
    k = 10
    protos = rng.standard_normal((k, a.shape[0])) * 2.0
    labels = rng.integers(0, k, 500)
    z = protos[labels] + 0.3 * rng.standard_normal((500, a.shape[0]))
    imgs = z @ a + 0.3 * rng.standard_normal((500, a.shape[1]))
    captions = protos[:, None, :] @ b + 0.3 * rng.standard_normal((k, 5, b.shape[1]))
    acc = float(np.mean(zero_shot_classify(m, imgs, captions) == labels))
    assert acc > 0.8, acc


def test_linear_encoders_on_centred_data_have_no_cone():
    rng = np.random.default_rng(5)
    x, y, *_ = make_pairs(2000, rng)
    m = DualEncoder(x.shape[1], y.shape[1])
    assert modality_gap(m, x, y) < 0.1
