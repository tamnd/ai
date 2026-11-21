# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/contrastive-language-image-pretraining/ (ai://NEW-contrastive-language-image-pretraining).
"""Tests that the dual encoder learns to match views, classifies zero-shot, and shows the temperature effects."""

import math

import torch
from clip import (
    DualEncoder,
    make_pairs,
    mean_cosine_within,
    modality_gap,
    retrieval_accuracy,
    sigmoid_loss,
    train,
    zero_shot_accuracy,
)

X, Y, _, MIX = make_pairs(8000)
XT, YT, _, _ = make_pairs(512, gen=torch.Generator().manual_seed(9), mix=MIX)


def fresh(**kw):
    torch.manual_seed(0)
    return DualEncoder(X.shape[1], Y.shape[1], **kw)


def test_relu_encoders_start_in_separate_cones():
    m = fresh()
    with torch.no_grad():
        assert mean_cosine_within(m.embed_images(XT)) > 0.3
        assert mean_cosine_within(m.embed_texts(YT)) > 0.3
    assert modality_gap(m, XT, YT) > 0.5


def test_clip_loss_learns_retrieval_and_zero_shot():
    m = fresh()
    assert retrieval_accuracy(m, XT, YT) < 0.05
    train(m, X, Y)
    assert retrieval_accuracy(m, XT, YT) > 0.95
    gen = torch.Generator().manual_seed(3)
    a, b = MIX
    protos = 2.0 * torch.randn(10, a.shape[0], generator=gen)
    labels = torch.randint(0, 10, (500,), generator=gen)
    z = protos[labels] + 0.3 * torch.randn(500, a.shape[0], generator=gen)
    images = z @ a + 0.3 * torch.randn(500, a.shape[1], generator=gen)
    captions = protos[:, None, :] @ b + 0.3 * torch.randn(10, 5, b.shape[1], generator=gen)
    assert zero_shot_accuracy(m, images, labels, captions) > 0.8


def test_sigmoid_loss_learns_retrieval():
    m = fresh(loss="sigmoid")
    train(m, X, Y)
    assert retrieval_accuracy(m, XT, YT) > 0.95


def test_sigmoid_initial_loss_with_bias():
    n = 512
    u = torch.zeros(n, 2)
    u[:, 0] = 1
    v = torch.zeros(n, 2)
    v[:, 1] = 1
    got = sigmoid_loss(u, v, torch.tensor(10.0), torch.tensor(-10.0)).item()
    assert math.isclose(got, math.log1p(math.exp(10)) + (n - 1) * math.log1p(math.exp(-10)), rel_tol=1e-5)


def test_low_temperature_keeps_the_gap_and_high_temperature_closes_it():
    cold = fresh(learn_scale=False, scale=100.0)
    train(cold, X, Y, steps=200)
    warm = fresh(learn_scale=False, scale=1.0)
    losses = train(warm, X, Y, steps=200)
    assert modality_gap(cold, XT, YT) > 0.3
    assert modality_gap(warm, XT, YT) < 0.1
    floor = math.log1p(255 / math.e)  # batch 256, scale 1, orthogonal negatives
    assert abs(sum(losses[-20:]) / 20 - floor) < 0.2
