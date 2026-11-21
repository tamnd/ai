# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/text-to-image-generation/ (ai://NEW-text-to-image-generation).
"""Tests for the autoregressive text-to-image model: binding, guidance and reranking on a token world."""

import math

import pytest
import torch
from t2i import TextToImage, World, accuracy, drop_text, entropy_per_image, log_likelihood, rerank_sample, sample, train

torch.manual_seed(0)


def trained(world, cond, steps=600):
    torch.manual_seed(0)
    m = TextToImage(world, cond)
    losses = train(m, steps=steps)
    return m, losses


@pytest.fixture(scope="module")
def clean():
    w = World(colors=6, slots=4, noise=0.0)
    return w, trained(w, "sequence"), trained(w, "pooled", steps=400)


@pytest.fixture(scope="module")
def noisy():
    w = World(colors=6, slots=4, noise=0.5)
    return w, trained(w, "sequence")


def test_world_captions_describe_their_images():
    w = World()
    text, img = w.sample(200, torch.Generator().manual_seed(1))
    assert torch.equal(w.target(text), img)
    sw = w.swap(text)
    assert torch.equal(w.target(sw)[:, [1, 0, 2, 3]], img)
    assert torch.equal(sw[:, 1::2], text[:, 1::2])  # same words in the same places, colours exchanged


def test_text_dropout_rate():
    w = World()
    text = torch.zeros(20000, w.text_len, dtype=torch.long)
    dropped = (drop_text(text, 0.1, w.PAD, torch.Generator().manual_seed(2)) == w.PAD).all(1).float().mean()
    assert abs(dropped.item() - 0.1) < 0.01


def test_sequence_conditioning_binds_and_pooled_cannot(clean):
    w, (seq, seq_losses), (pool, _) = clean
    text, _ = w.sample(4000, torch.Generator().manual_seed(3))
    assert seq_losses[-1] < seq_losses[0] / 5
    acc_seq = accuracy(w, text, sample(seq, text, gen=torch.Generator().manual_seed(4)))
    acc_pool = accuracy(w, text, sample(pool, text, gen=torch.Generator().manual_seed(4)))
    assert acc_seq > 0.98
    # A pooled caption fixes only the multiset of colours; the best any model can then do is 7/72 (derivation).
    assert torch.allclose(pool.text_context(text), pool.text_context(w.swap(text)), atol=1e-6)
    assert acc_pool < 7 / 72 + 0.02


def test_guidance_trades_diversity_for_alignment(noisy):
    w, (m, _) = noisy
    text, _ = w.sample(1000, torch.Generator().manual_seed(5))
    acc = {s: accuracy(w, text, sample(m, text, scale=s, gen=torch.Generator().manual_seed(6))) for s in (1.0, 2.0, 3.0)}
    ent = {s: entropy_per_image(m, text[:50], s, gen=torch.Generator().manual_seed(7)) for s in (1.0, 3.0)}
    # Half the training captions are wrong, so the conditional model is right about half the time ...
    assert 0.4 < acc[1.0] < 0.6
    # ... and guidance concentrates it on the caption's image.
    assert acc[2.0] > acc[1.0] + 0.3 and acc[3.0] > 0.95
    # The conditional entropy is near that of the data, about 4.27 nats; guidance removes most of it.
    p = 0.5 + 0.5 / 6**4
    h_data = -(p * math.log(p) + (6**4 - 1) * (0.5 / 6**4) * math.log(0.5 / 6**4))
    assert abs(ent[1.0] - h_data) < 0.4
    assert ent[3.0] < 0.5


def test_guided_likelihood_at_scale_one_is_the_conditional_model(noisy):
    w, (m, _) = noisy
    text, img = w.sample(8, torch.Generator().manual_seed(8))
    a = log_likelihood(m, text, img, 1.0)
    lc = sum(torch.log_softmax(m(text, img[:, :-1]), -1)[:, k].gather(1, img[:, k : k + 1]).squeeze(1) for k in range(4))
    assert torch.allclose(a, lc, atol=1e-5)


def test_reranking_follows_best_of_n(noisy):
    w, (m, _) = noisy
    text, _ = w.sample(1000, torch.Generator().manual_seed(9))

    def oracle(t, img):  # stands in for a contrastive scorer such as CLIP or CoCa
        return (img == w.target(t)).sum(1).float()

    p1 = accuracy(w, text, sample(m, text, gen=torch.Generator().manual_seed(10)))
    p4 = accuracy(w, text, rerank_sample(m, text, 4, oracle, gen=torch.Generator().manual_seed(11)))
    assert abs(p4 - (1 - (1 - p1) ** 4)) < 0.05
    assert p4 > p1 + 0.3
