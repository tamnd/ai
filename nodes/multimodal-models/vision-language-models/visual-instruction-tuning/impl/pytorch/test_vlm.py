# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/visual-instruction-tuning/ (ai://NEW-visual-instruction-tuning).
"""Tests for LLaVA-style two-stage training, the gated cross-attention layer, the resampler and POPE."""

import torch
from torch import nn
from vlm import VLM, Connector, GatedXAttn, PerceiverResampler, TinyLM, freeze, pope_metrics, train

torch.manual_seed(0)
VOCAB, D, D_IMG, P = 32, 64, 16, 4
OBJ = list(range(8, 16))  # eight object words
BOS, WHAT, IS, YES, NO = 1, 2, 3, 4, 5
GEN = torch.Generator().manual_seed(0)
PROTO = torch.randn(VOCAB, D_IMG, generator=GEN)


def images(obj, gen):
    return PROTO[obj][:, None, :] + 0.3 * torch.randn(len(obj), P, D_IMG, generator=gen)


def pretrain_lm():
    """Teach the LM a little text first: 'what is <obj>' -> '<obj>', so it has useful word embeddings."""
    lm = TinyLM(VOCAB, D)
    opt = torch.optim.AdamW(lm.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    for _ in range(150):
        o = torch.tensor(OBJ)[torch.randint(len(OBJ), (64,), generator=g)]
        ids = torch.stack([torch.full_like(o, BOS), o, torch.full_like(o, WHAT), torch.full_like(o, IS), o], 1)
        loss = nn.functional.cross_entropy(lm(ids)[:, -2], ids[:, -1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lm


LM = pretrain_lm()


def caption_batches(seed):
    g = torch.Generator().manual_seed(seed)

    def batch(_):
        o = torch.tensor(OBJ)[torch.randint(len(OBJ), (64,), generator=g)]
        ids = torch.stack([torch.full_like(o, WHAT), torch.full_like(o, IS), o], 1)
        return images(o, g), ids

    return batch


def build(kind="linear"):
    lm = TinyLM(VOCAB, D)
    lm.load_state_dict(LM.state_dict())
    enc = nn.Identity()  # stands in for a frozen CLIP encoder: the patch features are given
    return VLM(lm, enc, Connector(D_IMG, D, kind))


def accuracy(model, seed=99):
    g = torch.Generator().manual_seed(seed)
    o = torch.tensor(OBJ)[torch.randint(len(OBJ), (400,), generator=g)]
    ids = torch.stack([torch.full_like(o, WHAT), torch.full_like(o, IS), o], 1)
    with torch.no_grad():
        pred = model(images(o, g), ids)[:, 1].argmax(-1)
    return (pred == o).float().mean().item()


def test_stage_one_trains_only_the_connector():
    model = build()
    freeze(model.lm)
    before = {k: v.clone() for k, v in model.lm.state_dict().items()}
    acc0 = accuracy(model)
    losses = train(model, caption_batches(2), steps=200, lr=1e-2, answer_from=2)
    assert all(torch.equal(before[k], v) for k, v in model.lm.state_dict().items())
    assert losses[-1] < 0.3 * losses[0]
    assert acc0 < 0.4 and accuracy(model) > 0.95


def test_mlp_connector_also_aligns():
    model = build("mlp")
    freeze(model.lm)
    train(model, caption_batches(3), steps=200, lr=3e-3, answer_from=2)
    assert accuracy(model) > 0.95


def yes_no_batches(seed, only_yes=False):
    """'is <obj> ?' -> yes if the image shows <obj>; with only_yes the data never contain a no answer."""
    g = torch.Generator().manual_seed(seed)

    def batch(_):
        shown = torch.tensor(OBJ)[torch.randint(len(OBJ), (64,), generator=g)]
        asked = shown.clone()
        if not only_yes:
            flip = torch.rand(64, generator=g) < 0.5
            other = torch.tensor(OBJ)[torch.randint(len(OBJ), (64,), generator=g)]
            other = torch.where(other == shown, torch.tensor(OBJ)[(torch.tensor([OBJ.index(int(s)) for s in shown]) + 1) % len(OBJ)], other)
            asked = torch.where(flip, other, shown)
        ans = torch.where(asked == shown, torch.full_like(asked, YES), torch.full_like(asked, NO))
        ids = torch.stack([torch.full_like(asked, IS), asked, ans], 1)
        return images(shown, g), ids

    return batch


def pope_eval(model, seed=7):
    pb = yes_no_batches(seed)
    patches, ids = pb(0)
    with torch.no_grad():
        logits = model(patches, ids)[:, 1]
    pred = logits[:, YES] > logits[:, NO]
    return pope_metrics(pred, ids[:, 2] == YES)


def test_stage_two_instruction_tuning_and_the_yes_bias():
    balanced = build()
    freeze(balanced.lm)
    train(balanced, caption_batches(4), steps=150, lr=1e-2, answer_from=2)
    biased = build()
    biased.connector.load_state_dict(balanced.connector.state_dict())
    freeze(balanced.lm, False)
    freeze(biased.lm, False)
    train(balanced, yes_no_batches(5), steps=300, lr=1e-3, answer_from=2)
    train(biased, yes_no_batches(5, only_yes=True), steps=300, lr=1e-3, answer_from=2)
    good, bad = pope_eval(balanced), pope_eval(biased)
    assert good["f1"] > 0.9 and abs(good["yes"] - 0.5) < 0.15
    assert bad["yes"] > 0.95 and abs(bad["f1"] - 2 / 3) < 0.05 and abs(bad["accuracy"] - 0.5) < 0.1


def test_gated_layer_is_identity_at_init_and_opens_with_training():
    layer = GatedXAttn(D)
    y, x = torch.randn(3, 5, D), torch.randn(3, 7, D)
    assert torch.allclose(layer(y, x), y)
    target = y + x.mean(1, keepdim=True)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    out = layer(y, x)
    ((out - target) ** 2).mean().backward()
    assert layer.alpha_attn.grad.abs() > 0
    assert torch.all(layer.attn.in_proj_weight.grad == 0) and torch.all(layer.ffw[0].weight.grad == 0)
    for _ in range(200):
        opt.zero_grad()
        ((layer(y, x) - target) ** 2).mean().backward()
        opt.step()
    assert layer.alpha_attn.abs() > 0.05
    assert ((layer(y, x) - target) ** 2).mean() < 0.5 * ((y - target) ** 2).mean()


def test_resampler_fixed_output_and_order_free():
    r = PerceiverResampler(D, num_latents=8).eval()
    for n in (5, 49, 257):
        x = torch.randn(2, n, D)
        out = r(x)
        assert out.shape == (2, 8, D)
        perm = torch.randperm(n)
        assert torch.allclose(out, r(x[:, perm]), atol=1e-5)


def test_pope_metrics_always_yes():
    r = pope_metrics([True] * 10, [True] * 5 + [False] * 5)
    assert r["accuracy"] == 0.5 and r["precision"] == 0.5 and r["recall"] == 1.0 and round(r["f1"], 4) == 0.6667
