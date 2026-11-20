# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Tests of the PyTorch version: the shared fixture and its gradients, the cache of keys and values and greedy
decoding, torch.nn.TransformerEncoderLayer and scaled_dot_product_attention with a causal mask, gradcheck, the tied
embedding, causality, and a learning smoke test."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from decoder_lm import DecoderLM, config, encoder_layer, nll, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["vocab"], k["n_ctx"], k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], **kw)


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    model = DecoderLM.from_params(params, config())
    ids = inp["ids"]
    logits, _ = model(ids)
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.embed(ids), want["tokens"], rtol=1e-12, atol=1e-13)
    loss = nll(logits, ids)
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=1e-13)
    loss.backward()
    for name, p in model.p.items():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)


def test_cache_and_greedy_decoding_match_the_fixture():
    params, inp, want = fixture()
    model = DecoderLM.from_params(params, config())
    ids = inp["ids"]
    with torch.no_grad():
        first, kv = model(ids[:, :5])
        second, kv = model(ids[:, 5:], kv)
    torch.testing.assert_close(torch.cat([first, second], 1), want["logits_cached"], rtol=1e-12, atol=1e-13)
    assert kv[1][0].shape == (2, 2, 8, 4)
    assert torch.equal(model.generate(ids[:, :3], 5), want["greedy"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    assert param_count(cfg_of(c)) == c["model"]["params"]
    if c["kind"] != "named":  # the named models are too large to build in a test
        assert sum(p.numel() for p in DecoderLM(cfg_of(c)).parameters()) == c["model"]["params"]


@pytest.mark.parametrize("c", [c for c in SHAPES if c["kind"] != "named"], ids=lambda c: c["name"])
def test_shapes(c):
    cfg, B, blocks = cfg_of(c), c["config"]["B"], c["blocks"]
    g = torch.Generator().manual_seed(0)
    model = DecoderLM(cfg, generator=g)
    with torch.no_grad():
        logits, kv = model(torch.randint(0, cfg["vocab"], (B, cfg["n_ctx"]), generator=g))
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert len(kv) == cfg["n_layers"]
    for keys, values in kv:
        assert list(keys.shape) == blocks["attn"]["internals"]["k"]
        assert list(values.shape) == blocks["attn"]["internals"]["v"]
    assert sum(k.numel() + v.numel() for k, v in kv) == c["model"]["kv_cache"]


def test_layer_matches_torch_transformer_encoder_layer_with_a_causal_mask():
    g = torch.Generator().manual_seed(0)
    cfg = config(vocab=13, n_ctx=9, d_model=12, n_heads=3, d_ff=20, n_layers=2)
    model = DecoderLM(cfg, scale=0.4, generator=g)
    z = torch.randn(3, 9, 12, generator=g)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(9)
    for li in range(2):
        ref = encoder_layer(model, li)
        with torch.no_grad():
            torch.testing.assert_close(model.layer(li, z)[0], ref(z, src_mask=mask, is_causal=True), rtol=1e-12,
                                       atol=1e-12)  # fmt: skip


def test_attention_matches_scaled_dot_product_attention():
    g = torch.Generator().manual_seed(1)
    cfg = config(vocab=7, n_ctx=10, d_model=12, n_heads=4, d_ff=8, n_layers=1)
    model = DecoderLM(cfg, scale=0.5, generator=g)
    h = torch.randn(2, 10, 12, generator=g)
    p = model.p
    with torch.no_grad():
        q, k, v = (h @ p["l0_qkv_W"] + p["l0_qkv_b"]).reshape(2, 10, 3, 4, 3).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 12)
        want = o @ p["l0_out_W"] + p["l0_out_b"]
        torch.testing.assert_close(model.attention(0, h, None)[0], want, rtol=1e-12, atol=1e-12)


def test_cache_of_keys_and_values():
    g = torch.Generator().manual_seed(2)
    for pos in ("learned", "none"):
        cfg = config(vocab=9, n_ctx=10, d_model=8, n_heads=4, d_ff=12, n_layers=3, pos=pos)
        model = DecoderLM(cfg, scale=0.5, generator=g)
        ids = torch.randint(0, 9, (2, 10), generator=g)
        with torch.no_grad():
            full, _ = model(ids)
            pieces, kv = [], None
            for t in range(10):
                logits, kv = model(ids[:, t : t + 1], kv)
                pieces.append(logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-12, atol=1e-12)
        seq = ids[:, :3]
        with torch.no_grad():
            for _ in range(6):
                seq = torch.cat([seq, model(seq)[0][:, -1].argmax(-1, keepdim=True)], 1)
        assert torch.equal(model.generate(ids[:, :3], 6), seq[:, 3:])


def test_gradient():
    # torch.autograd.gradcheck compares autograd with central finite differences on every parameter.
    g = torch.Generator().manual_seed(3)
    for kw in ({}, {"tied": False, "pos": "none"}):
        cfg = config(vocab=5, n_ctx=4, d_model=4, n_heads=2, d_ff=6, n_layers=2, **kw)
        model = DecoderLM(cfg, scale=0.5, generator=g)
        ids = torch.randint(0, 5, (2, 4), generator=g)
        names = list(model.p)

        def f(*ws, names=names, model=model, ids=ids):
            params = dict(zip(names, ws, strict=True))
            return nll(torch.func.functional_call(model, {f"p.{k}": v for k, v in params.items()}, (ids,))[0], ids)

        ws = tuple(model.p[k].detach().clone().requires_grad_(True) for k in names)
        assert torch.autograd.gradcheck(f, ws, eps=1e-6, atol=1e-7)


def test_tied_embedding_is_one_tensor():
    model = DecoderLM(config())
    logits, _ = model(torch.tensor([[1, 2, 3]]))
    nll(logits, torch.tensor([[1, 2, 3]])).backward()
    assert "head_W" not in model.p
    untied = DecoderLM(config(tied=False))
    assert param_count(untied.cfg) - param_count(model.cfg) == 8 * 11


def test_causality():
    g = torch.Generator().manual_seed(4)
    cfg = config(vocab=9, n_ctx=7, d_model=8, n_heads=2, d_ff=12, n_layers=3)
    model = DecoderLM(cfg, scale=0.5, generator=g)
    ids = torch.randint(0, 9, (3, 7), generator=g)
    with torch.no_grad():
        base, _ = model(ids)
        for t in range(7):
            other = ids.clone()
            other[:, t] = (other[:, t] + 1) % 9
            changed, _ = model(other)
            assert torch.equal(changed[:, :t], base[:, :t])
            assert (changed[:, t:] - base[:, t:]).abs().amax(-1).min() > 0


def test_context_limit():
    model = DecoderLM(config())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))
    ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        assert math.isfinite(nll(model(ids)[0], ids).item())


def test_learns():
    # Adam fits 8 counting sequences, each token one more than the last modulo 8, in 150 steps, the task of the NumPy
    # test_learns. A wrong sign or a gradient that does not reach a weight leaves the loss far above the bound of
    # 0.05; a uniform guess is log 8 = 2.08.
    cfg = config(vocab=8, n_ctx=8, d_model=16, n_heads=2, d_ff=32, n_layers=1)
    model = DecoderLM(cfg, generator=torch.Generator().manual_seed(0))
    ids = torch.as_tensor((np.random.default_rng(1).integers(0, 8, 8)[:, None] + np.arange(8)) % 8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        loss = nll(model(ids)[0], ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 1.0
    assert losses[-1] < 0.05, losses[-1]
