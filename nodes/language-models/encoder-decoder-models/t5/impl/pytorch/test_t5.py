# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Tests of the PyTorch version: the shared fixture and its gradients, the cache and greedy decoding, the counts of
shapes.json, the attention against an explicit softmax, F.rms_norm against its formula, gradcheck, causality, the
encoder without position biases, and a learning smoke test."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from t5 import KEYS, T5, bucket, config, nll, param_count, shift_right

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    return config(*(c["config"][k] for k in KEYS), **kw)


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    model = T5.from_params(params, config())
    logits, e = model(inp["src"], inp["tgt"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(e, want["e"], rtol=1e-12, atol=1e-13)
    loss = nll(logits, inp["tgt"])
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=1e-13)
    loss.backward()
    for name, p in model.p.items():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)


def test_cache_and_greedy_decoding_match_the_fixture():
    params, inp, want = fixture()
    model = T5.from_params(params, config())
    with torch.no_grad():
        e = model.encode(inp["src"])
        ids = shift_right(inp["tgt"])
        first, kv = model.decode(e, ids[:, :2])
        second, kv = model.decode(e, ids[:, 2:], kv, start=2)
    torch.testing.assert_close(torch.cat([first, second], 1), want["logits_cached"], rtol=1e-12, atol=1e-13)
    assert kv[1][0].shape == (2, 2, 5, 3)
    assert torch.equal(model.generate(inp["src"], 6), want["greedy"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":  # the arrays of the paper's models are too large for a test
        model = T5(cfg, generator=torch.Generator().manual_seed(0))
        assert sum(p.numel() for p in model.parameters()) == c["model"]["params"]


@pytest.mark.parametrize("c", [c for c in SHAPES if c["kind"] != "named"], ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    model = T5(cfg, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(1)
    src = torch.randint(0, cfg["vocab"], (k["B"], k["S"]), generator=g)
    tgt = torch.randint(0, cfg["vocab"], (k["B"], k["T"]), generator=g)
    with torch.no_grad():
        logits, e = model(src, tgt)
        _, kv = model.decode(e, shift_right(tgt))
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(e.shape) == blocks["cross"]["inputs"]["e"]
    assert len(kv) == cfg["n_layers"]
    for keys, values in kv:
        assert list(keys.shape) == blocks["dec_self"]["internals"]["k"]
        assert list(values.shape) == blocks["dec_self"]["internals"]["v"]


def test_attention_against_an_explicit_softmax():
    torch.manual_seed(1)
    model = T5(config(n_heads=3, d_kv=2))
    xq, xkv = torch.randn(2, 4, 8), torch.randn(2, 6, 8)
    bias = torch.randn(3, 4, 6)
    bias[:, 0, 3:] = float("-inf")
    w = ("d0_cq", "d0_ck", "d0_cv", "d0_co")
    got = model.attention(w, xq, xkv, bias)[0]
    p = model.p
    q, k, v = model.heads(xq @ p[w[0]]), model.heads(xkv @ p[w[1]]), model.heads(xkv @ p[w[2]])
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(2) + bias, -1)
    want = (a @ v).transpose(1, 2).reshape(2, 4, 6) @ p[w[3]]
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-13)


def test_rms_norm_is_the_formula():
    torch.manual_seed(2)
    x, g = torch.randn(3, 5, 7), torch.randn(7)
    want = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g
    torch.testing.assert_close(F.rms_norm(x, (7,), g, 1e-6), want, rtol=1e-13, atol=1e-14)


def test_buckets_match_the_reference_values():
    got = bucket(torch.arange(-20, 21), True, 8, 16)
    want = [3] * 15 + [2] * 4 + [1, 0, 5] + [6] * 4 + [7] * 15
    assert got.tolist() == want
    far = bucket(-torch.arange(0, 300), False, 32, 128)
    assert far[112] == 30 and far[113] == 31 and bool((far[128:] == 31).all())


def test_gradient():
    # torch.autograd.gradcheck compares autograd with central finite differences on every parameter.
    torch.manual_seed(3)
    cfg = config(vocab=6, d_model=4, n_heads=2, d_kv=2, d_ff=3, n_layers=1, buckets=4, max_distance=6)
    model = T5(cfg)
    src, tgt = torch.randint(0, 6, (2, 4)), torch.randint(0, 6, (2, 3))
    names = list(model.p)

    def f(*ws):
        saved = {n: model.p[n] for n in names}
        for n, w in zip(names, ws, strict=True):
            model.p._parameters[n] = w  # pyright: ignore[reportIndexIssue]
        try:
            return nll(model(src, tgt)[0], tgt)
        finally:
            for n in names:
                model.p._parameters[n] = saved[n]  # pyright: ignore[reportIndexIssue]

    ws = tuple(model.p[n].detach().clone().requires_grad_() for n in names)
    assert torch.autograd.gradcheck(f, ws, eps=1e-6, atol=1e-6)


def test_decoder_is_causal():
    torch.manual_seed(4)
    model = T5(config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2))
    src, tgt = torch.randint(0, 9, (3, 6)), torch.randint(0, 9, (3, 7))
    with torch.no_grad():
        base = model(src, tgt)[0]
        for t in range(7):
            other = tgt.clone()
            other[:, t] = (other[:, t] + 1) % 9
            assert torch.equal(model(src, other)[0][:, : t + 1], base[:, : t + 1])


def test_without_positions_the_encoder_sees_a_set():
    torch.manual_seed(5)
    for pos, equivariant in (("none", True), ("relative", False)):
        model = T5(config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2, pos=pos))
        src = torch.randint(0, 9, (2, 7))
        perm = torch.randperm(7)
        with torch.no_grad():
            same = torch.allclose(model.encode(src[:, perm]), model.encode(src)[:, perm], rtol=1e-12, atol=1e-13)
        assert same == equivariant


def test_cache_matches_the_whole_prefix():
    torch.manual_seed(6)
    model = T5(config(vocab=9, d_model=8, n_heads=4, d_kv=2, d_ff=12, n_layers=3))
    src, tgt = torch.randint(0, 9, (2, 6)), torch.randint(0, 9, (2, 9))
    with torch.no_grad():
        e = model.encode(src)
        ids = shift_right(tgt)
        full = model.decode(e, ids)[0]
        pieces, kv = [], None
        for t in range(9):
            logits, kv = model.decode(e, ids[:, t : t + 1], kv, start=t)
            pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-12, atol=1e-13)


def test_bad_inputs():
    with pytest.raises(ValueError):
        config(buckets=8, max_distance=4)
    model = T5(config())
    with pytest.raises(ValueError):
        model(torch.tensor([[0, 13]]), torch.tensor([[1]]))


def test_learns():
    # Adam fits 8 copies of 5 source tokens in 150 steps, the task of the NumPy test_learns. A wrong sign or a gradient
    # that does not reach a weight leaves the loss far above the bound of 0.05; a uniform guess among the 6 source
    # tokens is log 6 = 1.79. The decoder cannot copy its own input, the target shifted right, so it must read the
    # encoder.
    cfg = config(vocab=8, d_model=16, n_heads=2, d_kv=8, d_ff=32, n_layers=1)
    model = T5(cfg, generator=torch.Generator().manual_seed(0))
    src = torch.as_tensor(np.random.default_rng(1).integers(2, 8, (8, 5)))
    tgt = src.clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        loss = nll(model(src, tgt)[0], tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 1.0
    assert losses[-1] < 0.05, losses[-1]
