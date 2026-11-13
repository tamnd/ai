# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradient, and torch's own attention."""

import json
import pathlib

import numpy as np
import pytest
import torch
from transformer import Transformer, attention, config, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], k["vocab"])


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return Transformer.from_params(params, cfg_of(SHAPES[0])), inp, want


def test_shared_fixture():
    model, inp, want = fixture()
    logits, cache = model(inp["src"], inp["tgt"], inp["src_pad"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["memory"], want["memory"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["enc"][0]["weights"], want["enc/0/weights"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["dec"][0]["self"]["weights"], want["dec/0/self/weights"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["dec"][0]["cross"]["weights"], want["dec/0/cross/weights"], rtol=1e-10, atol=1e-12)


def test_worked_example():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
    out, _ = attention(q, q, v)
    torch.testing.assert_close(out[2], torch.tensor([0.6698, 0.8879]), rtol=0, atol=5e-5)
    out, _ = attention(q, q, v, torch.ones(4, 4, dtype=torch.bool).tril())
    torch.testing.assert_close(out[2], torch.tensor([0.7517, 0.7517]), rtol=0, atol=5e-5)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    torch.manual_seed(0)
    model = Transformer(cfg)
    src = torch.randint(0, k["V"], (k["B"], k["S"]))
    tgt = torch.randint(0, k["V"], (k["B"], k["T"]))
    logits, cache = model(src, tgt)
    assert list(logits.shape) == blocks["logits"]["outputs"]["logits"]
    got = {"enc_attn": cache["enc"][0], "dec_attn": cache["dec"][0]["self"], "cross_attn": cache["dec"][0]["cross"],
           "enc_ffn": cache["enc"][0], "dec_ffn": cache["dec"][0]}
    for block, arrays in got.items():
        for name, shape in blocks[block]["internals"].items():
            assert list(arrays[name].shape) == shape, (block, name)
    assert sum(p.numel() for p in model.parameters()) == c["model"]["params"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    assert param_count(cfg_of(c)) == c["model"]["params"]


def test_gradient():
    # Autograd against central finite differences, on one small config and a few parameters of each kind.
    cfg = config(d_model=4, n_heads=2, d_ff=6, n_layers=1, vocab=7)
    torch.manual_seed(1)
    model = Transformer(cfg)
    src, tgt = torch.randint(0, 7, (1, 3)), torch.randint(0, 7, (1, 2))
    pad = torch.tensor([[False, False, True]])
    w = torch.randn(1, 2, 7)
    names = ["embed", "enc.0.attn.W_q", "dec.0.cross.W_k", "dec.0.ffn.b_1", "dec.0.norm2.gamma"]
    params = dict(model.named_parameters())

    def loss():
        return (model(src, tgt, pad)[0] * w).sum()

    for name in names:
        p = params[name]
        model.zero_grad()
        loss().backward()
        grad = p.grad.clone()
        for i in range(min(p.numel(), 6)):
            flat = p.data.view(-1)
            old = flat[i].item()
            h = 1e-6 * max(1.0, abs(old))
            flat[i] = old + h
            up = loss().item()
            flat[i] = old - h
            down = loss().item()
            flat[i] = old
            assert abs((up - down) / (2 * h) - grad.view(-1)[i].item()) < 1e-6 * max(1.0, abs(grad.view(-1)[i].item()))


def test_agrees_with_scaled_dot_product_attention():
    # The second oracle (8.7.3): torch's own attention on random inputs, with and without a mask.
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(2, 3, 5, 4, generator=g), torch.randn(2, 3, 7, 4, generator=g),
               torch.randn(2, 3, 7, 4, generator=g))
    allowed = torch.rand(2, 1, 5, 7, generator=g) > 0.3
    allowed[..., 0] = True  # every query sees a key, which sdpa needs to avoid NaN
    sdpa = torch.nn.functional.scaled_dot_product_attention
    torch.testing.assert_close(attention(q, k, v)[0], sdpa(q, k, v), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(attention(q, k, v, allowed)[0], sdpa(q, k, v, attn_mask=allowed), rtol=1e-10, atol=1e-12)


def test_agrees_with_multihead_attention():
    # torch.nn.MultiheadAttention computes x W^T; the reference computes x W, so the weights go in transposed.
    model = fixture()[0]
    layer = model.dec[0].attn
    M, H = 16, 4
    mha = torch.nn.MultiheadAttention(M, H, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([layer.W_q.T, layer.W_k.T, layer.W_v.T]))
        mha.in_proj_bias.copy_(torch.cat([layer.b_q, layer.b_k, layer.b_v]))
        mha.out_proj.weight.copy_(layer.W_o.T)
        mha.out_proj.bias.copy_(layer.b_o)
    x = torch.randn(2, 5, M, generator=torch.Generator().manual_seed(3))
    blocked = torch.ones(5, 5, dtype=torch.bool).triu(1)  # True is a position that may not be attended to
    want, _ = mha(x, x, x, attn_mask=blocked, need_weights=False)
    got, _ = layer(x, x, ~blocked)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


def test_agrees_with_layer_norm():
    model, _, _ = fixture()
    norm = model.enc[1].norm2
    x = torch.randn(2, 6, 16, generator=torch.Generator().manual_seed(4)) * 3 + 1
    want = torch.nn.functional.layer_norm(x, (16,), norm.gamma, norm.beta, eps=1e-6)
    torch.testing.assert_close(norm(x), want, rtol=1e-10, atol=1e-12)


def test_fully_masked_row_is_zero():
    q = torch.randn(4, 2)
    allowed = torch.ones(4, 4, dtype=torch.bool)
    allowed[1] = False
    out, w = attention(q, q, q, allowed)
    assert torch.all(out[1] == 0) and torch.all(w[1] == 0) and torch.all(torch.isfinite(out))
