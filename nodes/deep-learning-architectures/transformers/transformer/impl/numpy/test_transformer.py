# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""Tests of the NumPy reference: the worked example, the shape contract, the shared fixture and the properties."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from transformer import (
    attention,
    config,
    flops,
    forward,
    init,
    layer_norm,
    param_count,
    positional_encoding,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]  # forward runs on the example and the samples


def cfg_of(c):
    k = c["config"]
    return config(k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], k["vocab"])


def unflatten(flat):
    tree = {}
    for path, v in flat.items():
        *head, last = path.split("/")
        node = tree
        for part in head:
            node = node.setdefault(part, {})
        node[last] = v

    def lists(t):
        if not isinstance(t, dict):
            return t
        if t and all(k.isdigit() for k in t):
            return [lists(t[str(i)]) for i in range(len(t))]
        return {k: lists(v) for k, v in t.items()}

    return lists(tree)


def leaves(tree):
    items = tree.values() if isinstance(tree, dict) else tree
    for v in items:
        if isinstance(v, (dict, list)):
            yield from leaves(v)
        else:
            yield v


# The worked example of the page.
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
CAUSAL = np.tril(np.ones((4, 4), dtype=bool))


def test_worked_example():
    out, w = attention(Q, Q, V)
    np.testing.assert_allclose(w[2], [0.2212, 0.2212, 0.4486, 0.1091], atol=5e-5)
    np.testing.assert_allclose(out[2], [0.6698, 0.8879], atol=5e-5)
    np.testing.assert_allclose(w[3], [0.25] * 4, rtol=1e-12)
    np.testing.assert_allclose(out[3], [0.5, 1.0], rtol=1e-12)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-12)


def test_worked_example_causal():
    out, w = attention(Q, Q, V, CAUSAL)
    np.testing.assert_allclose(out[0], [1.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(w[2, :3], [0.2483, 0.2483, 0.5035], atol=5e-5)
    np.testing.assert_allclose(out[2], [0.7517, 0.7517], atol=5e-5)
    assert np.all(w[~CAUSAL] == 0.0)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    rng = np.random.default_rng(0)
    params = init(rng, cfg)
    src = rng.integers(0, k["V"], (k["B"], k["S"]))
    tgt = rng.integers(0, k["V"], (k["B"], k["T"]))
    logits, cache = forward(params, src, tgt, cfg)
    assert list(logits.shape) == blocks["logits"]["outputs"]["logits"]
    assert list(cache["memory"].shape) == blocks["enc_layer"]["outputs"]["y"]
    got = {"enc_attn": cache["enc"][0], "dec_attn": cache["dec"][0]["self"], "cross_attn": cache["dec"][0]["cross"],
           "enc_ffn": cache["enc"][0], "dec_ffn": cache["dec"][0]}
    for block, arrays in got.items():
        for name, shape in blocks[block]["internals"].items():
            assert list(arrays[name].shape) == shape, (block, name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count_and_flops(c):
    cfg, k = cfg_of(c), c["config"]
    assert param_count(cfg) == c["model"]["params"]
    assert flops(cfg, k["B"], k["S"], k["T"]) == c["model"]["flops_forward"]
    if c["kind"] != "named":  # the arrays of the paper's models are too large for a test
        assert sum(a.size for a in leaves(init(np.random.default_rng(0), cfg))) == c["model"]["params"]


def test_shared_fixture():
    params = unflatten(dict(np.load(FIXTURES / "params-small.npz")))
    inp = np.load(FIXTURES / "input-small.npz")
    want = np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits, cache = forward(params, inp["src"], inp["tgt"], cfg, src_pad=inp["src_pad"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache["memory"], want["memory"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cache["dec"][0]["cross"]["weights"], want["dec/0/cross/weights"], rtol=1e-12, atol=1e-14)
    assert np.all(cache["dec"][0]["cross"]["weights"][1, :, :, -2:] == 0.0)  # the padding gets no weight


def small():
    cfg = cfg_of(SHAPES[0])
    rng = np.random.default_rng(1)
    return cfg, init(rng, cfg), rng.integers(0, cfg["vocab"], (2, 6)), rng.integers(0, cfg["vocab"], (2, 5))


def test_decoder_is_causal():
    # c-causal-mask: changing the target token at t changes no logits before t.
    cfg, params, src, tgt = small()
    base, _ = forward(params, src, tgt, cfg)
    for t in range(tgt.shape[1]):
        other = tgt.copy()
        other[:, t] = (other[:, t] + 1) % cfg["vocab"]
        logits, _ = forward(params, src, other, cfg)
        np.testing.assert_allclose(logits[:, :t], base[:, :t], rtol=1e-12, atol=1e-12)
        assert not np.allclose(logits[:, t], base[:, t])


def test_padding_is_ignored():
    cfg, params, src, tgt = small()
    pad = np.zeros(src.shape, dtype=bool)
    pad[:, -2:] = True
    base, _ = forward(params, src, tgt, cfg, src_pad=pad)
    other = src.copy()
    other[:, -2:] = (other[:, -2:] + 7) % cfg["vocab"]
    logits, _ = forward(params, other, tgt, cfg, src_pad=pad)
    np.testing.assert_allclose(logits, base, rtol=1e-12, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_attention_is_permutation_equivariant(seed):
    rng = np.random.default_rng(seed)
    q, k, v = rng.normal(size=(5, 3)), rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
    pq, pk = rng.permutation(5), rng.permutation(7)
    out, _ = attention(q, k, v)
    moved, _ = attention(q[pq], k[pk], v[pk])  # permuting the queries permutes the rows; keys and values together do nothing
    np.testing.assert_allclose(moved, out[pq], rtol=1e-12, atol=1e-12)


def test_attention_is_stable_on_large_scores():
    out, w = attention(Q * 1e4, Q * 1e4, V)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-12)


def test_fully_masked_row_is_zero():
    allowed = np.ones((4, 4), dtype=bool)
    allowed[1] = False
    out, w = attention(Q, Q, V, allowed)
    assert np.all(w[1] == 0.0) and np.all(out[1] == 0.0)
    np.testing.assert_allclose(out[0], attention(Q, Q, V)[0][0], rtol=1e-12)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_pe_offset_is_linear(k):
    # c-pe-linear-offset: PE[pos + k] is PE[pos] times a block rotation that depends only on k.
    M = 16
    pe = positional_encoding(40, M)
    j = np.arange(M // 2)
    a = k / 10000.0 ** (2 * j / M)
    s, co = pe[:-k, 0::2], pe[:-k, 1::2]
    np.testing.assert_allclose(pe[k:, 0::2], s * np.cos(a) + co * np.sin(a), atol=1e-12)
    np.testing.assert_allclose(pe[k:, 1::2], co * np.cos(a) - s * np.sin(a), atol=1e-12)


def test_pe_odd_width():
    pe = positional_encoding(3, 5)
    assert pe.shape == (3, 5)
    np.testing.assert_allclose(pe[0], [0, 1, 0, 1, 0], atol=0)


def test_layer_norm_statistics():
    x = np.random.default_rng(2).normal(3.0, 5.0, (2, 4, 16))
    y = layer_norm({"gamma": np.ones(16), "beta": np.zeros(16)}, x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, rtol=1e-6)


def test_dot_product_variance():
    # c-dot-variance: with unit-variance entries, q . k has variance d_k, and the scale brings it back to 1.
    rng = np.random.default_rng(3)
    d = 64
    q, k = rng.normal(size=(200000, d)), rng.normal(size=(200000, d))
    dots = (q * k).sum(axis=1)
    assert abs(dots.var() / d - 1.0) < 0.02
    assert abs((dots / np.sqrt(d)).var() - 1.0) < 0.02


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        config(d_model=10, n_heads=4)
    with pytest.raises(ValueError):
        config(n_layers=0)
