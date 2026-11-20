# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Tests of the NumPy reference: shapes and counts, the five models of the paper, the position buckets, what the
position biases change, causality of the decoder, the shared fixture, the gradient of eq-loss, the cache of keys and
values, and span corruption."""

import json
import math
import pathlib

import numpy as np
import pytest

from t5 import (
    backward,
    bucket,
    config,
    decode,
    encode,
    flops,
    forward,
    generate,
    generate_without_cache,
    init,
    loss,
    param_count,
    param_shapes,
    position_bias,
    random_spans_mask,
    rms_norm,
    shift_right,
    span_corrupt,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("vocab", "d_model", "n_heads", "d_kv", "d_ff", "n_layers", "buckets", "max_distance")


def cfg_of(c, **kw):
    return config(*(c["config"][k] for k in KEYS), **kw)


def pair(rng, cfg, B, S, T):
    return rng.integers(0, cfg["vocab"], (B, S)), rng.integers(0, cfg["vocab"], (B, T))


def stack_params(cfg, prefix):
    rel = {"e": "enc_rel", "d": "dec_rel"}[prefix]
    ln = {"e": "enc_ln", "d": "dec_ln"}[prefix]
    return sum(
        math.prod(s) for k, s in param_shapes(cfg).items() if k[0] == prefix and k[1].isdigit() or k in (rel, ln)
    )


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg = cfg_of(c)
    m, k = c["model"], c["config"]
    assert param_count(cfg) == m["params"]
    assert stack_params(cfg, "e") == m["params_encoder"]
    assert stack_params(cfg, "d") == m["params_decoder"]
    assert flops(cfg, k["B"], k["S"], k["T"]) == m["flops_forward"]
    if c["kind"] != "named":
        rng = np.random.default_rng(0)
        src, tgt = pair(rng, cfg, k["B"], k["S"], k["T"])
        logits, cache = forward(init(rng, cfg), cfg, src, tgt)
        blocks = c["blocks"]
        assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
        assert list(cache["e"].shape) == blocks["cross"]["inputs"]["e"]
        enc, dec = cache["enc"]["layers"][0], cache["dec"]["layers"][0]
        for name in ("q", "k", "v", "weights"):
            assert list(enc["attn"][name].shape) == blocks["enc_attn"]["internals"][name]
            assert list(dec["attn"][name].shape) == blocks["dec_self"]["internals"][name]
        assert list(dec["cross"]["weights"].shape) == blocks["cross"]["internals"]["weights"]
        assert list(enc["u"].shape) == blocks["enc_ff"]["internals"]["hidden"]
        assert list(dec["u"].shape) == blocks["dec_ff"]["internals"]["hidden"]


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_the_models_of_the_paper():
    counts = {c["name"]: param_count(cfg_of(c)) for c in SHAPES if c["kind"] == "named"}
    assert counts == {
        "t5_small": 60_441_088,  # "about 60 million"
        "t5_base": 222_805_248,  # "roughly 220 million"
        "t5_large": 737_537_024,  # "around 770 million"
        "t5_3b": 2_851_467_264,  # "around 2.8 billion"
        "t5_11b": 11_307_190_272,  # "about 11 billion"
    }
    # An output layer of its own, V M more, would bring Large to 770 million but Base to 247 million.
    assert round((counts["t5_large"] + 32000 * 1024) / 1e6) == 770
    assert round((counts["t5_base"] + 32000 * 768) / 1e6) == 247
    base = cfg_of(next(c for c in SHAPES if c["name"] == "t5_base"))
    bert_base = 12 * (4 * 768 * 768 + 2 * 768 * 3072)  # the matrices of 12 BERT-BASE layers
    enc = stack_params(base, "e")
    assert enc / bert_base == pytest.approx(1, abs=1e-3)  # the encoder is BERT-BASE-sized
    # p. 18 puts the attention to the encoder at "about 10%" of all the parameters; these counts give 12.7 to 14.2%.
    shares = {}
    for c in SHAPES:
        if c["kind"] == "named":
            cfg = cfg_of(c)
            cross = sum(math.prod(s) for k, s in param_shapes(cfg).items() if k[0] == "d" and k.split("_")[1][0] == "c")
            shares[c["name"]] = cross / param_count(cfg)
    assert round(shares["t5_base"], 3) == 0.127 and round(shares["t5_11b"], 3) == 0.142
    assert all(0.12 < v < 0.15 for k, v in shares.items() if k != "t5_small")


def test_buckets():
    # Bidirectional, 8 buckets: 4 a side, 2 of them exact; key before the query in 0..3, after it in 4..7.
    got = bucket(np.arange(-20, 21), True, 8, 16)
    want = [3] * 15 + [2] * 4 + [1, 0, 5] + [6] * 4 + [7] * 15
    np.testing.assert_array_equal(got, want)
    # Unidirectional, 8 buckets, 4 exact: offsets 0..3, then 4..5, 6..7, 8..11 and 12 on.
    got = bucket(-np.arange(21), False, 8, 16)
    np.testing.assert_array_equal(got, [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6] + [7] * 9)
    np.testing.assert_array_equal(bucket(np.arange(1, 6), False, 8, 16), 0)  # later keys, all masked
    # The paper's 32 buckets reaching 128: every offset from 128 on shares the last bucket, which the log scale
    # in fact reaches at 113, where 16 + 16 log(n / 16) / log 8 first passes 31.
    far = bucket(-np.arange(0, 2000), False, 32, 128)
    assert np.all(far[128:] == 31) and far[112] == 30 and far[113] == 31
    assert len(np.unique(far)) == 32
    two = bucket(np.arange(-2000, 2001), True, 32, 128)
    assert two[2000] == 0 and two[2001] == 17 and 16 not in two  # offset 0 is in the first half; 16 stays empty
    assert len(np.unique(two)) == 31
    assert bucket(-20, False, 32, 128) == 17 and bucket(-10, True, 32, 128) == 8 and bucket(10, True, 32, 128) == 24
    assert np.all(np.diff(bucket(-np.arange(0, 300), False, 32, 128)) >= 0)  # never decreases with distance


def test_rms_norm_has_no_centring():
    g = np.array([1.0, 2.0, 0.5, 1.0])
    x = np.array([[3.0, 3.0, 3.0, 3.0], [1.0, -1.0, 1.0, -1.0]])
    y, _ = rms_norm(x, g)
    want = np.array([[1, 2, 0.5, 1], [1, -2, 0.5, -1]]) / np.sqrt([[1 + 1e-6 / 9], [1 + 1e-6]])
    np.testing.assert_allclose(y, want, rtol=1e-15)  # a constant row is not sent to zero, as a layer norm would


def test_without_positions_the_encoder_sees_a_set():
    rng = np.random.default_rng(1)
    for pos, equivariant in (("none", True), ("relative", False)):
        cfg = config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2, pos=pos)
        p = init(rng, cfg)
        src, tgt = pair(rng, cfg, 2, 7, 4)
        perm = rng.permutation(7)
        e = encode(p, cfg, src)[0]
        e_perm = encode(p, cfg, src[:, perm])[0]
        assert np.allclose(e_perm, e[:, perm], rtol=1e-12, atol=1e-13) == equivariant
        same = np.allclose(forward(p, cfg, src[:, perm], tgt)[0], forward(p, cfg, src, tgt)[0], atol=1e-13)
        assert same == equivariant  # the decoder cannot tell the source order without the encoder's biases


def test_the_bias_depends_on_the_offset_alone():
    rng = np.random.default_rng(2)
    cfg = config(buckets=8, max_distance=16)
    table = rng.standard_normal((8, 2))
    q, k = np.arange(5), np.arange(9)
    for shift in (1, 7, 300):
        for bidirectional in (True, False):
            a, _ = position_bias(table, cfg, q, k, bidirectional)
            b, _ = position_bias(table, cfg, q + shift, k + shift, bidirectional)
            np.testing.assert_array_equal(a, b)
    # The decoder at positions 5 and 6 with a cache uses the rows of a decoder run on all 7 positions.
    p = init(rng, cfg)
    e = encode(p, cfg, rng.integers(0, 13, (1, 4)))[0]
    ids = rng.integers(0, 13, (1, 7))
    _, full = decode(p, cfg, e, ids)
    _, head = decode(p, cfg, e, ids[:, :5])
    _, tail = decode(p, cfg, e, ids[:, 5:], past=head["kv"], start=5)
    np.testing.assert_array_equal(tail["buckets"], full["buckets"][5:])


def test_decoder_is_causal_and_the_target_is_shifted():
    rng = np.random.default_rng(3)
    cfg = config(vocab=9, d_model=8, n_heads=2, d_kv=4, d_ff=12, n_layers=2)
    p = init(rng, cfg)
    src, tgt = pair(rng, cfg, 3, 6, 7)
    np.testing.assert_array_equal(shift_right(tgt)[:, 1:], tgt[:, :-1])
    assert np.all(shift_right(tgt)[:, 0] == 0)
    base = forward(p, cfg, src, tgt)[0]
    for t in range(7):
        other = tgt.copy()
        other[:, t] = (other[:, t] + 1) % 9
        changed = forward(p, cfg, src, other)[0]
        np.testing.assert_array_equal(changed[:, : t + 1], base[:, : t + 1])  # token t is seen from t + 1 on
        if t < 6:
            assert np.abs(changed[:, t + 1 :] - base[:, t + 1 :]).min(axis=-1).max() > 0


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    assert set(params) == set(param_shapes(cfg))
    src, tgt = inp["src"], inp["tgt"]
    logits, cache = forward(params, cfg, src, tgt)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["e"], want["e"], rtol=1e-13, atol=1e-14)
    assert loss(params, cfg, src, tgt) == pytest.approx(float(want["loss"]), rel=1e-13)
    for k, v in backward(params, cfg, cache, logits, tgt).items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-12, atol=1e-14, err_msg=k)
    ids = shift_right(tgt)
    first, c1 = decode(params, cfg, cache["e"], ids[:, :2])
    second, _ = decode(params, cfg, cache["e"], ids[:, 2:], past=c1["kv"], start=2)
    np.testing.assert_allclose(np.concatenate([first, second], 1), want["logits_cached"], rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(generate(params, cfg, src, 6), want["greedy"])


@pytest.mark.parametrize("pos", ["relative", "none"])
def test_gradient_by_finite_differences(pos):
    rng = np.random.default_rng(4)
    cfg = config(vocab=7, d_model=6, n_heads=3, d_kv=2, d_ff=5, n_layers=2, buckets=6, max_distance=9, pos=pos)
    params = init(rng, cfg)
    src, tgt = pair(rng, cfg, 3, 7, 5)
    logits, cache = forward(params, cfg, src, tgt)
    grads = backward(params, cfg, cache, logits, tgt)
    assert set(grads) == set(params)
    eps = 1e-6
    for name, g in grads.items():
        for idx in [tuple(rng.integers(0, s) for s in g.shape) for _ in range(3)]:
            plus, minus = dict(params), dict(params)
            plus[name], minus[name] = params[name].copy(), params[name].copy()
            plus[name][idx] += eps
            minus[name][idx] -= eps
            fd = (loss(plus, cfg, src, tgt) - loss(minus, cfg, src, tgt)) / (2 * eps)
            assert g[idx] == pytest.approx(fd, rel=1e-5, abs=1e-8), name


def test_cache_of_keys_and_values():
    rng = np.random.default_rng(5)
    for pos in ("relative", "none"):
        cfg = config(vocab=9, d_model=8, n_heads=4, d_kv=2, d_ff=12, n_layers=3, pos=pos)
        params = init(rng, cfg)
        src, tgt = pair(rng, cfg, 2, 6, 9)
        e = encode(params, cfg, src)[0]
        ids = shift_right(tgt)
        full = decode(params, cfg, e, ids)[0]
        for cuts in [(1, 2, 3, 4, 5, 6, 7, 8), (4,), (3, 7)]:
            pieces, kv = [], None
            for a, b in zip((0, *cuts), (*cuts, 9), strict=True):
                logits, c = decode(params, cfg, e, ids[:, a:b], past=kv, start=a)
                pieces.append(logits)
                kv = c["kv"]
            np.testing.assert_allclose(np.concatenate(pieces, 1), full, rtol=1e-12, atol=1e-13)
            assert kv is not None and kv[0][0].shape == (2, 4, 9, 2)
        np.testing.assert_array_equal(generate(params, cfg, src, 8), generate_without_cache(params, cfg, src, 8))


def test_span_corruption_of_the_paper():
    words = "Thank you for inviting me to your party last week .".split()
    mask = [w in ("for", "inviting", "last") for w in words]
    inputs, targets = span_corrupt(words, mask, ["<X>", "<Y>", "<Z>"])
    assert " ".join(inputs) == "Thank you <X> me to your party <Y> week ."
    assert " ".join(targets) == "<X> for inviting <Y> last <Z>"


def test_random_spans():
    rng = np.random.default_rng(6)
    for n, density, mean in [(500, 0.15, 3), (512, 0.15, 3), (100, 0.5, 10), (60, 0.25, 2)]:
        for _ in range(5):
            mask = random_spans_mask(n, density, mean, rng)
            noise, spans = round(n * density), round(round(n * density) / mean)
            starts = mask & ~np.concatenate([[False], mask[:-1]])
            assert mask.sum() == noise and starts.sum() == spans
            assert not mask[0]  # the runs alternate, kept tokens first
            inputs, targets = span_corrupt(np.arange(n), mask, list(range(1000, 1000 + spans + 1)))
            assert len(inputs) == n - noise + spans and len(targets) == noise + spans + 1
    assert round(500 * 0.15) == 75 and 75 / 3 == 25  # p. 23: 75 corrupted tokens, 25 spans of mean length 3


def test_bad_configs():
    for kw in [{"buckets": 3}, {"buckets": 8, "max_distance": 4}, {"pos": "absolute"}, {"n_layers": 0}]:
        with pytest.raises(ValueError):
            config(**kw)
    assert config(buckets=8, max_distance=4, pos="none")["pos"] == "none"
    p = init(np.random.default_rng(0), config())
    with pytest.raises(ValueError):
        forward(p, config(), np.array([[0, 13]]), np.array([[1]]))
    with pytest.raises(ValueError):
        span_corrupt(np.arange(6), [1, 0, 1, 0, 1, 0], [100, 101, 102])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    k = c["config"]
    return forward, (init(rng, cfg), cfg, *pair(rng, cfg, k["B"], k["S"], k["T"]))
