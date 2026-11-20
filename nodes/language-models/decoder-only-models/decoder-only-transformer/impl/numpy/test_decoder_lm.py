# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Tests of the NumPy reference: shapes and counts, the models of the papers, causality, the shared fixture, the
gradient of eq-loss, the cache of keys and values, greedy decoding, and what the model can tell about order without
position embeddings."""

import json
import math
import pathlib

import numpy as np
import pytest

from decoder_lm import (
    backward,
    causal_mask,
    config,
    embed,
    flops,
    forward,
    generate,
    generate_without_cache,
    init,
    kv_cache_size,
    loss,
    msa,
    non_embedding_params,
    param_count,
    param_shapes,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["vocab"], k["n_ctx"], k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], **kw)


def ids_of(rng, cfg, B, n=None):
    return rng.integers(0, cfg["vocab"], (B, cfg["n_ctx"] if n is None else n))


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg = cfg_of(c)
    m = c["model"]
    assert param_count(cfg) == m["params"]
    assert sum(math.prod(s) for s in param_shapes(cfg).values()) == m["params"]
    assert non_embedding_params(cfg) == m["params_non_embedding"]
    B = c["config"]["B"]
    assert flops(cfg, B, cfg["n_ctx"]) == m["flops_forward"]
    assert B * kv_cache_size(cfg, cfg["n_ctx"]) == m["kv_cache"]
    if c["kind"] != "named":
        rng = np.random.default_rng(0)
        logits, cache = forward(init(rng, cfg), cfg, ids_of(rng, cfg, B))
        blocks = c["blocks"]
        assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
        assert list(cache["tokens"].shape) == blocks["embed"]["outputs"]["tokens"]
        attn = cache["layers"][0]["attn"]
        for name in ("q", "k", "v", "weights"):
            assert list(attn[name].shape) == blocks["attn"]["internals"][name]
        assert list(cache["layers"][0]["u"].shape) == blocks["mlp"]["internals"]["hidden"]
        assert sum(k.size + v.size for k, v in cache["kv"]) == m["kv_cache"]


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_the_models_of_the_papers():
    named = {c["name"]: cfg_of(c) for c in SHAPES if c["kind"] == "named"}
    counts = {k: param_count(v) for k, v in named.items()}
    assert counts["gpt2_117m"] == 124_439_808  # Table 2 of GPT-2 prints 117M
    assert counts["gpt3_small"] - counts["gpt2_117m"] == (2048 - 1024) * 768
    table_d1 = {"gpt3_small": (125, 1), "gpt3_medium": (356, 1), "gpt3_large": (760, 1), "gpt3_xl": (1320, 10),
                "gpt3_2_7b": (2650, 10), "gpt3_6_7b": (6660, 10), "gpt3_13b": (12850, 10),
                "gpt3_175b": (174600, 100)}  # fmt: skip  # millions and the unit of the last digit, p. 46
    for name, (millions, unit) in table_d1.items():
        assert abs(counts[name] / 1e6 - millions) <= unit / 2, name
    assert counts["gpt3_175b"] == 174_604_259_328
    as_printed = config(50257, 2048, 5140, 20, 4 * 5140, 40)  # the d_model of 5140 printed in Table 2.1
    assert abs(param_count(as_printed) / 1e6 - 12850) > 100
    big = named["gpt3_175b"]
    assert abs(non_embedding_params(big) / (12 * 96 * 12288**2) - 1) < 1e-4  # Kaplan's N = 12 n_layer d_model^2
    assert param_count(config(50257, 1024, 768, 1, 3072, 12)) == counts["gpt2_117m"]  # the heads change no count


def test_flops_per_token_is_about_twice_the_parameters():
    cfg = cfg_of(next(c for c in SHAPES if c["name"] == "gpt3_175b"))
    n, N = cfg["n_ctx"], non_embedding_params(cfg)
    per_token = flops(cfg, 1, n) / n
    attention = 4 * cfg["n_layers"] * n * cfg["d_model"]
    assert per_token == pytest.approx(2 * (N - cfg["n_layers"] * (9 * 12288 + 4 * 12288)) + attention
                                      + 2 * 12288 * 50257)  # fmt: skip
    assert 2.0 < per_token / N < 2.1


def test_causal_mask():
    np.testing.assert_array_equal(causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(causal_mask(2, 3), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])


def test_worked_example():
    cfg = config(vocab=5, n_ctx=4, d_model=4, n_heads=1, d_ff=4, n_layers=1)
    p = init(np.random.default_rng(1), cfg)
    for k in ("l0_qkv_W", "l0_qkv_b"):
        p[k] = np.zeros_like(p[k])  # every score is 0: each position averages the values it may see
    _, c = msa(p, 0, np.ones((1, 4, 4)), 1)
    want = np.array([[1, 0, 0, 0], [1 / 2, 1 / 2, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
    np.testing.assert_allclose(c["weights"][0, 0], want, rtol=1e-15)
    gpt3 = config(50257, 2048, 12288, 96, 49152, 96)
    per_token = kv_cache_size(gpt3, 1)
    assert per_token == 2 * 96 * 12288 == 2_359_296
    assert per_token * 2 / 2**20 == 4.5  # MiB per token at two bytes a number
    assert kv_cache_size(gpt3, 2048) * 2 / 2**30 == 9.0  # GiB for a full context


def test_causality():
    rng = np.random.default_rng(2)
    cfg = config(vocab=9, n_ctx=7, d_model=8, n_heads=2, d_ff=12, n_layers=3)
    params = init(rng, cfg, scale=0.5)
    ids = ids_of(rng, cfg, 3)
    base = forward(params, cfg, ids)[0]
    for t in range(7):
        other = ids.copy()
        other[:, t] = (other[:, t] + 1) % 9
        changed = forward(params, cfg, other)[0]
        np.testing.assert_array_equal(changed[:, :t], base[:, :t])
        assert np.abs(changed[:, t:] - base[:, t:]).min(axis=-1).max() > 0


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    assert set(params) == set(param_shapes(cfg))
    ids = inp["ids"]
    logits, cache = forward(params, cfg, ids)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["tokens"], want["tokens"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["final"], want["final"], rtol=1e-13, atol=1e-14)
    assert loss(params, cfg, ids) == pytest.approx(float(want["loss"]), rel=1e-13)
    for k, v in backward(params, cfg, cache, ids).items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-12, atol=1e-14, err_msg=k)
    first, c1 = forward(params, cfg, ids[:, :5])
    second, _ = forward(params, cfg, ids[:, 5:], past=c1["kv"])
    np.testing.assert_allclose(np.concatenate([first, second], 1), want["logits_cached"], rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(generate(params, cfg, ids[:, :3], 5), want["greedy"])


@pytest.mark.parametrize(
    "kw",
    [{}, {"tied": False}, {"pos": "none"}, {"tied": False, "pos": "none"}],
    ids=["tied", "untied", "no-pos", "both"],
)
def test_gradient_by_finite_differences(kw):
    rng = np.random.default_rng(3)
    cfg = config(vocab=7, n_ctx=6, d_model=6, n_heads=3, d_ff=5, n_layers=2, **kw)
    params = init(rng, cfg, scale=0.5)
    ids = ids_of(rng, cfg, 3, 5)
    grads = backward(params, cfg, forward(params, cfg, ids)[1], ids)
    assert set(grads) == set(params)
    eps = 1e-6
    for name, g in grads.items():
        for idx in [tuple(rng.integers(0, s) for s in g.shape) for _ in range(3)]:
            plus, minus = dict(params), dict(params)
            plus[name], minus[name] = params[name].copy(), params[name].copy()
            plus[name][idx] += eps
            minus[name][idx] -= eps
            fd = (loss(plus, cfg, ids) - loss(minus, cfg, ids)) / (2 * eps)
            assert g[idx] == pytest.approx(fd, rel=1e-5, abs=1e-8), name
    if cfg["pos"] == "learned":
        np.testing.assert_array_equal(grads["wpe"][5:], 0.0)  # positions the sequence does not reach


def test_tied_gradient_is_the_sum_of_both_uses():
    rng = np.random.default_rng(4)
    tied, untied = config(tied=True), config(tied=False)
    params = init(rng, tied, scale=0.5)
    ids = ids_of(rng, tied, 3)
    as_untied = dict(params) | {"head_W": params["wte"].T.copy()}
    g_tied = backward(params, tied, forward(params, tied, ids)[1], ids)
    g_untied = backward(as_untied, untied, forward(as_untied, untied, ids)[1], ids)
    np.testing.assert_allclose(
        forward(params, tied, ids)[0], forward(as_untied, untied, ids)[0], rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(g_tied["wte"], g_untied["wte"] + g_untied["head_W"].T, rtol=1e-12, atol=1e-15)


def test_cache_of_keys_and_values():
    rng = np.random.default_rng(5)
    for kw in [{}, {"pos": "none"}]:
        cfg = config(vocab=9, n_ctx=10, d_model=8, n_heads=4, d_ff=12, n_layers=3, **kw)
        params = init(rng, cfg, scale=0.5)
        ids = ids_of(rng, cfg, 2)
        full = forward(params, cfg, ids)[0]
        for cuts in [(1, 2, 3, 4, 5, 6, 7, 8, 9), (4,), (3, 7)]:
            pieces, kv = [], None
            for a, b in zip((0, *cuts), (*cuts, 10), strict=True):
                logits, cache = forward(params, cfg, ids[:, a:b], past=kv)
                pieces.append(logits)
                kv = cache["kv"]
            np.testing.assert_allclose(np.concatenate(pieces, 1), full, rtol=1e-12, atol=1e-13)
            assert kv is not None and kv[0][0].shape == (2, cfg["n_heads"], 10, 2)
        prompt = ids[:, :3]
        np.testing.assert_array_equal(generate(params, cfg, prompt, 7), generate_without_cache(params, cfg, prompt, 7))


def test_context_limit():
    cfg = config()
    params = init(np.random.default_rng(6), cfg)
    with pytest.raises(ValueError):
        forward(params, cfg, np.zeros((1, 9), np.int64))
    with pytest.raises(ValueError):
        embed(params, cfg, np.zeros((1, 3), np.int64), start=6)
    none = config(pos="none")
    assert forward(init(np.random.default_rng(6), none), none, np.zeros((1, 12), np.int64))[0].shape == (1, 12, 11)


def test_one_layer_without_positions_sees_a_bag_of_earlier_tokens():
    rng = np.random.default_rng(7)
    for _ in range(5):
        one, two = (config(vocab=9, n_ctx=8, d_model=8, n_heads=2, d_ff=12, n_layers=L, pos="none") for L in (1, 2))
        ids = ids_of(rng, one, 3)
        perm = np.concatenate([rng.permutation(7), [7]])  # shuffle every token but the last
        p1 = init(rng, one, scale=0.5)
        np.testing.assert_allclose(forward(p1, one, ids[:, perm])[0][:, -1], forward(p1, one, ids)[0][:, -1],
                                   rtol=1e-12, atol=1e-13)  # fmt: skip
        p2 = init(rng, two, scale=0.5)
        assert np.abs(forward(p2, two, ids[:, perm])[0][:, -1] - forward(p2, two, ids)[0][:, -1]).max() > 1e-3
    learned = config(vocab=9, n_ctx=8, d_model=8, n_heads=2, d_ff=12, n_layers=1)
    p = init(rng, learned, scale=0.5)
    assert np.abs(forward(p, learned, ids[:, perm])[0][:, -1] - forward(p, learned, ids)[0][:, -1]).max() > 1e-3


def test_the_causal_mask_alone_reveals_position():
    rng = np.random.default_rng(8)
    cfg = config(vocab=4, n_ctx=8, d_model=8, n_heads=2, d_ff=12, n_layers=1, pos="none")
    params = init(rng, cfg, scale=0.5)
    same = forward(params, cfg, np.full((1, 6), 2))[0][0]
    np.testing.assert_allclose(same, np.broadcast_to(same[0], same.shape), rtol=1e-12)  # nothing tells them apart
    marked = forward(params, cfg, np.array([[0, 2, 2, 2, 2, 2]]))[0][0, 1:]
    gaps = [np.abs(marked[i] - marked[j]).max() for i in range(5) for j in range(i)]
    assert min(gaps) > 1e-4  # one distinct first token, and every later position differs


def test_residual_scaled_init():
    cfg = config(d_model=64, n_heads=4, d_ff=256, n_layers=8)
    p = init(np.random.default_rng(9), cfg, scale=0.02, residual_scaled=True)
    assert p["l3_out_W"].std() == pytest.approx(0.02 / 4, rel=0.05)
    assert p["l3_fc2_W"].std() == pytest.approx(0.02 / 4, rel=0.05)
    assert p["l3_fc1_W"].std() == pytest.approx(0.02, rel=0.05)


def test_bad_configs():
    for kw in [{"n_heads": 3}, {"vocab": 1}, {"pos": "sinusoidal"}, {"n_layers": 0}]:
        with pytest.raises(ValueError):
            config(**kw)
    with pytest.raises(ValueError):
        forward(init(np.random.default_rng(0), config()), config(), np.array([[0, 11]]))


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), cfg, ids_of(rng, cfg, c["config"]["B"]))
