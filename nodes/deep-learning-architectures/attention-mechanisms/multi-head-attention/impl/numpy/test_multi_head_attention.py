# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Tests of the NumPy reference: the worked example, the masks, the loop over heads, the invariances,
the gradient against finite differences, the counts, and the shared fixture."""

import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from multi_head_attention import (
    attention,
    backward,
    causal_mask,
    config,
    flops,
    forward,
    init,
    padding_mask,
    param_count,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())

# The worked example of the page: three tokens in R^2, identity projections.
X_EX = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # (B, T, M)
I2 = {"W_q": np.eye(2), "W_k": np.eye(2), "W_v": np.eye(2), "W_o": np.eye(2)}
E = np.e


def random_case(seed, B=2, T=5, S=None, M=8, H=2):
    rng = np.random.default_rng(seed)
    cfg = config(M, H)
    params = init(rng, cfg)
    x = rng.normal(size=(B, T, M))
    mem = None if S is None else rng.normal(size=(B, S, M))
    return cfg, params, x, mem, rng


def test_worked_example_two_heads():
    y, cache = forward(I2, X_EX, config(2, 2))
    big, small = E / (2 * E + 1), 1 / (2 * E + 1)
    np.testing.assert_allclose(cache["a"][0, 0, 0], [big, small, big], rtol=1e-12)
    np.testing.assert_allclose(cache["a"][0, 1, 0], [1 / 3, 1 / 3, 1 / 3], rtol=1e-12)
    np.testing.assert_allclose(y[0, 0], [2 * big, 2 / 3], rtol=1e-12)
    np.testing.assert_allclose(np.round(y[0, 0], 4), [0.8446, 0.6667])


def test_worked_example_one_head():
    y, cache = forward(I2, X_EX, config(2, 1))
    w = np.exp([1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)])
    np.testing.assert_allclose(cache["a"][0, 0, 0], w / w.sum(), rtol=1e-12)
    np.testing.assert_allclose(np.round(cache["a"][0, 0, 0], 4), [0.4011, 0.1978, 0.4011])
    np.testing.assert_allclose(np.round(y[0, 0], 4), [0.8022, 0.5989])


def test_worked_example_causal():
    y, cache = forward(I2, X_EX, config(2, 2), mask=causal_mask(3))
    np.testing.assert_allclose(y[0, 0], [1.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(cache["a"][0, 1, 1, :2], [1 / (1 + E), E / (1 + E)], rtol=1e-12)
    np.testing.assert_allclose(np.round(y[0, 1], 4), [0.5, 0.7311])
    np.testing.assert_allclose(y[0, 2], forward(I2, X_EX, config(2, 2))[0][0, 2], rtol=1e-12)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10**6), st.integers(1, 6), st.integers(1, 6))
def test_rows_sum_to_one(seed, T, S):
    cfg, params, x, mem, _ = random_case(seed, T=T, S=S)
    _, cache = forward(params, x, cfg, memory=mem)
    a = cache["a"]
    assert a.shape == (2, 2, T, S)
    assert np.all(a >= 0)
    np.testing.assert_allclose(a.sum(axis=-1), 1.0, rtol=0, atol=1e-12)


def test_causal_mask_gives_zero_weight_to_the_future():
    cfg, params, x, _, rng = random_case(1, T=6)
    y, cache = forward(params, x, cfg, mask=causal_mask(6))
    future = ~causal_mask(6)
    assert np.all(cache["a"][..., future] == 0.0)
    np.testing.assert_allclose(cache["a"].sum(axis=-1), 1.0, atol=1e-12)
    x2 = x.copy()
    x2[:, 4:] = rng.normal(size=x2[:, 4:].shape)  # change the last two positions only
    y2, _ = forward(params, x2, cfg, mask=causal_mask(6))
    np.testing.assert_allclose(y2[:, :4], y[:, :4], rtol=1e-12, atol=1e-12)
    assert not np.allclose(y2[:, 4:], y[:, 4:])


def test_causal_mask_with_a_cache_offset():
    m = causal_mask(2, 5)  # two new queries after three cached keys
    np.testing.assert_array_equal(m, [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])


def test_padding_mask_matches_the_short_sequence():
    cfg, params, x, mem, _ = random_case(2, T=3, S=5)
    lengths = [5, 3]
    y, cache = forward(params, x, cfg, memory=mem, mask=padding_mask(lengths, 5))
    assert np.all(cache["a"][1, :, :, 3:] == 0.0)
    y_short, _ = forward(params, x[1:], cfg, memory=mem[1:, :3])
    np.testing.assert_allclose(y[1], y_short[0], rtol=1e-12, atol=1e-12)


def test_all_masked_row_gives_zero():
    rng = np.random.default_rng(3)
    q, k, v = rng.normal(size=(1, 1, 2, 4)), rng.normal(size=(1, 1, 3, 4)), rng.normal(size=(1, 1, 3, 4))
    mask = np.array([[True, False, True], [False, False, False]])
    y, a = attention(q, k, v, mask)
    np.testing.assert_array_equal(a[0, 0, 1], 0.0)
    np.testing.assert_array_equal(y[0, 0, 1], 0.0)
    assert np.all(np.isfinite(y))


def loop_reference(params, x, mem, cfg, mask=None):
    """Multi-head attention with explicit loops over the batch, the heads, the queries and the keys."""
    H, Dk, Dv, M = cfg["n_heads"], cfg["d_k"], cfg["d_v"], cfg["d_model"]
    B, T, S = x.shape[0], x.shape[1], mem.shape[1]
    y = np.zeros((B, T, M))
    for b in range(B):
        for r in range(H):
            Wq = params["W_q"][:, r * Dk : (r + 1) * Dk]
            Wk = params["W_k"][:, r * Dk : (r + 1) * Dk]
            Wv = params["W_v"][:, r * Dv : (r + 1) * Dv]
            Wo = params["W_o"][r * Dv : (r + 1) * Dv, :]
            for t in range(T):
                q = x[b, t] @ Wq
                keep = [s for s in range(S) if mask is None or mask[t, s]]
                scores = [float(q @ (mem[b, s] @ Wk)) / np.sqrt(Dk) for s in keep]
                top = max(scores)
                w = [np.exp(z - top) for z in scores]
                head = sum(wi / sum(w) * (mem[b, s] @ Wv) for wi, s in zip(w, keep, strict=True))
                y[b, t] += head @ Wo  # the concatenation times W^O is a sum over the heads
    return y


@pytest.mark.parametrize(("M", "H"), [(8, 1), (8, 2), (12, 3), (8, 8)])
def test_heads_match_a_direct_loop(M, H):
    cfg, params, x, mem, _ = random_case(4, T=4, S=3, M=M, H=H)
    assert cfg["d_k"] == M // H
    y, _ = forward(params, x, cfg, memory=mem)
    np.testing.assert_allclose(y, loop_reference(params, x, mem, cfg), rtol=1e-10, atol=1e-12)
    ys, _ = forward(params, x, cfg, mask=causal_mask(4))
    np.testing.assert_allclose(ys, loop_reference(params, x, x, cfg, causal_mask(4)), rtol=1e-10, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6), st.permutations(range(5)))
def test_self_attention_is_permutation_equivariant(seed, perm):
    cfg, params, x, _, _ = random_case(seed, T=5)
    perm = np.array(perm)
    y, _ = forward(params, x, cfg)
    y_perm, _ = forward(params, x[:, perm], cfg)
    np.testing.assert_allclose(y_perm, y[:, perm], rtol=1e-10, atol=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10**6), st.permutations(range(4)))
def test_cross_attention_ignores_the_order_of_the_memory(seed, perm):
    cfg, params, x, mem, _ = random_case(seed, T=3, S=4)
    y, _ = forward(params, x, cfg, memory=mem)
    y_perm, _ = forward(params, x, cfg, memory=mem[:, np.array(perm)])
    np.testing.assert_allclose(y_perm, y, rtol=1e-10, atol=1e-12)


def test_large_scores_stay_finite():
    cfg, params, x, _, _ = random_case(5, T=4)
    y, cache = forward(params, 1e4 * x, cfg)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(cache["a"].sum(axis=-1), 1.0, atol=1e-12)


def test_scaled_scores_have_unit_variance():
    rng = np.random.default_rng(6)
    for d_k in (4, 64, 512):
        q, k = rng.normal(size=(20000, d_k)), rng.normal(size=(20000, d_k))
        dots = np.sum(q * k, axis=1)
        assert abs(dots.var() / d_k - 1.0) < 0.05
        assert abs((dots / np.sqrt(d_k)).var() - 1.0) < 0.05


def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        up = f()
        x[i] = old - h
        down = f()
        x[i] = old
        g[i] = (up - down) / (2 * h)
    return g


@pytest.mark.parametrize("cross", [False, True])
def test_gradient(cross):
    cfg, params, x, mem, rng = random_case(7, B=2, T=3, S=4 if cross else None, M=4, H=2)
    mask = padding_mask([4, 2], 4) if cross else causal_mask(3)
    dy = rng.normal(size=(2, 3, 4))

    def loss():
        return float(np.sum(forward(params, x, cfg, memory=mem, mask=mask)[0] * dy))

    _, cache = forward(params, x, cfg, memory=mem, mask=mask)
    dx, dmem, dparams = backward(params, cache, dy)
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=1e-6, atol=1e-8)
    if cross:
        np.testing.assert_allclose(dmem, numeric_grad(loss, mem), rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(dmem[1, 2:], 0.0)  # padded positions get no gradient
    else:
        assert dmem is None
    for name in ("W_q", "W_k", "W_v", "W_o"):
        np.testing.assert_allclose(dparams[name], numeric_grad(loss, params[name]), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(("M", "H", "d_k", "d_v"), [(8, 2, None, None), (6, 3, 5, 7), (512, 8, None, None)])
def test_counts(M, H, d_k, d_v):
    cfg = config(M, H, d_k, d_v)
    params = init(np.random.default_rng(0), cfg)
    assert param_count(cfg) == sum(p.size for p in params.values())
    if d_k is None:
        assert param_count(cfg) == 4 * M * M
        assert flops(cfg, 2, 7) == 8 * 2 * 7 * M * M + 4 * 2 * 7 * 7 * M
    B, T, S = 2, 3, 5
    Dk, Dv = cfg["d_k"], cfg["d_v"]
    by_hand = 2 * B * (T * M * H * Dk + S * M * H * Dk + S * M * H * Dv + H * T * S * Dk + H * T * S * Dv + T * H * Dv * M)
    assert flops(cfg, B, T, S) == by_hand


def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        config(10, 4)
    assert config(10, 4, d_k=3, d_v=2)["d_v"] == 2


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = np.load(FIXTURES / "input-small.npz")
    want = np.load(FIXTURES / "expected-small.npz")
    cfg = config(int(inp["d_model"]), int(inp["n_heads"]))
    y, cache = forward(p, inp["x"], cfg, mask=inp["causal"])
    np.testing.assert_allclose(y, want["self/y"], rtol=1e-12, atol=1e-14)
    dx, _, dp = backward(p, cache, inp["dy"])
    np.testing.assert_allclose(dx, want["self/dx"], rtol=1e-12, atol=1e-14)
    y, cache = forward(p, inp["x"], cfg, memory=inp["memory"], mask=inp["padding"])
    np.testing.assert_allclose(y, want["cross/y"], rtol=1e-12, atol=1e-14)
    dx, dmem, dp = backward(p, cache, inp["dy"])
    np.testing.assert_allclose(dmem, want["cross/dmemory"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dp["W_k"], want["cross/dW_k"], rtol=1e-12, atol=1e-14)


def test_float16_limit_of_the_pitfalls():
    assert float(np.finfo(np.float16).max) == 65504.0
    assert np.isinf(np.float16(-1e9))
