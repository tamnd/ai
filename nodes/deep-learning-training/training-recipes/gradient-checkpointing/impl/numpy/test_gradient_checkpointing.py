# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
"""Tests of the NumPy reference: exact gradients, the memory and compute counts, the optimal k, BPTT-HSM and the
transformer memory formula."""

import math

import numpy as np
import pytest

import gradient_checkpointing as gc

# The shared fixture: three layers of width 2 and a batch of 2. The PyTorch tests check the same numbers.
W = [[[0.5, -0.25], [0.75, 0.5]], [[-0.5, 1.0], [0.25, 0.25]], [[1.0, 0.5], [-0.75, 0.5]]]
B = [[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]]
X = [[1.0, -1.0], [0.5, 2.0]]
Y = [[-0.16063384074179027, 0.5653530619863063], [0.6832818232403335, -0.11798989504262712]]
DX = [[-0.22916355262070015, -0.22191162909668455], [0.01690646129811597, 0.05965517658867096]]
DB = [
    [0.03557438564506993, -0.3067257121934922],
    [-0.1884971146317812, 0.19116142814553794],
    [0.2077863024765885, 0.26830532190736106],
]
DW0 = [[0.0770465591221502, -0.2844074265075516], [-0.345644802043623, 0.5402402512942772]]


@pytest.mark.parametrize(("policy", "k"), [("full", None), ("segments", 1), ("segments", 2), ("recursive", None)])
def test_shared_fixture(policy, k):
    params = {"W": np.array(W), "b": np.array(B)}
    cfg = gc.config(3, 2, policy, k)
    y, cache = gc.forward(params, np.array(X), cfg)
    dx, grads, _ = gc.backward(params, cache, y)
    for got, want in [(y, Y), (dx, DX), (grads["b"], DB), (grads["W"][0], DW0)]:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def run(n, policy, k=None, width=3, seed=0):
    rng = np.random.default_rng(seed)
    cfg = gc.config(n, width, policy, k)
    params = gc.init(rng, cfg)
    x = rng.normal(size=(2, width))  # (B, D)
    y, cache = gc.forward(params, x, cfg)
    dx, grads, meter = gc.backward(params, cache, y)  # the loss is |y|^2 / 2, so dL/dy = y
    return y, dx, grads, meter


def test_worked_example():
    # n = 9 layers, k = 3 segments: checkpoints h_0, h_3, h_6, h_9.
    assert gc.segment_sizes(9, 3) == [3, 3, 3]
    _, _, _, full = run(9, "full")
    _, _, _, seg = run(9, "segments", 3)
    assert (full.peak, full.forwards) == (10, 9)
    assert (seg.peak, seg.forwards) == (6, 15)
    assert (gc.peak_memory(9, 3), gc.recompute_count(9, 3)) == (6, 15)
    # n = 16 layers by bisection: 2 + log2(16) saved, 16 + 8 * 4 forwards.
    _, _, _, rec = run(16, "recursive")
    assert (rec.peak, rec.forwards) == (6, 48)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_gradients_equal_full_storage(n):
    y0, dx0, g0, _ = run(n, "full")
    for policy, k in [("recursive", None)] + [("segments", k) for k in range(1, n + 1)]:
        y, dx, g, meter = run(n, policy, k)
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(dx, dx0)
        np.testing.assert_array_equal(g["W"], g0["W"])
        np.testing.assert_array_equal(g["b"], g0["b"])
        assert meter.live == 0


def test_gradient_finite_differences():
    rng = np.random.default_rng(1)
    cfg = gc.config(5, 3, "segments", 2)
    params = gc.init(rng, cfg)
    x = rng.normal(size=(2, 3))

    def loss(p, xx):
        return 0.5 * float(np.sum(gc.forward(p, xx, cfg)[0] ** 2))

    y, cache = gc.forward(params, x, cfg)
    dx, grads, _ = gc.backward(params, cache, y)
    eps = 1e-6
    for name in ("W", "b"):
        for idx in np.ndindex(params[name].shape):
            hi = {k: v.copy() for k, v in params.items()}
            lo = {k: v.copy() for k, v in params.items()}
            hi[name][idx] += eps
            lo[name][idx] -= eps
            fd = (loss(hi, x) - loss(lo, x)) / (2 * eps)
            assert abs(fd - grads[name][idx]) <= 1e-6 * max(1.0, abs(fd))
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = eps
        fd = (loss(params, x + e) - loss(params, x - e)) / (2 * eps)
        assert abs(fd - dx[idx]) <= 1e-6 * max(1.0, abs(fd))


def test_counts_match_formulas():
    for n in range(1, 26):
        for k in range(1, n + 1):
            _, _, _, meter = run(n, "segments", k, width=2)
            assert meter.peak == gc.peak_memory(n, k) == max(j + s for j, s in enumerate(gc.segment_sizes(n, k), 1))
            assert meter.forwards == gc.recompute_count(n, k)
        _, _, _, full = run(n, "full", width=2)
        assert (full.peak, full.forwards) == (n + 1, n)


def test_segment_sizes():
    assert gc.segment_sizes(10, 3) == [4, 3, 3]
    assert gc.segment_sizes(7, 7) == [1] * 7
    assert gc.segment_sizes(7, 1) == [7]


def test_optimal_k_by_enumeration():
    for q in range(1, 31):
        n = q * q
        mem = [gc.peak_memory(n, k) for k in range(1, n + 1)]
        assert min(mem) == 2 * q == gc.peak_memory(n, q)
        assert min(k + n / k for k in range(1, n + 1)) == 2 * q
    for n in range(1, 301):
        k = gc.optimal_segments(n)
        best = min(gc.peak_memory(n, j) for j in range(1, n + 1))
        assert gc.peak_memory(n, k) == best == math.isqrt(4 * n + 1)
        assert all(gc.peak_memory(n, j) > best for j in range(k + 1, n + 1))
    assert gc.optimal_segments(100) == 13
    assert gc.optimal_segments(1) == 1


def test_recursive_counts():
    for n in range(1, 70):
        _, _, _, meter = run(n, "recursive", width=2)
        assert meter.peak == 2 + math.ceil(math.log2(n))
        assert meter.forwards == gc.recursive_forwards(n)
    for p in range(11):
        n = 2**p
        assert gc.recursive_forwards(n) == n + n * p // 2


def test_hsm_table_boundaries_and_bound():
    C, D = gc.hsm_table(300, 12)
    for t in range(1, 301):
        assert C[t][1] == t * (t + 1) // 2
        for m in range(1, 13):
            if m >= t:
                assert C[t][m] == 2 * t - 1
            assert C[t][m] <= m * t ** (1 + 1 / m) + 1e-9
            if m > 1:
                assert C[t][m] <= C[t][m - 1]
    assert (C[4][4], C[10][4], D[10][4]) == (7, 24, 1)
    assert C[100][2] == min(y + C[100 - y][1] + C[y][2] for y in range(1, 100))


@pytest.mark.parametrize(("t", "m"), [(1, 1), (6, 1), (7, 2), (10, 3), (12, 4), (5, 9)])
def test_hsm_executor(t, m):
    C, D = gc.hsm_table(t, m)
    rng = np.random.default_rng(2)
    cfg = gc.config(t, 3, "full")
    params = gc.init(rng, cfg)
    x = rng.normal(size=(2, 3))
    y0, cache = gc.forward(params, x, cfg)
    dx0, g0, _ = gc.backward(params, cache, y0)
    y, dx, g, meter = gc.bptt_hsm(params, x, lambda h: h, m, D)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(dx, dx0)
    np.testing.assert_array_equal(g["W"], g0["W"])
    np.testing.assert_array_equal(g["b"], g0["b"])
    assert meter.forwards == C[t][m]
    assert meter.peak <= m
    assert meter.live == 1


def test_transformer_layer_bytes():
    for s, b, h, a in [(2048, 1, 12288, 96), (2048, 4, 6144, 64), (2048, 1, 20480, 128), (16, 2, 8, 2)]:
        terms = gc.layer_activation_bytes(s, b, h, a)
        assert sum(terms.values()) * h == s * b * h * (34 * h + 5 * a * s)
        assert sum(terms[k] for k in gc.SELECTIVE) == 5 * a * s * s * b
        assert sum(v for k, v in terms.items() if k not in gc.SELECTIVE) == 34 * s * b * h
    gpt3 = gc.layer_activation_bytes(2048, 1, 12288, 96)
    assert sum(gpt3.values()) == 2048 * 12288 * 114
    assert round(sum(gpt3[k] for k in gc.SELECTIVE) / sum(gpt3.values()), 2) == 0.70
    mtnlg = gc.layer_activation_bytes(2048, 1, 20480, 128)
    assert round(sum(mtnlg[k] for k in gc.SELECTIVE) / sum(mtnlg.values()), 2) == 0.65


def test_config_errors():
    with pytest.raises(ValueError):
        gc.config(4, 2, "none")
    with pytest.raises(ValueError, match="at least 1"):
        gc.config(0, 2)
    with pytest.raises(ValueError, match="at least 1"):
        gc.config(4, 0)
    assert gc.config(1, 1)["segments"] == 1
    with pytest.raises(ValueError):
        gc.config(4, 2, "segments", 5)
    with pytest.raises(ValueError):
        gc.config(4, 2, "segments", 0)
    assert gc.config(9, 2)["segments"] == 5
    assert gc.config() == {"n_layers": 8, "width": 4, "policy": "segments", "segments": gc.optimal_segments(8)}
