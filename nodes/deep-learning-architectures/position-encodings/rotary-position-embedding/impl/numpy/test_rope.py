# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/rotary-position-embedding/ (ai://0V8E5T).
"""Tests of the NumPy reference: the worked example, the three forms, the relative property, norms and the gradient."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rope import (
    angles,
    backward,
    config,
    forward,
    frequencies,
    init,
    ntk_aware_base,
    rotate_complex,
    rotation_matrix,
    scores,
)

# The worked example of the page: d = 4, base 10000, so theta = (1, 0.01).
Q = np.array([[1.0, 2.0, 0.0, 1.0]])
K = np.array([[2.0, 1.0, 1.0, 0.0]])
Q_AT_1 = [-1.1426, 1.9221, -0.0100, 1.0000]
SCORES = [
    ((0, 0), 4.0),
    ((5, 5), 4.0),
    ((1, 0), -0.3732),
    ((2, 1), -0.3732),
    ((7, 6), -0.3732),
    ((0, 1), 4.6956),
    ((3, 0), -4.4133),
    ((10, 7), -4.4133),
]

# The shared fixture. The PyTorch and JAX tests hold the same numbers.
X = np.array([[0.5, -1.5, 2.0, 0.25, -0.75, 1.0], [1.0, 3.0, -2.5, 0.0, 4.0, -1.0], [-1.25, 0.5, 1.5, -3.0, 2.25, 0.5]])
POS = np.array([0.0, 3.0, 17.0])
BASE = 1000.0  # theta = (1, 0.1, 0.01) for d = 6

vectors = st.lists(st.floats(-10, 10), min_size=8, max_size=8).map(np.array)
positions = st.integers(-5000, 5000)


def test_worked_example():
    cfg = config(4)
    np.testing.assert_allclose(frequencies(cfg), [1.0, 0.01], rtol=1e-15)
    y, _ = forward(init(None, cfg), Q, [1], cfg)
    np.testing.assert_allclose(y[0], Q_AT_1, atol=5e-5)
    for (m, n), s in SCORES:
        assert round(float(scores(Q, K, [m], [n], cfg)[0, 0]), 4) == s


def test_shared_fixture_forms_agree():
    cfg = config(6, base=BASE)
    y, _ = forward({}, X, POS, cfg)
    by_matrix = np.stack([rotation_matrix(m, cfg) @ x for m, x in zip(POS, X, strict=True)])
    np.testing.assert_allclose(y, by_matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y, rotate_complex(X, POS, cfg), rtol=1e-12, atol=1e-12)


def test_fixture_values():
    # Row 1 at position 3: the first pair turns by 3 rad, the second by 3 / 10, the third by 3 / 100.
    cfg = config(6, base=BASE)
    y, _ = forward({}, X, POS, cfg)
    np.testing.assert_array_equal(y[0], X[0])
    for i, a in enumerate([3.0, 0.3, 0.03]):
        x1, x2 = X[1, 2 * i], X[1, 2 * i + 1]
        np.testing.assert_allclose(
            y[1, 2 * i : 2 * i + 2], [x1 * np.cos(a) - x2 * np.sin(a), x1 * np.sin(a) + x2 * np.cos(a)]
        )


def test_position_zero_is_identity():
    cfg = config(8)
    np.testing.assert_array_equal(rotation_matrix(0, cfg), np.eye(8))
    y, _ = forward({}, X[:, :4], [0, 0, 0], config(4))
    np.testing.assert_array_equal(y, X[:, :4])


@settings(max_examples=60, deadline=None)
@given(x=vectors, m=positions)
def test_norm_preserved(x, m):
    y, _ = forward({}, x[None], [m], config(8))
    np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), rtol=1e-12, atol=1e-12)


@settings(max_examples=60, deadline=None)
@given(q=vectors, k=vectors, m=positions, n=positions, t=positions)
def test_score_depends_on_m_minus_n(q, k, m, n, t):
    cfg = config(8)
    a = scores(q[None], k[None], [m], [n], cfg)
    b = scores(q[None], k[None], [m + t], [n + t], cfg)
    scale = 1.0 + np.linalg.norm(q) * np.linalg.norm(k)
    np.testing.assert_allclose(a, b, atol=1e-9 * scale)
    # q^T R_{n-m} k, the form of eq-score.
    np.testing.assert_allclose(a[0, 0], q @ rotation_matrix(n - m, cfg) @ k, atol=1e-9 * scale)


def test_matrix_is_orthogonal_and_composes():
    cfg = config(6, base=BASE)
    r3, r5 = rotation_matrix(3, cfg), rotation_matrix(5, cfg)
    np.testing.assert_allclose(r3.T @ r3, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(r3 @ r5, rotation_matrix(8, cfg), atol=1e-12)
    np.testing.assert_allclose(r3.T @ r5, rotation_matrix(2, cfg), atol=1e-12)


def test_backward_is_transpose_and_matches_finite_differences():
    cfg = config(6, base=BASE)
    dy = np.arange(18.0).reshape(3, 6) / 7 - 1
    y, cache = forward({}, X, POS, cfg)
    dx, dparams = backward({}, cache, dy)
    assert dparams == {}
    expect = np.stack([rotation_matrix(m, cfg).T @ g for m, g in zip(POS, dy, strict=True)])
    np.testing.assert_allclose(dx, expect, rtol=1e-12, atol=1e-12)
    eps = 1e-6
    fd = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        e = np.zeros_like(X)
        e[idx] = eps
        fd[idx] = ((forward({}, X + e, POS, cfg)[0] - forward({}, X - e, POS, cfg)[0]) * dy).sum() / (2 * eps)
    np.testing.assert_allclose(dx, fd, rtol=1e-6, atol=1e-8)


def test_interpolation_is_fractional_positions():
    # Position interpolation with s = 4 at position m is plain RoPE at position m / 4.
    plain, pi = config(6, base=BASE), config(6, base=BASE, scale=4.0)
    np.testing.assert_allclose(forward({}, X, POS, pi)[0], forward({}, X, POS / 4, plain)[0], rtol=1e-12)
    np.testing.assert_allclose(angles(8.0, pi), angles(2.0, plain), rtol=1e-15)


def test_ntk_aware_base():
    d, s, b = 128, 8.0, 10000.0
    new = config(d, base=ntk_aware_base(b, s, d))
    old = config(d, base=b)
    th_new, th_old = frequencies(new), frequencies(old)
    assert th_new[0] == th_old[0] == 1.0
    np.testing.assert_allclose(th_new[-1], th_old[-1] / s, rtol=1e-12)
    # theta'_i / theta_i = s^(-2i / (d - 2)): 1 for the highest frequency, 1 / s for the lowest.
    i = np.arange(d // 2)
    np.testing.assert_allclose(th_new / th_old, s ** (-2.0 * i / (d - 2)), rtol=1e-12)
    with pytest.raises(ValueError):
        ntk_aware_base(b, s, 2)


def test_half_pairing_is_a_permutation():
    # Pairing j with j + d/2 is RoPE on permuted dimensions, so the scores are the same.
    cfg, d = config(6, base=BASE), 6
    perm = np.array([0, 3, 1, 4, 2, 5])
    q, k = X[1], X[2]
    a = scores(q[perm][None], k[perm][None], [9], [2], cfg)
    half = np.concatenate([np.arange(0, d, 2), np.arange(1, d, 2)])
    inv = np.argsort(half)
    rq, _ = forward({}, q[perm][None], [9], cfg)
    rk, _ = forward({}, k[perm][None], [2], cfg)
    np.testing.assert_allclose(rq[0][half][inv], rq[0], rtol=1e-12)
    np.testing.assert_allclose(a[0, 0], rq[0] @ rk[0], rtol=1e-12)


def test_decay_bound():
    # |sum_i h_i e^{i r theta_i}| <= max_i |h_{i+1} - h_i| * sum_j |S_j|, with h_{d/2} = 0.
    rng = np.random.default_rng(5)
    d = 64
    th = frequencies(config(d))
    for r in [0, 1, 7, 50, 300]:
        q, k = rng.normal(size=d), rng.normal(size=d)
        h = np.append((q[0::2] + 1j * q[1::2]) * (k[0::2] - 1j * k[1::2]), 0)
        total = np.sum(h[:-1] * np.exp(1j * r * th))
        assert np.isclose(total.real, scores(q[None], k[None], [r], [0], config(d))[0, 0])
        bound = np.abs(np.diff(h)).max() * np.abs(np.cumsum(np.exp(1j * r * th))).sum()
        assert abs(total) <= bound + 1e-9


@pytest.mark.parametrize(("d", "base", "scale"), [(3, 10000.0, 1.0), (0, 10000.0, 1.0), (4, 1.0, 1.0), (4, 10.0, 0.0)])
def test_bad_config(d, base, scale):
    with pytest.raises(ValueError):
        config(d, base, scale)


def test_wrong_width():
    with pytest.raises(ValueError):
        forward({}, X, POS, config(4))


def test_batch_and_heads_shapes():
    x = np.ones((2, 3, 5, 8))
    y, _ = forward({}, x, np.arange(5), config(8))
    assert y.shape == x.shape
    np.testing.assert_allclose(y[0, 0], y[1, 2])
