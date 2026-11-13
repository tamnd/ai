# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Tests of the NumPy reference: the worked example, the expectation, the gradient and the edge cases."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dropout import backward, config, forward, init, sample_mask

# The worked example of the page.
H = np.array([2.0, 4.0, -1.0, 3.0])
EXAMPLES = [
    (0.5, [1, 0, 1, 1], [4.0, 0.0, -2.0, 6.0], [2.0, 0.0, 2.0, 2.0]),
    (0.2, [1, 1, 0, 1], [2.5, 5.0, 0.0, 3.75], [1.25, 1.25, 0.0, 1.25]),
]

# The shared fixture. The PyTorch and JAX tests hold the same numbers.
X = np.array([[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]])
M = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 1]], dtype=np.float64)
DY = np.array([[1.0, 2.0, -1.0, 0.5, 3.0], [-2.0, 1.0, 0.0, 4.0, -1.0], [0.25, -0.5, 2.0, 1.0, -3.0]])
P = 0.4


@pytest.mark.parametrize(("p", "mask", "y", "dx"), EXAMPLES)
def test_worked_example(p, mask, y, dx):
    cfg = config(p)
    out, cache = forward(init(None, cfg), H, cfg, mask=np.array(mask, dtype=np.float64))
    np.testing.assert_allclose(out, y, rtol=1e-12)
    grad, _ = backward({}, cache, np.ones(4))
    np.testing.assert_allclose(grad, dx, rtol=1e-12)


def test_shared_fixture():
    cfg = config(P)
    y, cache = forward({}, X, cfg, mask=M)
    np.testing.assert_allclose(y, X * M / 0.6, rtol=1e-12)
    dx, dparams = backward({}, cache, DY)
    np.testing.assert_allclose(dx, DY * M / 0.6, rtol=1e-12)
    assert dparams == {}


def test_eval_is_identity():
    cfg = config(0.5)
    rng = np.random.default_rng(0)
    y, cache = forward({}, X, cfg, rng=rng, training=False)
    np.testing.assert_array_equal(y, X)
    dx, _ = backward({}, cache, DY)
    np.testing.assert_array_equal(dx, DY)


def test_masks_differ():
    rng = np.random.default_rng(1)
    a = sample_mask(rng, (64,), 0.5)
    b = sample_mask(rng, (64,), 0.5)
    assert not np.array_equal(a, b)


def test_expectation():
    # The mean of many outputs is the input (c-inverted-scaling). The tolerance is 5 standard errors.
    rng = np.random.default_rng(2)
    p, n = 0.3, 200_000
    x = np.broadcast_to(H, (n, 4))
    y, _ = forward({}, x, config(p), rng=rng)
    se = np.abs(H) * np.sqrt(p / (1 - p) / n)
    assert np.all(np.abs(y.mean(axis=0) - H) <= 5 * se)


def test_variance():
    # The variance of an output is p / (1 - p) times the square of its input (c-noise-variance).
    rng = np.random.default_rng(3)
    p, n = 0.3, 200_000
    y, _ = forward({}, np.broadcast_to(H, (n, 4)), config(p), rng=rng)
    np.testing.assert_allclose(y.var(axis=0), p / (1 - p) * H**2, rtol=0.03)


def test_drop_rate():
    rng = np.random.default_rng(4)
    m = sample_mask(rng, (100_000,), 0.25)
    assert abs((1 - m.mean()) - 0.25) < 0.005


def test_gradient():
    # Central differences of sum(y * DY) for a fixed mask.
    cfg = config(P)
    _, cache = forward({}, X, cfg, mask=M)
    dx, _ = backward({}, cache, DY)
    eps = 1e-6
    num = np.zeros_like(X)
    for i in np.ndindex(X.shape):
        e = np.zeros_like(X)
        e[i] = eps
        up, _ = forward({}, X + e, cfg, mask=M)
        down, _ = forward({}, X - e, cfg, mask=M)
        num[i] = np.sum((up - down) * DY) / (2 * eps)
    np.testing.assert_allclose(dx, num, rtol=1e-8, atol=1e-10)


def test_p_zero_is_identity():
    rng = np.random.default_rng(5)
    y, _ = forward({}, X, config(0.0), rng=rng)
    np.testing.assert_array_equal(y, X)


def test_p_close_to_one():
    rng = np.random.default_rng(6)
    p = 0.999
    y, cache = forward({}, np.ones(10_000), config(p), rng=rng)
    kept = y != 0
    assert 0 < kept.sum() < 50
    np.testing.assert_allclose(y[kept], 1 / (1 - p), rtol=1e-9)
    scale = cache["scale"]
    assert scale is not None
    assert np.all(np.isfinite(scale))


class FixedDraws:
    """A generator whose draws are given, to pin the boundary of the mask."""

    def __init__(self, u):
        self.u = np.asarray(u, dtype=np.float64)

    def random(self, shape):
        return self.u.reshape(shape)


def test_mask_keeps_a_draw_equal_to_p():
    # A unit is kept when its uniform draw u in [0, 1) has u >= p, so it is kept with probability 1 - p.
    m = sample_mask(FixedDraws([0.0, 0.24, 0.25, 0.99]), (4,), 0.25)
    np.testing.assert_array_equal(m, [0.0, 0.0, 1.0, 1.0])
    assert m.dtype == np.float64


def test_config_is_plain_data():
    # The configuration goes into the fixtures as JSON, so p is a Python float, whatever number type came in.
    cfg = config(0)
    assert type(cfg["p"]) is float


def test_lists_and_integers():
    # forward and backward take anything that NumPy can read as an array, and compute in float64.
    y, cache = forward({}, [2, 4, -1, 3], config(0.5), mask=[1, 0, 1, 1])
    assert isinstance(y, np.ndarray) and y.dtype == np.float64
    np.testing.assert_array_equal(y, [4.0, 0.0, -2.0, 6.0])
    dx, _ = backward({}, cache, [1, 1, 1, 1])
    np.testing.assert_array_equal(dx, [2.0, 0.0, 2.0, 2.0])
    y, cache = forward({}, [1, 2], config(0.5), training=False)
    assert isinstance(y, np.ndarray) and y.dtype == np.float64
    dx, _ = backward({}, cache, [3, 4])
    assert isinstance(dx, np.ndarray) and dx.dtype == np.float64
    np.testing.assert_array_equal(dx, [3.0, 4.0])


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_invalid_p(p):
    with pytest.raises(ValueError):
        config(p)


def test_training_needs_rng_or_mask():
    with pytest.raises(ValueError):
        forward({}, X, config(0.5))


def test_mask_shape():
    with pytest.raises(ValueError):
        forward({}, X, config(0.5), mask=np.ones(3))


@settings(max_examples=50, deadline=None)
@given(
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=6),
    st.floats(min_value=0.0, max_value=0.9),
    st.integers(min_value=0, max_value=2**32 - 1),
)
def test_shape_and_values(rows, cols, p, seed):
    # Each output is 0 or the input times 1 / (1 - p), and the shape stays the same.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, cols))
    y, _ = forward({}, x, config(p), rng=rng)
    assert y.shape == x.shape
    ok = np.isclose(y, 0.0) | np.isclose(y, x / (1 - p))
    assert np.all(ok)
