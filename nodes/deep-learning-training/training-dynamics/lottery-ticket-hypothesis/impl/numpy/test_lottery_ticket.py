# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
"""Tests of the NumPy reference: the schedule, the masks on a fixture worked by hand, one round of IMP, and the
error barrier."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lottery_ticket import (
    error_barrier,
    imp_round,
    interpolate,
    magnitude_masks,
    remaining_fraction,
    rounds_to_reach,
    sparsity,
)

# The shared fixture. The PyTorch tests hold the same numbers.
W1 = np.array([[0.5, -0.1, 0.9], [-0.3, 0.7, 0.2]])
W2 = np.array([[0.05, -0.8], [0.6, -0.15], [0.4, 0.25]])
ONES = [np.ones((2, 3)), np.ones((3, 2))]
# Layerwise at 0.5: each layer drops its 3 smallest. W1 drops 0.1, 0.2, 0.3; W2 drops 0.05, 0.15, 0.25.
LAYER_50 = [np.array([[1, 0, 1], [0, 1, 0]]), np.array([[0, 1], [1, 0], [1, 0]])]
# Layerwise at 0.25: each layer drops round(1.5) = 2. W1 drops 0.1, 0.2; W2 drops 0.05, 0.15.
LAYER_25 = [np.array([[1, 0, 1], [1, 1, 0]]), np.array([[0, 1], [1, 0], [1, 1]])]
# Global at 0.25: the 12 weights drop their 3 smallest, 0.05, 0.1 and 0.15. W1 keeps 0.2.
GLOBAL_25 = [np.array([[1, 0, 1], [1, 1, 1]]), np.array([[0, 1], [1, 0], [1, 1]])]
# Layerwise at 0.5 for W1 and 0.25 for W2, as when the output layer is pruned at half the rate.
MIXED = [LAYER_50[0], LAYER_25[1]]


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_remaining_fraction():
    assert remaining_fraction(0.2, 0) == 1.0
    assert remaining_fraction(0.2, 1) == pytest.approx(0.8, abs=1e-15)
    assert remaining_fraction(0.2, 11) == pytest.approx(0.8**11, rel=1e-12)
    assert remaining_fraction(0.2, 11) == pytest.approx(0.0858993, rel=1e-6)
    assert remaining_fraction(0.0, 5) == 1.0
    assert remaining_fraction(0.5, 3) == 0.125


@pytest.mark.parametrize(("rate", "rounds"), [(-0.1, 1), (1.0, 1), (0.2, -1)])
def test_remaining_fraction_rejects(rate, rounds):
    with pytest.raises(ValueError):
        remaining_fraction(rate, rounds)


@pytest.mark.parametrize(
    ("rate", "target", "n"),
    [(0.2, 0.1, 11), (0.2, 0.036, 15), (0.5, 0.25, 2), (0.5, 0.26, 2), (0.5, 0.24, 3), (0.2, 1.0, 0), (0.2, 0.8, 1)],
)
def test_rounds_to_reach(rate, target, n):
    assert rounds_to_reach(rate, target) == n


@pytest.mark.parametrize(("rate", "target"), [(0.0, 0.5), (1.0, 0.5), (0.2, 0.0), (0.2, 1.5)])
def test_rounds_to_reach_rejects(rate, target):
    with pytest.raises(ValueError):
        rounds_to_reach(rate, target)


@given(st.floats(0.01, 0.9), st.floats(1e-4, 1.0))
@settings(max_examples=200, deadline=None)
def test_rounds_is_least(rate, target):
    n = rounds_to_reach(rate, target)
    assert remaining_fraction(rate, n) <= target
    assert n == 0 or remaining_fraction(rate, n - 1) > target
    # The closed form of the derivation agrees away from the boundary.
    exact = math.log(target) / math.log(1 - rate)
    if abs(exact - round(exact)) > 1e-9:
        assert n == max(0, math.ceil(exact))


@pytest.mark.parametrize(
    ("rate", "scope", "want"),
    [(0.5, "layer", LAYER_50), (0.25, "layer", LAYER_25), (0.25, "global", GLOBAL_25), ([0.5, 0.25], "layer", MIXED)],
)
def test_masks_fixture(rate, scope, want):
    got = magnitude_masks([W1, W2], ONES, rate, scope)
    same(got, want)
    assert all(g.dtype == np.float64 for g in got)


def test_global_differs_from_layer():
    # W1 keeps its 0.2 under global pruning because the smallest weights sit in W2.
    a = magnitude_masks([W1, W2], ONES, 0.25, "global")
    b = magnitude_masks([W1, W2], ONES, 0.25, "layer")
    assert a[0][1, 2] == 1 and b[0][1, 2] == 0


def test_global_count():
    got = magnitude_masks([W1, W2], ONES, 0.5, "global")
    assert sum(int(m.sum()) for m in got) == 6
    same(got, LAYER_50)


def test_pruned_stay_pruned():
    # The trained weights grow where the mask is 0. Those entries stay 0, and the rate applies to the alive ones.
    trained = [W1 + 10.0 * (1 - LAYER_50[0]), W2 + 10.0 * (1 - LAYER_50[1])]
    got = magnitude_masks(trained, LAYER_50, 1 / 3, "layer")
    # W1 alive: 0.5, 0.9, 0.7. One goes: 0.5. W2 alive: 0.8, 0.6, 0.4. One goes: 0.4.
    same(got, [np.array([[0, 0, 1], [0, 1, 0]]), np.array([[0, 1], [1, 0], [0, 0]])])
    got = magnitude_masks(trained, LAYER_50, 1 / 3, "global")
    # Six alive, two go: 0.4 and 0.5.
    same(got, [np.array([[0, 0, 1], [0, 1, 0]]), np.array([[0, 1], [1, 0], [0, 0]])])


def test_rounding_half_up():
    # rate * alive is rounded half up: 0.48 -> 0, 0.5 -> 1, 1.5 -> 2.
    w = [np.array([3.0, 1.0])]
    same(magnitude_masks(w, [np.ones(2)], 0.24, "layer"), [np.array([1.0, 1.0])])
    same(magnitude_masks(w, [np.ones(2)], 0.25, "layer"), [np.array([1.0, 0.0])])
    same(magnitude_masks(w, [np.ones(2)], 0.74, "layer"), [np.array([1.0, 0.0])])
    same(magnitude_masks(w, [np.ones(2)], 0.75, "layer"), [np.array([0.0, 0.0])])
    same(magnitude_masks(w, [np.ones(2)], 0.0, "layer"), [np.array([1.0, 1.0])])


def test_ties_by_position():
    w = [np.array([2.0, -2.0, 2.0, 2.0])]
    same(magnitude_masks(w, [np.ones(4)], 0.5, "layer"), [np.array([0.0, 0.0, 1.0, 1.0])])


def test_sign_ignored():
    w = [np.array([-5.0, 1.0, -0.5, 4.0])]
    same(magnitude_masks(w, [np.ones(4)], 0.5, "layer"), [np.array([1.0, 0.0, 0.0, 1.0])])


def test_masks_reject():
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES[:1], 0.2)
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], [np.ones((3, 2)), np.ones((2, 3))], 0.2)
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES, [0.2], "layer")
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES, [0.2, 0.2], "global")
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES, 0.2, "row")
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES, 1.0, "layer")
    with pytest.raises(ValueError):
        magnitude_masks([W1, W2], ONES, -0.1, "global")


def test_imp_round_reset():
    theta0 = [np.full((2, 3), 0.5), np.arange(6.0).reshape(3, 2)]
    masks, start = imp_round([W1, W2], theta0, ONES, 0.5)
    same(masks, LAYER_50)
    same(start, [0.5 * LAYER_50[0], np.arange(6.0).reshape(3, 2) * LAYER_50[1]])


def test_imp_round_global_rewind():
    theta_k = [-W1, 2.0 * W2]
    masks, start = imp_round([W1, W2], theta_k, ONES, 0.25, "global")
    same(masks, GLOBAL_25)
    same(start, [-W1 * GLOBAL_25[0], 2.0 * W2 * GLOBAL_25[1]])


def test_sparsity():
    assert sparsity(ONES) == 1.0
    assert sparsity(LAYER_50) == 0.5
    assert sparsity(GLOBAL_25) == 0.75
    assert sparsity([np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0])]) == pytest.approx(1 / 6)


@given(st.integers(0, 2**31 - 1), st.floats(0.05, 0.9), st.sampled_from(["layer", "global"]))
@settings(max_examples=60, deadline=None)
def test_mask_properties(seed, rate, scope):
    rng = np.random.default_rng(seed)
    ws = [rng.standard_normal((4, 5)), rng.standard_normal((5, 3))]
    ms = [(rng.random(w.shape) < 0.8).astype(float) for w in ws]
    got = magnitude_masks(ws, ms, rate, scope)
    for g, m in zip(got, ms, strict=True):
        assert np.all(g <= m)  # pruned stays pruned
    if scope == "global":
        alive = sum(int(m.sum()) for m in ms)
        assert sum(int(g.sum()) for g in got) == alive - math.floor(rate * alive + 0.5)
        kept = np.concatenate([np.abs(w)[g > 0] for w, g in zip(ws, got, strict=True)])
        gone = np.concatenate([np.abs(w)[(m > 0) & (g == 0)] for w, g, m in zip(ws, got, ms, strict=True)])
        if kept.size and gone.size:
            assert gone.max() <= kept.min()
    else:
        for g, m in zip(got, ms, strict=True):
            alive = int(m.sum())
            assert int(g.sum()) == alive - math.floor(rate * alive + 0.5)


def test_interpolate():
    a, b = [np.array([1.0, 2.0])], [np.array([5.0, -2.0])]
    np.testing.assert_allclose(interpolate(a, b, 0.25)[0], [4.0, -1.0], rtol=1e-15)
    np.testing.assert_array_equal(interpolate(a, b, 1.0)[0], a[0])
    np.testing.assert_array_equal(interpolate(a, b, 0.0)[0], b[0])


def test_barrier_concave():
    # E(w) = 1 - w^2 between w = 1 and w = -1: 0 at both ends and 1 at the middle.
    def err(ws):
        return 1.0 - float(ws[0][0]) ** 2

    h, alphas, errors = error_barrier(err, [np.array([1.0])], [np.array([-1.0])], num=3)
    assert h == 1.0
    np.testing.assert_array_equal(alphas, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(errors, [0.0, 1.0, 0.0])
    # With 30 points the middle is missed; the nearest alphas are 14/29 and 15/29, at w = -1/29 and 1/29.
    h, alphas, errors = error_barrier(err, [np.array([1.0])], [np.array([-1.0])])
    assert len(alphas) == 30 and len(errors) == 30
    assert h == pytest.approx(1.0 - 1.0 / 841.0, rel=1e-12)


def test_barrier_uses_mean_of_ends():
    # E(w) = w is linear: the path has no bump, but the barrier is measured from the mean of the ends.
    def err(ws):
        return float(ws[0][0])

    h, _, errors = error_barrier(err, [np.array([3.0])], [np.array([1.0])], num=5)
    assert h == pytest.approx(1.0, abs=1e-15)
    assert errors[0] == 1.0 and errors[-1] == 3.0


def test_barrier_flat_is_zero():
    def err(ws):
        return float(np.sum(ws[0] ** 2))

    h, _, _ = error_barrier(err, [np.array([1.0, 0.0])], [np.array([-1.0, 0.0])], num=11)
    assert h == 0.0


def test_barrier_rejects():
    with pytest.raises(ValueError):
        error_barrier(lambda ws: 0.0, [np.zeros(1)], [np.zeros(1)], num=1)


@given(st.integers(0, 2**31 - 1))
@settings(max_examples=30, deadline=None)
def test_barrier_nonnegative(seed):
    rng = np.random.default_rng(seed)
    a, b = [rng.standard_normal(3)], [rng.standard_normal(3)]
    c = rng.standard_normal(3)

    def err(ws):
        return float(np.sin(ws[0] @ c))

    h, _, _ = error_barrier(err, a, b, num=7)
    assert h >= 0.0


def test_barrier_two_points_and_rate_bounds():
    # num = 2 is the smallest grid: only the two ends, so the barrier is half the gap between them.
    h, alphas, _ = error_barrier(lambda ws: float(ws[0][0]), [np.array([3.0])], [np.array([1.0])], num=2)
    assert h == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_array_equal(alphas, [0.0, 1.0])
    for rate in (0.0, 1.0):
        with pytest.raises(ValueError):
            rounds_to_reach(rate, 0.5)
