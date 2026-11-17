# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://NEW-early-stopping).
"""Tests of the NumPy reference: the worked example, gradient descent and ridge as spectral filters, the gradient flow
limit, the patience rule with restore-best, and the criteria GL, PQ and UP."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from early_stopping import (
    EarlyStopping,
    flow_filter,
    gd_filter,
    generalization_loss,
    gl_stop,
    gradient_descent,
    matched_lambda,
    pq_stop,
    ridge,
    ridge_filter,
    spectral_fit,
    spectrum,
    training_progress,
    up_stop,
)

# The worked example of the page, shared with the PyTorch tests.
S = [4.0, 1.0, 0.25]
ETA, T = 0.1, 10
GD = [0.9939533824, 0.6513215599, 0.2236703800]
RIDGE = [0.8, 0.5, 0.2]
FLOW = [0.9816843611, 0.6321205588, 0.2211992169]
VAL = [1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74]
GL = [0.0, 0.0, 0.0, 2.857142857, 0.0, 2.898550725, 5.797101449, 7.246376812]
TRAIN = [2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.82, 0.80, 0.79, 0.785]
P5 = [466.6666667, 282.3529412, 163.4146341, 92.5, 53.16455696, 30.57324841]


def design(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.linspace(0.2, 2.0, p)
    y = X @ rng.normal(size=p) + rng.normal(size=n)
    return X, y


def test_worked_example():
    assert matched_lambda(ETA, T) == pytest.approx(1.0)
    assert gd_filter(S, ETA, T) == pytest.approx(GD, abs=1e-9)
    assert ridge_filter(S, matched_lambda(ETA, T)) == pytest.approx(RIDGE, abs=1e-12)
    assert flow_filter(S, ETA * T) == pytest.approx(FLOW, abs=1e-9)


@pytest.mark.parametrize(("n", "p"), [(40, 6), (8, 12)])
def test_gradient_descent_is_the_filter(n, p):
    X, y = design(n, p, 1)
    s, _ = spectrum(X)
    eta = 0.9 / s.max()
    for t in (1, 2, 7, 60):
        w = gradient_descent(X, y, eta, t)
        assert np.allclose(w, spectral_fit(X, y, lambda v, t=t: gd_filter(v, eta, t)), rtol=1e-9, atol=1e-11)
    assert np.allclose(gradient_descent(X, y, eta, 0), 0.0)


def test_gradient_descent_from_w0():
    X, y = design(30, 4, 2)
    w0 = np.array([1.0, -2.0, 0.5, 3.0])
    w = gradient_descent(X, y, 0.05, 1, w0=w0)
    assert np.allclose(w, w0 + 0.05 * X.T @ (y - X @ w0) / 30)
    assert np.allclose(w0, [1.0, -2.0, 0.5, 3.0])


@pytest.mark.parametrize(("n", "p"), [(40, 6), (8, 12)])
def test_ridge_is_the_filter(n, p):
    X, y = design(n, p, 3)
    for lam in (0.01, 0.5, 4.0):
        assert np.allclose(ridge(X, y, lam), spectral_fit(X, y, lambda v, lam=lam: ridge_filter(v, lam)), atol=1e-10)


def test_spectral_fit_skips_null_directions():
    X, y = design(5, 9, 4)
    w = spectral_fit(X, y, lambda v: np.ones_like(v))
    assert np.allclose(w, np.linalg.pinv(X) @ y, atol=1e-9)


def test_gradient_flow_is_the_limit():
    X, y = design(50, 5, 5)
    tau = 3.0
    flow = spectral_fit(X, y, lambda v: flow_filter(v, tau))
    errs = [np.linalg.norm(gradient_descent(X, y, tau / k, k) - flow) for k in (30, 300)]
    assert errs[1] < errs[0] / 8
    assert errs[1] < 1e-2 * np.linalg.norm(flow)


@settings(max_examples=200, deadline=None)
@given(x=st.floats(0.0, 1.0), t=st.integers(1, 400))
def test_filter_sandwich(x, t):
    # with eta s = x in [0, 1]: ridge at 1 / (eta t) <= gd <= min(1, eta t s) <= 2 ridge.
    eta = 0.01
    s = x / eta
    g = float(gd_filter(s, eta, t))
    r = float(ridge_filter(s, matched_lambda(eta, t)))
    assert r <= g + 1e-12
    assert g <= min(1.0, t * x) + 1e-12
    assert g <= 2.0 * r + 1e-12


@settings(max_examples=200, deadline=None)
@given(u=st.floats(0.0, 60.0))
def test_flow_ridge_ratio(u):
    # with u = s tau: u / (1 + u) <= 1 - exp(-u) <= 1.2985 u / (1 + u).
    f = float(flow_filter(u, 1.0))
    r = float(ridge_filter(u, 1.0))
    assert r <= f + 1e-12
    assert f <= 1.2985 * r + 1e-12


def test_same_slope_at_zero():
    h = 1e-7
    eta, t = 0.05, 40
    assert gd_filter(h, eta, t) / h == pytest.approx(eta * t, rel=1e-4)
    assert ridge_filter(h, matched_lambda(eta, t)) / h == pytest.approx(eta * t, rel=1e-4)
    assert flow_filter(h, eta * t) / h == pytest.approx(eta * t, rel=1e-4)


def test_matched_lambda_rejects_bad_input():
    for eta, t in ((0.0, 5), (0.1, 0), (-0.1, 5)):
        with pytest.raises(ValueError):
            matched_lambda(eta, t)


def run(values, **kw):
    rule = EarlyStopping(**kw)
    for i, v in enumerate(values):
        if rule.step(v, {"epoch": i + 1}):
            break
    return rule


def test_patience_worked_example():
    rule = run(VAL, patience=2)
    assert (rule.epoch, rule.best_epoch, rule.best) == (7, 5, 0.69)
    assert rule.restore() == {"epoch": 5}
    assert rule.stopped


def test_min_delta_worked_example():
    rule = run(VAL, patience=2, min_delta=0.02)
    assert (rule.epoch, rule.best_epoch, rule.best) == (5, 3, 0.7)


def test_min_delta_boundary():
    rule = run([1.0, 0.75, 0.5], patience=1, min_delta=0.25)
    assert (rule.epoch, rule.best_epoch) == (2, 1)


def test_mode_max():
    rule = run([0.5, 0.6, 0.59, 0.61, 0.6, 0.605], patience=2, mode="max", min_delta=0.005)
    assert (rule.epoch, rule.best_epoch, rule.best) == (6, 4, 0.61)
    rule = run([0.5, 0.5], patience=1, mode="max")
    assert (rule.epoch, rule.best_epoch) == (2, 1)


def test_nan_never_improves():
    rule = run([float("nan"), 1.0, float("nan"), float("nan")], patience=2)
    assert (rule.epoch, rule.best_epoch, rule.best) == (4, 2, 1.0)


def test_ties_do_not_improve():
    rule = run([1.0, 1.0, 1.0], patience=2)
    assert (rule.epoch, rule.best_epoch) == (3, 1)


def test_restore_is_a_copy():
    rule = EarlyStopping(patience=3)
    w = {"w": np.array([1.0, 2.0])}
    rule.step(0.5, w)
    w["w"][0] = 99.0
    rule.step(0.9, w)
    got = rule.restore()
    assert np.array_equal(got["w"], [1.0, 2.0])
    got["w"][1] = -5.0
    assert np.array_equal(rule.restore()["w"], [1.0, 2.0])


def test_bad_use_raises():
    rule = EarlyStopping(patience=1)
    with pytest.raises(RuntimeError):
        rule.restore()
    rule.step(1.0)
    rule.step(2.0)
    with pytest.raises(RuntimeError):
        rule.step(0.5)
    bad = [(0, 0.0, "min"), (1.5, 0.0, "min"), (2, -0.1, "min"), (2, 0.0, "mid")]
    for patience, min_delta, mode in bad:
        with pytest.raises(ValueError):
            EarlyStopping(patience, min_delta=min_delta, mode=mode)
    assert EarlyStopping(patience=2.0).patience == 2


@settings(max_examples=300, deadline=None)
@given(
    vals=st.lists(st.floats(0.0, 10.0), min_size=1, max_size=40),
    p=st.integers(1, 6),
    delta=st.sampled_from([0.0, 0.1, 0.5]),
)
def test_guarantees(vals, p, delta):
    rule = run(vals, patience=p, min_delta=delta)
    assert rule.best_epoch is not None and rule.best is not None
    seen = vals[: rule.epoch]
    if rule.stopped:
        assert rule.epoch == rule.best_epoch + p
    else:
        assert rule.epoch == len(vals)
    assert rule.best == seen[rule.best_epoch - 1]
    assert rule.best <= min(seen) + delta
    assert all(v > rule.best for v in seen[: rule.best_epoch - 1])
    assert all(v >= rule.best - delta for v in seen[rule.best_epoch :])
    assert rule.restore() == {"epoch": rule.best_epoch}


@pytest.mark.parametrize("p", [1, 3, 10])
def test_unimodal_curve_gives_the_global_minimum(p):
    e = np.arange(1, 60)
    vals = list((e - 17.0) ** 2 / 100 + 1)
    rule = run(vals, patience=p)
    assert (rule.best_epoch, rule.epoch) == (17, 17 + p)


@pytest.mark.parametrize("p", [1, 3, 10])
def test_plateau_defeats_any_patience(p):
    vals = [1.0, 0.5] + [0.5] * p + [0.1]
    rule = run(vals, patience=p)
    assert rule.stopped and rule.best == 0.5 and min(vals) == 0.1


def test_generalization_loss():
    assert generalization_loss(VAL) == pytest.approx(GL, abs=1e-8)
    assert gl_stop(VAL, 5.0) == 7
    assert gl_stop(VAL, 2.857142857) == 4
    assert gl_stop(VAL, 7.3) is None
    with pytest.raises(ValueError):
        generalization_loss([1.0, 0.0])


def test_training_progress():
    pk = training_progress(TRAIN, 5)
    assert np.isnan(pk[:4]).all()
    assert pk[4:] == pytest.approx(P5, rel=1e-8)
    assert training_progress([1.0, 1.0, 1.0], 2)[1:] == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError):
        training_progress([1.0, -1.0], 1)


def test_pq_stop():
    # GL(5) = 0 and GL(10) = 100 (0.66 / 0.6 - 1) = 10; P_5(10) = 1000 (4.95 / 4.75 - 1) = 42.105.
    val = [1.0, 0.9, 0.8, 0.7, 0.6, 0.62, 0.63, 0.64, 0.65, 0.66]
    train = [1.0, 0.9, 0.8, 0.7, 0.6, 1.0, 1.0, 1.0, 1.0, 0.95]
    assert pq_stop(val, train, 0.2, k=5) == 10
    assert pq_stop(val, train, 0.3, k=5) is None
    assert pq_stop(val[:9], train[:9], 0.0, k=5) is None
    assert pq_stop([1.0, 2.0], [1.0, 1.0], 0.5, k=2) == 2


def test_up_stop():
    val = [5, 4, 3, 2, 1, 1, 2, 3, 4, 3, 3, 3, 3, 3, 4]
    # strip ends 3, 6, 9, 12, 15 with errors 3, 1, 4, 3, 4: rises at 9 and at 15, falls at 6 and 12.
    assert up_stop(val, 1, k=3) == 9
    assert up_stop(val, 2, k=3) is None
    assert up_stop([1, 1, 2, 2, 3, 3], 2, k=2) == 6
    assert up_stop([1, 1, 1, 1], 1, k=2) is None
    assert up_stop([2, 1, 0.5, 3], 1, k=1) == 4


def test_gl_matches_the_ratio_definition():
    val = np.array([3.0, 2.0, 2.5, 1.0, 1.5])
    assert generalization_loss(val)[-1] == pytest.approx(50.0)
    assert math.isclose(generalization_loss(val)[2], 25.0)
