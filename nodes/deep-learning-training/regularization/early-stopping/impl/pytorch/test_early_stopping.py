# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/early-stopping/ (ai://VR3ZRP).
"""Tests of the PyTorch reference: the worked example and fixture of the NumPy tests, torch.optim.SGD on least squares
as the spectral filter, the patience rule on a module with restore-best, and the criteria GL and PQ."""

import pytest
import torch
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
    training_progress,
)

torch.set_default_dtype(torch.float64)

# The worked example and the fixture of the NumPy tests.
S = [4.0, 1.0, 0.25]
ETA, T = 0.1, 10
GD = [0.9939533824, 0.6513215599, 0.2236703800]
RIDGE = [0.8, 0.5, 0.2]
FLOW = [0.9816843611, 0.6321205588, 0.2211992169]
VAL = [1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74]
GL = [0.0, 0.0, 0.0, 2.857142857, 0.0, 2.898550725, 5.797101449, 7.246376812]
TRAIN = [2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.82, 0.80, 0.79, 0.785]
P5 = [466.6666667, 282.3529412, 163.4146341, 92.5, 53.16455696, 30.57324841]


def close(a, b, tol=1e-9):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b, dtype=torch.float64), rtol=tol, atol=tol)


def design(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=g) * torch.linspace(0.2, 2.0, p)
    y = X @ torch.randn(p, generator=g) + torch.randn(n, generator=g)
    return X, y


def spectral(X, y, filt):
    n = X.shape[0]
    s, V = torch.linalg.eigh(X.T @ X / n)
    c = V.T @ (X.T @ y / n)
    keep = s > 1e-10 * s.max()
    coef = torch.zeros_like(s)
    coef[keep] = filt(s[keep]) * c[keep] / s[keep]
    return V @ coef


def test_worked_example():
    assert matched_lambda(ETA, T) == pytest.approx(1.0)
    assert close(gd_filter(S, ETA, T), GD)
    assert close(ridge_filter(S, 1.0), RIDGE)
    assert close(flow_filter(S, ETA * T), FLOW)


@pytest.mark.parametrize(("n", "p"), [(40, 6), (8, 12)])
def test_sgd_is_the_filter(n, p):
    X, y = design(n, p, 1)
    eta = 0.9 / float(torch.linalg.eigvalsh(X.T @ X / n).max())
    for t in (1, 5, 40):
        w = gradient_descent(X, y, eta, t)
        assert close(w, spectral(X, y, lambda v, t=t: gd_filter(v, eta, t)), 1e-8)


def test_ridge_is_the_filter():
    X, y = design(30, 5, 2)
    for lam in (0.1, 2.0):
        assert close(ridge(X, y, lam), spectral(X, y, lambda v, lam=lam: ridge_filter(v, lam)), 1e-9)


def test_flow_is_the_limit():
    X, y = design(50, 5, 3)
    flow = spectral(X, y, lambda v: flow_filter(v, 2.0))
    errs = [float(torch.linalg.norm(gradient_descent(X, y, 2.0 / k, k) - flow)) for k in (20, 200)]
    assert errs[1] < errs[0] / 8


@settings(max_examples=100, deadline=None)
@given(x=st.floats(0.0, 1.0), t=st.integers(1, 300))
def test_filter_sandwich(x, t):
    s, eta = x / 0.01, 0.01
    g = float(gd_filter(s, eta, t))
    r = float(ridge_filter(s, matched_lambda(eta, t)))
    assert r <= g + 1e-12 and g <= 2.0 * r + 1e-12


def test_matched_lambda_rejects_bad_input():
    for eta, t in ((0.0, 5), (0.1, 0)):
        with pytest.raises(ValueError):
            matched_lambda(eta, t)


def train_with_rule(values, **kw):
    model = torch.nn.Linear(2, 1)
    rule = EarlyStopping(**kw)
    for i, v in enumerate(values):
        with torch.no_grad():
            model.weight.fill_(float(i + 1))
        if rule.step(torch.tensor(v), model):
            break
    return rule, model


def test_patience_restores_the_best_module():
    rule, model = train_with_rule(VAL, patience=2)
    assert (rule.epoch, rule.best_epoch, rule.best) == (7, 5, pytest.approx(0.69))
    assert float(model.weight.detach()[0, 0]) == 7.0
    rule.restore(model)
    assert torch.equal(model.weight, torch.full((1, 2), 5.0))


def test_min_delta_and_mode():
    rule, _ = train_with_rule(VAL, patience=2, min_delta=0.02)
    assert (rule.epoch, rule.best_epoch) == (5, 3)
    rule, _ = train_with_rule([0.5, 0.6, 0.59, 0.61, 0.6, 0.605], patience=2, mode="max", min_delta=0.005)
    assert (rule.epoch, rule.best_epoch) == (6, 4)
    rule, _ = train_with_rule([1.0, 0.75, 0.5], patience=1, min_delta=0.25)
    assert (rule.epoch, rule.best_epoch) == (2, 1)
    rule, _ = train_with_rule([0.5, 0.5], patience=1, mode="max")
    assert (rule.epoch, rule.best_epoch) == (2, 1)


def test_nan_never_improves():
    rule, _ = train_with_rule([float("nan"), 1.0, float("nan"), float("nan")], patience=2)
    assert (rule.epoch, rule.best_epoch, rule.best) == (4, 2, 1.0)


def test_kept_state_is_a_clone():
    model = torch.nn.Linear(2, 1)
    rule = EarlyStopping(patience=3)
    before = model.weight.detach().clone()
    rule.step(0.5, model)
    with torch.no_grad():
        model.weight.add_(1.0)
    rule.step(0.9, model)
    rule.restore(model)
    assert torch.equal(model.weight, before)


def test_bad_use_raises():
    model = torch.nn.Linear(1, 1)
    rule = EarlyStopping(patience=1)
    with pytest.raises(RuntimeError):
        rule.restore(model)
    rule.step(1.0, model)
    rule.step(2.0, model)
    with pytest.raises(RuntimeError):
        rule.step(0.5, model)
    bad = [(0, 0.0, "min"), (1.5, 0.0, "min"), (2, -0.1, "min"), (2, 0.0, "x")]
    for patience, min_delta, mode in bad:
        with pytest.raises(ValueError):
            EarlyStopping(patience, min_delta=min_delta, mode=mode)
    assert EarlyStopping(patience=2.0).patience == 2
    rule = EarlyStopping(patience=1)
    rule.step(1.0)
    with pytest.raises(RuntimeError):
        rule.restore(model)


def test_generalization_loss_and_gl_stop():
    assert close(generalization_loss(VAL), GL, 1e-8)
    assert gl_stop(VAL, 5.0) == 7
    assert gl_stop(VAL, 7.3) is None
    with pytest.raises(ValueError):
        generalization_loss([1.0, 0.0])


def test_training_progress_and_pq():
    pk = training_progress(TRAIN, 5)
    assert torch.isnan(pk[:4]).all()
    assert close(pk[4:], P5, 1e-8)
    assert torch.isnan(training_progress([1.0, 2.0], 3)).all()
    with pytest.raises(ValueError):
        training_progress([1.0, -1.0], 1)
    val = [1.0, 0.9, 0.8, 0.7, 0.6, 0.62, 0.63, 0.64, 0.65, 0.66]
    train = [1.0, 0.9, 0.8, 0.7, 0.6, 1.0, 1.0, 1.0, 1.0, 0.95]
    assert pq_stop(val, train, 0.2, k=5) == 10
    assert pq_stop(val, train, 0.3, k=5) is None
    assert pq_stop([1.0, 2.0], [1.0, 1.0], 0.5, k=2) == 2
