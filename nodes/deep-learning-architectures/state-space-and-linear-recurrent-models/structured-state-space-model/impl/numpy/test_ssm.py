# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Tests of the NumPy reference: the two modes agree, the discretizations, the kernel, causality and the worked example."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ssm import (
    causal_conv,
    config,
    continuous,
    discretize,
    discretize_dense,
    expm,
    forward_conv,
    forward_scan,
    hippo_legs,
    init,
    init_state,
    kernel,
    kernel_dense,
    s4d_lin,
    scan_dense,
    selective_scan,
    step,
)

METHODS = ["zoh", "bilinear"]


def layer(seed, method, h=3, n=4):
    cfg = config(h=h, n=n, method=method, dt_min=0.01, dt_max=0.5)
    rng = np.random.default_rng(seed)
    params = init(rng, cfg)
    # Move the parameters away from the initialization, so that every channel and entry differs.
    params["log_a_re"] = params["log_a_re"] + 0.3 * rng.standard_normal(params["log_a_re"].shape)
    params["a_im"] = params["a_im"] + rng.standard_normal(params["a_im"].shape)
    params["b"] = params["b"] + 0.5 * (rng.standard_normal(params["b"].shape) + 1j * rng.standard_normal(params["b"].shape))
    return cfg, params


# The worked example of the page: a = -1, b = c = 1. ZOH at dt = ln 2 and bilinear at dt = 2/3 both give Ab = Bb = 1/2.
U = np.array([1.0, 2.0, 0.0, -1.0])
Y = np.array([0.5, 1.25, 0.625, -0.1875])
K4 = np.array([0.5, 0.25, 0.125, 0.0625])


@pytest.mark.parametrize("method, dt", [("zoh", math.log(2.0)), ("bilinear", 2.0 / 3.0)])
def test_worked_example(method, dt):
    ab, bb = discretize_dense([[-1.0]], [1.0], dt, method)
    assert np.allclose(ab, [[0.5]]) and np.allclose(bb, [0.5])
    assert np.allclose(kernel_dense(ab, bb, np.array([1.0]), 4), K4)
    assert np.allclose(scan_dense(ab, bb, np.array([1.0]), U), Y)
    assert np.allclose(causal_conv(U[None, :, None], K4[None, :])[0, :, 0], Y)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), method=st.sampled_from(METHODS), length=st.integers(1, 40))
def test_recurrence_equals_convolution(seed, method, length):
    cfg, params = layer(seed, method)
    u = np.random.default_rng(seed + 1).standard_normal((2, length, cfg["h"]))
    assert np.allclose(forward_scan(params, u, cfg), forward_conv(params, u, cfg), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_impulse_response_is_kernel(method):
    cfg, params = layer(1, method)
    length = 30
    u = np.zeros((cfg["h"], length, cfg["h"]))
    for h in range(cfg["h"]):
        u[h, 0, h] = 1.0
    k = kernel(params, length, cfg)
    for mode in (forward_scan, forward_conv):
        y = mode(params, u, cfg)
        for h in range(cfg["h"]):
            expected = k[h].copy()
            expected[0] += params["d"][h]
            assert np.allclose(y[h, :, h], expected, atol=1e-12)
            others = [g for g in range(cfg["h"]) if g != h]
            assert np.allclose(y[h][:, others], 0.0, atol=1e-12)  # the channels do not mix


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("t", [0, 7, 18])
def test_causality(method, t):
    cfg, params = layer(2, method)
    rng = np.random.default_rng(3)
    u = rng.standard_normal((2, 20, cfg["h"]))
    v = u.copy()
    v[:, t + 1 :, :] += rng.standard_normal(v[:, t + 1 :, :].shape)
    for mode in (forward_scan, forward_conv):
        a, b = mode(params, u, cfg), mode(params, v, cfg)
        assert np.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-12)
        if t + 1 < 20:
            assert not np.allclose(a[:, t + 1 :], b[:, t + 1 :])


def test_kernel_matches_dense():
    cfg, params = layer(4, "zoh", h=1, n=3)
    a, b, dt = continuous(params)
    ab, bb = discretize(a[0], b[0], dt[0], "zoh")
    # The same SSM written as a dense complex system of size 2n, with the conjugate pairs made explicit.
    abig = np.concatenate([ab, ab.conj()])
    bbig = np.concatenate([bb, bb.conj()])
    cbig = np.concatenate([params["c"][0], params["c"][0].conj()])
    dense = np.array([cbig @ (abig**i * bbig) for i in range(12)])
    assert np.allclose(dense.imag, 0.0)
    assert np.allclose(kernel(params, 12, cfg)[0], dense.real)


def test_zoh_scalar_matches_exponential():
    a, b, dt = -0.7, 1.3, 0.25
    ab, bb = discretize(np.array([a + 0j]), np.array([b + 0j]), dt, "zoh")
    assert np.isclose(ab[0], math.exp(a * dt))
    assert np.isclose(bb[0], (math.exp(a * dt) - 1.0) / a * b)
    ab_d, bb_d = discretize_dense([[a]], [b], dt, "zoh")
    assert np.isclose(ab_d[0, 0], math.exp(a * dt)) and np.isclose(bb_d[0], (math.exp(a * dt) - 1.0) / a * b)
    # The integral of e^(a s) b over [0, dt] by the midpoint rule with many points.
    s = (np.arange(20000) + 0.5) * dt / 20000
    assert np.isclose(bb[0].real, np.sum(np.exp(a * s)) * dt / 20000 * b, rtol=1e-8)


def test_zoh_dense_matches_eigendecomposition():
    rng = np.random.default_rng(5)
    m = rng.standard_normal((4, 4))
    a = -(m @ m.T) - 0.5 * np.eye(4) + (m - m.T)
    w, v = np.linalg.eig(a)
    exact = (v @ np.diag(np.exp(0.3 * w)) @ np.linalg.inv(v)).real
    assert np.allclose(expm(0.3 * a), exact, atol=1e-10)
    b = rng.standard_normal(4)
    ab, bb = discretize_dense(a, b, 0.3, "zoh")
    assert np.allclose(ab, exact, atol=1e-10)
    assert np.allclose(bb, np.linalg.solve(a, (exact - np.eye(4)) @ b), atol=1e-10)


def test_bilinear_scalar():
    a, b, dt = -2.0, 0.5, 0.1
    ab, bb = discretize(np.array([a + 0j]), np.array([b + 0j]), dt, "bilinear")
    assert np.isclose(ab[0], (1 + dt * a / 2) / (1 - dt * a / 2))
    assert np.isclose(bb[0], dt * b / (1 - dt * a / 2))


@pytest.mark.parametrize("dt", [1e-3, 0.1, 1.0, 10.0, 1e4])
def test_bilinear_keeps_stable(dt):
    rng = np.random.default_rng(6)
    for _ in range(20):
        m = rng.standard_normal((5, 5))
        a = -(m @ m.T) - 0.01 * np.eye(5) + 3.0 * (m - m.T)
        assert np.linalg.eigvals(a).real.max() < 0
        ab, _ = discretize_dense(a, np.ones(5), dt, "bilinear")
        assert np.abs(np.linalg.eigvals(ab)).max() < 1.0
    a = -np.exp(rng.standard_normal(50)) + 1j * rng.standard_normal(50) * 100
    ab, _ = discretize(a, np.ones(50), dt, "bilinear")
    assert np.abs(ab).max() < 1.0
    ab, _ = discretize(a, np.ones(50), dt, "zoh")
    assert np.abs(ab).max() < 1.0


def test_bilinear_maps_unstable_outside():
    ab, _ = discretize(np.array([0.3 + 2j]), np.array([1.0 + 0j]), 0.5, "bilinear")
    assert abs(ab[0]) > 1.0
    ab, _ = discretize(np.array([0.0 + 2j]), np.array([1.0 + 0j]), 0.5, "bilinear")
    assert np.isclose(abs(ab[0]), 1.0)


def test_rules_agree_to_second_order():
    a = np.array([-1.0 + 3j])
    errs = []
    for dt in (0.04, 0.02, 0.01):
        z, _ = discretize(a, a, dt, "zoh")
        w, _ = discretize(a, a, dt, "bilinear")
        errs.append(abs(z[0] - w[0]))
    # The difference of e^z and (1 + z/2) / (1 - z/2) is z^3 / 12 + O(z^4): halving dt divides it by 8.
    assert 7.5 < errs[0] / errs[1] < 8.5 and 7.5 < errs[1] / errs[2] < 8.5
    assert np.isclose(errs[2], abs((0.01 * a[0]) ** 3) / 12, rtol=0.02)


@pytest.mark.parametrize("method", METHODS)
def test_step_streams_like_scan(method):
    cfg, params = layer(7, method)
    u = np.random.default_rng(8).standard_normal((2, 15, cfg["h"]))
    full = forward_conv(params, u, cfg)
    x = init_state(cfg, 2)
    assert x.shape == (2, cfg["h"], cfg["n"])
    for t in range(15):
        x, y_t = step(params, x, u[:, t], cfg)
        assert np.allclose(y_t, full[:, t], atol=1e-10)


def test_conjugation_invariance():
    rng = np.random.default_rng(9)
    m = rng.standard_normal((4, 4))
    a = -(m @ m.T) - np.eye(4)
    b, c = rng.standard_normal(4), rng.standard_normal(4)
    v = rng.standard_normal((4, 4)) + 3 * np.eye(4)
    vi = np.linalg.inv(v)
    for method in METHODS:
        k1 = kernel_dense(*discretize_dense(a, b, 0.2, method), c, 10)
        k2 = kernel_dense(*discretize_dense(vi @ a @ v, vi @ b, 0.2, method), c @ v, 10)
        assert np.allclose(k1, k2, atol=1e-10)


def test_hippo_legs_is_normal_plus_low_rank():
    for n in (2, 5, 16):
        a = hippo_legs(n)
        assert a[1, 0] == -math.sqrt(3.0) and a[0, 0] == -1.0 and a[0, 1] == 0.0
        p = np.sqrt(np.arange(n) + 0.5)
        s = a + np.outer(p, p)
        assert np.allclose(s @ s.T, s.T @ s)
        assert np.allclose(s + s.T, -np.eye(n))
        assert np.allclose(np.linalg.eigvals(s).real, -0.5)
        assert np.allclose(sorted(np.linalg.eigvals(a).real), -np.arange(n, 0, -1))


def test_s4d_lin_and_init():
    assert np.allclose(s4d_lin(3), [-0.5, -0.5 + 1j * math.pi, -0.5 + 2j * math.pi])
    cfg = config(h=4, n=3, dt_min=0.01, dt_max=0.1)
    params = init(np.random.default_rng(0), cfg)
    a, _, dt = continuous(params)
    assert np.allclose(a, s4d_lin(3)[None, :])
    assert np.all((dt >= 0.01 - 1e-12) & (dt <= 0.1 + 1e-12))
    assert params["c"].shape == (4, 3) and params["d"].shape == (4,)


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        config(method="euler")
    with pytest.raises(ValueError):
        config(h=0)
    with pytest.raises(ValueError):
        config(dt_min=0.2, dt_max=0.1)
    with pytest.raises(ValueError):
        discretize(np.array([-1.0 + 0j]), np.array([1.0 + 0j]), 0.1, "euler")
    with pytest.raises(ValueError):
        discretize_dense([[-1.0]], [1.0], 0.1, "euler")


def test_selective_scan_reduces_to_lti_and_breaks_shift_invariance():
    rng = np.random.default_rng(10)
    length, n = 12, 3
    a = -np.exp(rng.standard_normal(n))
    b, c = rng.standard_normal(n), rng.standard_normal(n)
    u = rng.standard_normal(length)
    y = selective_scan(u, np.full(length, 0.3), a, np.tile(b, (length, 1)), np.tile(c, (length, 1)))
    ab, bb = discretize_dense(np.diag(a), b, 0.3, "zoh")
    assert np.allclose(y, scan_dense(ab, bb, c, u))
    # With a step size that depends on the input, the map is no longer linear, so no kernel K gives y = K * u.
    delta = lambda u: np.log1p(np.exp(2.0 * u))  # noqa: E731, softplus
    e0, e3 = np.eye(length)[0], np.eye(length)[3]
    bs, cs = np.tile(b, (length, 1)), np.tile(c, (length, 1))
    r0 = selective_scan(e0, delta(e0), a, bs, cs)
    r3 = selective_scan(e3, delta(e3), a, bs, cs)
    assert np.allclose(r3[3:], r0[: length - 3])  # a lone impulse still shifts in time
    two = selective_scan(e0 + e3, delta(e0 + e3), a, bs, cs)
    assert not np.allclose(two, r0 + r3)  # but the response to a sum is not the sum


def test_selective_gate_form():
    # Theorem 1 of Gu and Dao: N = 1, A = -1, B = 1 and delta = softplus(z) give h = (1 - g) h + g x, g = sigmoid(z).
    rng = np.random.default_rng(11)
    x, z = rng.standard_normal(10), rng.standard_normal(10)
    y = selective_scan(x, np.log1p(np.exp(z)), np.array([-1.0]), np.ones((10, 1)), np.ones((10, 1)))
    g = 1.0 / (1.0 + np.exp(-z))
    h, ref = 0.0, []
    for k in range(10):
        h = (1 - g[k]) * h + g[k] * x[k]
        ref.append(h)
    assert np.allclose(y, ref)
