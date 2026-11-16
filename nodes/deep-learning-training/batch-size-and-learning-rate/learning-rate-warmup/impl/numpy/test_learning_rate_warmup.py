# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/learning-rate-warmup/ (ai://NEW-learning-rate-warmup).
"""Tests of the NumPy reference: the worked example of the page (Goyal et al.'s ramp, the Transformer schedule and
RAdam's rectification), the algebra of the derivation, and the shared RAdam fixture."""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_rate_warmup import (
    first_adaptive_step,
    gradual_warmup,
    linear_warmup,
    noam,
    noam_peak,
    radam_init,
    radam_rectification,
    radam_rho,
    radam_rho_inf,
    radam_step,
    variance_first_order,
)

# The shared fixture: one parameter vector and six gradients. The PyTorch tests check AFTER6_TORCH against torch.optim.RAdam.
P0 = [0.5, -1.5, 2.0]
GRADS = [
    [1.0, -2.0, 0.5],
    [0.5, 1.0, -0.25],
    [-1.5, 0.5, 0.0],
    [2.0, -0.5, 1.0],
    [0.25, 0.75, -1.0],
    [1.0, 1.0, 1.0],
]
CFG = {"lr": 0.1, "beta1": 0.9, "beta2": 0.5}
# RAdam with the threshold of PyTorch (rho_t > 5), lr 0.1, betas (0.9, 0.999), eps 1e-8, after the six gradients.
AFTER6_TORCH = [0.23675198784933882, -1.2330182831375551, 1.8971525192034477]


def rect(t, beta2=0.999, threshold=4.0):
    r = radam_rectification(t, beta2, threshold)
    assert r is not None
    return r


def run(threshold, cfg=CFG, grads=GRADS):
    p, s = np.asarray(P0, dtype=np.float64), radam_init(P0)
    for g in grads:
        p, s = radam_step(p, g, s, threshold=threshold, **cfg)
    return p


def test_goyal_ramp():
    iters = 1.28e6 / 8192 * 5
    assert iters == 781.25
    inc = (3.2 - 0.1) / iters
    assert inc == pytest.approx(0.003968, abs=1e-6)
    assert gradual_warmup(0, iters, 0.1, 3.2) == pytest.approx(0.1)
    assert gradual_warmup(156, iters, 0.1, 3.2) == pytest.approx(0.72, abs=0.005)
    assert gradual_warmup(391, iters, 0.1, 3.2) == pytest.approx(1.65, abs=0.005)
    assert gradual_warmup(iters, iters, 0.1, 3.2) == pytest.approx(3.2)
    assert gradual_warmup(10**6, iters, 0.1, 3.2) == pytest.approx(3.2)


def test_noam_worked_example():
    peak = noam_peak(512, 4000)
    assert peak == pytest.approx(6.99e-4, abs=5e-7)
    assert noam(4000, 512, 4000) == pytest.approx(peak)
    assert noam(1, 512, 4000) == pytest.approx(1.75e-7, abs=5e-10)
    assert noam(1, 512, 4000) / peak == pytest.approx(1 / 4000)
    assert noam(1000, 512, 4000) / peak == pytest.approx(0.25)
    assert noam(16000, 512, 4000) / peak == pytest.approx(0.5)
    assert noam(100000, 512, 4000) == pytest.approx(1.40e-4, abs=5e-7)
    assert noam(100000, 512, 4000) / peak == pytest.approx(0.2)
    assert noam(300000, 512, 4000) == pytest.approx(8.07e-5, abs=5e-8)
    assert noam(300000, 512, 4000) / peak == pytest.approx(0.115, abs=5e-4)


@settings(max_examples=200, deadline=None)
@given(st.integers(64, 4096), st.integers(1, 20000), st.integers(1, 200000))
def test_noam_peak_and_branches(d, w, s):
    lr = noam(s, d, w)
    assert lr <= noam_peak(d, w) * (1 + 1e-12)
    if s <= w:
        assert lr == pytest.approx(d**-0.5 * s * w**-1.5)
    else:
        assert lr == pytest.approx((d * s) ** -0.5)


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 10000), st.floats(1e-4, 10.0), st.floats(0.0, 1.0))
def test_ramps(w, peak, frac):
    t = int(frac * w)
    lr = linear_warmup(t, w, peak)
    assert 0.0 <= lr <= peak * (1 + 1e-12)
    assert lr == pytest.approx(peak * t / w)
    assert linear_warmup(t + 1, w, peak) >= lr
    assert gradual_warmup(w, w, 0.5 * peak, peak) == pytest.approx(peak)


def test_bad_arguments():
    with pytest.raises(ValueError):
        gradual_warmup(0, 0, 0.1, 1.0)
    with pytest.raises(ValueError):
        noam(0, 512, 4000)
    with pytest.raises(ValueError):
        radam_rho_inf(1.0)
    with pytest.raises(ValueError):
        radam_rho(0, 0.999)


@settings(max_examples=200, deadline=None)
@given(st.floats(0.01, 0.9999))
def test_rho_one_at_first_step(beta2):
    assert radam_rho(1, beta2) == pytest.approx(1.0, abs=1e-6)


def test_rho_close_to_t_early():
    for t in [2, 5, 10, 20]:
        assert radam_rho(t, 0.999) == pytest.approx(t, rel=0.01)
    assert radam_rho(100, 0.999) == pytest.approx(98.33, abs=0.01)


def test_radam_worked_example():
    assert radam_rho_inf(0.999) == pytest.approx(1999)
    assert [round(radam_rho(t, 0.999), 4) for t in (2, 3, 4, 5)] == [
        1.9995,
        2.9987,
        3.9975,
        4.996,
    ]
    assert radam_rectification(4, 0.999) is None
    want = {5: 0.017, 100: 0.215, 556: 0.50, 1000: 0.645, 2000: 0.828, 5000: 0.983}
    for t, r in want.items():
        assert radam_rectification(t, 0.999) == pytest.approx(r, abs=6e-4)
    assert rect(555) < 0.5 <= rect(556)
    assert math.sqrt(100 / 1999) == pytest.approx(0.224, abs=5e-4)


def test_first_adaptive_step_paper_and_torch():
    assert first_adaptive_step(0.999, 4.0) == 5
    assert first_adaptive_step(0.999, 5.0) == 6
    assert radam_rho(6, 0.999) == pytest.approx(5.994, abs=5e-4)
    assert radam_rectification(6, 0.999, 5.0) == pytest.approx(0.026, abs=5e-4)
    assert radam_rectification(5, 0.999, 5.0) is None


def test_degenerate_and_rho_inf():
    assert [radam_rho_inf(b) for b in (0.6, 0.9, 0.98, 0.99)] == pytest.approx([4, 19, 99, 199])
    for b in (0.3, 0.5, 0.6):
        assert first_adaptive_step(b, 4.0) is None


def test_rectification_is_variance_ratio():
    rho_inf = radam_rho_inf(0.999)
    for t in [5, 10, 100, 1000]:
        rho = radam_rho(t, 0.999)
        ratio = math.sqrt(variance_first_order(rho_inf, 2.0) / variance_first_order(rho, 2.0))
        assert radam_rectification(t, 0.999) == pytest.approx(ratio)
    assert variance_first_order(5) / variance_first_order(500) == pytest.approx(823, abs=1)
    assert variance_first_order(10) == pytest.approx(10 / 96)
    assert variance_first_order(6, sigma=0.5) == pytest.approx(1.5)


def test_rectification_increases_to_one():
    r = [rect(t) for t in range(5, 20000, 37)]
    assert all(a < b for a, b in itertools.pairwise(r))
    assert 0 < r[0] and r[-1] < 1


def test_first_inverse_moment_is_large():
    # E[1/g^2] diverges for g ~ N(0, 1): the Monte Carlo mean keeps growing with the sample size.
    rng = np.random.default_rng(0)
    means = [float(np.mean(1.0 / rng.standard_normal(n) ** 2)) for n in (10**3, 10**5, 10**7)]
    assert means[-1] > 100


def test_radam_fixture_thresholds():
    # beta2 = 0.5: rho_inf = 3, so no step is adaptive and RAdam is SGD with bias-corrected momentum.
    p = run(4.0)
    m, q = np.zeros(3), np.asarray(P0, dtype=np.float64)
    for t, g in enumerate(GRADS, start=1):
        m = 0.9 * m + 0.1 * np.asarray(g)
        q = q - 0.1 * m / (1 - 0.9**t)
    np.testing.assert_allclose(p, q, rtol=0, atol=1e-12)
    cfg = {"lr": 0.1, "beta1": 0.9, "beta2": 0.999}
    np.testing.assert_allclose(run(4.0, cfg, GRADS[:4]), run(5.0, cfg, GRADS[:4]), rtol=0, atol=0)
    assert not np.allclose(run(4.0, cfg, GRADS[:5]), run(5.0, cfg, GRADS[:5]))
    np.testing.assert_allclose(run(5.0, cfg), AFTER6_TORCH, rtol=0, atol=1e-12)
