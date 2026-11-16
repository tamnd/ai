# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://NEW-adam).
"""Tests of the NumPy reference: the worked example, the fixture shared with the PyTorch and JAX tests, the bias
correction, the bound on the step, the invariance to the scale of the gradients, epsilon hat, weight decay, AMSGrad,
the counterexample of Reddi et al., and agreement with torch.optim.Adam and torch.optim.AdamW when PyTorch is
installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adam import config, init_state, ratio_bound, step, step_eps_hat, step_uncorrected

# The worked example of the page: one parameter, theta_0 = 1, lr = 0.1, betas (0.5, 0.96), eps = 0, gradients 2, -2.
WCFG = {"lr": 0.1, "betas": (0.5, 0.96), "eps": 0.0}
WG = [2.0, -2.0]
WM = [1.0, -0.5]
WV = [0.16, 0.3136]
WMHAT = [2.0, -2.0 / 3.0]
WVHAT = [4.0, 4.0]
WTHETA = [0.9, 14.0 / 15.0]

# The shared fixture. The PyTorch and JAX tests hold the same numbers, and the PyTorch tests check them against
# torch.optim.Adam and torch.optim.AdamW.
P = {"w": [0.5, -1.5, 2.0], "b": [0.25]}
G = [{"w": [1.0, -2.0, 0.5], "b": [0.5]}, {"w": [0.5, 1.0, -0.25], "b": [-1.0]}, {"w": [-1.5, 0.5, 0.0], "b": [2.0]}]
CONFIGS = {"adam": {"lr": 0.1}, "l2": {"lr": 0.1, "weight_decay": 0.1},
           "adamw": {"lr": 0.1, "weight_decay": 0.1, "decoupled": True}, "amsgrad": {"lr": 0.1, "amsgrad": True}}
AFTER3 = {
    "adam": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527783689874684], "b": [0.14464479271810454]},
    "l2": {"w": [0.3108766333376067, -1.3489621467423747, 1.778235325837831], "b": [0.14103715505555778]},
    "adamw": {"w": [0.3030513976144773, -1.324932247497697, 1.7956327059805899], "b": [0.1388434391558989]},
    "amsgrad": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527886655267884], "b": [0.14464479271810454]},
}


def arrays(d):
    return {k: np.array(v, dtype=np.float64) for k, v in d.items()}


def run(cfg, params, grads):
    state = init_state(params)
    for g in grads:
        params, state = step(cfg, params, g, state)
    return params, state


def test_worked_example():
    cfg = config(**WCFG)
    p, s = {"x": np.array(1.0)}, init_state({"x": np.array(1.0)})
    for t in range(2):
        p, s = step(cfg, p, {"x": np.array(WG[t])}, s)
        np.testing.assert_allclose(s["m"]["x"], WM[t], rtol=1e-14)
        np.testing.assert_allclose(s["v"]["x"], WV[t], rtol=1e-14)
        np.testing.assert_allclose(s["m"]["x"] / (1 - 0.5 ** (t + 1)), WMHAT[t], rtol=1e-14)
        np.testing.assert_allclose(s["v"]["x"] / (1 - 0.96 ** (t + 1)), WVHAT[t], rtol=1e-14)
        np.testing.assert_allclose(p["x"], WTHETA[t], rtol=1e-14)
    # Without the bias correction the first step is 2.5 times the learning rate.
    p, _ = step_uncorrected(cfg, {"x": np.array(1.0)}, {"x": np.array(2.0)}, init_state({"x": np.array(1.0)}))
    np.testing.assert_allclose(p["x"], 0.75, rtol=1e-14)


def test_worked_example_weight_decay():
    x0 = {"x": np.array(1.0)}
    pw, _ = step(config(**WCFG, weight_decay=0.1, decoupled=True), x0, {"x": np.array(2.0)}, init_state(x0))
    pl, _ = step(config(**WCFG, weight_decay=0.1), x0, {"x": np.array(2.0)}, init_state(x0))
    np.testing.assert_allclose(pw["x"], 0.89, rtol=1e-14)  # 1 * (1 - 0.1 * 0.1) - 0.1
    np.testing.assert_allclose(pl["x"], 0.9, rtol=1e-14)  # the L2 term is normalized away at the first step


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(name):
    p, _ = run(config(**CONFIGS[name]), arrays(P), [arrays(g) for g in G])
    for k in P:
        np.testing.assert_allclose(p[k], AFTER3[name][k], rtol=1e-13)


@settings(max_examples=60, deadline=None)
@given(st.floats(0.0, 0.99), st.floats(0.5, 0.9999), st.floats(-100, 100).filter(lambda g: abs(g) > 1e-3),
       st.integers(1, 30))
def test_constant_gradient_step_is_lr(b1, b2, g, steps):
    cfg = config(lr=0.01, betas=(b1, b2), eps=0.0)
    x = {"x": np.array(0.0)}
    s = init_state(x)
    for _ in range(steps):
        new, s = step(cfg, x, {"x": np.array(g)}, s)
        np.testing.assert_allclose(x["x"] - new["x"], 0.01 * np.sign(g), rtol=1e-9)
        x = new


@settings(max_examples=60, deadline=None)
@given(st.floats(0.0, 0.95), st.floats(0.5, 0.9999), st.integers(1, 40), st.integers(0, 2**31))
def test_ratio_never_exceeds_bound(b1, b2, t, seed):
    g = np.random.default_rng(seed).standard_normal(t) * np.exp(np.random.default_rng(seed + 1).normal(0, 2, t))
    cfg = config(lr=1.0, betas=(b1, b2), eps=0.0)
    x = {"x": np.array(0.0)}
    s = init_state(x)
    for gi in g:
        new, s = step(cfg, x, {"x": np.array(gi)}, s)
        delta, x = abs(float(x["x"] - new["x"])), new
    assert delta <= ratio_bound(b1, b2, t) * (1 + 1e-9)


@pytest.mark.parametrize("b1,b2,limit", [(0.9, 0.999, 7.27), (0.9, 0.99, 2.35)])
def test_bound_attained_above_paper(b1, b2, limit):
    """The sequence g_{t-k} = (beta1/beta2)^k attains the bound; for the defaults it exceeds the paper's 3.16."""
    t = 20000
    g = (b1 / b2) ** np.arange(t)[::-1]
    cfg = config(lr=1.0, betas=(b1, b2), eps=1e-300)
    _, s = run(cfg, {"x": np.array(0.0)}, [{"x": np.array(gi)} for gi in g])
    ratio = s["m"]["x"] / (1 - b1**t) / np.sqrt(s["v"]["x"] / (1 - b2**t))
    np.testing.assert_allclose(ratio, ratio_bound(b1, b2, t), rtol=1e-9)
    assert abs(ratio - limit) < 0.01
    assert ratio > max(1.0, (1 - b1) / np.sqrt(1 - b2)) * 2


def test_single_spike():
    """A gradient that is zero at all steps but the last gives the ratio (1 - b1)/sqrt(1 - b2) of Section 2.1."""
    t = 2000
    g = [{"x": np.array(0.0)}] * (t - 1) + [{"x": np.array(1.0)}]
    _, s = run(config(lr=1.0, eps=1e-300), {"x": np.array(0.0)}, g)
    ratio = s["m"]["x"] / (1 - 0.9**t) / np.sqrt(s["v"]["x"] / (1 - 0.999**t))
    np.testing.assert_allclose(ratio, 0.1 / np.sqrt(0.001) * np.sqrt(1 - 0.999**t) / (1 - 0.9**t), rtol=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.floats(1e-3, 1e3), st.integers(0, 2**31))
def test_scale_invariance(c, seed):
    g = np.random.default_rng(seed).standard_normal((5, 4))
    cfg = config(lr=0.1, eps=0.0)
    a, _ = run(cfg, {"x": np.zeros(4)}, [{"x": gi} for gi in g])
    b, _ = run(cfg, {"x": np.zeros(4)}, [{"x": c * gi} for gi in g])
    np.testing.assert_allclose(a["x"], b["x"], rtol=1e-9)


def test_uncorrected_first_step():
    p, _ = step_uncorrected(config(lr=1.0, eps=0.0), {"x": np.array(0.0)}, {"x": np.array(1.0)},
                            init_state({"x": np.array(0.0)}))
    np.testing.assert_allclose(-p["x"], 0.1 / np.sqrt(0.001), rtol=1e-12)  # about 3.16, not 1


def test_eps_hat():
    """A Keras epsilon e_hat at step t is a PyTorch eps of e_hat / sqrt(1 - beta2^t): about 31.6 times larger at t = 1."""
    rng = np.random.default_rng(20261115)
    g = [{"x": rng.standard_normal(3) * 1e-6} for _ in range(4)]
    x = {"x": np.zeros(3)}
    sa, sb = init_state(x), init_state(x)
    pa = pb = x
    for t, gi in enumerate(g, start=1):
        pa, sa = step_eps_hat(config(lr=1.0, eps=1e-7), pa, gi, sa)
        pb, sb = step(config(lr=1.0, eps=1e-7 / np.sqrt(1 - 0.999**t)), pb, gi, sb)
        np.testing.assert_allclose(pa["x"], pb["x"], rtol=1e-12)
    p1, _ = step(config(lr=1.0, eps=1e-7), x, g[0], init_state(x))
    p2, _ = step_eps_hat(config(lr=1.0, eps=1e-7), x, g[0], init_state(x))
    assert not np.allclose(p1["x"], p2["x"], rtol=1e-3)


def test_l2_is_not_decoupled():
    """With L2 the decay passes through the preconditioner: a weight with large gradients is decayed less."""
    x = {"x": np.array([1.0, 1.0])}
    grads = [{"x": np.array([100.0, 0.01])}, {"x": np.array([-100.0, -0.01])}] * 5
    zero = run(config(lr=0.01), x, grads)[0]["x"]
    l2 = run(config(lr=0.01, weight_decay=0.1), x, grads)[0]["x"]
    w = run(config(lr=0.01, weight_decay=0.1, decoupled=True), x, grads)[0]["x"]
    decay_l2, decay_w = zero - l2, zero - w
    assert decay_l2[0] < 1e-2 * decay_l2[1]
    np.testing.assert_allclose(decay_w[0], decay_w[1], rtol=0.05)


def test_amsgrad_max_is_nondecreasing():
    rng = np.random.default_rng(20261115)
    cfg = config(lr=0.1, amsgrad=True)
    x = {"x": np.zeros(5)}
    s = init_state(x)
    prev = np.zeros(5)
    for _ in range(50):
        x, s = step(cfg, x, {"x": rng.standard_normal(5) * rng.uniform(0, 3)}, s)
        assert np.all(s["v_max"]["x"] >= prev) and np.all(s["v_max"]["x"] >= s["v"]["x"])
        prev = s["v_max"]["x"]


def reddi(amsgrad, steps=30000, c=3.0, alpha=0.5):
    """The online problem of Section 3 of Reddi et al.: f_t(x) = c x if t mod 3 = 1, else -x, on [-1, 1], with
    beta1 = 0, beta2 = 1 / (1 + c^2), the step alpha / sqrt(t), no bias correction and a projection, as in the paper."""
    b2, x, v, vmax = 1.0 / (1.0 + c * c), 0.0, 0.0, 0.0
    for t in range(1, steps + 1):
        g = c if t % 3 == 1 else -1.0
        v = b2 * v + (1 - b2) * g * g
        vmax = max(vmax, v)
        x = min(1.0, max(-1.0, x - alpha / np.sqrt(t) * g / np.sqrt(vmax if amsgrad else v)))
    return x


def test_reddi_counterexample():
    assert reddi(False) == 1.0  # Adam ends at the worst point of [-1, 1]
    assert reddi(True) < -0.99  # AMSGrad goes to the minimizer x = -1


@pytest.mark.parametrize("kw", [{"lr": -1.0}, {"betas": (1.0, 0.999)}, {"betas": (0.9, -0.1)}, {"eps": -1e-8},
                                {"weight_decay": -0.1}])
def test_config_rejects(kw):
    with pytest.raises(ValueError):
        config(**kw)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_matches_torch(name):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20261115)
    grads = [{"w": rng.standard_normal((3, 2)), "b": rng.standard_normal(3)} for _ in range(6)]
    params = {"w": rng.standard_normal((3, 2)), "b": rng.standard_normal(3)}
    kw = dict(CONFIGS[name])
    mine, _ = run(config(**kw), params, grads)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    cls = torch.optim.AdamW if kw.pop("decoupled", False) else torch.optim.Adam
    opt = cls(list(tp.values()), **kw)
    for g in grads:
        for k in tp:
            tp[k].grad = torch.tensor(g[k])
        opt.step()
    for k in params:
        np.testing.assert_allclose(mine[k], tp[k].detach().numpy(), rtol=1e-12, atol=1e-14)
