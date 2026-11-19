# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://2WYA9X).
"""Tests of the NumPy reference of Mamba: the fixtures, the gradients, the recurrent mode, the selection mechanism
and the counts of the page."""

import json
import math
import pathlib

import numpy as np
import pytest

import mamba as m

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]
SIZES = ("vocab", "d_model", "n_layers", "d_state", "d_conv", "expand", "dt_rank")


def cfg_of(c, **kw):
    return m.config(**{k: c["config"][k] for k in SIZES}, **kw)


def setup(disc="simple", seed=0, jitter=True, **kw):
    cfg = m.config(disc=disc, **kw)
    rng = np.random.default_rng(seed)
    p = m.init(rng, cfg, scale=0.3)
    if jitter:  # move the structured parameters off their initial values, so that their gradients are generic
        for k in p:
            if k.endswith(("A_log", "_D", "norm", "norm_f")):
                p[k] = p[k] + rng.normal(0.0, 0.2, p[k].shape)
    return cfg, p, rng


def test_fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    inp = np.load(FIX / "input-small.npz")
    exp = np.load(FIX / "expected-small.npz")
    cfg = m.config()
    logits, cache = m.forward(params, cfg, inp["tokens"][:, :-1])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.isclose(m.nll(logits, inp["tokens"][:, 1:]), exp["loss"], atol=1e-12)
    assert np.allclose(cache["layers"][-1]["mixer"]["hs"][:, -1], exp["last_state"], atol=1e-12)
    grads = m.backward(params, cfg, cache, logits, inp["tokens"][:, 1:])
    for k, v in grads.items():
        assert np.allclose(v, exp["d" + k], atol=1e-12), k


@pytest.mark.parametrize("disc", ["simple", "zoh"])
def test_gradients_match_finite_differences(disc):
    cfg, p, rng = setup(disc)
    tok = rng.integers(0, cfg["vocab"], (2, 8))
    logits, cache = m.forward(p, cfg, tok[:, :-1])
    g = m.backward(p, cfg, cache, logits, tok[:, 1:])
    for k, v in p.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in v.shape)
            old = v[idx]
            v[idx] = old + 1e-6
            a = m.lm_loss(p, cfg, tok)
            v[idx] = old - 1e-6
            b = m.lm_loss(p, cfg, tok)
            v[idx] = old
            assert abs((a - b) / 2e-6 - g[k][idx]) < 1e-7, k


@pytest.mark.parametrize("disc", ["simple", "zoh"])
def test_step_mode_equals_the_parallel_forward(disc):
    cfg, p, rng = setup(disc)
    tok = rng.integers(0, cfg["vocab"], (3, 9))
    logits, _ = m.forward(p, cfg, tok)
    state, out = m.init_state(cfg, 3), []
    for t in range(tok.shape[1]):
        o, state = m.step(p, cfg, tok[:, t], state)
        out.append(o)
    assert np.allclose(np.stack(out, 1), logits, atol=1e-12)


def test_state_size_does_not_grow_with_the_sequence():
    cfg, p, rng = setup()
    state = m.init_state(cfg, 1)
    sizes = []
    for t in range(50):
        _, state = m.step(p, cfg, rng.integers(0, cfg["vocab"], 1), state)
        sizes.append(sum(a.size + b.size for a, b in state))
    assert len(set(sizes)) == 1
    assert sizes[0] == cfg["n_layers"] * cfg["d_inner"] * (cfg["d_conv"] - 1 + cfg["d_state"])


def test_causal():
    cfg, p, rng = setup()
    tok = rng.integers(0, cfg["vocab"], (1, 10))
    a, _ = m.forward(p, cfg, tok)
    tok2 = tok.copy()
    tok2[0, 6] = (tok2[0, 6] + 1) % cfg["vocab"]
    b, _ = m.forward(p, cfg, tok2)
    assert np.allclose(a[:, :6], b[:, :6], atol=1e-14)
    assert not np.allclose(a[:, 6:], b[:, 6:])


def test_parameters_depend_on_the_input():
    """Selection: Delta, B and C differ between positions, so the SSM is time-varying (Algorithm 2)."""
    cfg, p, rng = setup()
    _, cache = m.forward(p, cfg, rng.integers(0, cfg["vocab"], (1, 6)))
    c = cache["layers"][0]["mixer"]
    for name in ("dt", "Bm", "C"):
        assert np.ptp(c[name], axis=1).max() > 1e-3, name


def test_not_a_convolution():
    """A time-invariant SSM is linear in its input and equals a convolution; the selective scan is not linear."""
    rng = np.random.default_rng(1)
    I, N, T = 3, 4, 7
    A = -np.exp(rng.normal(size=(I, N)))
    Wb, Wc, wd = rng.normal(size=(I, N)), rng.normal(size=(I, N)), rng.normal(size=I)

    def ssm(x):
        dt = m.softplus(x * wd)  # Delta from the input, as in s_Delta
        Bm, C = np.tanh(x @ Wb), np.tanh(x @ Wc)
        dA, dB = m.discretize(dt[None], A, Bm[None], "zoh")
        return m.selective_scan(dA, dB, x[None], C[None])[0][0]

    x1, x2 = rng.normal(size=(T, I)), rng.normal(size=(T, I))
    assert not np.allclose(ssm(x1 + x2), ssm(x1) + ssm(x2), atol=1e-3)


def test_theorem1_gate():
    """With N = 1, A = -1, B = 1 and zero-order hold, the recurrence is h = (1 - g) h + g x with g = sigmoid(w x)."""
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=(1, 12, 1)), 1.7
    s = w * x[..., 0]
    dA, dB = m.discretize(m.softplus(s)[..., None], -np.ones((1, 1)), np.ones((1, 12, 1)), "zoh")
    y, _ = m.selective_scan(dA, dB, x, np.ones((1, 12, 1)))
    h, gate = 0.0, []
    for t in range(12):
        g = 1.0 / (1.0 + math.exp(-s[0, t]))
        h = (1 - g) * h + g * x[0, t, 0]
        gate.append(h)
    assert np.allclose(y[0, :, 0], gate, atol=1e-13)


def test_simple_rule_is_zoh_to_first_order():
    """B_bar = Delta B and the ZOH B_bar = (exp(Delta A) - 1) / A B differ by a factor 1 + Delta A / 2 + (Delta A)^2 / 6 + ...."""
    A, Bm = -np.array([[1.0, 3.0]]), np.ones((1, 1, 2))
    for d in (1e-2, 1e-3, 1e-4):
        _, simple = m.discretize(np.full((1, 1, 1), d), A, Bm, "simple")
        _, zoh = m.discretize(np.full((1, 1, 1), d), A, Bm, "zoh")
        assert np.allclose(zoh / simple, 1 + d * A / 2, atol=2 * d * d)


def test_large_delta_resets_and_small_delta_ignores():
    """Section 3.5.2: a large Delta resets the state to the current input; a small one keeps the state."""
    A, Bm, x = -np.ones((1, 1)), np.ones((1, 2, 1)), np.array([[[5.0], [-2.0]]])
    for dt, want in ((np.array([[[50.0], [50.0]]]), -2.0), (np.array([[[50.0], [1e-9]]]), 5.0)):
        dA, dB = m.discretize(dt, A, Bm, "zoh")
        y, _ = m.selective_scan(dA, dB, x, np.ones((1, 2, 1)))
        assert np.isclose(y[0, 1, 0], want, atol=1e-6)


def test_init():
    cfg, p, _ = setup(jitter=False, d_state=16)
    assert np.allclose(-np.exp(p["l0_A_log"][0]), -np.arange(1, 17))  # S4D-Real: A_n = -(n + 1)
    dt = m.softplus(p["l0_dt_b"])
    assert dt.min() >= 1e-3 - 1e-12 and dt.max() <= 1e-1 + 1e-12


def test_param_count_matches_shapes():
    for kw in ({}, {"d_model": 20, "d_state": 7, "d_conv": 4, "expand": 3}, {"dt_rank": 5, "n_layers": 3}):
        cfg = m.config(**kw)
        assert m.param_count(cfg) == sum(math.prod(s) for s in m.param_shapes(cfg).values())


def test_released_model_sizes():
    """eq-params on the released configs; the 2.8B count is the one that Hugging Face reports for its weights."""
    cfg = m.config(vocab=50280, d_model=2560, n_layers=64, d_state=16, d_conv=4)
    assert cfg["dt_rank"] == 160 and m.param_count(cfg) == 2_768_345_600
    assert m.param_count(m.config(vocab=50280, d_model=768, n_layers=24, d_state=16, d_conv=4)) == 129_135_360


def test_ablation_sizes():
    """The parameter counts of Tables 9 and 10 of the paper, the 350M setting with 48 blocks of width 1024."""
    base = dict(vocab=50280, d_model=1024, n_layers=48, d_conv=4)
    assert round(m.param_count(m.config(d_state=16, dt_rank=64, **base)) / 1e6, 1) == 371.5
    assert round(m.param_count(m.config(d_state=1, dt_rank=64, **base)) / 1e6, 1) == 367.1
    assert round(m.param_count(m.config(d_state=16, dt_rank=1, **base)) / 1e6, 1) == 359.1


def test_two_blocks_match_a_transformer_layer():
    """Section 3.4: a block has 3 E D^2 parameters in its projections, so two blocks with E = 2 have 12 D^2."""
    D = 64
    cfg = m.config(d_model=D, expand=2)
    s = m.block_shapes(cfg)
    assert 2 * sum(math.prod(s[k]) for k in ("in_proj", "out_proj")) == 12 * D * D


def test_flops_by_counting():
    cfg, p, rng = setup()
    T = 5
    D, I, N, K, R, V = (cfg[k] for k in ("d_model", "d_inner", "d_state", "d_conv", "dt_rank", "vocab"))
    per_token = 2 * (D * 2 * I + I * K + I * (R + 2 * N) + R * I + 2 * I * N + I * D)
    assert m.flops(cfg, 2, T) == 2 * T * (cfg["n_layers"] * per_token + 2 * D * V)


def test_worked_example():
    """The worked example of the page: N = 1, A = -1, B = C = 1, x = (5, -2), Delta = (log 2, log 4)."""
    A, ones, x = -np.ones((1, 1)), np.ones((1, 2, 1)), np.array([[[5.0], [-2.0]]])
    dt = np.log([[[2.0], [4.0]]])
    assert np.allclose(m.softplus(np.array([0.0, math.log(3.0)])), np.log([2.0, 4.0]))
    dA, dB = m.discretize(dt, A, ones, "zoh")
    assert np.allclose(dA[0, :, 0, 0], [0.5, 0.25]) and np.allclose(dB[0, :, 0, 0], [0.5, 0.75])
    y, _ = m.selective_scan(dA, dB, x, ones)
    assert np.allclose(y[0, :, 0], [2.5, -0.875])
    S = np.cumsum(dt[0, :, 0])
    M = np.tril(np.exp(-(S[:, None] - S[None, :])) * dB[0, None, :, 0, 0])
    assert np.allclose(M, [[0.5, 0.0], [0.125, 0.75]]) and np.allclose(M @ x[0, :, 0], [2.5, -0.875])
    dA, dB = m.discretize(dt, A, ones, "simple")
    y, _ = m.selective_scan(dA, dB, x, ones)
    assert np.allclose(y[0, :, 0], [5 * math.log(2), 5 * math.log(2) / 4 - 2 * math.log(4)])
    assert np.isclose(y[0, 1, 0], -1.9062, atol=1e-4)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg, k, mod = cfg_of(c), c["config"], c["model"]
    assert cfg["d_inner"] == k["d_inner"]
    assert m.param_count(cfg) == mod["params"] == sum(math.prod(s) for s in m.param_shapes(cfg).values())
    assert m.flops(cfg, k["B"], k["T"]) == mod["flops_forward"]
    s = m.block_shapes(cfg)
    assert cfg["n_layers"] * sum(math.prod(s[n]) for n in ("in_proj", "out_proj")) == mod["projection_params"]
    state = m.init_state(cfg, k["B"])
    assert sum(a.size + b.size for a, b in state) == mod["recurrent_state"]
    if c["kind"] != "named":
        blocks = c["blocks"]
        rng = np.random.default_rng(0)
        p = m.init(rng, cfg)
        logits, cache = m.forward(p, cfg, rng.integers(0, cfg["vocab"], (k["B"], k["T"])))
        assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
        mix, want = cache["layers"][0]["mixer"], blocks["mixer"]["internals"]
        for name, key in (("x", "x"), ("z", "z"), ("delta", "dt"), ("B_t", "Bm"), ("C_t", "C")):
            assert list(mix[key].shape) == want[name], name
        assert list(mix["hs"][:, -1].shape) == want["h"]
        assert list(state[0][0].shape) == want["conv_state"] and list(state[0][1].shape) == want["h"]


def test_default_config_is_the_example():
    assert m.config() == cfg_of(SHAPES[0])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    return m.forward, (m.init(rng, cfg), cfg, rng.integers(0, cfg["vocab"], (c["config"]["B"], c["config"]["T"])))
