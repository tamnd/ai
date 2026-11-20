# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Tests of the NumPy reference: shapes and counts against the shape contract and the paper, the shared fixture, the
gradients of the static network and of the HyperRNN against finite differences, the layout of the kernels and the
tiling of eq. 3, the affine map from embeddings to kernels, the general and the memory-efficient HyperRNN, the
initialization of the HyperLSTM, and the worked example."""

import json
import math
import pathlib

import numpy as np
import pytest

import hypernetwork as H

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())
# aik flopcount imports this module in a directory of its own, without the fixtures, and calls flop_case only.
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"] if (FIXTURES / "shapes.json").exists() else []


def test_the_shape_contract_is_there():
    assert len(SHAPES) >= 3


def cfg_of(c):
    k = c["config"]
    return H.config(k["c_in"], k["n_basic"], k["mult"], k["fsize"], k["depth"], k["n_z"], k["size"], k["classes"])


def images(rng, cfg, B):
    return rng.normal(size=(B, cfg["c_in"], cfg["size"], cfg["size"]))


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    return H.forward, (H.init(rng, cfg), images(rng, cfg, c["config"]["B"]), cfg)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg, m, k = cfg_of(c), c["model"], c["config"]
    assert H.param_count(cfg) == m["params"]
    assert sum(math.prod(s) for s in H.param_shapes(cfg).values()) == m["params"]
    assert H.flops(cfg, k["B"]) == m["flops_forward"]
    N, f = cfg["n_basic"], cfg["fsize"]
    assert H.hyper_param_count(cfg["n_z"], cfg["d"], N, N, f, H.n_embeddings(cfg)) == m["hyper_params"]
    assert H.one_layer_param_count(cfg["n_z"], N, N, f) == m["one_layer_params"]
    rng = np.random.default_rng(0)
    p = H.init(rng, cfg)
    ws, gen = H.kernels(p, cfg)
    assert sum(w.size for w in ws) == m["generated_weights"]
    if c["kind"] != "named":
        b = c["blocks"]
        logits, cache = H.forward(p, images(rng, cfg, k["B"]), cfg)
        assert list(cache["hs"][0].shape) == b["stem"]["outputs"]["h"]
        assert list(gen["a"].shape) == b["hyper"]["internals"]["a"]
        assert list(H.generate(p, p["z"])[1].shape) == b["hyper"]["outputs"]["K"]
        assert list(ws[0].shape) == b["conv"]["internals"]["w"]
        assert list(cache["pooled"].shape) == b["head"]["internals"]["pooled"]
        assert list(logits.shape) == b["head"]["outputs"]["logits"]


def test_default_config_is_the_example():
    ex = next(c for c in SHAPES if c["name"] == "example")
    assert H.config() == cfg_of(ex)


def test_the_counts_of_the_paper():
    """Section 4.1 and Table 7: a 7x7x16x16 kernel of 12,544 weights from N_z = 4, by a hypernetwork of 4,240
    parameters, 4,244 with the embedding; a one-layer hypernetwork would need N_z times the kernel."""
    assert 7 * 7 * 16 * 16 == 12544
    assert H.hyper_param_count(4, 4, 16, 16, 7, 1) == 4244
    assert H.hyper_param_count(4, 4, 16, 16, 7, 1) - 4 == 4240
    assert H.one_layer_param_count(4, 16, 16, 7) == 4 * 12544 == 50176


def fixture():
    p, i, e = (np.load(FIXTURES / f"{n}-small.npz") for n in ("params", "input", "expected"))
    return dict(p), dict(i), dict(e)


def part(d, prefix):
    return {k[len(prefix) + 1 :]: v for k, v in d.items() if k.startswith(prefix + ".")}


def test_shared_fixture():
    P, Inp, E = fixture()
    cfg, sp, se = H.config(), part(P, "static"), part(E, "static")
    assert set(sp) == set(H.param_shapes(cfg))
    logits, cache = H.forward(sp, Inp["static.x"], cfg)
    assert np.allclose(logits, se["logits"], atol=1e-12)
    assert np.isclose(H.loss(logits, Inp["static.y"]), se["loss"], atol=1e-12)
    g, dx = H.backward(sp, cfg, cache, H.loss_grad(logits, Inp["static.y"]))
    for k, v in g.items():
        assert np.allclose(v, se["d" + k], atol=1e-12), k
    assert np.allclose(dx, se["dx"], atol=1e-12)
    rp, re = part(P, "rnn"), part(E, "rnn")
    out, steps = H.hyper_rnn(rp, Inp["rnn.x"], return_cache=True)
    assert np.allclose(out, re["out"], atol=1e-12)
    assert np.isclose(H.rnn_loss(out, Inp["rnn.target"]), re["loss"], atol=1e-12)
    g, dx = H.hyper_rnn_backward(rp, Inp["rnn.x"], steps, (out - Inp["rnn.target"]) / out.size)
    for k, v in g.items():
        assert np.allclose(v, re["d" + k], atol=1e-12), k
    assert np.allclose(dx, re["dx"], atol=1e-12)
    lp = part(P, "lstm")
    assert np.allclose(H.hyper_lstm(lp, Inp["rnn.x"], ln=True), E["lstm.out_ln"], atol=1e-12)
    assert np.allclose(H.hyper_lstm(lp, Inp["rnn.x"], ln=False), E["lstm.out"], atol=1e-12)


def numeric(fn, p, k, idx, eps=1e-6):
    q = {a: b.copy() for a, b in p.items()}
    q[k][idx] += eps
    up = fn(q)
    q[k][idx] -= 2 * eps
    return (up - fn(q)) / (2 * eps)


@pytest.mark.parametrize("kw", [{}, {"mult": 1, "depth": 3}, {"fsize": 5, "n_z": 3, "c_in": 2}])
def test_static_gradient_by_finite_differences(kw):
    cfg = H.config(**kw)
    rng = np.random.default_rng(1)
    p = H.init(rng, cfg)
    x, y = images(rng, cfg, 3), rng.integers(0, cfg["classes"], 3)
    logits, cache = H.forward(p, x, cfg)
    g, dx = H.backward(p, cfg, cache, H.loss_grad(logits, y))
    fn = lambda q: H.loss(H.forward(q, x, cfg)[0], y)  # noqa: E731
    for k in p:
        for _ in range(3):
            idx = tuple(int(rng.integers(0, s)) for s in p[k].shape)
            assert np.isclose(g[k][idx], numeric(fn, p, k, idx), rtol=1e-5, atol=1e-8), k
    xq = {"x": x}
    num = numeric(lambda q: H.loss(H.forward(p, q["x"], cfg)[0], y), xq, "x", (1, 0, 2, 3))
    assert np.isclose(dx[1, 0, 2, 3], num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("seed", [2, 3])
def test_hyper_rnn_gradient_by_finite_differences(seed):
    cfg = H.config_rnn(n_x=2, n_h=4, n_hat=3, n_z=3)
    rng = np.random.default_rng(seed)
    p = H.init_rnn(rng, cfg, scale=0.8)
    x, tg = rng.normal(size=(2, 6, 2)), rng.normal(size=(2, 6, 4))
    out, steps = H.hyper_rnn(p, x, return_cache=True)
    g, dx = H.hyper_rnn_backward(p, x, steps, (out - tg) / out.size)
    fn = lambda q: H.rnn_loss(H.hyper_rnn(q, x), tg)  # noqa: E731
    for k in p:
        for _ in range(3):
            idx = tuple(int(rng.integers(0, s)) for s in p[k].shape)
            assert np.isclose(g[k][idx], numeric(fn, p, k, idx), rtol=1e-5, atol=1e-9), k
    num = numeric(lambda q: H.rnn_loss(H.hyper_rnn(p, q["x"]), tg), {"x": x}, "x", (0, 1, 1))
    assert np.isclose(dx[0, 1, 1], num, rtol=1e-5, atol=1e-9)


def test_kernel_layout():
    """The paper's K^j, (N_in f) x (N_out f), holds the weight of input i, output o and position (u, v) at row i f + u
    and column o f + v, and the generated kernel convolves like any other."""
    rng = np.random.default_rng(4)
    N_in, N_out, f, Nz = 3, 2, 3, 2
    p = {"W_in": rng.normal(size=(N_in, Nz, Nz)), "B_in": rng.normal(size=(N_in, Nz)),
         "W_out": rng.normal(size=(f, N_out * f, Nz)), "B_out": rng.normal(size=(f, N_out * f))}  # fmt: skip
    _, K = H.generate(p, rng.normal(size=(1, Nz)))
    w, Kj = H.to_filters(K, f)[0], H.kernel_matrix(K)[0]
    assert Kj.shape == (N_in * f, N_out * f) and w.shape == (N_out, N_in, f, f)
    for i in range(N_in):
        for o in range(N_out):
            assert np.array_equal(Kj[i * f : (i + 1) * f, o * f : (o + 1) * f], w[o, i])
    assert np.array_equal(H.from_filters(H.to_filters(K, f), f), K)
    x = rng.normal(size=(1, N_in, 4, 4))
    y = H.conv2d(x, w, np.zeros(N_out))
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    assert np.isclose(y[0, 1, 2, 3], sum(w[1, i, u, v] * xp[0, i, 2 + u, 3 + v]
                                         for i in range(N_in) for u in range(f) for v in range(f)))  # fmt: skip


def test_tiling_of_eq3():
    """Eq. 3: a kernel with N_in = 32 and N_out = 64 is tiled from 8 basic kernels of 16, two rows of four."""
    rng = np.random.default_rng(5)
    basic = rng.normal(size=(8, 16, 16, 3, 3))
    w = H.tile(basic, 2, 4)
    assert w.shape == (64, 32, 3, 3)
    for r in range(2):
        for c in range(4):
            assert np.array_equal(w[16 * c : 16 * (c + 1), 16 * r : 16 * (r + 1)], basic[4 * r + c])
    assert np.array_equal(H.untile(w, 2, 4), basic)
    with pytest.raises(ValueError):
        H.tile(basic, 3)


def test_kernels_are_an_affine_function_of_the_embedding():
    """The two-layer linear hypernetwork is the one-layer map z -> (W_out W_i) z + (W_out B_i + B_out): every slice
    of every kernel lies in the affine space B_out + span of the d slices of W_out (the derivation linear-hypernetwork)."""
    cfg = H.config(n_basic=3, mult=2, depth=3, n_z=2, fsize=3)
    rng = np.random.default_rng(6)
    p = H.init(rng, cfg)
    _, K = H.generate(p, p["z"])
    one = np.einsum("uwd,idz->iuwz", p["W_out"], p["W_in"])
    bias = np.einsum("uwd,id->iuw", p["W_out"], p["B_in"]) + p["B_out"]
    assert np.allclose(K, np.einsum("iuwz,ez->eiuw", one, p["z"]) + bias)
    slices = (K - p["B_out"]).reshape(-1, p["B_out"].size)  # every K_i of every embedding, flattened
    assert slices.shape[0] == H.n_embeddings(cfg) * 3 > cfg["d"]
    assert np.linalg.matrix_rank(slices) == cfg["d"]


def test_eq7_is_an_elementwise_product():
    rng = np.random.default_rng(7)
    W, d, h = rng.normal(size=(5, 4)), rng.normal(size=5), rng.normal(size=4)
    assert np.allclose((d[:, None] * W) @ h, d * (W @ h))
    assert np.allclose(np.diag(d) @ W @ h, d * (W @ h))


def test_the_scaled_form_is_the_full_form_with_tied_slices():
    """eq-scaled is eq-full with slice k of W_hz equal to diag(W_hz[:, k]) W_h, and the same for W_x."""
    cfg = H.config_rnn()
    rng = np.random.default_rng(8)
    p = H.init_rnn(rng, cfg)
    x = rng.normal(size=(3, 5, cfg["n_x"]))
    full, Whs, Wxs = H.hyper_rnn_full(H.full_from_scaled(p), x)
    assert np.allclose(full, H.hyper_rnn(p, x), atol=1e-14)
    # what the full form builds at every step, against the vectors of the scaled form
    assert Whs.size + Wxs.size == 3 * 5 * cfg["n_h"] * (cfg["n_h"] + cfg["n_x"])


def test_memory_of_the_two_forms():
    """Section 3.2: the parameters of eq-full grow as N_z times those of a basic RNN; those of eq-scaled add 3 N_h N_z
    to them."""
    for Nx, Nh, Nhat, Nz in [(3, 5, 4, 2), (64, 1000, 128, 4), (205, 1800, 128, 16)]:
        cfg = H.config_rnn(Nx, Nh, Nhat, Nz)
        hyper = Nhat * Nhat + Nhat * (Nh + Nx) + Nhat + 3 * Nz * Nhat + 2 * Nz
        basic = Nh * Nh + Nh * Nx + Nh
        assert H.rnn_param_count(cfg, full=True) == Nz * (Nh * Nh + Nh * Nx) + Nh * Nz + Nh + hyper
        assert H.rnn_param_count(cfg) == basic + 3 * Nh * Nz + hyper
    cfg = H.config_rnn(64, 1000, 128, 4)
    assert H.rnn_param_count(cfg, full=True) > 4 * (1000 * 1000 + 1000 * 64)


def lstm_step_reference(p, h, c, x, ln):
    norm = H.layer_norm if ln else (lambda a: a)
    gates = norm(np.einsum("ynj,bj->ybn", p["W_h"], h) + np.einsum("ynj,bj->ybn", p["W_x"], x) + p["b_0"][:, None])
    c = H.sigmoid(gates[2]) * c + H.sigmoid(gates[0]) * np.tanh(gates[1])
    return H.sigmoid(gates[3]) * np.tanh(norm(c)), c


@pytest.mark.parametrize("ln", [True, False])
def test_the_hyper_lstm_with_unit_scales_is_an_lstm(ln):
    """With every d equal to 1 and b(z_b) = b_0, the HyperLSTM is the (layer-normalized) LSTM of W_h, W_x and b_0."""
    cfg = H.config_rnn(n_x=3, n_h=6, n_hat=4, n_z=3)
    rng = np.random.default_rng(9)
    p = {k: rng.normal(0.0, 0.5, s) for k, s in H.lstm_param_shapes(cfg).items()}
    p |= {"W_hh": np.zeros_like(p["W_hh"]), "b_hh": np.ones_like(p["b_hh"]), "W_hx": np.zeros_like(p["W_hx"]),
          "b_hx": np.ones_like(p["b_hx"]), "W_hz": np.full_like(p["W_hz"], 1 / 3), "W_xz": np.full_like(p["W_xz"], 1 / 3),
          "W_bz": np.zeros_like(p["W_bz"])}  # fmt: skip
    x = rng.normal(size=(2, 7, 3))
    h, c, outs = np.zeros((2, 6)), np.zeros((2, 6)), []
    for t in range(7):
        h, c = lstm_step_reference(p, h, c, x[:, t], ln)
        outs.append(h)
    assert np.allclose(H.hyper_lstm(p, x, ln=ln), np.stack(outs, 1), atol=1e-12)


def test_the_initialization_of_appendix_a23():
    """W_hz = W_xz = 0.1 / N_z with z_h = z_x = 1 make every weight scaling vector 0.1, at every step until W_hh and
    W_hx move from zero; W_bz = 0 and b_0 = 0 make the generated biases zero; the fixed matrices are orthogonal."""
    cfg = H.config_rnn(n_x=5, n_h=8, n_hat=4, n_z=4)
    rng = np.random.default_rng(10)
    p = H.init_lstm(rng, cfg)
    _, ds = H.hyper_lstm(p, rng.normal(size=(3, 6, 5)), return_scales=True)
    assert np.allclose(ds, 0.1, atol=1e-15)
    for k in ("W_h", "W_hhat"):
        for y in range(4):
            assert np.allclose(p[k][y] @ p[k][y].T, np.eye(p[k].shape[1]))
    for y in range(4):
        assert np.allclose(p["W_x"][y].T @ p["W_x"][y], np.eye(5))
    assert np.abs(p["W_hb"]).max() < 0.06 and not p["W_bz"].any() and not p["b_0"].any()


def test_worked_example():
    """K = W_out (W_1 z + B_1) + B_out with W_1 = 2, B_1 = 1, W_out = (1, -1), B_out = (0.5, 0)."""
    ex = H.worked_example()
    assert np.allclose(ex[0.0], [1.5, -1.0])
    assert np.allclose(ex[1.0], [3.5, -3.0])
    assert np.allclose(ex[-0.5], [0.5, 0.0])


def test_config_errors():
    bad: list[dict] = [{"fsize": 4}, {"depth": 0}, {"n_z": 1.5}]
    for kw in bad:
        with pytest.raises(ValueError):
            H.config(**kw)
    with pytest.raises(ValueError):
        H.config_rnn(n_h=0)
