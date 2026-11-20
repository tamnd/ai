# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://S263A8).
"""Tests of the NumPy reference: shapes and counts, the models of the paper, the shared fixture, the gradient of
eq-loss, the invariance and the equivariance, masks and padding, Lemma 3 and the worked example."""

import json
import math
import pathlib

import numpy as np
import pytest

from deepsets import (
    ACTS,
    POOLS,
    backward,
    config,
    equivariant,
    equivariant_matrix,
    flops,
    forward,
    init,
    loss,
    param_count,
    param_shapes,
    pool,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d_in"], k["width"], k["n_layers"], k["n_classes"], **kw)


def sets_of(rng, cfg, B, M):
    return rng.normal(size=(B, M, cfg["d_in"]))


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg = cfg_of(c)
    m, k = c["model"], c["config"]
    assert param_count(cfg) == m["params"]
    assert sum(math.prod(s) for s in param_shapes(cfg).values()) == m["params"]
    assert flops(cfg, k["B"], k["set_size"]) == m["flops_forward"]
    if c["kind"] != "named":
        rng = np.random.default_rng(0)
        logits, cache = forward(init(rng, cfg), cfg, sets_of(rng, cfg, k["B"], k["set_size"]))
        blocks = c["blocks"]
        assert list(logits.shape) == blocks["readout"]["outputs"]["logits"]
        assert list(cache["layers"][0]["p"].shape) == blocks["first"]["internals"]["pooled"]
        assert list(cache["layers"][0]["a"].shape) == blocks["first"]["outputs"]["h"]
        if cfg["n_layers"] > 1:
            assert list(cache["layers"][1]["p"].shape) == blocks["equivariant"]["internals"]["pooled"]
        assert list(cache["s"].shape) == blocks["readout"]["internals"]["s"]
        assert list(cache["r"].shape) == blocks["readout"]["internals"]["r"]


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_the_models_of_the_paper():
    named = {c["name"]: cfg_of(c) for c in SHAPES if c["kind"] == "named"}
    assert param_count(named["pointcloud_1000"]) == 340_520
    assert param_count(named["pointcloud_5000"]) == 1_336_360
    assert flops(named["pointcloud_1000"], 1, 1000) == 264_095_232
    # the products with Gamma, once per set
    no_gamma = config(3, 256, 3, 40, interact=False)
    assert flops(named["pointcloud_1000"], 1, 1000) - flops(no_gamma, 1, 1000) == 263_680
    # the count does not depend on the set size, and the cost is linear in it
    c = named["pointcloud_1000"]
    assert flops(c, 1, 2000) - flops(c, 1, 1000) == flops(c, 1, 1000) - flops(c, 1, 0)


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    assert set(params) == set(param_shapes(cfg))
    logits, cache = forward(params, cfg, inp["x"], inp["mask"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["features"], want["features"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["s"], want["pooled"], rtol=1e-13, atol=1e-14)
    assert loss(params, cfg, inp["x"], inp["labels"], inp["mask"]) == pytest.approx(float(want["loss"]), rel=1e-13)
    grads, dx = backward(params, cfg, cache, logits, inp["labels"], inp["x"])
    assert set(grads) == set(params)
    for k, v in grads.items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-12, atol=1e-14, err_msg=k)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-12, atol=1e-14)
    assert np.all(dx[1, 4:] == 0)


CASES = [
    dict(pool=p, readout=r, act=a, interact=i, head=h)
    for p in POOLS
    for r, a, i, h in [("sum", "tanh", True, "set"), ("mean", "relu", True, "set"), ("max", "tanh", False, "set"),
                       ("sum", "tanh", True, "element"), ("sum", "relu", False, "element")]  # fmt: skip
]


@pytest.mark.parametrize("kw", CASES, ids=lambda kw: "-".join(str(v) for v in kw.values()))
def test_gradient_by_finite_differences(kw):
    rng = np.random.default_rng(4)
    cfg = config(3, 4, 2, 3, **kw)
    params = init(rng, cfg, scale=0.7)
    B, M = 3, 5
    mask = np.ones((B, M))
    mask[1, 3:] = 0
    mask[2, 1:] = 0
    x = rng.normal(size=(B, M, 3)) * mask[..., None]
    labels = rng.integers(0, 3, B) if cfg["head"] == "set" else np.array([0, 2, 0])
    logits, cache = forward(params, cfg, x, mask)
    grads, dx = backward(params, cfg, cache, logits, labels, x)
    eps = 1e-6
    for name, p in params.items():
        num = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = loss(params, cfg, x, labels, mask)
            p[idx] = old - eps
            down = loss(params, cfg, x, labels, mask)
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)
        np.testing.assert_allclose(grads[name], num, rtol=1e-5, atol=1e-8, err_msg=name)
    num = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        if mask[idx[:2]] == 0:
            continue
        old = x[idx]
        x[idx] = old + eps
        up = loss(params, cfg, x, labels, mask)
        x[idx] = old - eps
        down = loss(params, cfg, x, labels, mask)
        x[idx] = old
        num[idx] = (up - down) / (2 * eps)
    np.testing.assert_allclose(dx, num, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("p", POOLS)
@pytest.mark.parametrize("act", ACTS)
def test_invariance_and_equivariance(p, act):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 7, 3))
    perm = rng.permutation(7)
    cfg = config(3, 6, 3, 4, pool=p, readout=p, act=act)
    params = init(rng, cfg, 0.5)
    np.testing.assert_allclose(forward(params, cfg, x[:, perm])[0], forward(params, cfg, x)[0], rtol=1e-12, atol=1e-13)
    cfg = config(3, 6, 3, 4, pool=p, act=act, head="element")
    params = init(rng, cfg, 0.5)
    s = forward(params, cfg, x)[0]
    np.testing.assert_allclose(forward(params, cfg, x[:, perm])[0], s[:, perm], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("p", POOLS)
def test_padding_does_not_change_the_output(p):
    rng = np.random.default_rng(6)
    for head in ("set", "element"):
        cfg = config(2, 5, 2, 3, pool=p, readout=p, head=head)
        params = init(rng, cfg, 0.5)
        x = rng.normal(size=(1, 4, 2))
        pad = np.concatenate([x, rng.normal(size=(1, 3, 2))], 1)  # the padded slots hold anything
        mask = np.array([[1, 1, 1, 1, 0, 0, 0]])
        alone, padded = forward(params, cfg, x)[0], forward(params, cfg, pad, mask)[0]
        if head == "set":
            np.testing.assert_allclose(padded, alone, rtol=1e-12, atol=1e-13)
        else:
            np.testing.assert_allclose(padded[:, :4], alone, rtol=1e-12, atol=1e-13)
            assert np.all(np.isneginf(padded[:, 4:]))


def test_pools_skip_the_padding():
    h = np.array([[[1.0, -2.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])
    np.testing.assert_array_equal(pool(h, mask, "sum"), [[4.0, 3.0]])
    np.testing.assert_array_equal(pool(h, mask, "mean"), [[2.0, 1.5]])
    np.testing.assert_array_equal(pool(h, mask, "max"), [[3.0, 5.0]])


def test_lemma3_dense_matrix():
    rng = np.random.default_rng(7)
    for M in range(1, 8):
        lam, gam = rng.normal(size=2)
        x = rng.normal(size=M)
        Theta = equivariant_matrix(lam, gam, M)
        z, _ = equivariant(x[None, :, None], np.ones((1, M)), np.array([[lam]]), np.array([[-gam]]), np.zeros(1), "sum")
        np.testing.assert_allclose(z[0, :, 0], Theta @ x, rtol=1e-12)
        P = np.eye(M)[rng.permutation(M)]
        np.testing.assert_allclose(P @ Theta, Theta @ P, rtol=1e-12)
    # a matrix with unequal diagonal entries does not commute with a transposition
    Theta = np.diag([1.0, 2.0, 3.0])
    P = np.eye(3)[[1, 0, 2]]
    assert not np.allclose(P @ Theta, Theta @ P)


def test_worked_example():
    x = np.array([1.0, 2.0, 4.0])
    Theta = equivariant_matrix(1.0, -1.0 / 3.0, 3)
    np.testing.assert_allclose(np.diag(Theta), [2 / 3] * 3)
    y = Theta @ x
    np.testing.assert_allclose(y, [-4 / 3, -1 / 3, 5 / 3])
    np.testing.assert_allclose(Theta @ x[[2, 0, 1]], [5 / 3, -4 / 3, -1 / 3])
    mask = np.ones((1, 3))
    z, _ = equivariant(x[None, :, None], mask, np.array([[1.0]]), np.array([[1.0]]), np.zeros(1), "mean")
    np.testing.assert_allclose(z[0, :, 0], y)
    assert abs(y.sum()) < 1e-12 and y.max() == pytest.approx(5 / 3)
    assert pool(np.stack([np.ones(3), x], -1)[None], mask, "sum")[0, 1] / 3 == pytest.approx(7 / 3)
    assert [float((x**k).sum()) for k in (1, 2, 3)] == [7.0, 21.0, 73.0]
    np.testing.assert_allclose(np.poly(x), [1, -7, 14, -8])


def test_config_errors():
    for kw in ({"width": 0}, {"pool": "median"}, {"act": "gelu"}, {"head": "pair"}, {"n_classes": 0}):
        with pytest.raises(ValueError):
            config(**kw)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    k = c["config"]
    return forward, (init(rng, cfg), cfg, sets_of(rng, cfg, k["B"], k["set_size"]))
