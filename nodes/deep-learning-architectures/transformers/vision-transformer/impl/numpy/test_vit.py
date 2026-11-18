# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://DV9G9W).
"""Tests of the NumPy reference: shapes and counts, the patches, the worked example, the shared fixture, the gradient
of eq-loss, the bag of patches without position embeddings and the resizing of the position embeddings."""

import json
import math
import pathlib

import numpy as np
import pytest

from vit import (
    backward,
    config,
    embed,
    embed_as_conv,
    embed_from_conv,
    flops,
    forward,
    init,
    interpolation_matrix,
    loss,
    n_tokens,
    param_count,
    param_shapes,
    patchify,
    resize_pos,
    unpatchify,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["image_size"], k["patch_size"], k["channels"], k["d_model"], k["n_heads"], k["d_ff"],
                  k["n_layers"], k["n_classes"], **kw)  # fmt: skip


def images_of(rng, cfg, B):
    S = cfg["image_size"]
    return rng.normal(size=(B, S, S, cfg["channels"]))


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    cfg = cfg_of(c)
    m = c["model"]
    assert param_count(cfg) == m["params"]
    assert sum(math.prod(s) for s in param_shapes(cfg).values()) == m["params"]
    assert n_tokens(cfg) == m["tokens"]
    B = c["config"]["B"] if c["kind"] != "named" else 1
    if c["kind"] != "named":
        assert flops(cfg, B) == m["flops_forward"]
        rng = np.random.default_rng(0)
        logits, cache = forward(init(rng, cfg), cfg, images_of(rng, cfg, B))
        blocks = c["blocks"]
        assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
        assert list(cache["patches"].shape) == blocks["embed"]["internals"]["patches"]
        assert list(cache["tokens"].shape) == blocks["embed"]["outputs"]["tokens"]
        attn = cache["layers"][0]["attn"]
        for name in ("q", "k", "v", "weights"):
            assert list(attn[name].shape) == blocks["attn"]["internals"][name]
        assert list(cache["layers"][0]["u"].shape) == blocks["mlp"]["internals"]["hidden"]
        assert list(cache["y"].shape) == blocks["head"]["internals"]["y"]


def test_default_config_is_the_example():
    assert config() == cfg_of(SHAPES[0])


def test_the_models_of_the_paper():
    named = {c["name"]: cfg_of(c) for c in SHAPES if c["kind"] == "named"}
    counts = {k: param_count(v) for k, v in named.items()}
    assert counts["vit_b16"] == 86_567_656 and round(counts["vit_b16"] / 1e6) == 87
    assert round(counts["vit_h14"] / 1e6) == 632 and round(counts["vit_l32"] / 1e6) == 307
    assert counts["vit_l16"] == 304_326_632
    assert counts["vit_b16_384"] - counts["vit_b16"] == (577 - 197) * 768
    assert flops(named["vit_b16"], 1) == 35_127_656_448
    assert flops(named["vit_b16_384"], 1) == 110_968_700_928
    assert 3.9 < flops(named["vit_b16"], 1) / flops(named["vit_b32"], 1) < 4.0


def test_patchify_is_raster_order_and_invertible():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 6, 6, 2))
    p = patchify(x, 3)
    assert p.shape == (2, 4, 18)
    np.testing.assert_array_equal(p[:, 1].reshape(2, 3, 3, 2), x[:, 0:3, 3:6])  # top right
    np.testing.assert_array_equal(p[:, 2].reshape(2, 3, 3, 2), x[:, 3:6, 0:3])  # bottom left
    np.testing.assert_array_equal(unpatchify(p, 3, 2), x)
    with pytest.raises(ValueError):
        patchify(x, 4)


def test_worked_example():
    image = np.arange(1.0, 17.0).reshape(1, 4, 4, 1)
    p = patchify(image, 2)
    np.testing.assert_array_equal(p[0], [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]])
    np.testing.assert_array_equal(p[0].mean(1), [3.5, 5.5, 11.5, 13.5])
    mirror = unpatchify(p[:, [1, 0, 3, 2]], 2, 1)
    np.testing.assert_array_equal(mirror[0, :, :, 0], [[3, 4, 1, 2], [7, 8, 5, 6], [11, 12, 9, 10], [15, 16, 13, 14]])
    cfg = config(4, 2, 1, 4, 2, 8, 2, 3, pos="none")
    params = init(np.random.default_rng(2), cfg, scale=0.5)
    np.testing.assert_allclose(forward(params, cfg, mirror)[0], forward(params, cfg, image)[0], rtol=1e-12)
    b16, b16_384 = config(224, 16, 3, 768, 12, 3072, 12, 1000), config(384, 16, 3, 768, 12, 3072, 12, 1000)
    assert (n_tokens(b16), n_tokens(b16_384), param_shapes(b16)["embed_W"][0]) == (197, 577, 768)
    assert param_shapes(b16_384)["pos"] == (577, 768)
    pos = np.random.default_rng(3).normal(size=(197, 4))
    assert resize_pos(pos, 14, 24).shape == (577, 4)


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    assert set(params) == set(param_shapes(cfg))
    logits, cache = forward(params, cfg, inp["images"])
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["tokens"], want["tokens"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(cache["final"], want["final"], rtol=1e-13, atol=1e-14)
    assert loss(params, cfg, inp["images"], inp["labels"]) == pytest.approx(float(want["loss"]), rel=1e-13)
    grads, dimages = backward(params, cfg, cache, inp["labels"])
    for k, v in grads.items():
        np.testing.assert_allclose(v, want["d" + k], rtol=1e-12, atol=1e-14, err_msg=k)
    np.testing.assert_allclose(dimages, want["dimages"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "kw",
    [{}, {"pool": "gap"}, {"pos": "none"}, {"head_hidden": 3}, {"pool": "gap", "pos": "none", "head_hidden": 2}],
    ids=["cls", "gap", "no-pos", "tanh-head", "gap-no-pos-tanh"],
)
def test_gradient_by_finite_differences(kw):
    rng = np.random.default_rng(4)
    cfg = config(4, 2, 2, 6, 3, 5, 2, 4, **kw)
    params = init(rng, cfg, scale=0.5)
    images = images_of(rng, cfg, 3)
    labels = rng.integers(0, cfg["n_classes"], 3)
    grads, dimages = backward(params, cfg, forward(params, cfg, images)[1], labels)
    eps = 1e-6
    for name, g in grads.items():
        for idx in [tuple(rng.integers(0, s) for s in g.shape) for _ in range(3)]:
            plus, minus = dict(params), dict(params)
            plus[name], minus[name] = params[name].copy(), params[name].copy()
            plus[name][idx] += eps
            minus[name][idx] -= eps
            fd = (loss(plus, cfg, images, labels) - loss(minus, cfg, images, labels)) / (2 * eps)
            assert g[idx] == pytest.approx(fd, rel=1e-5, abs=1e-8), name
    for idx in [tuple(rng.integers(0, s) for s in images.shape) for _ in range(5)]:
        plus, minus = images.copy(), images.copy()
        plus[idx] += eps
        minus[idx] -= eps
        fd = (loss(params, cfg, plus, labels) - loss(params, cfg, minus, labels)) / (2 * eps)
        assert dimages[idx] == pytest.approx(fd, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("pool", ["cls", "gap"])
def test_no_positions_means_a_bag_of_patches(pool):
    rng = np.random.default_rng(5)
    for _ in range(5):
        cfg = config(6, 2, 2, 8, 2, 12, 3, 4, pool=pool, pos="none")
        params = init(rng, cfg, scale=0.5)
        images = images_of(rng, cfg, 4)
        perm = rng.permutation(9)
        shuffled = unpatchify(patchify(images, 2)[:, perm], 2, 2)
        np.testing.assert_allclose(forward(params, cfg, shuffled)[0], forward(params, cfg, images)[0], rtol=1e-11,
                                   atol=1e-13)  # fmt: skip
    learned = config(6, 2, 2, 8, 2, 12, 3, 4, pool=pool)
    params = init(rng, learned, scale=0.5)
    assert np.abs(forward(params, learned, shuffled)[0] - forward(params, learned, images)[0]).max() > 1e-3


def test_resize_pos():
    rng = np.random.default_rng(6)
    pos = rng.normal(size=(1 + 5 * 5, 3))
    np.testing.assert_allclose(resize_pos(pos, 5, 5), pos, rtol=0, atol=1e-15)
    for old, new in [(14, 24), (5, 3), (4, 2), (3, 7)]:
        r, c = np.meshgrid(np.arange(old), np.arange(old), indexing="ij")
        field = np.stack([2.0 * r - c + 1.0, 0.5 * c], -1).reshape(old * old, 2)
        out = resize_pos(np.concatenate([[[9.0, 9.0]], field]), old, new)
        np.testing.assert_array_equal(out[0], [9.0, 9.0])
        src = np.clip((np.arange(new) + 0.5) * old / new - 0.5, 0, old - 1)
        R, Cc = np.meshgrid(src, src, indexing="ij")
        want = np.stack([2.0 * R - Cc + 1.0, 0.5 * Cc], -1).reshape(new * new, 2)
        np.testing.assert_allclose(out[1:], want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(interpolation_matrix(7, 4).sum(1), 1.0)
    with pytest.raises(ValueError):
        resize_pos(pos, 4, 6)


def test_pool_and_positions_change_the_parameters():
    base = config()
    assert "cls" in param_shapes(base) and param_shapes(base)["pos"] == (17, 8)
    gap = config(pool="gap")
    assert "cls" not in param_shapes(gap) and param_shapes(gap)["pos"] == (16, 8)
    assert "pos" not in param_shapes(config(pos="none"))
    assert param_count(base) - param_count(config(pool="gap", pos="none")) == 8 + 17 * 8
    assert param_count(config(head_hidden=3)) - param_count(base) == 8 * 3 + 3 + 3 * 5 - 8 * 5


def test_embed_as_conv_round_trip():
    rng = np.random.default_rng(7)
    W = rng.normal(size=(3 * 3 * 2, 5))
    w = embed_as_conv(W, 3, 2)
    assert w.shape == (5, 2, 3, 3)
    np.testing.assert_array_equal(w[4, 1, 2, 0], W[(2 * 3 + 0) * 2 + 1, 4])
    np.testing.assert_array_equal(embed_from_conv(w), W)


def test_bad_configs():
    for kw in [{"image_size": 7}, {"n_heads": 3}, {"n_classes": 1}, {"pool": "max"}, {"pos": "sincos"}]:
        with pytest.raises(ValueError):
            config(**kw)


def test_embed_puts_the_class_token_first():
    rng = np.random.default_rng(8)
    cfg = config()
    params = init(rng, cfg)
    z, x = embed(params, cfg, images_of(rng, cfg, 2))
    np.testing.assert_allclose(z[:, 0], np.broadcast_to(params["cls"] + params["pos"][0], (2, 8)))
    np.testing.assert_allclose(z[:, 1:], x @ params["embed_W"] + params["embed_b"] + params["pos"][1:])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), cfg, images_of(rng, cfg, c["config"]["B"]))
