# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://P3DWR1).
"""Tests of the NumPy reference: each stage of the addressing against its equation, the erase and add of a write,
a copy and a lookup wired by hand, the pitfalls, the shapes and the shared fixture."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ntm import (
    EPS,
    MEMORY_INIT,
    address,
    bits_per_sequence,
    config,
    content_weights,
    cosine,
    flops,
    forward,
    init,
    initial_state,
    interpolate,
    n_head_outputs,
    param_count,
    read,
    sharpen,
    shift,
    softmax,
    split_heads,
    step,
    write,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("x_dim", "y_dim", "hidden", "n_loc", "width", "n_read", "n_write", "shift")


def cfg_of(c):
    return config(**{k: c["config"][k] for k in KEYS})


def simplex(rng, *shape):
    return softmax(rng.normal(size=shape) * 2.0)


def one_hot(i, n, b=1):
    w = np.zeros((b, n))
    w[:, i] = 1.0
    return w


def test_cosine_is_the_definition():
    rng = np.random.default_rng(0)
    mem, key = rng.normal(size=(2, 5, 3)), rng.normal(size=(2, 3))
    want = [[mem[b, i] @ key[b] / np.linalg.norm(mem[b, i]) / np.linalg.norm(key[b]) for i in range(5)] for b in range(2)]
    np.testing.assert_allclose(cosine(mem, key), want, rtol=1e-6)


def test_content_weights_beta():
    rng = np.random.default_rng(1)
    mem, key = rng.normal(size=(1, 6, 4)), rng.normal(size=(1, 4))
    np.testing.assert_allclose(content_weights(mem, key, np.zeros(1)), np.full((1, 6), 1 / 6))
    sharp = content_weights(mem, key, np.array([1e4]))
    assert sharp.argmax() == cosine(mem, key).argmax() and sharp.max() > 1 - 1e-9
    k = cosine(mem, key)
    np.testing.assert_allclose(content_weights(mem, key, np.array([2.0])), np.exp(2 * k) / np.exp(2 * k).sum())


def test_interpolation_gate_extremes():
    rng = np.random.default_rng(2)
    wc, wp = simplex(rng, 3, 5), simplex(rng, 3, 5)
    np.testing.assert_array_equal(interpolate(wc, wp, np.ones(3)), wc)
    np.testing.assert_array_equal(interpolate(wc, wp, np.zeros(3)), wp)


def test_shift_is_the_circular_convolution():
    rng = np.random.default_rng(3)
    w, s = simplex(rng, 2, 7), simplex(rng, 2, 5)
    want = np.zeros((2, 7))
    for b in range(2):
        for i in range(7):
            for j in range(7):
                d = (i - j) % 7  # the shift from j to i, read in -2..2 when it is small
                d = d - 7 if d > 3 else d
                if -2 <= d <= 2:
                    want[b, i] += w[b, j] * s[b, d + 2]
    np.testing.assert_allclose(shift(w, s), want, rtol=1e-12)


def test_shift_direction_and_wrap():
    s_plus = np.array([[0.0, 0.0, 1.0]])  # all on shift +1
    np.testing.assert_array_equal(shift(one_hot(2, 5), s_plus), one_hot(3, 5))
    np.testing.assert_array_equal(shift(one_hot(4, 5), s_plus), one_hot(0, 5))  # the last location wraps to the first
    np.testing.assert_array_equal(shift(one_hot(0, 5), np.array([[1.0, 0.0, 0.0]])), one_hot(4, 5))


def test_blurry_shift_disperses_and_sharpening_refocuses():
    """The example of section 3.3.2 of Graves et al.: shifts -1, 0, 1 with weights 0.1, 0.8, 0.1 spread a sharp
    weighting; repeated, the weighting keeps spreading unless gamma sharpens it."""
    s = np.array([[0.1, 0.8, 0.1]])
    w = one_hot(5, 11)
    np.testing.assert_allclose(shift(w, s)[0, 4:7], [0.1, 0.8, 0.1])
    blur, kept = w, w
    for _ in range(5):
        blur = shift(blur, s)
        kept = sharpen(shift(kept, s), np.array([8.0]))
    assert blur.max() < 0.5 and kept.max() > 0.99 and kept.argmax() == 5


def test_sharpen():
    rng = np.random.default_rng(4)
    w = simplex(rng, 3, 6)
    np.testing.assert_allclose(sharpen(w, np.ones(3)), w)
    big = sharpen(w, np.full(3, 200.0))
    np.testing.assert_array_equal(big.argmax(-1), w.argmax(-1))
    assert np.all(big.max(-1) > 0.99)
    with_zero = sharpen(np.array([[0.0, 0.5, 0.5]]), np.array([3.0]))
    assert np.all(np.isfinite(with_zero)) and with_zero[0, 0] == 0.0


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000))
def test_address_is_a_weighting(seed):
    rng = np.random.default_rng(seed)
    cfg = config(n_loc=7, width=3, shift=2)
    o = rng.normal(size=(2, n_head_outputs(cfg))) * 3.0
    head = split_heads(o, cfg)[0]
    w = address(rng.normal(size=(2, 7, 3)), simplex(rng, 2, 7), head)
    assert np.all(w >= 0)
    np.testing.assert_allclose(w.sum(-1), 1.0)


def test_head_outputs_are_in_range():
    cfg = config(n_read=2, n_write=1)
    o = np.random.default_rng(5).normal(size=(4, n_head_outputs(cfg))) * 10.0
    heads = split_heads(o, cfg)
    assert len(heads) == 3 and "erase" not in heads[0] and "erase" in heads[2]
    for hd in heads:
        assert np.all(np.abs(hd["key"]) <= 1) and np.all(hd["beta"] >= 0)
        assert np.all((hd["g"] >= 0) & (hd["g"] <= 1)) and np.all(hd["gamma"] >= 1)
        np.testing.assert_allclose(hd["s"].sum(-1), 1.0)
    assert np.all((heads[2]["erase"] >= 0) & (heads[2]["erase"] <= 1)) and np.all(np.abs(heads[2]["add"]) <= 1)


def test_read_is_the_weighted_sum():
    rng = np.random.default_rng(6)
    mem, w = rng.normal(size=(2, 5, 3)), simplex(rng, 2, 5)
    np.testing.assert_allclose(read(mem, w), [sum(w[b, i] * mem[b, i] for i in range(5)) for b in range(2)])
    np.testing.assert_array_equal(read(mem, one_hot(3, 5, 2)), mem[:, 3])


def test_erase_and_add():
    rng = np.random.default_rng(7)
    mem, a = rng.normal(size=(1, 4, 3)), rng.normal(size=(1, 3))
    new = write(mem, [one_hot(1, 4)], [np.ones((1, 3))], [a])  # w = 1 and e = 1: the row is replaced
    np.testing.assert_allclose(new[0, 1], a[0])
    np.testing.assert_array_equal(new[0, [0, 2, 3]], mem[0, [0, 2, 3]])
    half = write(mem, [0.5 * one_hot(1, 4) + 0.5 * one_hot(2, 4)], [np.ones((1, 3))], [np.zeros((1, 3))])
    np.testing.assert_allclose(half[0, 1:3], 0.5 * mem[0, 1:3])  # a blurry erase only halves the rows
    added = write(mem, [one_hot(0, 4)], [np.zeros((1, 3))], [a])
    np.testing.assert_allclose(added[0, 0], mem[0, 0] + a[0])


def test_write_heads_commute():
    rng = np.random.default_rng(8)
    mem = rng.normal(size=(2, 5, 3))
    ws = [simplex(rng, 2, 5) for _ in range(3)]
    es = [rng.uniform(size=(2, 3)) for _ in range(3)]
    adds = [rng.normal(size=(2, 3)) for _ in range(3)]
    base = write(mem, ws, es, adds)
    for order in ((2, 0, 1), (1, 2, 0)):
        np.testing.assert_allclose(write(mem, [ws[i] for i in order], [es[i] for i in order], [adds[i] for i in order]), base)


def test_copy_by_location():
    """A write head that moves one location per step (g = 0, all of s on +1, gamma = 1, e = 1) stores the inputs
    in order; a read head that starts at the same place and moves the same way returns them. With 5 locations the
    sixth item overwrites the first."""
    n, m = 5, 3
    items = np.random.default_rng(9).normal(size=(7, 1, m))
    mem, w = np.zeros((1, n, m)), one_hot(n - 1, n)
    s, keep = np.array([[0.0, 0.0, 1.0]]), {"key": np.zeros((1, m)), "beta": np.zeros(1), "g": np.zeros(1),
                                            "gamma": np.ones(1)}
    for x in items:
        w = address(mem, w, dict(keep, s=s))
        mem = write(mem, [w], [np.ones((1, m))], [x])
    r = one_hot(n - 1, n)
    out = []
    for _ in range(n):
        r = address(mem, r, dict(keep, s=s))
        out.append(read(mem, r)[0])
    np.testing.assert_allclose(out, [items[5, 0], items[6, 0], items[2, 0], items[3, 0], items[4, 0]])


def test_content_then_shift_finds_the_next_item():
    """Associative recall in one step: a content lookup of item j (g = 1, a large beta), then a shift by +1,
    reads item j + 1; this is how Graves et al. describe the solution the NTM learns (section 4.3)."""
    rng = np.random.default_rng(10)
    mem = np.tanh(rng.normal(size=(1, 6, 8)) * 2)
    head = {"key": mem[:, 2].copy(), "beta": np.array([200.0]), "g": np.ones(1), "s": np.array([[0.0, 0.0, 1.0]]),
            "gamma": np.ones(1)}
    w = address(mem, one_hot(0, 6), head)
    np.testing.assert_allclose(read(mem, w), mem[:, 3], atol=1e-8)


def test_zero_memory_needs_the_epsilon():
    """The cosine of a row of zeros is 0/0; with EPS in the denominator it is 0 and the content weighting of a blank
    memory is uniform. A key of zeros gives the same."""
    k = cosine(np.zeros((1, 4, 3)), np.ones((1, 3)))
    assert np.all(np.isfinite(k)) and np.all(k == 0)
    np.testing.assert_allclose(content_weights(np.zeros((1, 4, 3)), np.ones((1, 3)), np.array([5.0])), 0.25)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.einsum("bnm,bm->bn", np.zeros((1, 4, 3)), np.ones((1, 3))) / 0.0
    assert np.all(np.isnan(raw)) and EPS > 0


def test_a_negative_beta_attends_to_the_least_similar_row():
    """beta must be at least 0, which the softplus ensures; with a negative beta the content weighting goes to the
    row least like the key."""
    rng = np.random.default_rng(11)
    mem, key = rng.normal(size=(1, 6, 4)), rng.normal(size=(1, 4))
    assert content_weights(mem, key, np.array([-100.0])).argmax() == cosine(mem, key).argmin()


def test_write_then_read_in_one_step():
    """The write heads act before the read heads, so a read of the location just written sees the new row."""
    cfg = config(x_dim=2, y_dim=1, hidden=3, n_loc=4, width=2)
    mem, ws, reads = initial_state(cfg, 1)
    assert np.all(mem == MEMORY_INIT) and np.all(ws[0] == one_hot(0, 4)) and np.all(reads[0] == 0)
    params = init(np.random.default_rng(12), cfg)
    (mem2, ws2, reads2), _ = step(params, (mem, ws, reads), np.ones((1, 2)), cfg)
    np.testing.assert_allclose(reads2[0], read(mem2, ws2[0]))


def test_param_count_does_not_depend_on_the_memory_size():
    counts = set()
    for n in (3, 6, 50):
        cfg = config(n_loc=n)
        params = init(np.random.default_rng(0), cfg)
        assert sum(v.size for v in params.values()) == param_count(cfg)
        counts.add(param_count(cfg))
        assert forward(params, np.ones((1, 3, cfg["x_dim"])), cfg).shape == (1, 3, cfg["y_dim"])
    assert len(counts) == 1


def test_bits_per_sequence():
    rng = np.random.default_rng(13)
    logits, targets = rng.normal(size=(2, 4, 3)), rng.integers(0, 2, (2, 4, 3)).astype(float)
    mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1]], float)
    p = 1 / (1 + np.exp(-logits))
    want = -(targets * np.log2(p) + (1 - targets) * np.log2(1 - p)) * mask[:, :, None]
    assert bits_per_sequence(logits, targets, mask) == pytest.approx(want.sum() / 2)


def test_config_checks():
    with pytest.raises(ValueError):
        config(n_loc=2, shift=1)
    with pytest.raises(ValueError):
        config(width=0)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(np.random.default_rng(0), cfg)
    x = np.zeros((k["B"], k["T"], k["X"]))
    state = initial_state(cfg, k["B"])
    blocks = c["blocks"]
    assert list(state[0].shape) == blocks["address"]["inputs"]["memory"]
    assert list(state[1][0].shape) == blocks["address"]["inputs"]["w_prev"]
    o = np.tanh(np.zeros((k["B"], cfg["hidden"]))) @ params["Wh"].T
    assert list(o.shape) == blocks["controller"]["outputs"]["o"]
    (mem, ws, reads), y = step(params, state, x[:, 0], cfg)
    assert list(mem.shape) == blocks["write"]["outputs"]["memory"]
    assert list(reads[0].shape) == blocks["read"]["outputs"]["r"]
    assert list(y.shape) == blocks["output"]["outputs"]["y_t"]
    logits = forward(params, x, cfg)
    assert list(logits.shape) == [k["B"], k["T"], k["Y"]]
    assert param_count(cfg) == c["model"]["params"] == sum(v.size for v in params.values())
    assert flops(cfg, k["B"], k["T"]) == c["model"]["flops_forward"]
    assert mem.size == c["model"]["memory_floats"]


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits, ws, mem = forward(params, inp["x"], cfg, trace=True)
    np.testing.assert_allclose(logits, want["logits"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ws, want["weights"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mem, want["memory"], rtol=1e-12, atol=1e-12)
    assert bits_per_sequence(logits, inp["targets"], inp["mask"]) == pytest.approx(float(want["loss"]), rel=1e-12)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), rng.normal(size=(k["B"], k["T"], k["X"])), cfg)
