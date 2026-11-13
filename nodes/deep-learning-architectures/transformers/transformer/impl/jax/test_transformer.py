# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""Tests of the JAX reference: the shared fixture, the shape contract, jit, and jax.grad against finite differences."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformer import apply, attention, config, init, param_count

jax.config.update("jax_enable_x64", True)  # before any array is made: the tests run in float64

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["d_model"], k["n_heads"], k["d_ff"], k["n_layers"], k["vocab"])


def unflatten(flat):
    tree = {}
    for path, v in flat.items():
        *head, last = path.split("/")
        node = tree
        for part in head:
            node = node.setdefault(part, {})
        node[last] = jnp.asarray(v)

    def lists(t):
        if not isinstance(t, dict):
            return t
        if t and all(k.isdigit() for k in t):
            return [lists(t[str(i)]) for i in range(len(t))]
        return {k: lists(v) for k, v in t.items()}

    return lists(tree)


def fixture():
    params = unflatten(dict(np.load(FIXTURES / "params-small.npz")))
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    return params, inp, dict(np.load(FIXTURES / "expected-small.npz")), cfg_of(SHAPES[0])


def close(a, b, rtol=1e-10, atol=1e-12):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_shared_fixture():
    params, inp, want, cfg = fixture()
    logits, cache = apply(params, inp["src"], inp["tgt"], cfg, inp["src_pad"])
    close(logits, want["logits"])
    close(cache["memory"], want["memory"])
    close(cache["enc"][0]["weights"], want["enc/0/weights"])
    close(cache["dec"][0]["self"]["weights"], want["dec/0/self/weights"])
    close(cache["dec"][0]["cross"]["weights"], want["dec/0/cross/weights"])


def test_jit_gives_the_same_logits():
    params, inp, want, cfg = fixture()
    run = jax.jit(lambda p, s, t, m: apply(p, s, t, cfg, m)[0])
    close(run(params, inp["src"], inp["tgt"], inp["src_pad"]), want["logits"])


def test_worked_example():
    q = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    v = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
    close(attention(q, q, v)[0][2], [0.6698, 0.8879], rtol=0, atol=5e-5)
    close(attention(q, q, v, jnp.tril(jnp.ones((4, 4), dtype=bool)))[0][2], [0.7517, 0.7517], rtol=0, atol=5e-5)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    params = init(jax.random.key(0), cfg)
    rng = np.random.default_rng(0)
    src = jnp.asarray(rng.integers(0, k["V"], (k["B"], k["S"])))
    tgt = jnp.asarray(rng.integers(0, k["V"], (k["B"], k["T"])))
    logits, cache = apply(params, src, tgt, cfg)
    assert list(logits.shape) == blocks["logits"]["outputs"]["logits"]
    got = {"enc_attn": cache["enc"][0], "dec_attn": cache["dec"][0]["self"], "cross_attn": cache["dec"][0]["cross"],
           "enc_ffn": cache["enc"][0], "dec_ffn": cache["dec"][0]}
    for block, arrays in got.items():
        for name, shape in blocks[block]["internals"].items():
            assert list(arrays[name].shape) == shape, (block, name)
    assert sum(a.size for a in jax.tree_util.tree_leaves(params)) == c["model"]["params"]


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    assert param_count(cfg_of(c)) == c["model"]["params"]


def test_gradient():
    # jax.grad against central finite differences, on a few entries of every parameter array.
    cfg = config(d_model=4, n_heads=2, d_ff=6, n_layers=1, vocab=7)
    params = init(jax.random.key(1), cfg)
    src, tgt = jnp.array([[1, 4, 2]]), jnp.array([[3, 0]])
    pad = jnp.array([[False, False, True]])
    w = jnp.asarray(np.random.default_rng(1).normal(size=(1, 2, 7)))

    @jax.jit
    def loss(p):
        return (apply(p, src, tgt, cfg, pad)[0] * w).sum()

    grads = jax.grad(loss)(params)
    leaves, tree = jax.tree_util.tree_flatten(params)
    for n, (leaf, g) in enumerate(zip(leaves, jax.tree_util.tree_leaves(grads))):
        for i in range(min(leaf.size, 3)):
            h = 1e-6 * max(1.0, abs(float(leaf.reshape(-1)[i])))

            def shifted(d, n=n, i=i, leaf=leaf):
                new = list(leaves)
                new[n] = leaf.reshape(-1).at[i].add(d).reshape(leaf.shape)
                return float(loss(jax.tree_util.tree_unflatten(tree, new)))

            fd = (shifted(h) - shifted(-h)) / (2 * h)
            assert abs(fd - float(g.reshape(-1)[i])) < 1e-6 * max(1.0, abs(fd))


def test_gradient_of_a_masked_row_is_finite():
    q = jnp.ones((3, 2))
    allowed = jnp.array([[True, False, False], [False, False, False], [True, True, True]])
    g = jax.grad(lambda q: attention(q, q, q, allowed)[0].sum())(q)
    assert bool(jnp.all(jnp.isfinite(g)))
