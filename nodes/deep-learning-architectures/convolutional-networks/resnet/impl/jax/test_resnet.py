# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Tests of the JAX reference: the shared fixture, the shape contract, jit, and jax.grad against finite differences."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from resnet import apply, config, init, param_count

jax.config.update("jax_enable_x64", True)  # before any array is made: the tests run in float64

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["width"], (k["n1"], k["n2"], k["n3"], k["n4"]), k["classes"])


def block_id(s, i):
    return f"s{s + 1}_{'first' if i == 0 else 'rest'}"


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
    state = unflatten(dict(np.load(FIXTURES / "state-small.npz")))
    x = jnp.asarray(np.load(FIXTURES / "input-small.npz")["x"])
    return params, state, x, dict(np.load(FIXTURES / "expected-small.npz")), cfg_of(SHAPES[0])


def test_shared_fixture():
    params, state, x, want, cfg = fixture()
    logits, cache = apply(params, x, cfg)
    np.testing.assert_allclose(logits, want["logits_train"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cache["stem"]["y"], want["stem_train"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cache["blocks"][(1, 0)]["c"], want["s2_first_c_train"], rtol=1e-10, atol=1e-12)
    logits, cache = apply(params, x, cfg, state)
    np.testing.assert_allclose(logits, want["logits_infer"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cache["blocks"][(1, 0)]["c"], want["s2_first_c_infer"], rtol=1e-10, atol=1e-12)


def test_jit_matches():
    params, state, x, want, cfg = fixture()
    jitted = jax.jit(lambda p, t: apply(p, t, cfg)[0])
    np.testing.assert_allclose(jitted(params, x), want["logits_train"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    params = init(jax.random.PRNGKey(0), cfg)
    logits, cache = apply(params, jax.random.normal(jax.random.PRNGKey(1), (k["B"], 3, k["R"], k["R"])), cfg)
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["pooled"].shape) == blocks["head"]["internals"]["pooled"]
    assert list(cache["stem"]["conv"].shape) == blocks["stem"]["internals"]["conv"]
    for (s, i), got in cache["blocks"].items():
        want = blocks[block_id(s, i)]
        assert list(got["y"].shape) == want["outputs"]["y"]
        for name, shape in want["internals"].items():
            assert list(got[name].shape) == shape, (s, i, name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        leaves = jax.tree_util.tree_leaves(init(jax.random.PRNGKey(0), cfg))
        assert sum(a.size for a in leaves) == c["model"]["params"]


def test_grad_matches_finite_differences():
    params, _, x, _, cfg = fixture()
    y = jnp.array([1, 3])

    def loss(p):
        logits, _ = apply(p, x, cfg)
        return -jax.nn.log_softmax(logits)[jnp.arange(2), y].mean()

    g = jax.grad(loss)(params)
    rng = np.random.default_rng(0)
    for path in (("stages", 1, 0, "conv2"), ("stages", 3, 0, "proj"), ("stem", "conv"), ("fc", "W")):
        leaf = params
        for k in path:
            leaf = leaf[k]
        idx = tuple(int(rng.integers(0, n)) for n in leaf.shape)
        eps = 1e-6

        def moved(d):
            p = jax.tree_util.tree_map(lambda a: a, params)
            node = p
            for k in path[:-1]:  # noqa: B023
                node = node[k]
            node[path[-1]] = leaf.at[idx].add(d)  # noqa: B023
            return loss(p)

        numeric = (moved(eps) - moved(-eps)) / (2 * eps)
        gl = g
        for k in path:
            gl = gl[k]
        np.testing.assert_allclose(gl[idx], numeric, rtol=1e-5, atol=1e-9, err_msg=str(path))


def test_zero_gamma_makes_blocks_identity():
    cfg = config(2, (2, 2, 2, 2), 5)
    params = init(jax.random.PRNGKey(0), cfg, zero_gamma=True)
    _, cache = apply(params, jax.random.normal(jax.random.PRNGKey(1), (2, 3, 32, 32)), cfg)
    for s in range(4):
        np.testing.assert_array_equal(cache["blocks"][(s, 1)]["y"], cache["blocks"][(s, 0)]["y"])
