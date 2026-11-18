# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""Tests of the JAX version: the shared fixture and its gradients, the shape contract, jit and vmap."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from seq2seq import EOS, config, encode, forward, greedy, init, loss, param_count  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("src_vocab", "tgt_vocab", "embed_size", "hidden_size", "n_layers")


def cfg_of(c, reverse=True):
    return config(*(c["config"][k] for k in KEYS), reverse=reverse)


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want


def test_shared_fixture():
    params, inp, want = fixture()
    cfg = config(reverse=True)
    np.testing.assert_allclose(forward(params, cfg, inp["src"], inp["tgt"]), want["logits"], rtol=1e-12, atol=1e-13)
    h, c = encode(params, cfg, inp["src"])
    np.testing.assert_allclose(h, want["v_h"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(c, want["v_c"], rtol=1e-12, atol=1e-13)
    fwd = forward(params, config(reverse=False), inp["src"], inp["tgt"])
    np.testing.assert_allclose(fwd, want["logits_forward"], rtol=1e-12, atol=1e-13)
    out = np.asarray(greedy(params, cfg, inp["src"], 6))
    for i in range(2):
        n = int((want["greedy"][i] >= 0).sum())
        assert out[i, :n].tolist() == want["greedy"][i, :n].tolist()


def test_shared_fixture_gradients():
    params, inp, want = fixture()
    cfg = config(reverse=True)
    value, grads = jax.value_and_grad(loss)(params, cfg, inp["src"], inp["tgt"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    for name, g in grads.items():
        np.testing.assert_allclose(g, want["d" + name], rtol=1e-10, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_shapes(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] == "named":
        return
    k = c["config"]
    params = init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in jax.tree_util.tree_leaves(params)) == c["model"]["params"]
    src = jnp.ones((k["B"], k["S"]), int).at[:, -1].set(EOS)
    tgt = jnp.ones((k["B"], k["T"]), int).at[:, -1].set(EOS)
    assert list(forward(params, cfg, src, tgt).shape) == c["blocks"]["softmax"]["outputs"]["logits"]
    h, _ = encode(params, cfg, src)
    assert list(h.shape) == c["blocks"]["encoder"]["outputs"]["v_h"]


def test_jit_and_vmap():
    params, inp, want = fixture()
    cfg = config(reverse=True)
    f = jax.jit(lambda p, s, t: loss(p, cfg, s, t))
    np.testing.assert_allclose(f(params, inp["src"], inp["tgt"]), want["loss"], rtol=1e-12)
    # vmap over single sentences gives the rows of the batched pass.
    one = jax.vmap(lambda s, t: forward(params, cfg, s[None], t[None])[0])(inp["src"], inp["tgt"])
    np.testing.assert_allclose(one, want["logits"], rtol=1e-12, atol=1e-13)
