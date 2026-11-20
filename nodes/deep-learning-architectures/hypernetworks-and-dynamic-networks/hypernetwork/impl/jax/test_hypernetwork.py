# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Tests of the JAX hypernetworks: the shared fixture and its gradients for the static network and the HyperRNN, the
HyperLSTM with and without layer normalization, the counts of the shape contract, jit, and vmap over embeddings."""

import json
import math
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import hypernetwork as H  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]


def fixture():
    p, i, e = (dict(np.load(FIX / f"{n}-small.npz")) for n in ("params", "input", "expected"))
    return p, i, e


def part(d, prefix):
    return {k[len(prefix) + 1 :]: jnp.asarray(v) for k, v in d.items() if k.startswith(prefix + ".")}


def test_static_fixture_and_gradients():
    P, Inp, E = fixture()
    p, e = part(P, "static"), part(E, "static")
    x, y = jnp.asarray(Inp["static.x"]), jnp.asarray(Inp["static.y"])
    assert np.allclose(H.forward(p, x), e["logits"], atol=1e-12)
    value, (g, dx) = jax.value_and_grad(H.loss, argnums=(0, 1))(p, x, y)
    assert np.isclose(value, e["loss"], atol=1e-12)
    for k, v in g.items():
        assert np.allclose(v, e["d" + k], atol=1e-12), k
    assert np.allclose(dx, e["dx"], atol=1e-12)


def test_hyper_rnn_fixture_and_gradients():
    P, Inp, E = fixture()
    p, e = part(P, "rnn"), part(E, "rnn")
    x, tg = jnp.asarray(Inp["rnn.x"]), jnp.asarray(Inp["rnn.target"])
    assert np.allclose(H.hyper_rnn(p, x), e["out"], atol=1e-12)
    value, (g, dx) = jax.value_and_grad(H.rnn_loss, argnums=(0, 1))(p, x, tg)
    assert np.isclose(value, e["loss"], atol=1e-12)
    for k, v in g.items():
        assert np.allclose(v, e["d" + k], atol=1e-12), k
    assert np.allclose(dx, e["dx"], atol=1e-12)


@pytest.mark.parametrize("ln", [True, False])
def test_hyper_lstm_fixture(ln):
    P, Inp, E = fixture()
    out = jax.jit(lambda p, x: H.hyper_lstm(p, x, ln=ln))(part(P, "lstm"), jnp.asarray(Inp["rnn.x"]))
    assert np.allclose(out, E["lstm.out_ln" if ln else "lstm.out"], atol=1e-12)


@pytest.mark.parametrize("c", [c for c in SHAPES if c["kind"] != "named"], ids=lambda c: c["name"])
def test_shape_contract(c):
    k, m = c["config"], c["model"]
    N, f, Nz, mult, D, Mw = k["n_basic"], k["fsize"], k["n_z"], k["mult"], k["depth"], k["width"]
    keys = iter(jax.random.split(jax.random.PRNGKey(0), 8))
    p = {"stem_w": jax.random.normal(next(keys), (Mw, k["c_in"], f, f)), "stem_b": jnp.zeros(Mw),
         "z": jax.random.normal(next(keys), (D * mult**2, Nz)), "W_in": jax.random.normal(next(keys), (N, Nz, Nz)),
         "B_in": jnp.zeros((N, Nz)), "W_out": jax.random.normal(next(keys), (f, N * f, Nz)),
         "B_out": jnp.zeros((f, N * f)), "head_w": jax.random.normal(next(keys), (Mw, k["classes"])),
         "head_b": jnp.zeros(k["classes"])}  # fmt: skip
    p |= {f"conv{j}_b": jnp.zeros(Mw) for j in range(D)}
    assert sum(math.prod(v.shape) for v in p.values()) == m["params"]
    x = jax.random.normal(next(keys), (k["B"], k["c_in"], k["size"], k["size"]))
    logits = jax.jit(lambda p, x: H.forward(p, x, mult, D))(p, x)
    assert list(logits.shape) == c["blocks"]["head"]["outputs"]["logits"]
    assert list(H.tile(H.generate(p, p["z"])[: mult**2], mult).shape) == c["blocks"]["conv"]["internals"]["w"]


def test_vmap_over_embeddings():
    """generate on a batch of embeddings is generate on each one."""
    P, _, _ = fixture()
    p = part(P, "static")
    one = jax.vmap(lambda z: H.generate(p, z[None])[0])(p["z"])
    assert np.allclose(one, H.generate(p, p["z"]), atol=1e-14)
