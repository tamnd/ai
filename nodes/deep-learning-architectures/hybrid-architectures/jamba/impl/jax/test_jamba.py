# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Tests of the JAX Jamba: the shared fixture and its gradients under jit, the dispatch of the tokens to the experts'
buffers, and what a capacity below the even share drops."""

import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jamba as M  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIX / "params-small.npz").items()}
    return params, jnp.asarray(np.load(FIX / "input-small.npz")["tokens"]), np.load(FIX / "expected-small.npz")


def test_fixture():
    params, tok, exp = fixture()
    cfg = M.config()
    logits, aux, dropped = jax.jit(lambda p, t: M.forward(p, cfg, t))(params, tok[:, :-1])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.isclose(aux, exp["aux"], atol=1e-12)
    assert int(dropped) == 0
    value, grads = jax.jit(jax.value_and_grad(lambda p, t: M.loss(p, cfg, t)))(params, tok)
    assert np.isclose(value, exp["loss"], atol=1e-12)
    assert set(grads) == set(params)
    for k, g in grads.items():
        assert np.allclose(g, exp["d" + k], atol=1e-11), k


def test_dispatch_gives_each_choice_its_own_slot():
    top = jnp.array([[0, 2], [2, 1], [0, 2], [1, 0]])
    d = M.dispatch(top, 3, 4)
    assert d.shape == (4, 2, 3, 4)
    assert (d.sum((2, 3)) == 1).all()  # every choice has one slot
    assert (d.sum((0, 1)) <= 1).all()  # and no slot has two
    assert [int(jnp.argmax(d[m, k, top[m, k]])) for m, k in ((0, 0), (2, 0), (3, 1))] == [0, 1, 2]
    assert int(M.dispatch(top, 3, 2).sum()) == 6  # experts 0 and 2 have three choices for two slots: each drops one


@pytest.mark.parametrize("capacity", [1.0, 1.25])
def test_a_small_capacity_drops_choices(capacity):
    """With the capacity of GShard and the Switch Transformer, C = ceil(c M K / n) slots, an unbalanced router fills
    some buffers and drops the choices that come after; the layer passes those tokens on with less of their mixture,
    and the dropped choices count."""
    params, tok, _ = fixture()
    full, _, _ = M.forward(params, M.config(), tok[:, :-1])
    part, _, dropped = M.forward(params, M.config(capacity=capacity), tok[:, :-1])
    assert int(dropped) > 0
    assert not np.allclose(full, part)
    _, _, none = M.forward(params, M.config(capacity=M.config()["n_experts"] / M.config()["top_k"]), tok[:, :-1])
    assert int(none) == 0


def test_kinds_change_the_mixers_only():
    params, tok, _ = fixture()
    cfg = M.config()
    a, _, _ = M.forward(params, cfg, tok, kinds=["mamba", "mamba", "attn", "mamba"])
    b, _, _ = M.forward(params, cfg, tok)
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        M.forward(params, cfg, tok, kinds=["mamba"] * 3)
