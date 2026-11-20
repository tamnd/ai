# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://B95BYJ).
"""Tests of the JAX RNNsearch: the shared fixture and its gradients, jit, the energies on the whole grid, the
precomputation of U_a h_j and the parameter count."""

import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from attention import config, encode, energies, forward, init, loss, shapes  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIX / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIX / "input-small.npz").items()}
    return params, inp, np.load(FIX / "expected-small.npz")


def test_fixture():
    p, inp, exp = fixture()
    logits, alpha, _ = forward(p, config(), inp["src"], inp["tgt"], inp["src_mask"])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.allclose(alpha, exp["alpha"], atol=1e-12)
    assert np.allclose(encode(p, inp["src"], inp["src_mask"]), exp["annotations"], atol=1e-12)
    value, grads = jax.value_and_grad(loss)(p, config(), inp["src"], inp["tgt"], inp["src_mask"], inp["tgt_mask"])
    assert np.isclose(value, exp["loss"], atol=1e-12)
    for k, g in grads.items():
        assert np.allclose(g, exp["d" + k], atol=1e-12), k
    plain = forward(p, config(attention=False), inp["src"], inp["tgt"], inp["src_mask"])[0]
    assert np.allclose(plain, exp["logits_encdec"], atol=1e-12)


def test_jit():
    p, inp, exp = fixture()
    f = jax.jit(lambda p, s, t, m: forward(p, config(), s, t, m)[0])
    assert np.allclose(f(p, inp["src"], inp["tgt"], inp["src_mask"]), exp["logits"], atol=1e-12)


def test_energies_on_the_grid_give_the_alignments():
    """The softmax over j of the (T, S) grid of energies, with the pads masked, is the alpha of every step."""
    p, inp, exp = fixture()
    _, alpha, states = forward(p, config(), inp["src"], inp["tgt"], inp["src_mask"])
    H = encode(p, inp["src"], inp["src_mask"])
    e = jnp.where(inp["src_mask"][:, None], energies(p, states, H), -jnp.inf)
    assert np.allclose(jax.nn.softmax(e, -1), exp["alpha"], atol=1e-12)


def test_precomputing_U_a_h_is_exact():
    """Bahdanau et al.'s A.1.2: U_a h_j does not depend on i, so it is computed once per sentence; tanh(W_a s + U_a h)
    with the product taken per step or once gives the same energies."""
    p, inp, _ = fixture()
    _, _, states = forward(p, config(), inp["src"], inp["tgt"], inp["src_mask"])
    H = encode(p, inp["src"], inp["src_mask"])
    per_step = jnp.stack([jnp.tanh((states[:, i] @ p["W_a"])[:, None] + H @ p["U_a"]) @ p["v_a"] for i in range(4)], 1)
    assert np.allclose(per_step, energies(p, states, H), atol=1e-14)


def test_parameter_count():
    cfg = config()
    assert sum(v.size for v in init(jax.random.PRNGKey(0), cfg).values()) == sum(
        int(np.prod(s)) for s in shapes(cfg).values()
    )
    assert sum(int(np.prod(s)) for s in shapes(config(30000, 30000, 620, 1000, 1000, 500)).values()) == 80_401_000
