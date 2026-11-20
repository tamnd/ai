# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://JBSH5S).
"""Tests of the JAX linear attention: the shared fixture and its gradients for every causal implementation, jit,
and the chunked form against the scan for several chunk lengths."""

import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from linear_attention import causal_chunked, causal_quadratic, causal_scan, elu_plus_one, forward, loss  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIX / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIX / "input-small.npz").items()}
    return params, inp, np.load(FIX / "expected-small.npz")


@pytest.mark.parametrize("causal,impl", [(True, "scan"), (True, "chunked"), (True, "quadratic"), (False, None)])
def test_fixture(causal, impl):
    p, inp, exp = fixture()
    tag = "causal" if causal else "full"
    kw = {"causal": causal, "impl": impl}
    assert np.allclose(forward(p, inp["x"], **kw), exp["out_" + tag], atol=1e-12)
    value, (grads, dx) = jax.value_and_grad(loss, argnums=(0, 1))(p, inp["x"], inp["target"], **kw)
    assert np.isclose(value, exp["loss_" + tag], atol=1e-12)
    for k, g in grads.items():
        assert np.allclose(g, exp[f"d{k}_{tag}"], atol=1e-12), k
    assert np.allclose(dx, exp["dx_" + tag], atol=1e-12)


def test_jit():
    p, inp, exp = fixture()
    f = jax.jit(lambda p, x: forward(p, x))
    assert np.allclose(f(p, inp["x"]), exp["out_causal"], atol=1e-12)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 6, 12])
def test_chunked_equals_the_scan(T):
    """eq-chunk for every chunk length that divides N = 12: T = 1 is the recurrence, T = N the quadratic form."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    fQ, fK = elu_plus_one(jax.random.normal(k1, (2, 3, 12, 4))), elu_plus_one(jax.random.normal(k2, (2, 3, 12, 4)))
    V = jax.random.normal(k3, (2, 3, 12, 5))
    want = causal_scan(fQ, fK, V)
    assert np.allclose(causal_chunked(fQ, fK, V, T), want, atol=1e-12)
    assert np.allclose(causal_quadratic(fQ, fK, V), want, atol=1e-12)
