# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://MDN3YW).
"""Tests of the JAX layer: conv, scan and the parallel scan agree, the impulse response, causality and the discretizations."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ssm import apply, discrete, init, kernel  # noqa: E402

METHODS = ["zoh", "bilinear"]


def layer(seed, h=3, n=4):
    params = init(jax.random.PRNGKey(seed), h, n, dt_min=0.01, dt_max=0.5)
    rng = np.random.default_rng(seed)
    params["log_a_re"] = params["log_a_re"] + 0.3 * rng.standard_normal((h, n))
    params["a_im"] = params["a_im"] + rng.standard_normal((h, n))
    params["b"] = params["b"] + 0.5 * (rng.standard_normal((h, n)) + 1j * rng.standard_normal((h, n)))
    return params


def reference(params, u, method):
    ab, bb = (np.asarray(t) for t in discrete(params, method))
    c, d, u = np.asarray(params["c"]), np.asarray(params["d"]), np.asarray(u)
    x = np.zeros((u.shape[0],) + ab.shape, dtype=complex)
    y = np.empty_like(u)
    for t in range(u.shape[1]):
        x = ab * x + bb * u[:, t, :, None]
        y[:, t] = 2 * np.einsum("hn,bhn->bh", c, x).real + d * u[:, t]
    return y


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("length", [1, 8, 31])
def test_three_modes_agree(method, length):
    params = layer(length)
    u = jnp.asarray(np.random.default_rng(length).standard_normal((2, length, 3)))
    ref = reference(params, u, method)
    for mode in ("conv", "scan", "parallel"):
        assert np.allclose(apply(params, u, method, mode), ref, atol=1e-10), mode


@pytest.mark.parametrize("method", METHODS)
def test_impulse_response_is_kernel(method):
    params = layer(1)
    u = jnp.zeros((3, 20, 3)).at[jnp.arange(3), 0, jnp.arange(3)].set(1.0)
    k = np.asarray(kernel(params, 20, method))
    for mode in ("conv", "parallel"):
        y = np.asarray(apply(params, u, method, mode))
        for h in range(3):
            expected = k[h].copy()
            expected[0] += float(params["d"][h])
            assert np.allclose(y[h, :, h], expected, atol=1e-12)


def test_causality():
    params = layer(2)
    rng = np.random.default_rng(2)
    u = rng.standard_normal((2, 16, 3))
    v = u.copy()
    v[:, 6:] += rng.standard_normal((2, 10, 3))
    for mode in ("conv", "scan", "parallel"):
        a, b = np.asarray(apply(params, jnp.asarray(u), "zoh", mode)), np.asarray(apply(params, jnp.asarray(v), "zoh", mode))
        assert np.allclose(a[:, :6], b[:, :6], atol=1e-12)
        assert not np.allclose(a[:, 6:], b[:, 6:])


def test_discretizations():
    params = {
        "log_dt": jnp.array([math.log(0.25)]),
        "log_a_re": jnp.array([[math.log(0.7)]]),
        "a_im": jnp.array([[0.0]]),
        "b": jnp.array([[1.3 + 0j]]),
        "c": jnp.array([[1.0 + 0j]]),
        "d": jnp.array([0.0]),
    }
    ab, bb = discrete(params, "zoh")
    assert np.isclose(complex(ab[0, 0]), math.exp(-0.175)) and np.isclose(complex(bb[0, 0]), (math.exp(-0.175) - 1) / -0.7 * 1.3)
    ab, bb = discrete(params, "bilinear")
    assert np.isclose(complex(ab[0, 0]), (1 - 0.0875) / (1 + 0.0875)) and np.isclose(complex(bb[0, 0]), 0.25 * 1.3 / 1.0875)
    big = layer(5, h=4, n=6)
    big["log_dt"] = jnp.array([-6.0, 0.0, 4.0, 9.0])
    big["a_im"] = 100.0 * big["a_im"]
    assert float(jnp.abs(discrete(big, "bilinear")[0]).max()) < 1.0
    with pytest.raises(ValueError):
        discrete(params, "euler")
    with pytest.raises(ValueError):
        apply(params, jnp.zeros((1, 2, 1)), "zoh", "fft")


def test_gradients_agree():
    params = layer(6, h=2, n=3)
    u = jnp.asarray(np.random.default_rng(6).standard_normal((1, 10, 2)))

    def loss(p, mode):
        return jnp.sum(apply(p, u, "zoh", mode) ** 2)

    gc, gp = jax.grad(loss)(params, "conv"), jax.grad(loss)(params, "parallel")
    for name in ("log_dt", "log_a_re", "d"):
        assert np.allclose(gc[name], gp[name], atol=1e-8), name
