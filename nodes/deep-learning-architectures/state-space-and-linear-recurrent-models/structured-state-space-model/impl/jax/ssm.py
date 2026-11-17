# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://NEW-structured-state-space-model).
"""A diagonal state space layer in JAX: pure functions init and apply, with three ways to compute it.

conv uses the kernel and the FFT, scan uses jax.lax.scan one step at a time, and parallel uses
jax.lax.associative_scan with the operator (a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2), which also applies
when the parameters change with the input.
"""

import math

import jax
import jax.numpy as jnp


def init(key, h=2, n=4, dt_min=1e-3, dt_max=1e-1):
    """S4D-Lin for A, log step sizes uniform in [log dt_min, log dt_max], B = 1, C complex normal."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k = jnp.arange(n)
    c = (jax.random.normal(k2, (h, n)) + 1j * jax.random.normal(k3, (h, n))) * math.sqrt(0.5)
    return {
        "log_dt": jax.random.uniform(k1, (h,), minval=math.log(dt_min), maxval=math.log(dt_max)),
        "log_a_re": jnp.full((h, n), math.log(0.5)),
        "a_im": jnp.tile(jnp.pi * k, (h, 1)).astype(jnp.float64),
        "b": jnp.ones((h, n), dtype=jnp.complex128),
        "c": c.astype(jnp.complex128),
        "d": jax.random.normal(k4, (h,)),
    }


def discrete(params, method="zoh"):
    """(Ab, Bb), each (H, N)."""
    a = -jnp.exp(params["log_a_re"]) + 1j * params["a_im"]
    dt = jnp.exp(params["log_dt"])[:, None]
    if method == "zoh":
        ab = jnp.exp(dt * a)
        return ab, (ab - 1.0) / a * params["b"]
    if method == "bilinear":
        den = 1.0 - dt * a / 2.0
        return (1.0 + dt * a / 2.0) / den, dt * params["b"] / den
    raise ValueError(f"method must be zoh or bilinear, got {method!r}")


def kernel(params, length, method="zoh"):
    """K[h, l] = 2 Re sum_n C Bb Ab^l, shape (H, L)."""
    ab, bb = discrete(params, method)
    powers = ab[:, :, None] ** jnp.arange(length)
    return 2.0 * jnp.einsum("hn,hnl->hl", params["c"] * bb, powers).real


def _combine(left, right):
    a1, b1 = left
    a2, b2 = right
    return a1 * a2, a2 * b1 + b2


def apply(params, u, method="zoh", mode="conv"):
    """u (B, L, H) to y (B, L, H); mode is conv, scan or parallel."""
    length = u.shape[1]
    if mode == "conv":
        k = kernel(params, length, method)
        uf = jnp.fft.rfft(u, n=2 * length, axis=1)
        kf = jnp.fft.rfft(k.T, n=2 * length, axis=0)
        return jnp.fft.irfft(uf * kf, n=2 * length, axis=1)[:, :length] + params["d"] * u
    ab, bb = discrete(params, method)
    inputs = bb * u[..., None]  # (B, L, H, N): Bb u_k
    if mode == "scan":

        def body(x, v):
            x = ab * x + v
            return x, x

        x0 = jnp.zeros((u.shape[0],) + ab.shape, dtype=jnp.complex128)
        _, xs = jax.lax.scan(body, x0, jnp.swapaxes(inputs, 0, 1))
        xs = jnp.swapaxes(xs, 0, 1)
    elif mode == "parallel":
        coeffs = jnp.broadcast_to(ab, inputs.shape)
        _, xs = jax.lax.associative_scan(_combine, (coeffs, inputs), axis=1)
    else:
        raise ValueError(f"mode must be conv, scan or parallel, got {mode!r}")
    return 2.0 * jnp.einsum("hn,blhn->blh", params["c"], xs).real + params["d"] * u
