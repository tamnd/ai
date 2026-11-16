# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/batch-normalization/ (ai://NEW-batch-normalization).
"""Batch normalization as pure functions, the JAX reference.

The input has the shape (N, C) or (N, C, ...), as in PyTorch. apply returns the output and the new running
statistics: in training it normalizes with the biased batch variance (eq-bn-forward) and updates the running averages
with the unbiased one (eq-running); in evaluation it uses the running averages (eq-inference). jax.grad gives the
backward pass, and the tests compare it with eq-bn-backward.
"""

import jax
import jax.numpy as jnp


def config(num_features, eps=1e-5, momentum=0.1):
    """Return the configuration. momentum is the PyTorch convention, the weight of the new batch. The Flax momentum
    rho is momentum = 1 - rho here."""
    num_features = int(num_features)
    if num_features < 1:
        raise ValueError(f"num_features must be positive, got {num_features}")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")
    return {"num_features": num_features, "eps": float(eps), "momentum": float(momentum)}


def init(key, cfg):
    """Return the parameters gamma = 1 and beta = 0."""
    del key
    c = cfg["num_features"]
    return {"gamma": jnp.ones(c), "beta": jnp.zeros(c)}


def init_state(cfg):
    """Return the running statistics, mean 0 and variance 1."""
    c = cfg["num_features"]
    return {"mean": jnp.zeros(c), "var": jnp.ones(c)}


def apply(params, state, x, cfg, training=True):
    """Return (y, new_state)."""
    if x.ndim < 2 or x.shape[1] != cfg["num_features"]:
        raise ValueError(f"x must have the shape (N, {cfg['num_features']}, ...), got {x.shape}")
    axes = (0,) + tuple(range(2, x.ndim))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    if training:
        count = x.size // x.shape[1]
        if count < 2:
            raise ValueError(f"training needs more than 1 value per channel, got {count}")
        mean = jnp.mean(x, axis=axes)
        var = jnp.mean((x - mean.reshape(shape)) ** 2, axis=axes)
        alpha = cfg["momentum"]
        bm, bv = jax.lax.stop_gradient(mean), jax.lax.stop_gradient(var)  # the running averages carry no gradient
        new_state = {"mean": (1 - alpha) * state["mean"] + alpha * bm,
                     "var": (1 - alpha) * state["var"] + alpha * bv * count / (count - 1)}
    else:
        mean, var, new_state = state["mean"], state["var"], state
    xhat = (x - mean.reshape(shape)) / jnp.sqrt(var.reshape(shape) + cfg["eps"])
    return params["gamma"].reshape(shape) * xhat + params["beta"].reshape(shape), new_state
