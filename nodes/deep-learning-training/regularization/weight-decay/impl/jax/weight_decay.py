# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://NEW-weight-decay).
"""Weight decay in JAX, as pure functions on pytrees.

sgd_update is torch.optim.SGD: with decoupled=False, weight_decay is an L2 term added to the gradient before the
momentum buffer (eq-l2-step); with decoupled=True the parameters are multiplied by 1 - lr * weight_decay beside the step
(eq-decay-step). mask is a pytree of booleans with the structure of the parameters; decay_mask selects the arrays with
ndim >= 2. l2_penalty is the explicit penalty, whose jax.grad is weight_decay * p.
"""

import jax
import jax.numpy as jnp


def l2_penalty(params, weight_decay, mask=None):
    """(weight_decay / 2) times the sum of the squares of the selected leaves."""
    leaves = jax.tree_util.tree_leaves(params)
    keep = [True] * len(leaves) if mask is None else jax.tree_util.tree_leaves(mask)
    return 0.5 * weight_decay * sum(jnp.sum(p * p) for p, k in zip(leaves, keep) if k)


def decay_mask(params):
    """True for leaves with ndim >= 2 (weights), False for biases and normalization parameters."""
    return jax.tree_util.tree_map(lambda p: jnp.ndim(p) >= 2, params)


def sgd_init(params):
    """The momentum buffers, zero, and the step count."""
    return {"t": 0, "buf": jax.tree_util.tree_map(jnp.zeros_like, params)}


def sgd_update(params, grads, state, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
               decoupled=False, mask=None):
    """Return (params, state) after one step. The first step sets the buffer to the gradient, as PyTorch does."""
    if mask is None:
        mask = jax.tree_util.tree_map(lambda _: True, params)
    first = state["t"] == 0

    def leaf(p, g, b, m):
        wd = weight_decay if m else 0.0
        if not decoupled:
            g = g + wd * p
        if momentum:
            b = jnp.where(first, g, momentum * b + (1 - dampening) * g)
            d = g + momentum * b if nesterov else b
        else:
            d = g
        if decoupled:
            p = p * (1 - lr * wd)
        return p - lr * d, b

    out = jax.tree_util.tree_map(leaf, params, grads, state["buf"], mask)
    new_p = jax.tree_util.tree_map(lambda _, o: o[0], params, out)
    new_b = jax.tree_util.tree_map(lambda _, o: o[1], params, out)
    return new_p, {"t": state["t"] + 1, "buf": new_b}


def map_ridge(X, y, noise_var, prior_var):
    """The MAP estimate of a linear model with Gaussian noise under the prior N(0, prior_var I)."""
    lam = noise_var / prior_var
    return jnp.linalg.solve(X.T @ X + lam * jnp.eye(X.shape[1]), X.T @ y)


def neg_log_posterior(theta, X, y, noise_var, prior_var):
    """The negative log posterior up to a constant: the Gaussian likelihood plus |theta|^2 / (2 prior_var)."""
    return 0.5 * jnp.sum((y - X @ theta) ** 2) / noise_var + 0.5 * jnp.sum(theta**2) / prior_var


def exp_lr_schedule(lr, weight_decay, t):
    """lr (1 - lr weight_decay)^(-2t - 1), the learning rate of Li and Arora with momentum 0."""
    return lr * (1 - lr * weight_decay) ** (-2 * t - 1)


def effective_lr(lr, w):
    """lr / |w|^2, the effective learning rate of the direction of scale-invariant weights."""
    return lr / jnp.sum(w * w)
