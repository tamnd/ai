# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""The message-passing network with sum aggregation and a sum readout, and the GCN layer, as pure JAX functions.

It computes eq-message, eq-update, eq-readout, eq-renorm and eq-gcn of the page as the NumPy reference does, and
leaves the gradient to jax.grad. init and apply use the parameter tree of the NumPy reference.
"""

import jax
import jax.numpy as jnp


def config(d_in=3, d_hidden=8, n_layers=2, classes=4):
    """Return the config: node feature width F, hidden width D, number of layers L and number of classes K."""
    if min(d_in, d_hidden, n_layers, classes) < 1:
        raise ValueError("the sizes must be at least 1")
    return {
        "d_in": d_in,
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "classes": classes,
    }


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    F, D, L, K = cfg["d_in"], cfg["d_hidden"], cfg["n_layers"], cfg["classes"]
    return (2 * F * D + D) + (L - 1) * (2 * D * D + D) + D * K + K


def init(key, cfg):
    """Return the parameter tree of the NumPy reference: weights with variance 1 / n_in and zero biases."""
    F, D, K = cfg["d_in"], cfg["d_hidden"], cfg["classes"]
    keys = iter(jax.random.split(key, 2 * cfg["n_layers"] + 1))
    layers = []
    for i in range(cfg["n_layers"]):
        n_in = F if i == 0 else D
        layers.append(
            {
                "W_self": jax.random.normal(next(keys), (n_in, D)) * n_in**-0.5,
                "W_nbr": jax.random.normal(next(keys), (n_in, D)) * n_in**-0.5,
                "b": jnp.zeros(D),
            }
        )
    return {
        "layers": layers,
        "head": {
            "W": jax.random.normal(next(keys), (D, K)) * D**-0.5,
            "b": jnp.zeros(K),
        },
    }


def from_flat(flat):
    """Rebuild the tree from the flat fixture ("layers/0/W_self", "head/W", ...)."""
    n = 1 + max(int(k.split("/")[1]) for k in flat if k.startswith("layers/"))
    layers = [{name: jnp.asarray(flat[f"layers/{i}/{name}"]) for name in ("W_self", "W_nbr", "b")} for i in range(n)]
    return {
        "layers": layers,
        "head": {"W": jnp.asarray(flat["head/W"]), "b": jnp.asarray(flat["head/b"])},
    }


def apply(params, x, adj):
    """Return (logits, cache) for node features x (B, N, F) and adjacency adj (B, N, N)."""
    h, states = x, []
    for p in params["layers"]:
        m = adj @ (h @ p["W_nbr"])  # (B, N, D); eq: eq-message
        h = jax.nn.relu(h @ p["W_self"] + m + p["b"])  # (B, N, D); eq: eq-update
        states.append(h)
    g = h.sum(axis=-2)  # (B, D); eq: eq-readout
    return g @ params["head"]["W"] + params["head"]["b"], {
        "h": states,
        "graph": g,
    }  # (B, K)


def normalized_adjacency(adj):
    """D~^(-1/2) (A + I) D~^(-1/2), eq-renorm."""
    a = adj + jnp.eye(adj.shape[-1])  # (B, N, N); eq: eq-renorm
    s = a.sum(axis=-1) ** -0.5  # (B, N)
    return s[..., :, None] * a * s[..., None, :]


def gcn_layer(W, h, adj_hat, activation=True):
    """sigma(A_hat H W), eq-gcn."""
    z = adj_hat @ (h @ W)  # (B, N, D); eq: eq-gcn
    return jax.nn.relu(z) if activation else z
