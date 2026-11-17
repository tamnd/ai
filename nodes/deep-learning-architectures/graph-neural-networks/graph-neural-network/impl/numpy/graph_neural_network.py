# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://PS4TCX).
"""A message-passing graph neural network with sum aggregation and a sum readout, and the GCN layer, in NumPy.

The model of the shape contract computes eq-message, eq-update and eq-readout of the page on a batch of graphs
with dense adjacency matrices. gcn_layer and normalized_adjacency compute eq-gcn and eq-renorm, aggregate gives the
sum, mean and max of eq-aggregate, and wl_histograms runs the colour refinement of eq-wl. There is no backward pass:
the page derives no gradient, and the PyTorch and JAX versions train by automatic differentiation.
"""

import numpy as np


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


def _weight(rng, n_in, n_out):
    """A weight with variance 1 / n_in. The sources do not fix the initialization of this model."""
    return rng.normal(0.0, n_in**-0.5, (n_in, n_out))


def init(rng, cfg):
    """Return the parameter tree: a list of layers with W_self, W_nbr and b, then the head W and b."""
    F, D, K = cfg["d_in"], cfg["d_hidden"], cfg["classes"]
    layers = []
    for i in range(cfg["n_layers"]):
        n_in = F if i == 0 else D
        layers.append(
            {
                "W_self": _weight(rng, n_in, D),
                "W_nbr": _weight(rng, n_in, D),
                "b": np.zeros(D),
            }
        )
    return {"layers": layers, "head": {"W": _weight(rng, D, K), "b": np.zeros(K)}}


def relu(z):
    return np.maximum(z, 0.0)


def aggregate(adj, h, how="sum"):
    """Combine the states of the neighbours of each node, eq-aggregate. adj[b, v, u] = 1 when u is a neighbour of v."""
    if how == "sum":
        return adj @ h  # (B, N, D)
    deg = adj.sum(axis=-1, keepdims=True)  # (B, N, 1)
    if how == "mean":
        return np.divide(
            adj @ h,
            deg,
            out=np.zeros(h.shape[:-2] + (adj.shape[-2], h.shape[-1])),
            where=deg > 0,
        )
    if how == "max":
        nbr = np.where(adj[..., None] > 0, h[..., None, :, :], -np.inf)  # (B, N, N, D): row v holds the neighbours of v
        return np.where(deg > 0, nbr.max(axis=-2), 0.0)  # a node with no neighbour gets 0
    raise ValueError(f"unknown aggregation {how!r}")


def mp_layer(p, h, adj):
    """One message-passing layer: the message W_nbr h_u summed over the neighbours, then the update."""
    m = adj @ (h @ p["W_nbr"])  # (B, N, D); eq: eq-message
    return relu(h @ p["W_self"] + m + p["b"])  # (B, N, D); eq: eq-update


def readout(h):
    """The graph vector: the sum of the node states, eq-readout."""
    return h.sum(axis=-2)  # (B, D); eq: eq-readout


def forward(params, x, adj, cfg):
    """Return (logits, cache) for node features x (B, N, F) and adjacency adj (B, N, N)."""
    if x.shape[-1] != cfg["d_in"] or len(params["layers"]) != cfg["n_layers"]:
        raise ValueError("the input or the parameters do not match the config")
    h = x  # (B, N, F)
    states = []
    for p in params["layers"]:
        h = mp_layer(p, h, adj)  # (B, N, D)
        states.append(h)
    g = readout(h)  # (B, D)
    logits = g @ params["head"]["W"] + params["head"]["b"]  # (B, K)
    return logits, {"h": states, "graph": g}


def normalized_adjacency(adj):
    """D~^(-1/2) (A + I) D~^(-1/2) of Kipf and Welling, eq-renorm. Every degree of A + I is at least 1."""
    n = adj.shape[-1]
    a = adj + np.eye(n)  # (B, N, N); eq: eq-renorm
    s = a.sum(axis=-1) ** -0.5  # (B, N)
    return s[..., :, None] * a * s[..., None, :]  # (B, N, N)


def gcn_layer(W, h, adj_hat, activation=True):
    """The GCN layer sigma(A_hat H W), eq-gcn, with A_hat from normalized_adjacency and no bias, as in the paper."""
    z = adj_hat @ (h @ W)  # (B, N, D); eq: eq-gcn
    return relu(z) if activation else z


def wl_histograms(adjs, rounds):
    """1-WL colour refinement, eq-wl, on several graphs (each (N_i, N_i)) that start with equal colours.

    Each round gives a node the pair (its colour, the sorted multiset of its neighbours' colours) and renames the
    distinct pairs by integers, in one table shared by all the graphs so that their colours can be compared.
    Return, for each graph, its sorted list of colours after each round 0, 1, ..., rounds.
    """
    colours = [np.zeros(a.shape[0], dtype=int) for a in adjs]
    out = [[tuple(np.sort(c).tolist())] for c in colours]
    for _ in range(rounds):
        sigs = [
            [(int(c[v]), tuple(sorted(c[np.flatnonzero(a[v])].tolist()))) for v in range(len(c))]
            for a, c in zip(adjs, colours, strict=True)
        ]
        table = {s: k for k, s in enumerate(sorted({s for g in sigs for s in g}))}  # eq: eq-wl
        colours = [np.array([table[s] for s in g], dtype=int) for g in sigs]
        for i, c in enumerate(colours):
            out[i].append(tuple(np.sort(c).tolist()))
    return out


def param_count(cfg):
    """The number of parameters, from the sizes alone."""
    F, D, L, K = cfg["d_in"], cfg["d_hidden"], cfg["n_layers"], cfg["classes"]
    return (2 * F * D + D) + (L - 1) * (2 * D * D + D) + D * K + K


def flops(cfg, B, N):
    """FLOPs of forward with a multiply-add as 2 and the dense adjacency; sums, biases and ReLU are not counted."""
    F, D, L, K = cfg["d_in"], cfg["d_hidden"], cfg["n_layers"], cfg["classes"]
    first = 4 * B * N * F * D + 2 * B * N * N * D
    rest = 4 * B * N * D * D + 2 * B * N * N * D
    return first + (L - 1) * rest + 2 * B * D * K
