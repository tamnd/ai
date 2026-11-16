# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/graph-neural-network/ (ai://NEW-graph-neural-network).
"""Write the shared fixtures of the graph neural network from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The parameter tree is stored flat, its path joined by "/".
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20170404  # the day Gilmer et al. posted arXiv 1704.01212
CONFIG = {
    "d_in": 3,
    "d_hidden": 8,
    "n_layers": 2,
    "classes": 4,
}  # the example config of shapes.yaml
B, N = 2, 6


def load_reference():
    spec = importlib.util.spec_from_file_location("gnn_ref", HERE.parent / "impl" / "numpy" / "graph_neural_network.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flatten(tree, prefix=""):
    out = {}
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for k, v in items:
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(flatten(v, path))
        else:
            out[path] = np.asarray(v)
    return out


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    flat = flatten(params)
    for k, v in flat.items():
        if k.endswith("/b"):  # the biases start at 0 in init; random values check that each one is used
            v[...] = rng.normal(0.0, 0.1, v.shape)
    adj = np.zeros((B, N, N))
    for i in range(N):  # graph 0: the 6-cycle with the chord 0 - 3; graph 1: the 5-path and an isolated node 5
        adj[0, i, (i + 1) % N] = adj[0, (i + 1) % N, i] = 1.0
        if i < N - 2:
            adj[1, i, i + 1] = adj[1, i + 1, i] = 1.0
    adj[0, 0, 3] = adj[0, 3, 0] = 1.0
    x = rng.normal(size=(B, N, CONFIG["d_in"]))
    W_gcn = rng.normal(size=(CONFIG["d_in"], CONFIG["d_hidden"]))
    logits, cache = ref.forward(params, x, adj, cfg)
    adj_hat = ref.normalized_adjacency(adj)
    expected = {
        "logits": logits,
        "graph": cache["graph"],
        "adj_hat": adj_hat,
        "gcn": ref.gcn_layer(W_gcn, x, adj_hat),
    }
    expected.update({f"h/{i}": h for i, h in enumerate(cache["h"])})
    files = {
        "params-small.npz": flat,
        "input-small.npz": {"x": x, "adj": adj, "W_gcn": W_gcn},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + "}",
        f'inputs: {{B: {B}, N: {N}, graphs: "a 6-cycle with one chord; a 5-path and an isolated node"}}',
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
