# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/transformer/ (ai://4W3NTQ).
"""Write the shared fixtures of the Transformer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The parameter tree is stored flat, its path joined by "/".
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20170612  # the day the paper appeared on arXiv
CONFIG = {"d_model": 16, "n_heads": 4, "d_ff": 64, "n_layers": 2, "vocab": 50}  # the example config of shapes.yaml
B, S, T = 2, 6, 5


def load_reference():
    spec = importlib.util.spec_from_file_location("transformer_ref", HERE.parent / "impl" / "numpy" / "transformer.py")
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
    # Biases and norms start at 0 and 1 in init; random values here make the fixture check that every one is used.
    flat = flatten(params)
    for k, v in flat.items():
        if k.split("/")[-1] in ("b_q", "b_k", "b_v", "b_o", "b_1", "b_2", "beta"):
            v[...] = rng.normal(0.0, 0.1, v.shape)
        elif k.endswith("/gamma"):
            v[...] = 1.0 + rng.normal(0.0, 0.1, v.shape)
    src = rng.integers(0, cfg["vocab"], (B, S))
    tgt = rng.integers(0, cfg["vocab"], (B, T))
    src_pad = np.zeros((B, S), dtype=bool)
    src_pad[1, -2:] = True  # the second source sentence is two tokens shorter
    logits, cache = ref.forward(params, src, tgt, cfg, src_pad=src_pad)
    expected = {"logits": logits, "memory": cache["memory"],
                "enc/0/weights": cache["enc"][0]["weights"], "dec/0/self/weights": cache["dec"][0]["self"]["weights"],
                "dec/0/cross/weights": cache["dec"][0]["cross"]["weights"]}
    files = {"params-small.npz": flat, "input-small.npz": {"src": src, "tgt": tgt, "src_pad": src_pad},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + "}",
             f"inputs: {{B: {B}, S: {S}, T: {T}, padded: \"the last 2 source tokens of sentence 1\"}}",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
