# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Write the shared fixtures of multi-head attention from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. There are two cases on the same weights: self-attention with a causal mask,
and cross-attention over a memory whose second sequence is padded after 3 keys.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20170612  # the day Vaswani et al. appeared on arXiv
CONFIG = {"d_model": 8, "n_heads": 2}
B, T, S = 2, 5, 4
LENGTHS = [4, 3]


def load_reference():
    spec = importlib.util.spec_from_file_location("mha_ref", HERE.parent / "impl" / "numpy" / "multi_head_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    x = rng.normal(size=(B, T, cfg["d_model"]))
    memory = rng.normal(size=(B, S, cfg["d_model"]))
    dy = rng.normal(size=(B, T, cfg["d_model"]))
    causal = ref.causal_mask(T)
    padding = ref.padding_mask(LENGTHS, S)
    expected = {}
    for case, mem, mask in (("self", None, causal), ("cross", memory, padding)):
        y, cache = ref.forward(params, x, cfg, memory=mem, mask=mask)
        dx, dmem, dparams = ref.backward(params, cache, dy)
        expected[f"{case}/y"] = y
        expected[f"{case}/a"] = cache["a"]
        expected[f"{case}/dx"] = dx
        if dmem is not None:
            expected[f"{case}/dmemory"] = dmem
        for name, g in dparams.items():
            expected[f"{case}/d{name}"] = g
    inputs = {"x": x, "memory": memory, "dy": dy, "causal": causal, "padding": padding,
              "d_model": np.array(cfg["d_model"]), "n_heads": np.array(cfg["n_heads"])}
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + "}",
             f"inputs: {{B: {B}, T: {T}, S: {S}, padded: \"memory sequence 1 has 3 real keys of 4\"}}",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
