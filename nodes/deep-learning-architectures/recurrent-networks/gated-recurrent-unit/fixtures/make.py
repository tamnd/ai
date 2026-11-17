# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gated-recurrent-unit/ (ai://NEW-gated-recurrent-unit).
"""Write the shared fixtures of the GRU layer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The gradients come from the backward pass of the reference, for the loss
sum(h * r) + sum(h_T * r_h) with the fixed weights r and r_h of input-small.npz. The arrays of the reset gate
applied after the recurrent matrix, as in torch.nn.GRU, have plain names; those of the reset gate applied before
it, as in eq. (8) of Cho et al., end in _before.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20140603  # the date of the first version of arXiv 1406.1078
CONFIG = {"input_size": 3, "hidden_size": 4}  # the example config of shapes.yaml
B, T = 2, 5


def load_reference():
    spec = importlib.util.spec_from_file_location("gru_ref", HERE.parent / "impl" / "numpy" / "gru.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    H = cfg["hidden_size"]
    x = rng.normal(0.0, 1.0, (B, T, cfg["input_size"]))
    h0 = rng.normal(0.0, 0.5, (B, H))
    r, r_h = rng.normal(0.0, 1.0, (B, T, H)), rng.normal(0.0, 1.0, (B, H))
    expected = {}
    for reset, sfx in (("after", ""), ("before", "_before")):
        h, cache = ref.forward(params, x, h0, reset)
        dx, dp, dh0 = ref.backward(params, cache, r, r_h)
        arrays = {"h": h, "h_T": cache["h_T"], "r": cache["r"], "z": cache["z"], "n": cache["n"]}
        arrays |= {"dx": dx, "dh0": dh0} | {"d" + k: v for k, v in dp.items()}
        expected |= {name + sfx: v for name, v in arrays.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"x": x, "h0": h0, "r": r, "r_h": r_h},
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
        f"inputs: {{B: {B}, T: {T}, resets: [after, before]}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
