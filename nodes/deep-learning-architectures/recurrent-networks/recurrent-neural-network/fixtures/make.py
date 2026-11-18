# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Write the shared fixtures of the simple recurrent layer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The gradients come from the backward pass of the reference, for the loss
sum(h * r) + sum(h_T * r_h) with the fixed weights r and r_h of input-small.npz. The arrays of the tanh layer have
plain names; those of the relu layer, with the same parameters and inputs, end in _relu.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 19900401  # the year of Elman's Finding Structure in Time
CONFIG = {"input_size": 3, "hidden_size": 4}  # the example config of shapes.yaml
B, T = 2, 5


def load_reference():
    spec = importlib.util.spec_from_file_location("rnn_ref", HERE.parent / "impl" / "numpy" / "rnn.py")
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
    for nl, sfx in (("tanh", ""), ("relu", "_relu")):
        h, cache = ref.forward(params, x, h0, nl)
        dx, dp, dh0 = ref.backward(params, cache, r, r_h)
        arrays = {"h": h, "h_T": cache["h_T"], "a": cache["a"], "dx": dx, "dh0": dh0}
        arrays |= {"d" + k: v for k, v in dp.items()}
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
        f"inputs: {{B: {B}, T: {T}, nonlinearities: [tanh, relu]}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
