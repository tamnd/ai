# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/long-short-term-memory/ (ai://NEW-long-short-term-memory).
"""Write the shared fixtures of the LSTM layer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The gradients come from the backward pass of the reference, for the loss
sum(h * r) + sum(h_T * r_h) + sum(c_T * r_c) with the fixed weights r, r_h and r_c of input-small.npz.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 19971115  # the issue date of Neural Computation 9(8)
CONFIG = {"input_size": 3, "hidden_size": 4}  # the example config of shapes.yaml
B, T = 2, 5


def load_reference():
    spec = importlib.util.spec_from_file_location("lstm_ref", HERE.parent / "impl" / "numpy" / "lstm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, forget_bias=1.0)
    H = cfg["hidden_size"]
    x = rng.normal(0.0, 1.0, (B, T, cfg["input_size"]))
    h0 = rng.normal(0.0, 0.5, (B, H))
    c0 = rng.normal(0.0, 0.5, (B, H))
    r, r_h, r_c = (
        rng.normal(0.0, 1.0, (B, T, H)),
        rng.normal(0.0, 1.0, (B, H)),
        rng.normal(0.0, 1.0, (B, H)),
    )
    h, cache = ref.forward(params, x, h0, c0)
    dx, dp, dh0, dc0 = ref.backward(params, cache, r, r_h, r_c)
    expected = {
        "h": h,
        "c": cache["c"],
        "h_T": cache["h_T"],
        "c_T": cache["c_T"],
        "i": cache["i"],
        "f": cache["f"],
        "g": cache["g"],
        "o": cache["o"],
        "dx": dx,
        "dh0": dh0,
        "dc0": dc0,
        "dW": dp["W"],
        "dU": dp["U"],
        "db": dp["b"],
    }
    files = {
        "params-small.npz": params,
        "input-small.npz": {"x": x, "h0": h0, "c0": c0, "r": r, "r_h": r_h, "r_c": r_c},
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
        f"inputs: {{B: {B}, T: {T}, forget_bias: 1.0}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
