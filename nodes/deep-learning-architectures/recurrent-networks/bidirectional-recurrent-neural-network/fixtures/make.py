# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Write the shared fixtures of the bidirectional recurrent network from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml, a two-layer bidirectional LSTM, on a
batch of three sequences of lengths 6, 4 and 1 padded to 6 steps; the padding holds random numbers, which must not
change anything. The gradients are those of eq-loss from the backward pass of the reference, dx included.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 19971101  # the issue of IEEE Transactions on Signal Processing in which the BRNN appeared, November 1997
CONFIG = {"input_size": 3, "hidden_size": 4, "n_layers": 2, "n_classes": 5}  # the example config
B, T = 3, 6
LENGTHS = [6, 4, 1]
SCALE = 0.5  # wider than a usual initialization, so that the outputs are far from uniform


def load_reference():
    spec = importlib.util.spec_from_file_location("birnn_ref", HERE.parent / "impl" / "numpy" / "birnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG, cell="lstm", bidirectional=True)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    x = rng.normal(size=(B, T, CONFIG["input_size"]))
    labels = rng.integers(0, CONFIG["n_classes"], (B, T))
    lengths = np.array(LENGTHS)
    logits, cache = ref.forward(params, cfg, x, lengths)
    grads, dx = ref.backward(params, cfg, cache, labels)
    expected = {"logits": logits, "loss": np.array(ref.nll(logits, labels, lengths)), "top": cache["top"], "dx": dx}
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"x": x, "labels": labels, "lengths": lengths},
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
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", cell: lstm, bidirectional: true}",
        f"inputs: {{B: {B}, T: {T}, lengths: {LENGTHS}, init_scale: {SCALE}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
