# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Write the shared fixtures of Deep Sets from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml, elements of 3 features, 2 equivariant
layers of width 8 with max pooling and tanh, a sum readout, a hidden layer of rho of 8 and 5 classes, on a batch of
two random sets of 6 slots, the second of which has 4 elements and 2 padded slots. The gradients are those of eq-loss
from the backward pass of the reference, the gradient with respect to the elements included.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 201703  # the month in which Deep Sets was posted to arXiv, March 2017
CONFIG = {"d_in": 3, "width": 8, "n_layers": 2, "n_classes": 5, "pool": "max", "readout": "sum", "act": "tanh"}
B, M = 2, 6
SCALE = 0.5  # wider than a usual initialization, so that the outputs are far from uniform


def load_reference():
    spec = importlib.util.spec_from_file_location("deepsets_ref", HERE.parent / "impl" / "numpy" / "deepsets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    mask = np.ones((B, M))
    mask[1, 4:] = 0.0
    x = rng.normal(size=(B, M, CONFIG["d_in"])) * mask[..., None]
    labels = rng.integers(0, CONFIG["n_classes"], B)
    logits, cache = ref.forward(params, cfg, x, mask)
    grads, dx = ref.backward(params, cfg, cache, logits, labels, x)
    expected = {
        "logits": logits,
        "loss": np.array(ref.nll(logits, labels)),
        "features": cache["features"],
        "pooled": cache["s"],
        "dx": dx,
    }
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"x": x, "mask": mask, "labels": labels},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", interact: true, head: set}",
        f"inputs: {{B: {B}, M: {M}, sizes: [6, 4], init_scale: {SCALE}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
