# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-equivariant-convolutional-network/ (ai://NEW-group-equivariant-convolutional-network).
"""Write the shared fixtures of the group equivariant CNN from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The gradients are those of the reference backward pass, which the automatic
differentiation of the PyTorch and JAX references reproduces to rounding.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20160224  # the day Cohen and Welling posted the paper to arXiv
CONFIG = {"c_in": 1, "c1": 4, "c2": 4, "k": 3, "n_classes": 3}  # the example config of shapes.yaml
B, N = 2, 5


def load_reference():
    spec = importlib.util.spec_from_file_location("gcnn_ref", HERE.parent / "impl" / "numpy" / "gcnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    for b in ("b1", "b2", "bc"):
        params[b] = rng.normal(0.0, 0.1, params[b].shape)  # nonzero biases, so the fixture checks them
    x = rng.normal(0.0, 1.0, (B, CONFIG["c_in"], N, N))
    labels = rng.integers(0, CONFIG["n_classes"], B)
    logits = ref.forward(params, x, cfg)
    loss, grads, dx = ref.loss_and_grads(params, x, labels, cfg)
    expected = {"logits": logits, "loss": np.array(loss), "dx": dx}
    expected.update({f"d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "labels": labels},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {c_in: 1, c1: 4, c2: 4, k: 3, n_classes: 3, group: p4, padding: zero}",
             f"inputs: {{B: {B}, N: {N}, x: \"standard Gaussian images\", labels: \"uniform over the classes\"}}",
             "loss: \"mean softmax cross-entropy of the logits\"",
             "gradients: \"of the loss, by the reference backward pass\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
