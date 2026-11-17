# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://NEW-convolutional-neural-network).
"""Write the shared fixtures of the convolutional network from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The parameter tree is stored flat, its paths joined by "/".
The expected arrays are the logits of the network, the maps after each layer, and one strided, padded
convolution with zero and with circular padding, so that every framework checks the same layer arithmetic.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 19981101  # the month of LeCun et al. (1998)
CONFIG = {"channels": 1, "c1": 2, "c2": 3, "f1": 5, "f2": 4, "classes": 3, "size": 16}  # the example of shapes.yaml
B = 2
LAYER = {"B": 2, "C": 3, "D": 4, "H": 7, "k": 3, "stride": 2, "pad": 1}  # one strided, padded convolution


def load_reference():
    spec = importlib.util.spec_from_file_location("cnn_ref", HERE.parent / "impl" / "numpy" / "cnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    flat = {}
    for name, p in params.items():
        p["b"][...] = rng.normal(0.0, 0.1, p["b"].shape)  # nonzero biases, so that the fixture checks each is added
        flat[f"{name}/w"], flat[f"{name}/b"] = p["w"], p["b"]
    x = rng.normal(0.0, 1.0, (B, CONFIG["channels"], CONFIG["size"], CONFIG["size"]))
    L = LAYER
    lx = rng.normal(0.0, 1.0, (L["B"], L["C"], L["H"], L["H"]))
    lw = rng.normal(0.0, 1.0, (L["D"], L["C"], L["k"], L["k"]))
    lb = rng.normal(0.0, 1.0, (L["D"],))
    logits, cache = ref.forward(params, x, cfg)
    expected = {"logits": logits, **{f"{k}": v for k, v in cache.items()},
                "layer_zeros": ref.conv2d(lx, lw, lb, L["stride"], L["pad"]),
                "layer_circular": ref.conv2d(lx, lw, lb, L["stride"], L["pad"], mode="circular")}
    files = {"params-small.npz": flat, "input-small.npz": {"x": x, "layer_x": lx, "layer_w": lw, "layer_b": lb},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    conf = ", ".join(f"{k}: {v}" for k, v in CONFIG.items())
    layer = ", ".join(f"{k}: {v}" for k, v in LAYER.items())
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}", f"config: {{{conf}}}",
             f"inputs: {{B: {B}, x: \"standard Gaussian pixels\"}}", f"layer: {{{layer}}}",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
