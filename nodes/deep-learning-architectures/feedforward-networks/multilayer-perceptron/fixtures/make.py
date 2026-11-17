# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://NEW-multilayer-perceptron).
"""Write the shared fixtures of the multilayer perceptron from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The weights are stored flat as W/0, W/1, ... and b/0, b/1, ...
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 19861009  # the day Rumelhart, Hinton and Williams appeared in Nature
CONFIG = {"d_in": 3, "d_hidden": 4, "d_out": 2, "n_hidden": 2, "activation": "tanh"}  # the example config of shapes.yaml
B = 5


def load_reference():
    spec = importlib.util.spec_from_file_location("mlp_ref", HERE.parent / "impl" / "numpy" / "multilayer_perceptron.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(tree):
    return {f"{k}/{i}": np.asarray(v) for k in ("W", "b") for i, v in enumerate(tree[k])}


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, "glorot")
    params["b"] = [rng.normal(0.0, 0.1, b.shape) for b in params["b"]]  # nonzero biases, so the fixture checks them
    x = rng.normal(0.0, 1.0, (B, CONFIG["d_in"]))
    t = rng.normal(0.0, 1.0, (B, CONFIG["d_out"]))
    y, cache = ref.forward(params, x, cfg)
    loss, dy = ref.mse_loss(y, t)
    dx, grads = ref.backward(params, cache, dy)
    expected = {"y": y, "loss": np.array(loss), "dx": dx}
    expected.update({f"d{k}": v for k, v in flat(grads).items()})
    files = {"params-small.npz": flat(params), "input-small.npz": {"x": x, "t": t}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_in: 3, d_hidden: 4, d_out: 2, n_hidden: 2, activation: tanh}",
             f"inputs: {{B: {B}, x: \"standard Gaussian\", t: \"standard Gaussian targets\"}}",
             "loss: \"eq-loss, half the mean squared error\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
