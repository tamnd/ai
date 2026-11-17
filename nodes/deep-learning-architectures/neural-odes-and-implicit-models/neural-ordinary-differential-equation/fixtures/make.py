# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""Write the shared fixtures of the neural ODE from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The gradients are those of the discrete RK4 map (backprop), which the automatic
differentiation of the PyTorch and JAX references reproduces to rounding.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20180619  # the day Chen, Rubanova, Bettencourt and Duvenaud posted the paper to arXiv
CONFIG = {"d": 2, "d_hidden": 4, "d_state": 3, "n_steps": 3}  # the example config of shapes.yaml
B = 5


def load_reference():
    spec = importlib.util.spec_from_file_location("node_ref", HERE.parent / "impl" / "numpy" / "neural_ode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    params["b1"] = rng.normal(0.0, 0.1, params["b1"].shape)  # nonzero biases, so the fixture checks them
    params["b2"] = rng.normal(0.0, 0.1, params["b2"].shape)
    x = rng.normal(0.0, 1.0, (B, CONFIG["d"]))
    t = rng.normal(0.0, 1.0, (B, CONFIG["d"]))
    y = ref.forward(params, x, cfg)
    loss, dyd = ref.mse_loss(y[:, :CONFIG["d"]], t)
    dy = np.zeros_like(y)
    dy[:, :CONFIG["d"]] = dyd
    dx, grads = ref.backprop(params, x, dy, cfg)
    expected = {"y": y, "loss": np.array(loss), "dx": dx}
    expected.update({f"d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "t": t}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d: 2, d_hidden: 4, d_state: 3, n_steps: 3, T: 1.0, method: rk4}",
             f"inputs: {{B: {B}, x: \"standard Gaussian\", t: \"standard Gaussian targets for the first d coordinates of h(T)\"}}",
             "loss: \"half the mean squared error of the first d coordinates of h(T)\"",
             "gradients: \"of the discrete RK4 map, by backprop\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
