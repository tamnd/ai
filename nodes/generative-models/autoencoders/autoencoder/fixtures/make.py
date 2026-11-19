# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://NEW-autoencoder).
"""Write the shared fixtures of the autoencoder from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are stored flat as enc_W/0, enc_b/0, ..., dec_W/0, dec_b/0, ...
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20060728  # the day Hinton and Salakhutdinov's paper appeared in Science
B = 4  # the example config of shapes.yaml: d_in 6, d_hidden 5, n_hidden 1, d_code 2


def load_reference():
    spec = importlib.util.spec_from_file_location("ae_ref", HERE.parent / "impl" / "numpy" / "autoencoder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(tree):
    return {f"{k}/{i}": np.asarray(v) for k in sorted(tree) for i, v in enumerate(tree[k])}


def main():
    ref = load_reference()
    cfg = ref.config()
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    for k in ("enc_b", "dec_b"):
        params[k] = [rng.normal(0.0, 0.3, b.shape) for b in params[k]]  # nonzero biases, so the fixture checks them
    x = rng.uniform(0.0, 1.0, (B, cfg["widths"][0]))
    out, cache = ref.forward(params, cfg, x)
    value, da = ref.loss(cfg, x, out["a"])
    grads, dx = ref.backward(params, cfg, cache, da)
    expected = {"y": out["y"], "a": out["a"], "z": out["z"], "loss": np.array(value), "dx": dx}
    expected.update({f"d{k}": v for k, v in flat(grads).items()})
    files = {"params-small.npz": flat(params), "input-small.npz": {"x": x}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        target = HERE / name
        with open(target, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_in: 6, d_hidden: 5, n_hidden: 1, d_code: 2, act: sigmoid, code_act: linear, out: sigmoid, tied: false}",
             f"inputs: {{B: {B}, x: \"uniform on [0, 1]\"}}",
             "loss: \"eq-cross-entropy, summed over the components and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
