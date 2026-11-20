# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Write the shared fixtures of the hypernetwork page from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. Three networks, their arrays under the prefixes static., rnn. and lstm.: the example
config of shapes.yaml with a static hypernetwork (two generated convolutions of 4 channels, each tiled from 2 x 2
basic kernels), on two images with labels, its logits, loss and every gradient; the HyperRNN of eq-scaled with
N_x = 3, N_h = 5, N_hat = 4 and N_z = 2 on two sequences of 4 steps, its states, the loss half the mean squared error
against a random target, and every gradient; and the HyperLSTM with random parameters on the same sequences, its
states with and without layer normalization.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20160927  # the day HyperNetworks appeared on arXiv
B, T = 2, 4


def load_reference():
    spec = importlib.util.spec_from_file_location("hyper_ref", HERE.parent / "impl" / "numpy" / "hypernetwork.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    cfg = ref.config()
    sp = ref.init(rng, cfg)
    x = rng.normal(size=(B, cfg["c_in"], cfg["size"], cfg["size"]))
    y = rng.integers(0, cfg["classes"], size=B)
    logits, cache = ref.forward(sp, x, cfg)
    sg, sdx = ref.backward(sp, cfg, cache, ref.loss_grad(logits, y))
    rc = ref.config_rnn()
    rp = ref.init_rnn(rng, rc)
    seq = rng.normal(size=(B, T, rc["n_x"]))
    target = rng.normal(size=(B, T, rc["n_h"]))
    out, steps = ref.hyper_rnn(rp, seq, return_cache=True)
    rg, rdx = ref.hyper_rnn_backward(rp, seq, steps, (out - target) / out.size)
    lp = {k: rng.normal(0.0, 0.5, s) for k, s in ref.lstm_param_shapes(rc).items()}
    params = {f"static.{k}": v for k, v in sp.items()} | {f"rnn.{k}": v for k, v in rp.items()}
    params |= {f"lstm.{k}": v for k, v in lp.items()}
    expected = {"static.logits": logits, "static.loss": np.array(ref.loss(logits, y)), "static.dx": sdx}
    expected |= {f"static.d{k}": v for k, v in sg.items()}
    expected |= {"rnn.out": out, "rnn.loss": np.array(ref.rnn_loss(out, target)), "rnn.dx": rdx}
    expected |= {f"rnn.d{k}": v for k, v in rg.items()}
    expected |= {"lstm.out_ln": ref.hyper_lstm(lp, seq, ln=True), "lstm.out": ref.hyper_lstm(lp, seq, ln=False)}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"static.x": x, "static.y": y, "rnn.x": seq, "rnn.target": target},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        out_path = HERE / name
        with open(out_path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {static: {" + ", ".join(f"{k}: {v}" for k, v in cfg.items()) + "}, rnn: {"
        + ", ".join(f"{k}: {v}" for k, v in rc.items()) + "}}",
        f"inputs: {{B: {B}, T: {T}, static_loss: mean cross-entropy, rnn_loss: half the mean squared error}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
