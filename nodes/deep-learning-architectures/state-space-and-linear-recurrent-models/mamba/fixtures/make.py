# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Write the shared fixtures of Mamba from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml: a vocabulary of 11, width 8, two
Mamba blocks with E = 2 (so 16 channels), a state of 4 per channel, a convolution of width 3 and a Delta projection
of rank 1, with the discretization of the reference code (A_bar = exp(Delta A), B_bar = Delta B). The input is two
random sequences of 9 tokens; the model reads the first 8 and is scored on the last 8. The weights are N(0, 0.5^2),
with the structured parameters (A_log, D and the RMSNorm weights) moved off their initial values by N(0, 0.2^2), so
that no gradient is special. The expected values are the scores, the loss, the SSM state of the last block after the
last position and the gradients of the loss from the backward pass of the reference.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 202312  # the month in which Mamba was posted to arXiv, December 2023
CONFIG = {"vocab": 11, "d_model": 8, "n_layers": 2, "d_state": 4, "d_conv": 3, "expand": 2, "disc": "simple"}
B, T = 2, 8
SCALE, JITTER = 0.5, 0.2


def load_reference():
    spec = importlib.util.spec_from_file_location("mamba_ref", HERE.parent / "impl" / "numpy" / "mamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    for k in sorted(params):
        if k.endswith(("A_log", "_D", "norm", "norm_f")):
            params[k] = params[k] + rng.normal(0.0, JITTER, params[k].shape)
    tokens = rng.integers(0, CONFIG["vocab"], (B, T + 1))
    logits, cache = ref.forward(params, cfg, tokens[:, :-1])
    grads = ref.backward(params, cfg, cache, logits, tokens[:, 1:])
    expected = {
        "logits": logits,
        "loss": np.array(ref.nll(logits, tokens[:, 1:])),
        "last_state": cache["layers"][-1]["mixer"]["hs"][:, -1],
    }
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"tokens": tokens},
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
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + f", dt_rank: {cfg['dt_rank']}}}",
        f"inputs: {{B: {B}, T: {T}, init_scale: {SCALE}, jitter: {JITTER}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
