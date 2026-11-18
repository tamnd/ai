# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-turing-machine/ (ai://NEW-neural-turing-machine).
"""Write the shared fixtures of the Neural Turing Machine from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The input is a small copy task: random bits, then a delimiter, then blanks while
the model should repeat the bits; the loss counts the blank steps only.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20141020  # the day Graves, Wayne and Danihelka posted the paper to arXiv
CONFIG = {"x_dim": 4, "y_dim": 3, "hidden": 8, "n_loc": 6, "width": 4, "n_read": 1, "n_write": 1, "shift": 1}
B, ITEMS = 2, 3  # T = 2 * ITEMS + 1 = 7 steps


def load_reference():
    spec = importlib.util.spec_from_file_location("ntm_ref", HERE.parent / "impl" / "numpy" / "ntm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_task(rng, batch, items, bits):
    """x has a delimiter channel last; the targets are the bits, shown again after the delimiter."""
    t = 2 * items + 1
    seq = rng.integers(0, 2, (batch, items, bits)).astype(float)
    x = np.zeros((batch, t, bits + 1))
    x[:, :items, :bits] = seq
    x[:, items, bits] = 1.0
    targets = np.zeros((batch, t, bits))
    targets[:, items + 1:] = seq
    mask = np.zeros((batch, t))
    mask[:, items + 1:] = 1.0
    return x, targets, mask


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=2.0)
    for b in ("bc", "bh", "by"):
        params[b] = rng.normal(0.0, 0.5, params[b].shape)  # nonzero biases, so the fixture checks them
    x, targets, mask = copy_task(rng, B, ITEMS, CONFIG["y_dim"])
    logits, weights, memory = ref.forward(params, x, cfg, trace=True)
    loss = ref.bits_per_sequence(logits, targets, mask)
    expected = {"logits": logits, "weights": weights, "memory": memory, "loss": np.array(loss)}
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "targets": targets, "mask": mask},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    cfg_text = ", ".join(f"{k}: {v}" for k, v in CONFIG.items())
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             f"config: {{{cfg_text}, controller: feedforward, memory_init: {ref.MEMORY_INIT}}}",
             f"inputs: {{B: {B}, T: {2 * ITEMS + 1}, x: \"a copy task of {ITEMS} random 3-bit items, a delimiter, then blanks\"}}",
             "expected: \"logits (B, T, Y), the weightings of every head at every step (T, R + W, B, N), the last memory (B, N, M)\"",
             "loss: \"bits per sequence of the sigmoid outputs over the steps after the delimiter, mean over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
