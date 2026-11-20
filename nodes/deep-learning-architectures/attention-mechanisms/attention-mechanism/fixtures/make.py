# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Write the shared fixtures of the attention page from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The model is RNNsearch at the default config of impl/numpy/attention.py; the second
source sentence has 3 real words of 5 and the second target 3 of 4, so the masks are exercised. logits_encdec is the
same parameters with c_i fixed (RNNencdec). The gradients are those of eq-loss from the backward pass of the reference.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20140901  # the day Neural Machine Translation by Jointly Learning to Align and Translate appeared on arXiv
B, S, T = 2, 5, 4
SCALE = 0.5  # wider than the 0.01 of the paper, so that the weights are far from uniform


def load_reference():
    spec = importlib.util.spec_from_file_location("attention_ref", HERE.parent / "impl" / "numpy" / "attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config()
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    src = rng.integers(1, cfg["src_vocab"], (B, S))
    tgt = rng.integers(1, cfg["tgt_vocab"], (B, T))
    src_mask, tgt_mask = np.ones((B, S), bool), np.ones((B, T), bool)
    src_mask[1, 3:], tgt_mask[1, 3:] = False, False
    src[1, 3:], tgt[1, 3:] = 0, 0
    tgt[0, -1], tgt[1, 2] = 0, 0  # <EOS>
    logits, alpha, cache = ref.forward(params, cfg, src, tgt, src_mask)
    grads = ref.backward(params, cfg, cache, tgt_mask)
    expected = {
        "logits": logits,
        "alpha": alpha,
        "loss": np.array(ref.nll(logits, tgt, tgt_mask)),
        "annotations": cache["H"],
        "logits_encdec": ref.forward(params, ref.config(attention=False), src, tgt, src_mask)[0],
    }
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"src": src, "tgt": tgt, "src_mask": src_mask, "tgt_mask": tgt_mask},
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
        "config: {" + ", ".join(f"{k}: {str(v).lower()}" for k, v in cfg.items()) + "}",
        f'inputs: {{B: {B}, S: {S}, T: {T}, init_scale: {SCALE}, padded: "source 2 has 3 real words of 5, target 2 has 3 of 4"}}',
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
