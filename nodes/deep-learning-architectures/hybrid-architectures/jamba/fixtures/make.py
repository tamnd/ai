# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Write the shared fixtures of Jamba from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml: a vocabulary of 13, width 8, one
Jamba block of 4 layers (Mamba, Mamba with MoE, attention, Mamba with MoE), 2 query heads of size 4 sharing one
key-value head, 4 experts of width 12 with the top 2 for each token, and Mamba layers with 16 channels, a state of 4,
a convolution of width 3 and a Delta projection of rank 2. The input is two random sequences of 7 tokens; the model
reads the first 6 and is scored on the last 6. The weights are N(0, 0.5^2), with A_log, D and the RMSNorm weights
moved off their initial values by N(0, 0.2^2), so that no gradient is special. The expected values are the scores,
the cross-entropy, the load-balancing term, the loss, the experts chosen in each MoE layer and the gradients of the
loss from the backward pass of the reference.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 202403  # the month in which Jamba was posted to arXiv, March 2024
B, T = 2, 6
SCALE, JITTER = 0.5, 0.2


def load_reference():
    spec = importlib.util.spec_from_file_location("jamba_ref", HERE.parent / "impl" / "numpy" / "jamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config()
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    for k in sorted(params):
        if k.endswith(("A_log", "_D", "norm", "norm1", "norm2", "norm_f")):
            params[k] = params[k] + rng.normal(0.0, JITTER, params[k].shape)
    tokens = rng.integers(0, cfg["vocab"], (B, T + 1))
    logits, aux, cache = ref.forward(params, cfg, tokens[:, :-1])
    grads = ref.backward(params, cfg, cache, logits, tokens[:, 1:])
    ce = ref.nll(logits, tokens[:, 1:])
    expected = {"logits": logits, "nll": np.array(ce), "aux": np.array(aux), "loss": np.array(ce + aux)}
    for i, (_, f) in enumerate(ref.layers(cfg)):
        if f == "moe":
            probs = cache["layers"][i]["ffn"]["probs"]
            expected[f"l{i}_top"] = np.argsort(-probs, axis=-1, kind="stable")[:, : cfg["top_k"]]
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {"params-small.npz": params, "input-small.npz": {"tokens": tokens}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    keys = ("vocab", "d_model", "n_blocks", "block_len", "moe_every", "n_experts", "top_k", "d_ff", "n_heads", "n_kv",
            "d_state", "d_conv", "expand", "dt_rank", "aux", "eps")  # fmt: skip
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {cfg[k]}" for k in keys) + "}",
        f"inputs: {{B: {B}, T: {T}, init_scale: {SCALE}, jitter: {JITTER}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
