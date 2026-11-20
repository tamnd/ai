# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Write the shared fixtures of the linear-attention page from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The layer is the default config of impl/numpy/linear_attention.py, 2 heads of width 4
with the elu + 1 feature map, run causal and non-causal on the same input; the loss is half the mean squared error
against a random target, and the gradients are those of the reference backward. out_softmax_causal is the same
projections with causal softmax attention, for comparison.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20200629  # the day Transformers are RNNs appeared on arXiv
B, N = 2, 6


def load_reference():
    spec = importlib.util.spec_from_file_location("linear_ref", HERE.parent / "impl" / "numpy" / "linear_attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config()
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    x = rng.normal(size=(B, N, cfg["d_model"]))
    target = rng.normal(size=(B, N, cfg["d_model"]))
    expected = {}
    for causal in (True, False):
        c = ref.config(causal=causal)
        tag = "causal" if causal else "full"
        out, cache = ref.forward(params, c, x)
        grads, dx = ref.backward(params, c, cache, (out - target) / out.size)
        expected |= {"out_" + tag: out, "loss_" + tag: np.array(ref.loss(out, target)), "dx_" + tag: dx}
        expected |= {f"d{k}_{tag}": v for k, v in grads.items()}
    H = cfg["n_heads"]
    q, k, v = (ref.split(x @ params[w], H) for w in ("W_q", "W_k", "W_v"))
    expected["out_softmax_causal"] = ref.merge(ref.softmax_attention(q, k, v, causal=True)) @ params["W_o"]
    files = {
        "params-small.npz": params,
        "input-small.npz": {"x": x, "target": target},
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
        "config: {" + ", ".join(f"{k}: {str(v).lower()}" for k, v in cfg.items() if k != "causal") + "}",
        f"inputs: {{B: {B}, N: {N}, loss: half the mean squared error, causal: [true, false]}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
