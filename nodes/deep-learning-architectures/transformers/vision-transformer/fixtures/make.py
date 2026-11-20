# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Write the shared fixtures of the Vision Transformer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml, 8 x 8 images of 3 channels in
4 x 4 patches of 2 x 2, width 8 with 2 heads, an MLP of 16, 2 layers and 5 classes, with a class token, learned
position embeddings and a linear head, on a batch of two random images. The gradients are those of eq-loss from the
backward pass of the reference, the gradient with respect to the images included.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 202010  # the month in which An Image is Worth 16x16 Words was posted to arXiv, October 2020
CONFIG = {"image_size": 8, "patch_size": 2, "channels": 3, "d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 2,
          "n_classes": 5}  # fmt: skip
B = 2
SCALE = 0.5  # wider than a usual initialization, so that the outputs are far from uniform


def load_reference():
    spec = importlib.util.spec_from_file_location("vit_ref", HERE.parent / "impl" / "numpy" / "vit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    S, C = CONFIG["image_size"], CONFIG["channels"]
    images = rng.normal(size=(B, S, S, C))
    labels = rng.integers(0, CONFIG["n_classes"], B)
    logits, cache = ref.forward(params, cfg, images)
    grads, dimages = ref.backward(params, cfg, cache, labels)
    expected = {
        "logits": logits,
        "loss": np.array(ref.nll(logits, labels)),
        "tokens": cache["tokens"],
        "final": cache["final"],
        "dimages": dimages,
    }
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"images": images, "labels": labels},
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
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", pool: cls, pos: learned}",
        f"inputs: {{B: {B}, init_scale: {SCALE}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
