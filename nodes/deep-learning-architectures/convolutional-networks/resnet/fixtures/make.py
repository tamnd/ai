# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Write the shared fixtures of ResNet from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, state-small.npz, input-small.npz,
expected-small.npz and manifest.yaml next to itself. The trees are stored flat, their paths joined by "/".
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20151210  # the day the paper appeared on arXiv
CONFIG = {"width": 2, "blocks": (1, 1, 1, 1), "classes": 5}  # the example config of shapes.yaml
B, R = 2, 32


def load_reference():
    spec = importlib.util.spec_from_file_location("resnet_ref", HERE.parent / "impl" / "numpy" / "resnet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flatten(tree, prefix=""):
    out = {}
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for k, v in items:
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(flatten(v, path))
        else:
            out[path] = np.asarray(v)
    return out


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    state = ref.init_state(params)
    # The norms start at 1 and 0 in init; random values here make the fixture check that every one is used.
    flat, flat_state = flatten(params), flatten(state)
    for k, v in flat.items():
        if k.endswith("/gamma"):
            v[...] = 1.0 + rng.normal(0.0, 0.1, v.shape)
        elif k.endswith("/beta") or k == "fc/b":
            v[...] = rng.normal(0.0, 0.1, v.shape)
    for k, v in flat_state.items():
        v[...] = rng.normal(0.0, 0.1, v.shape) if k.endswith("/mean") else rng.uniform(0.5, 2.0, v.shape)
    x = rng.normal(0.0, 1.0, (B, 3, R, R))
    train, cache = ref.forward(params, x, cfg)
    infer, cache_inf = ref.forward(params, x, cfg, state)
    expected = {"logits_train": train, "logits_infer": infer, "stem_train": cache["stem"]["y"],
                "s2_first_c_train": cache["blocks"][(1, 0)]["c"], "s2_first_c_infer": cache_inf["blocks"][(1, 0)]["c"]}
    files = {"params-small.npz": flat, "state-small.npz": flat_state, "input-small.npz": {"x": x},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             f"config: {{width: {CONFIG['width']}, blocks: [1, 1, 1, 1], classes: {CONFIG['classes']}, stride_on: 1x1}}",
             f"inputs: {{B: {B}, R: {R}, x: \"standard Gaussian pixels\"}}",
             "modes: {train: \"batch statistics\", infer: \"the running statistics of state-small.npz\"}",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
