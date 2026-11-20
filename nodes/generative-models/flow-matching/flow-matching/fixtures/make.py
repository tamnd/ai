# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Write the shared fixtures of flow matching from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are those of the velocity network of the example config, all of them
random. The input is a batch x0 of noise, a batch x1 of data, times t in (0, 1) and sigma_min. The expected values are
the time features feat, the point x_t of the path, the target u, the velocity v at x_t, its divergence div, loss and
d<param> of the conditional flow matching loss, x_euler after four Euler steps from x0 to t = 1 and x_midpoint after
two midpoint steps.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20221006  # the day Lipman, Chen, Ben-Hamu, Nickel and Le put Flow Matching on arXiv
B = 4  # the example config of shapes.yaml: d_data 3, d_hidden 5, d_time 4
SIGMA_MIN = 0.1


def load_reference():
    spec = importlib.util.spec_from_file_location("fm_ref", HERE.parent / "impl" / "numpy" / "flow_matching.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    cfg = ref.config(sigma_min=SIGMA_MIN)
    D = cfg["d_data"]
    params = {k: rng.normal(0.0, 0.5, s) for k, s in ref.param_shapes(cfg).items()}
    x0 = rng.standard_normal((B, D))
    x1 = rng.standard_normal((B, D)) + 1.0
    t = np.array([0.1, 0.35, 0.6, 0.9])
    xt = ref.path(cfg, x0, x1, t)
    v, _ = ref.velocity(params, cfg, xt, t)
    loss, grads = ref.loss_and_grads(params, cfg, x0, x1, t)

    def v_fn(x, s):
        return ref.velocity(params, cfg, x, s)[0]

    expected = {"feat": ref.time_features(t, cfg["d_time"]), "xt": xt, "u": ref.target(cfg, x0, x1), "v": v,
                "div": ref.divergence(params, cfg, xt, t), "loss": np.array(loss),
                "x_euler": ref.euler_sample(v_fn, x0, 4), "x_midpoint": ref.midpoint_sample(v_fn, x0, 2)}  # fmt: skip
    expected.update({f"d{k}": g for k, g in grads.items()})
    files = {"params-small.npz": params,
             "input-small.npz": {"x0": x0, "x1": x1, "t": t, "sigma_min": np.array(SIGMA_MIN)},
             "expected-small.npz": expected}  # fmt: skip
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             f"config: {{d_data: 3, d_hidden: 5, d_time: 4, sigma_min: {SIGMA_MIN}}}",
             f"inputs: {{B: {B}, x0: \"standard normal\", x1: \"standard normal plus 1\", t: \"[0.1, 0.35, 0.6, 0.9]\"}}",
             "loss: \"the conditional flow matching loss, the squared error of the velocity summed over coordinates and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
