# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""Write the shared fixtures of the diffusion model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are those of the noise-prediction network of the example config, all of
them random. The input is a batch x0 of data, integer steps t, the noise eps that makes x_t from x0, and the noise z of
a sampling step. The expected values are the time embedding emb, x_t, the prediction eps_pred at x_t, loss and
d<param> of the simple loss, post_mean and post_var of q(x_{t-1} | x_t, x_0), mu of the reverse step, x_prev of the
ancestral step with sigma_t^2 = beta_t, and x_ddim of a step from t to t // 2 with eta = 0.5.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20200619  # the day Ho, Jain and Abbeel put DDPM on arXiv
B = 4  # the example config of shapes.yaml: d_data 3, d_hidden 5, d_time 4, n_steps 10


def load_reference():
    spec = importlib.util.spec_from_file_location("diffusion_ref", HERE.parent / "impl" / "numpy" / "diffusion.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    cfg = ref.config()
    D = cfg["d_data"]
    params = {k: rng.normal(0.0, 0.5, s) for k, s in ref.param_shapes(cfg).items()}
    x0 = rng.standard_normal((B, D))
    t = np.array([1, 4, 7, 10])
    eps = rng.standard_normal((B, D))
    z = rng.standard_normal((B, D))
    xt = ref.q_sample(cfg, x0, t, eps)
    out, _ = ref.eps_model(params, cfg, xt, t)
    loss, grads = ref.loss_and_grads(params, cfg, x0, t, eps)
    pm, pv = ref.posterior(cfg, xt, x0, t)
    expected = {"emb": ref.time_embedding(t, cfg["d_time"]), "xt": xt, "eps_pred": out, "loss": np.array(loss),
                "post_mean": pm, "post_var": pv, "mu": ref.reverse_mean(cfg, xt, t, out),
                "x_prev": ref.ddpm_step(cfg, xt, t, out, z), "x_ddim": ref.ddim_step(cfg, xt, t, t // 2, out, z, 0.5)}  # fmt: skip
    expected.update({f"d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x0": x0, "t": t, "eps": eps, "z": z},
             "expected-small.npz": expected}  # fmt: skip
    hashes = {}
    for name, arrays in files.items():
        target = HERE / name
        with open(target, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_data: 3, d_hidden: 5, d_time: 4, n_steps: 10, beta_1: 0.02, beta_T: 0.5}",
             f"inputs: {{B: {B}, x0: \"standard normal\", t: \"[1, 4, 7, 10]\", eps: \"standard normal\", z: \"standard normal\"}}",
             "loss: \"the simple loss, the squared error of the noise prediction summed over coordinates and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
