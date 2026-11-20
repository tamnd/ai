# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Write the shared fixtures of the variational autoencoder from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are those of the Gaussian decoder; the Bernoulli decoder uses all of them
but olv_W and olv_b. The noise eps is part of the input, so every implementation computes the same codes. The expected
values are named <decoder>/<objective>/<name>: the loss and the gradient d<param> of each objective, and the forward
values once per decoder.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20131220  # the day Kingma and Welling put the paper on arXiv
B, S = 4, 3  # the example config of shapes.yaml: d_in 6, d_hidden 5, d_latent 2


def load_reference():
    spec = importlib.util.spec_from_file_location("vae_ref", HERE.parent / "impl" / "numpy" / "vae.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, ref.config(decoder="gaussian"))
    for k in params:
        if params[k].ndim == 1:
            params[k] = rng.normal(0.0, 0.3, params[k].shape)  # nonzero biases, so the fixture checks them
    x = (rng.uniform(0.0, 1.0, (B, 6)) > 0.5).astype(np.float64)
    eps = rng.standard_normal((S, B, 2))
    expected = {}
    for dec in ref.DECODERS:
        cfg = ref.config(decoder=dec)
        p = {k: v for k, v in params.items() if k in ref.param_shapes(cfg)}
        out, _ = ref.forward(p, cfg, x, eps)
        for k in ("mu", "lv", "z", "a") + (("olv",) if dec == "gaussian" else ()):
            expected[f"{dec}/{k}"] = out[k]
        expected[f"{dec}/kl"] = ref.kl(out["mu"], out["lv"])
        expected[f"{dec}/nll"] = ref.nll(cfg, x, out)
        expected[f"{dec}/log_w"] = ref.log_weights(cfg, x, out, eps)
        for kind in ref.OBJECTIVES:
            value, grads = ref.loss(p, cfg, x, eps, kind)
            expected[f"{dec}/{kind}/loss"] = np.array(value)
            expected.update({f"{dec}/{kind}/d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "eps": eps}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        target = HERE / name
        with open(target, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_in: 6, d_hidden: 5, d_latent: 2, decoder: [bernoulli, gaussian]}",
             f"inputs: {{B: {B}, S: {S}, x: \"binary, each pixel 1 with probability 1/2\", eps: \"standard normal\"}}",
             "loss: \"elbo, elbo_mc and iwae of eq-elbo, eq-sgvb-a and eq-iwae, negated and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
