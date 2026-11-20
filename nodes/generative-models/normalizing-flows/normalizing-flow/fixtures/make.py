# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/normalizing-flow/ (ai://NEW-normalizing-flow).
"""Write the shared fixtures of the normalizing flow from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are those of the coupling flow of the example config, all of them random
so that no layer is the identity, and the planar and autoregressive layers' parameters, prefixed planar_ and ar_. The
input is a batch x of data and a batch u of base noise. The expected values are z, logdet and log_prob of the coupling
flow at x, x_inv = f^-1(u), loss and d<param> and dx of the mean negative log-likelihood, and the planar layer's output
planar_f and planar_logdet at x with u_hat in place of u, and the autoregressive layer's ar_u and ar_logdet at x and
ar_x = its sample from u.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20141030  # the day Dinh, Krueger and Bengio put NICE on arXiv
B = 4  # the example config of shapes.yaml: d_data 3, d_hidden 5, n_layers 3


def load_reference():
    spec = importlib.util.spec_from_file_location("flow_ref", HERE.parent / "impl" / "numpy" / "flow.py")
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
    planar = {"w": rng.standard_normal(D), "u": rng.standard_normal(D), "b": rng.normal(0.0, 0.5, (1,))}
    ar = {"Wm": rng.normal(0.0, 0.5, (D, D)), "bm": rng.normal(0.0, 0.5, D),
          "Wa": rng.normal(0.0, 0.3, (D, D)), "ba": rng.normal(0.0, 0.3, D)}  # fmt: skip
    params.update({f"planar_{k}": v for k, v in planar.items()})
    params.update({f"ar_{k}": v for k, v in ar.items()})
    x = rng.standard_normal((B, D))
    u = rng.standard_normal((B, D))
    z, ld, _ = ref.forward(params, cfg, x)
    loss, grads, gx = ref.nll_and_grads(params, cfg, x)
    pf, pld = ref.planar(x, planar["w"], ref.u_hat(planar["w"], planar["u"]), planar["b"][0])
    au, ald = ref.ar_density(ar, x)
    expected = {"z": z, "logdet": ld, "log_prob": ref.log_prob(params, cfg, x), "x_inv": ref.inverse(params, cfg, u),
                "loss": np.array(loss), "dx": gx, "planar_f": pf, "planar_logdet": pld, "ar_u": au,
                "ar_logdet": ald, "ar_x": ref.ar_sample(ar, u)}  # fmt: skip
    expected.update({f"d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "u": u}, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        target = HERE / name
        with open(target, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_data: 3, d_hidden: 5, n_layers: 3}",
             f"inputs: {{B: {B}, x: \"standard normal\", u: \"standard normal\"}}",
             "loss: \"the negative log-likelihood in nats, averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
