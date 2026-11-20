# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Write the shared fixtures of the energy-based model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are an RBM with D = 4 visible and H = 3 hidden units (rbm_*) and a
network energy on R^2 with 5 hidden units (mlp_*), all random. The input is a batch v of binary vectors with hidden
states h, the uniforms u_h and u_v of one Gibbs step, points x with standard normal draws z, a step size and the
weight gamma of the quadratic term, the NCE parameters theta with data nce_x and noise nce_y, and classifier logits.
The expected values are the RBM's free energy, energy, p(h | v), the Gibbs step h1 and v1, the CD estimate cd_* with
v1 as the negative samples, the exact gradient ml_* and log Z; the network's energy, input gradient, mean parameter
gradient mlp_d*, score matching objective and Langevin step; the NCE objective and gradient; and the classifier energy.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 1903086890  # the arXiv number of Du and Mordatch, 1903.08689, and a 0


def load_reference():
    spec = importlib.util.spec_from_file_location("eb_ref", HERE.parent / "impl" / "numpy" / "energy_based.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    rbm = {k: rng.normal(0.0, 1.0, s) for k, s in ref.param_shapes(ref.config(4, 3)).items()}
    mlp = {k: rng.normal(0.0, 1.0, s) for k, s in ref.mlp_shapes(2, 5).items()}
    v = ref.all_binary(4)[[0, 5, 10, 15, 6]]
    h = ref.all_binary(3)[[1, 2, 4, 7, 3]]
    u_h, u_v = rng.random((5, 3)), rng.random((5, 4))
    x, z = rng.normal(0.0, 1.5, (6, 2)), rng.standard_normal((6, 2))
    step, gamma = 0.05, 0.1
    theta = np.array([1.3, 0.4, -1.2])
    nce_x, nce_y = rng.normal(1.0, 0.8, 7), rng.normal(0.0, 2.0, 7)
    logits = rng.normal(0.0, 2.0, (3, 5))
    h1, v1 = ref.gibbs_step(rbm, v, u_h, u_v)
    expected = {"free_energy": ref.free_energy(rbm, v), "energy": ref.energy(rbm, v, h), "p_h": ref.p_h(rbm, v),
                "h1": h1, "v1": v1, "log_z": np.array(ref.log_partition(rbm)),
                "mlp_energy": ref.mlp_energy(mlp, x, gamma), "mlp_grad_x": ref.mlp_grad_x(mlp, x, gamma),
                "mlp_sm": np.array(ref.mlp_sm_objective(mlp, x, gamma)),
                "langevin": ref.langevin_step(mlp, x, step, z, None, gamma),
                "nce_j": np.array(ref.nce_objective(theta, nce_x, nce_y, 0.0, 2.0)),
                "nce_grad": ref.nce_grad(theta, nce_x, nce_y, 0.0, 2.0), "jem_energy": ref.jem_energy(logits)}  # fmt: skip
    expected.update({f"cd_{k}": g for k, g in ref.grad_from_samples(rbm, v, v1).items()})
    expected.update({f"ml_{k}": g for k, g in ref.exact_grad(rbm, v).items()})
    expected.update({f"mlp_d{k}": g for k, g in ref.mlp_grad_params(mlp, x).items()})
    params = {f"rbm_{k}": a for k, a in rbm.items()}
    params.update({f"mlp_{k}": a for k, a in mlp.items()})
    inputs = {"v": v, "h": h, "u_h": u_h, "u_v": u_v, "x": x, "z": z, "step": np.array(step), "gamma": np.array(gamma),
              "theta": theta, "nce_x": nce_x, "nce_y": nce_y, "logits": logits}  # fmt: skip
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "rbm: {d_visible: 4, d_hidden: 3}", "mlp: {d: 2, H: 5, gamma: 0.1}",
             "inputs: {B: 5, v: \"binary vectors\", x: \"6 points of R^2\", step: 0.05, noise: \"N(0, 2^2) for NCE\"}",
             "loss: \"the gradients are of the mean log-likelihood of v (rbm) and of the mean energy of x (mlp)\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
