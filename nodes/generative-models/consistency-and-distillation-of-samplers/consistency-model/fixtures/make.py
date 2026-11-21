# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Write the shared fixtures of the consistency model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are a consistency model for samples of dimension 3 with 5 hidden units
and a time embedding of 4, random and perturbed so that no weight is zero, and a second, perturbed copy as the target
network. The input is a batch of four clean samples with noise and pairs of adjacent levels of the discretization of
N = 18, a two-component mixture for the teacher, a starting batch with noise for a two-step sample, and a pair of
variance-preserving levels for the target of progressive distillation. The expected values are the model's output,
one Heun step of the teacher, the distillation loss with the squared l2 metric and its gradient in every parameter,
the training loss with the Pseudo-Huber metric and the weight 1 / (t_{i+1} - t_i) and its gradient, one EMA update,
the two-step sample and the progressive-distillation target.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2303014690  # the arXiv number of Song, Dhariwal, Chen and Sutskever, 2303.01469, and a 0


def load_reference():
    spec = importlib.util.spec_from_file_location("cm_ref", HERE.parent / "impl" / "numpy" / "consistency_model.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    shape = (3, 5, 4)
    cfg = ref.config(*shape)
    params = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in ref.init(rng, cfg).items()}
    params_minus = {k: v + 0.05 * rng.standard_normal(v.shape) for k, v in params.items()}
    ts = ref.karras_steps(18)
    n = np.array([0, 4, 11, 16])
    x0, z = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
    means, weights = np.array([[1.0, -0.5, 0.0], [-1.0, 0.5, 0.25]]), np.array([0.3, 0.7])
    xT, noise = 80.0 * rng.standard_normal((5, 3)), rng.standard_normal((1, 5, 3))
    lam = np.array([1.5, 0.5, -0.5])
    z_pd = rng.standard_normal((4, 3))

    def score_fn(x, t):
        return ref.score(x, t, means, 0.1, weights)

    t_lo, t_hi = ts[n], ts[n + 1]
    x_hi, x_lo = ref.cd_pair(x0, z, t_lo, t_hi, score_fn)
    cd_loss, cd_grad = ref.consistency_loss(params, params_minus, cfg, x_hi, t_hi, x_lo, t_lo)
    c = 0.00054 * np.sqrt(3.0)
    xc_hi, xc_lo = ref.ct_pair(x0, z, t_lo, t_hi)
    ct_loss, ct_grad = ref.consistency_loss(params, params, cfg, xc_hi, t_hi, xc_lo, t_lo,
                                            ref.ict_weight(t_lo, t_hi), c)  # fmt: skip
    (a_t, s_t), (a_s, s_s), (a_s2, s_s2) = (ref.vp(v) for v in lam)
    x_teacher = np.tanh(z_pd)
    z_s = ref.ddim_step(z_pd, x_teacher, a_t, s_t, a_s, s_s)
    z_s2 = ref.ddim_step(z_s, 0.5 * x_teacher, a_s, s_s, a_s2, s_s2)
    expected = {"f": ref.forward(params, cfg, x_hi, t_hi), "x_lo": x_lo, "cd_loss": np.array(cd_loss),
                "ct_loss": np.array(ct_loss), "ema": ref.ema(params_minus, params, 0.95)["W1"],
                "sample": ref.sample(lambda x, t: ref.forward(params, cfg, x, t), xT, [0.8], noise),
                "pd_target": ref.pd_target(z_pd, z_s2, a_t, s_t, a_s2, s_s2)}  # fmt: skip
    expected.update({f"cd_d{k}": g for k, g in cd_grad.items()})
    expected.update({f"ct_d{k}": g for k, g in ct_grad.items()})
    inputs = {"cfg": np.array(shape), "n": n, "x0": x0, "z": z, "means": means, "weights": weights, "xT": xT,
              "noise": noise, "lam": lam, "z_pd": z_pd}  # fmt: skip
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    files["params-small.npz"] = {**params, **{f"minus_{k}": v for k, v in params_minus.items()}}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "model: {dim: 3, hidden: 5, temb: 4, sigma_data: 0.5, eps: 0.002, T: 80}",
             "inputs: {B: 4, levels: \"karras_steps(18), pairs (t_n, t_n+1) for n = 0, 4, 11, 16\", teacher: \"one Heun step of the ODE of a two-component mixture, s2 = 0.1, weights 0.3 and 0.7\", sample: \"5 samples, one step at T = 80 then tau = 0.8\", pd: \"log-SNR 1.5, 0.5, -0.5, teacher x-predictions tanh(z) and half of it\"}",
             "losses: \"distillation: squared l2 against the target network, lambda = 1; training: Pseudo-Huber with c = 0.00054 sqrt(3), lambda = 1 / (t_n+1 - t_n), the target the student itself\"",
             "target: \"the parameters minus_* are the target network theta-; ema is W1 after one update with mu = 0.95\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
