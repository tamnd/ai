# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/classifier-free-guidance/ (ai://NEW-classifier-free-guidance).
"""Write the shared fixtures of classifier-free guidance from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are a denoiser for samples of dimension 3 with 5 hidden units, a time
embedding of 4 and 2 classes plus the null class, random and perturbed so that no weight is zero. The input is a batch
of four noisy samples with times, classes (one of them the null class) and noise, a pair of conditional and
unconditional noise predictions of shape (2, 3, 4, 4), a batch of x-predictions for dynamic thresholding, and two
mixture means with starting noise for a guided DDIM run. The expected values are the predicted noise, the loss and its
gradient in every parameter, the guided noise at s = 3, its rescale with phi = 0.7, the dynamic threshold at the 90th
percentile, and a five-step DDIM sample guided at s = 2 in the interval (0.3, 3.0] with its number of evaluations.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2207125980  # the arXiv number of Ho and Salimans, 2207.12598, and a 0


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "cg_ref", HERE.parent / "impl" / "numpy" / "classifier_free_guidance.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    shape = (3, 5, 4, 2)
    cfg = ref.config(*shape)
    params = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in ref.init(rng, cfg).items()}
    z, t, y, eps = (
        rng.standard_normal((4, 3)),
        np.array([3, 250, 600, 990]),
        np.array([0, 1, 2, 1]),
        rng.standard_normal((4, 3)),
    )
    e_c, e_u = rng.standard_normal((2, 2, 3, 4, 4))
    x0 = 1.5 * rng.standard_normal((3, 10))
    means, zT = np.array([[1.0, -0.5], [-1.0, 0.5]]), rng.standard_normal((5, 2))
    abar = ref.alpha_bars(1000)
    w = np.array([0.5, 0.5])
    eps_fn, count = ref.guided_sampler(lambda zz, tt: ref.exact_eps_cond(zz, np.full(len(zz), abar[tt]), means, 0.1, 0),
                                       lambda zz, tt: ref.exact_eps(zz, np.full(len(zz), abar[tt]), means, 0.1, w),
                                       2.0, abar, interval=(0.3, 3.0))  # fmt: skip
    ddim = ref.ddim_sample(eps_fn, zT, abar, [999, 600, 300, 100, 0])
    guided = ref.guide(e_c, e_u, 3.0)
    expected = {"eps_hat": ref.forward(params, cfg, z, t, y), "loss": np.array(ref.eps_loss(params, cfg, z, t, y, eps)),
                "guided": guided, "rescaled": ref.rescale(e_c, guided, 0.7), "dyn": ref.dynamic_threshold(x0, 90.0),
                "ddim": ddim, "nfe": np.array(count["nfe"])}  # fmt: skip
    expected.update({f"d{k}": g for k, g in ref.eps_loss_grad(params, cfg, z, t, y, eps).items()})
    inputs = {"cfg": np.array(shape), "z": z, "t": t, "y": y, "eps": eps, "e_c": e_c, "e_u": e_u, "x0": x0,
              "means": means, "zT": zT}  # fmt: skip
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "denoiser: {dim: 3, hidden: 5, temb: 4, classes: 2}",
             "inputs: {B: 4, schedule: \"linear betas 1e-4 to 2e-2 over 1000 steps\", guidance: \"s = 3 on (2, 3, 4, 4) predictions, rescale phi = 0.7, dynamic threshold at the 90th percentile\", ddim: \"two-component mixture, s2 = 0.1, class 0, s = 2 in the interval (0.3, 3.0], steps 999, 600, 300, 100, 0\"}",
             "loss: \"the noise-prediction loss, the squared error summed over the sample and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
