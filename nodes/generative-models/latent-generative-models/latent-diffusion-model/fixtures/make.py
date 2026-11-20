# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""Write the shared fixtures of the latent diffusion model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are a denoiser for latents of 2 positions by 2 channels with 5 hidden
units, a time embedding of 4, 2 context tokens of width 3 per class and 2 classes plus the null class, random and
perturbed so that no weight is zero. The input is a batch of four noisy latents with times, classes and noise, six
signals of length 8 for a principal component autoencoder with f = 4 and c = 2, encoder outputs with a codebook and
a decoder gradient for vector quantization, and two mixture means with starting noise for DDIM. The expected values
are the predicted noise, the loss and its gradient in every parameter, the latents (up to the sign of each principal
component), the code indices, the straight-through gradients, one EMA step and a four-step DDIM sample.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2112107520  # the arXiv number of Rombach and colleagues, 2112.10752, and a 0


def load_reference():
    spec = importlib.util.spec_from_file_location("ld_ref", HERE.parent / "impl" / "numpy" / "latent_diffusion.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    shape = (2, 2, 5, 4, 2, 3, 2)
    cfg = ref.config(*shape)
    params = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in ref.init(rng, cfg).items()}
    z, t, y = rng.standard_normal((4, 2, 2)), np.array([3, 250, 600, 990]), np.array([0, 1, 2, 1])
    eps = rng.standard_normal((4, 4))
    x = rng.standard_normal((6, 8)) @ rng.standard_normal((8, 8))
    z_e, codebook, g_zq = rng.standard_normal((3, 2, 2)), rng.standard_normal((5, 2)), rng.standard_normal((3, 2, 2))
    means, zT = np.array([[1.0, -0.5], [-1.0, 0.5]]), rng.standard_normal((5, 2))
    idx, _ = ref.quantize(z_e, codebook)
    g_e, g_cb = ref.vq_grads(z_e, codebook, idx, g_zq, 0.25)
    abar = ref.alpha_bars(1000)
    ddim = ref.ddim_sample(lambda zz, tt: ref.exact_eps(zz, np.full(len(zz), abar[tt]), means, 0.1, np.array([0.5, 0.5])),
                           zT, abar, [999, 600, 300, 0])  # fmt: skip
    expected = {"eps_hat": ref.forward(params, cfg, z, t, y), "loss": np.array(ref.eps_loss(params, cfg, z, t, y, eps)),
                "latent": ref.encode(ref.fit_autoencoder(x, 4, 2), x, 4), "idx": idx, "g_e": g_e, "g_cb": g_cb,
                "ema_e": ref.ema_update(ref.ema_init(codebook), z_e, idx, 0.99)["e"], "ddim": ddim}  # fmt: skip
    expected.update({f"d{k}": g for k, g in ref.eps_loss_grad(params, cfg, z, t, y, eps).items()})
    inputs = {"cfg": np.array(shape), "z": z, "t": t, "y": y, "eps": eps, "x": x, "z_e": z_e, "codebook": codebook,
              "g_zq": g_zq, "means": means, "zT": zT}  # fmt: skip
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "denoiser: {positions: 2, channels: 2, hidden: 5, temb: 4, ctx_tokens: 2, attn: 3, classes: 2}",
             "inputs: {B: 4, schedule: \"linear betas 1e-4 to 2e-2 over 1000 steps\", autoencoder: \"f = 4, c = 2 on six signals of length 8\", vq: \"beta = 0.25, gamma = 0.99\", ddim: \"two-component mixture, s2 = 0.1, steps 999, 600, 300, 0\"}",
             "loss: \"the noise-prediction loss of eq-ldm, the squared error summed over the latent and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
