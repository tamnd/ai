# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Write the shared fixtures of the generative adversarial network from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The input is a batch of data x, a batch of noise z and the mixing weights alpha of the
gradient penalty, so every implementation computes the same fakes and the same points x_hat. The expected values are
the forward values, named fake, a_real, a_fake and grad_real, and for each loss <kind>/d/loss and <kind>/d/d<param>, the
discriminator's loss and gradients, and <kind>/g/loss and <kind>/g/d<param>, the generator's.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20140610  # the day Goodfellow et al. put the paper on arXiv
B = 4  # the example config of shapes.yaml: d_noise 2, d_hidden 5, d_data 3, d_critic 4
LAM, GAMMA = 10.0, 1.0


def load_reference():
    spec = importlib.util.spec_from_file_location("gan_ref", HERE.parent / "impl" / "numpy" / "gan.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    cfg = ref.config()
    params = ref.init(rng, cfg)
    for k in ("gen_b1", "gen_b2", "dis_b1", "dis_b2"):
        params[k] = rng.normal(0.0, 0.3, params[k].shape)  # nonzero biases, so the fixture checks them
    x = rng.normal(0.0, 1.0, (B, 3))
    z = rng.standard_normal((B, 2))
    alpha = rng.uniform(0.0, 1.0, B)
    fake = ref.generate(params, z)[0]
    a_real, h_real = ref.critic(params, x)
    expected = {"fake": fake, "a_real": a_real, "a_fake": ref.critic(params, fake)[0],
                "grad_real": ref.input_grad(params, h_real)}  # fmt: skip
    for kind in ref.LOSSES:
        value, grads = ref.d_step(params, x, z, kind, alpha, LAM, GAMMA)
        expected[f"{kind}/d/loss"] = np.array(value)
        expected.update({f"{kind}/d/d{k}": v for k, v in grads.items()})
        value, grads = ref.g_step(params, z, kind)
        expected[f"{kind}/g/loss"] = np.array(value)
        expected.update({f"{kind}/g/d{k}": v for k, v in grads.items()})
    files = {"params-small.npz": params, "input-small.npz": {"x": x, "z": z, "alpha": alpha},
             "expected-small.npz": expected}  # fmt: skip
    hashes = {}
    for name, arrays in files.items():
        target = HERE / name
        with open(target, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_noise: 2, d_hidden: 5, d_data: 3, d_critic: 4}",
             f"inputs: {{B: {B}, x: \"standard normal\", z: \"standard normal\", alpha: \"uniform on [0, 1]\"}}",
             f"loss: \"minimax, nonsat, wgan, wgan_gp with lambda {LAM:g} and r1 with gamma {GAMMA:g}, averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
