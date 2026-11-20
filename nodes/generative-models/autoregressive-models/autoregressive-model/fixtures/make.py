# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Write the shared fixtures of the autoregressive model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are a MADE of the example config (made_*), a NADE with D = 4 and H = 3
(nade_*) and a WaveNet stack with K = 5 values, C = 3 channels, kernel 2 and dilations 1, 2 (wave_*), all random. The
input is a batch x of binary vectors, the ordering of the MADE, integer sequences seq, the dilations and audio samples
in [-1, 1]. The expected values are the MADE logits, log_prob, loss and d<param> of the mean negative log-likelihood,
the NADE logits nade_logits, the WaveNet logits wave_logits and the mu-law codes mu_codes of the audio.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 150203509  # the arXiv number of MADE, 1502.03509
ORDER = np.array([3, 1, 4, 2])
DILATIONS = np.array([1, 2])


def load_reference():
    spec = importlib.util.spec_from_file_location("ar_ref", HERE.parent / "impl" / "numpy" / "autoregressive.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    cfg = ref.config()
    made = {k: rng.normal(0.0, 0.7, s) for k, s in ref.param_shapes(cfg).items()}
    nade = {k: rng.normal(0.0, 0.7, s) for k, s in ref.nade_shapes(4, 3).items()}
    wave = {k: rng.normal(0.0, 0.6, s) for k, s in ref.wavenet_shapes(5, 3, 2, tuple(DILATIONS)).items()}
    x = ref.all_binary(4)[[0, 5, 10, 15, 6]]
    seq = rng.integers(0, 5, (2, 9))
    audio = np.array([-1.0, -0.5, -0.01, 0.0, 0.001, 0.2, 0.5, 1.0])
    mk = ref.masks(cfg, ref.degrees(cfg, ORDER))
    a, _ = ref.logits(made, cfg, mk, x)
    loss, grads = ref.loss_and_grads(made, cfg, mk, x)
    expected = {"logits": a, "log_prob": ref.log_prob(made, cfg, mk, x), "loss": np.array(loss),
                "nade_logits": ref.nade_logits(nade, x),
                "wave_logits": ref.wavenet_logits(wave, seq, tuple(int(d) for d in DILATIONS)),
                "mu_codes": ref.mu_law_encode(audio)}  # fmt: skip
    expected.update({f"d{k}": g for k, g in grads.items()})
    params = {f"made_{k}": v for k, v in made.items()}
    params.update({f"nade_{k}": v for k, v in nade.items()})
    params.update({f"wave_{k}": v for k, v in wave.items()})
    files = {"params-small.npz": params,
             "input-small.npz": {"x": x, "order": ORDER, "seq": seq, "dilations": DILATIONS, "audio": audio},
             "expected-small.npz": expected}  # fmt: skip
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "config: {d_data: 4, hidden: [6], direct: true, order: [3, 1, 4, 2]}",
             "nade: {D: 4, H: 3}", "wavenet: {K: 5, C: 3, kernel: 2, dilations: [1, 2]}",
             "inputs: {B: 5, x: \"binary vectors\", seq: \"2 integer sequences of length 9 in 0..4\", audio: \"8 samples in [-1, 1]\"}",
             "loss: \"the negative log-likelihood of the MADE, summed over variables and averaged over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
