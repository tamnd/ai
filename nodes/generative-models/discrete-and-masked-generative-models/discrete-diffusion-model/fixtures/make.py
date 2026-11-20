# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Write the shared fixtures of the discrete diffusion model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The parameters are a denoiser for L = 3 tokens over K = 3 data symbols with 5 hidden
units, random and scaled up so that its outputs are far from uniform. The input is a batch x of four sequences with
times t and uniforms u that mask them under the cosine schedule, a partly masked batch z, a noisy token xt with a
predicted distribution p0 for the uniform D3PM reverse step, and a distribution p_se with positive scores for score
entropy. The expected values are the logits and SUBS probabilities on z, the masked batch, the one-sample bound as a
weighted cross-entropy with its gradient in every parameter and per sequence, the reverse step, the exact bound of
the first sequence and the score entropy.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2406075240  # the arXiv number of Sahoo and colleagues, 2406.07524, and a 0


def load_reference():
    spec = importlib.util.spec_from_file_location("dd_ref", HERE.parent / "impl" / "numpy" / "discrete_diffusion.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    L, K, H = 3, 3, 5
    cfg = ref.config(L, K, H)
    params = ref.init(rng, cfg, scale=3.0)
    params = {k: v + 0.5 * rng.standard_normal(v.shape) for k, v in params.items()}
    x = rng.integers(0, K, (4, L))
    t, u = np.array([0.15, 0.4, 0.7, 0.95]), rng.random((4, L))
    z = np.array([[3, 0, 1], [2, 3, 3], [3, 3, 3], [0, 1, 2]])
    xt, p0 = 2, rng.dirichlet(np.ones(4))
    p_se, scores = rng.dirichlet(np.ones(4)), np.exp(rng.normal(size=(4, 4)))
    logits = ref.forward(params, z, cfg)
    Z, X, Wt = ref.mc_batch(x, t, u, K, "cosine")
    lp = ref.log_softmax(ref.forward(params, Z, cfg))
    expected = {"logits": logits, "probs": ref.subs_probs(logits, z, K), "z_masked": Z,
                "mc_loss": np.array(ref.weighted_ce(params, cfg, Z, X, Wt)),
                "per_seq": ref.mdlm_estimate(lp, Z, X, t, K, "cosine"),
                "reverse_uniform": ref.posterior(ref.one_hot(xt, 4), p0, ref.uniform_q(0.3, 4), ref.uniform_qbar(0.6, 4)),
                "nelbo_x0": np.array(ref.exact_nelbo(ref.denoise_fn(params, cfg), x[0], K)),
                "score_entropy": np.array(ref.score_entropy(scores, p_se, np.ones_like(scores)))}  # fmt: skip
    expected.update({f"d{k}": g for k, g in ref.weighted_ce_grad(params, cfg, Z, X, Wt).items()})
    inputs = {"cfg": np.array([L, K, H]), "x": x, "t": t, "u": u, "z": z, "xt": np.array(xt), "p0": p0,
              "p_se": p_se, "scores": scores}  # fmt: skip
    files = {"params-small.npz": params, "input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             "denoiser: {seq_len: 3, vocab: 3, hidden: 5}",
             "inputs: {B: 4, schedule: cosine, z: \"a partly masked batch, the mask is 3\", uniform_d3pm: \"K = 4, beta = 0.3, alpha_bar_prev = 0.6\"}",
             "loss: \"the one-sample masked diffusion bound averaged over the batch, as a weighted cross-entropy\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
