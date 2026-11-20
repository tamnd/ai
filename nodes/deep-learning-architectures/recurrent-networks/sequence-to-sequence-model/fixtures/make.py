# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://NEW-sequence-to-sequence-model).
"""Write the shared fixtures of the sequence-to-sequence model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The model is the example config of shapes.yaml with the source reversed; the arrays
ending in _forward come from the same parameters reading the source in its own order. The gradients are those of
eq-loss from the backward pass of the reference. Greedy outputs are padded with -1 to max_len.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20140910  # the day Sequence to Sequence Learning with Neural Networks appeared on arXiv
CONFIG = {"src_vocab": 7, "tgt_vocab": 6, "embed_size": 3, "hidden_size": 4, "n_layers": 2}  # the example config
B, S, T = 2, 5, 4
SCALE = 0.5  # wider than the 0.08 of the paper, so that the outputs are far from uniform
BEAM, MAX_LEN = 3, 6


def load_reference():
    spec = importlib.util.spec_from_file_location("seq2seq_ref", HERE.parent / "impl" / "numpy" / "seq2seq.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sentences(rng, n, length, vocab):
    """n token sequences of the given length, ending in <EOS> (0), with the other tokens in 1..vocab-1."""
    out = rng.integers(1, vocab, (n, length))
    out[:, -1] = 0
    return out


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG, reverse=True)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    src = sentences(rng, B, S, CONFIG["src_vocab"])
    tgt = sentences(rng, B, T, CONFIG["tgt_vocab"])
    logits, cache = ref.forward(params, cfg, src, tgt)
    loss, grads = ref.nll(logits, tgt), ref.backward(params, cfg, cache)
    greedy = np.full((B, MAX_LEN), -1)
    for i, y in enumerate(ref.greedy_decode(params, cfg, src, MAX_LEN)):
        greedy[i, : len(y)] = y
    beam_tokens, beam_logp = ref.beam_search(params, cfg, src[0], BEAM, MAX_LEN)
    beam = np.full(MAX_LEN, -1)
    beam[: len(beam_tokens)] = beam_tokens
    fwd = ref.config(**CONFIG, reverse=False)
    expected = {
        "logits": logits,
        "loss": np.array(loss),
        "v_h": cache["v"][0],
        "v_c": cache["v"][1],
        "seq_logp": ref.sequence_log_prob(params, cfg, src, tgt),
        "greedy": greedy,
        "beam": beam,
        "beam_logp": np.array(beam_logp),
        "logits_forward": ref.forward(params, fwd, src, tgt)[0],
    }
    expected |= {"d" + k: v for k, v in grads.items()}
    files = {
        "params-small.npz": params,
        "input-small.npz": {"src": src, "tgt": tgt},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        path = HERE / name
        with open(path, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", reverse: true}",
        f"inputs: {{B: {B}, S: {S}, T: {T}, init_scale: {SCALE}, beam: {BEAM}, max_len: {MAX_LEN}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
