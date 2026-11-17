# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://NEW-end-to-end-memory-network).
"""Write the shared fixtures of the end-to-end memory network from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz
and manifest.yaml next to itself. The input is a toy task in the form of the first bAbI task: sentences
"<person> went <place>", the most recent first, and the question "where is <person>".
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20150331  # the day Sukhbaatar, Szlam, Weston and Fergus posted the paper to arXiv
CONFIG = {"vocab": 12, "dim": 6, "hops": 3, "mem": 5, "sent_len": 4}
B = 2
WHERE, IS, WENT = 1, 2, 3
PERSONS, PLACES = (4, 5, 6), (7, 8, 9, 10, 11)


def load_reference():
    spec = importlib.util.spec_from_file_location("memn2n_ref", HERE.parent / "impl" / "numpy" / "memn2n.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy_batch(rng, batch, n, j):
    story = np.zeros((batch, n, j), dtype=np.int64)
    query = np.zeros((batch, j), dtype=np.int64)
    answer = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        who, where = rng.choice(PERSONS, n), rng.choice(PLACES, n)
        story[b, :, 0], story[b, :, 1], story[b, :, 2] = who, WENT, where
        asked = who[rng.integers(n)]
        query[b, :3] = WHERE, IS, asked
        answer[b] = where[np.flatnonzero(who == asked)[0]]
    story[1, -1] = 0  # an empty memory, as the padding of a short story leaves
    return story, query, answer


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, sigma=0.5)
    story, query, answer = toy_batch(rng, B, CONFIG["mem"], CONFIG["sent_len"])
    logits, attn = ref.forward(params, story, query, cfg, trace=True)
    loss = ref.cross_entropy(logits, answer)
    expected = {"logits": logits, "attention": attn, "loss": np.array(loss)}
    files = {"params-small.npz": params, "input-small.npz": {"story": story, "query": query, "answer": answer},
             "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    cfg_text = ", ".join(f"{k}: {v}" for k, v in CONFIG.items())
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             f"config: {{{cfg_text}, tying: adjacent, encoding: pe, temporal: true}}",
             f"inputs: {{B: {B}, story: \"(B, N, J) word ids, the most recent sentence first, 0 the null word\", query: \"(B, J)\", answer: \"(B,) word ids\"}}",
             "expected: \"logits (B, V) and the attention of every hop (K, B, N)\"",
             "loss: \"the cross-entropy of the answers, summed over the batch\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
