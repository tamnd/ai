# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/byte-pair-encoding/ (ai://ZRQW82).
"""Write the shared fixtures of byte-pair encoding from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes input-small.npz, expected-small.npz and manifest.yaml next to
itself. The training table holds 60 distinct words of 2 to 8 letters over a six-letter alphabet, built from a few stems
and suffixes so that pairs recur, with frequencies falling as 1000 / rank; the test words are 20 more words of the same
kind, some of them unseen. The expected values are the 40 merges in learned order, the final segmentation of every
training word, the ordered and the priority segmentation of every test word, and the BPE-dropout segmentation of every
test word at p = 0.3 with one Generator seeded by SEED + 1 and used for the test words in order. A segmentation is
stored as its symbols joined by single spaces, so the arrays are plain Unicode and load without pickle."""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2016
STEMS = ("ab", "ac", "bad", "cab", "dec", "fed", "ace", "bead")
SUFFIXES = ("", "e", "ed", "s", "es", "fa", "de", "bb")
N_TRAIN, N_TEST, MERGES, P = 60, 20, 40, 0.3


def load_reference():
    path = HERE.parent / "impl" / "numpy" / "byte_pair_encoding.py"
    spec = importlib.util.spec_from_file_location("bpe_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(rng, n, avoid=()):
    """n distinct words stem + suffix, or two stems, not in avoid, in the order drawn."""
    out = []
    while len(out) < n:
        a = STEMS[rng.integers(len(STEMS))]
        b = (SUFFIXES + STEMS)[rng.integers(len(SUFFIXES) + len(STEMS))]
        w = a + b
        if w not in out and w not in avoid:
            out.append(w)
    return out


def main():
    bpe = load_reference()
    rng = np.random.default_rng(SEED)
    train = words(rng, N_TRAIN)
    freqs = np.array([1000 // r for r in range(1, N_TRAIN + 1)], dtype=np.int64)
    test = words(rng, N_TEST // 2, train) + [train[i] for i in rng.choice(N_TRAIN, N_TEST // 2, replace=False)]
    merges, vocab = bpe.train_bpe(dict(zip(train, freqs.tolist(), strict=True)), MERGES)
    drop = np.random.default_rng(SEED + 1)
    expected = {
        "merges": np.array(merges, dtype=str),
        "train_segments": np.array([" ".join(seg) for seg in vocab], dtype=str),
        "encode": np.array([" ".join(bpe.encode(w, merges)) for w in test], dtype=str),
        "encode_priority": np.array([" ".join(bpe.encode_priority(w, merges)) for w in test], dtype=str),
        "dropout": np.array([" ".join(bpe.bpe_dropout_encode(w, merges, P, drop)) for w in test], dtype=str),
    }
    inputs = {
        "words": np.array(train, dtype=str),
        "freqs": freqs,
        "test_words": np.array(test, dtype=str),
        "num_merges": np.array(MERGES),
        "p": np.array(P),
        "dropout_seed": np.array(SEED + 1),
    }
    shas = {}
    for name, arrays in (("input-small.npz", inputs), ("expected-small.npz", expected)):
        np.savez(HERE / name, **arrays)
        shas[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        f"source: {{train_words: {N_TRAIN}, test_words: {N_TEST}, stems: {len(STEMS)}, suffixes: {len(SUFFIXES)}}}",
        f"training: {{merges: {MERGES}, learned: {len(merges)}, frequency: 1000 // rank}}",
        f"dropout: {{p: {P}, seed: {SEED + 1}}}",
        "dtype: unicode and int64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {k}: {{sha256: {v}}}" for k, v in shas.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
