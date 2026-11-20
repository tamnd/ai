# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""Write the shared fixtures of the n-gram language model from the plain Python reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes input-small.npz, expected-small.npz and manifest.yaml next to
itself. The input is a training corpus of 6000 sentences and a test corpus of 300 sentences drawn from a known
second-order Markov source over 2000 words named w0 to w1999, stored as word ids with sentence lengths, together with
the natural log of the source's own probability of every test token (EOS included), so that the true cross-entropy
of the test text is known. The source ends a sentence with probability 0.12 after each word; otherwise it draws the
next word from 0.25 G + 0.35 A(v) + 0.4 B(u, v), where u, v are the two previous words (BOS at the start). G is the
Zipf distribution with exponent 1.2 over the word ids. A and B are Zipf distributions with exponent 2 over a context's
own ranking of the words: word i is ranked by log(i + 1) plus normal noise of standard deviation 2.5 drawn with a seed
derived from the context, so that each context prefers its own few words and rare words stay rare overall.

The expected values are, for orders 1 to 4 and both interpolated and modified Kneser-Ney, the discounts and the log
probability of every test token; the Stupid Backoff scores of the test tokens for order 3; the infinity-gram
distribution and effective n for five contexts; and the n1 / N leftover of the training trigrams.
"""

import hashlib
import importlib.util
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 1312300  # the arXiv number of the One Billion Word benchmark, 1312.3005, without its last digits
V, P_END, ZIPF_G, ZIPF_C, SPREAD, MIX = 2000, 0.12, 1.2, 2.0, 2.5, (0.25, 0.35, 0.4)
N_TRAIN, N_TEST = 6000, 300


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ng_ref", HERE.parent / "impl" / "python" / "n_gram_language_model.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Source:
    """The second-order Markov source. Context ids run over 0..V-1 with V standing for BOS."""

    def __init__(self, seed):
        self.seed, self.cache = seed, {}
        g, c = 1.0 / np.arange(1, V + 1) ** ZIPF_G, 1.0 / np.arange(1, V + 1) ** ZIPF_C
        self.g, self.c = g / g.sum(), c / c.sum()

    def _dist(self, key):
        if key in self.cache:
            return self.cache[key]
        noise = np.random.default_rng([self.seed, *key]).normal(0.0, SPREAD, V)
        d = np.empty(V)
        d[np.argsort(np.log(np.arange(1, V + 1)) + noise, kind="stable")] = self.c
        if len(key) == 1:
            self.cache[key] = d
        return d

    def next_dist(self, u, v):
        """The distribution of the next word id given the previous two (V for BOS), before the end decision."""
        return MIX[0] * self.g + MIX[1] * self._dist((v,)) + MIX[2] * self._dist((u, v))

    def sentences(self, rng, n):
        out = []
        for _ in range(n):
            u, v, s = V, V, []
            while True:
                s.append(int(rng.choice(V, p=self.next_dist(u, v))))
                u, v = v, s[-1]
                if rng.random() < P_END:
                    break
            out.append(s)
        return out

    def token_logprobs(self, sents):
        """The log probability of each word and of each EOS, sentence by sentence."""
        lp = []
        for s in sents:
            u, v = V, V
            for i, w in enumerate(s):
                lp.append(math.log(self.next_dist(u, v)[w]) + (math.log(1 - P_END) if i > 0 else 0.0))
                u, v = v, w
            lp.append(math.log(P_END))
        return np.array(lp)


def words(ids):
    return [f"w{i}" for i in ids]


def flatten(sents):
    return np.array([w for s in sents for w in s], dtype=np.int16), np.array([len(s) for s in sents], dtype=np.int16)


CONTEXTS = [[], ["w3"], ["w0", "w5"], ["w9", "w9", "w9"], ["zzz", "w1", "w2"]]


def main():
    ref = load_reference()
    rng = np.random.default_rng(SEED)
    src = Source(SEED)
    train, test = src.sentences(rng, N_TRAIN), src.sentences(rng, N_TEST)
    train_w, test_w = [words(s) for s in train], [words(s) for s in test]
    tr_ids, tr_len = flatten(train)
    te_ids, te_len = flatten(test)
    inputs = {"train_ids": tr_ids, "train_len": tr_len, "test_ids": te_ids, "test_len": te_len,
              "true_logprob": src.token_logprobs(test)}  # fmt: skip
    expected = {}
    for order in (1, 2, 3, 4):
        for kind in ("interpolated", "modified"):
            m = ref.KneserNey(train_w, order, kind)
            lp = []
            for s in test_w:
                toks = ref.pad(s, order)
                lp += [math.log(m.prob(toks[i], toks[i - order + 1 : i])) for i in range(order - 1, len(toks))]
            expected[f"{kind}{order}_logprob"] = np.array(lp)
            expected[f"{kind}{order}_discounts"] = np.array([m.discounts[n] for n in range(1, order + 1)])
    sb = ref.StupidBackoff(train_w, 3)
    sc = []
    for s in test_w:
        toks = ref.pad(s, 3)
        sc += [sb.score(toks[i], toks[i - 2 : i]) for i in range(2, len(toks))]
    expected["sb3_score"] = np.array(sc)
    ig = ref.InfiniGram(train_w)
    for j, ctx in enumerate(CONTEXTS):
        dist, n_eff = ig.distribution(ctx)
        keys = sorted(dist, key=lambda w: int(w[1:]))
        expected[f"inf{j}_ids"] = np.array([int(w[1:]) for w in keys])
        expected[f"inf{j}_prob"] = np.array([dist[w] for w in keys])
        expected[f"inf{j}_n"] = np.array(n_eff)
    expected["gt_leftover3"] = np.array(ref.good_turing_leftover(ref.count_ngrams(train_w, 3)))
    files = {"input-small.npz": inputs, "expected-small.npz": expected}
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    lines = ["# Written by fixtures/make.py; do not edit.", f"seed: {SEED}",
             f"source: {{words: {V}, end: {P_END}, zipf_global: {ZIPF_G}, zipf_context: {ZIPF_C}, spread: {SPREAD}, mix: {list(MIX)}, train: {N_TRAIN}, test: {N_TEST}}}",
             "models: \"Kneser-Ney orders 1 to 4, interpolated and modified; Stupid Backoff order 3, alpha 0.4; infinity-gram\"",
             "contexts: \"" + "; ".join(" ".join(c) if c else "(empty)" for c in CONTEXTS) + "\"",
             "dtype: float64", f"numpy: \"{np.__version__}\"", "files:"]  # fmt: skip
    lines += [f"  {name}: {{sha256: {hh}}}" for name, hh in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
