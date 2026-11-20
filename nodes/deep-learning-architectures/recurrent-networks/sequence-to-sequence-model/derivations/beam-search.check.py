# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://NEW-sequence-to-sequence-model).
"""Numeric check of the derivation beam-search: beam 1 is greedy, a beam that never prunes is exact, the example."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import seq2seq  # noqa: E402

EOS = seq2seq.EOS


def brute_force(params, cfg, src, n):
    best = None
    for t in range(n):
        for pre in itertools.product(range(1, cfg["tgt_vocab"]), repeat=t):
            y = list(pre) + [EOS]
            lp = seq2seq.sequence_log_prob(params, cfg, src[None], np.array([y]))[0]
            if best is None or lp > best[1]:
                best = (y, lp)
    return best


def beam_no_early_stop(params, cfg, src, beam, n):
    """The same search without the early stop of statement 3."""
    (h, c), _ = seq2seq.encode(params, cfg, src[None])
    hyps, scores, complete = [[]], np.zeros(1), []
    for _ in range(n):
        logp, (h, c) = seq2seq.decoder_step(params, np.array([y[-1] if y else EOS for y in hyps]), (h, c))
        total = scores[:, None] + logp
        keep = []
        for k in np.argsort(-total, axis=None, kind="stable")[:beam]:
            j, tok = divmod(int(k), total.shape[1])
            (complete.append((hyps[j] + [tok], total[j, tok])) if tok == EOS else keep.append((j, tok)))
        if not keep:
            break
        rows = [j for j, _ in keep]
        hyps, scores = [hyps[j] + [tok] for j, tok in keep], np.array([total[j, tok] for j, tok in keep])
        h, c = h[:, rows], c[:, rows]
    return max(complete, key=lambda p: p[1]) if complete else None


rng = np.random.default_rng(0)
for _ in range(40):
    V, n = int(rng.integers(2, 5)), int(rng.integers(1, 5))
    cfg = seq2seq.config(5, V, 2, 3, int(rng.integers(1, 3)))
    params, src = seq2seq.init(rng, cfg, scale=1.5), rng.integers(1, 5, 3)
    assert seq2seq.beam_search(params, cfg, src, 1, n)[0] == seq2seq.greedy_decode(params, cfg, src[None], n)[0]
    wide = V * (V - 1) ** (n - 1)
    want = brute_force(params, cfg, src, n)
    got = seq2seq.beam_search(params, cfg, src, wide, n)
    assert got[0] == want[0] and abs(got[1] - want[1]) < 1e-12
    for beam in (1, 2, 3, wide):
        a, b = seq2seq.beam_search(params, cfg, src, beam, n), beam_no_early_stop(params, cfg, src, beam, n)
        if b is not None:
            assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-12

# The worked example.
cfg = seq2seq.config(5, 3, 3, 4, 1)
rng = np.random.default_rng(0)
params, src = seq2seq.init(rng, cfg, scale=1.5), rng.integers(1, 5, 4)
state, _ = seq2seq.encode(params, cfg, src[None])
first, state = seq2seq.decoder_step(params, np.array([EOS]), state)
assert np.allclose(np.exp(first[0]), [0.2207, 0.4918, 0.2875], atol=5e-5)
second, _ = seq2seq.decoder_step(params, np.array([1]), state)
assert np.allclose(np.exp(second[0]), [0.3646, 0.3707, 0.2646], atol=5e-5)
for beam, tokens, p in ((1, [1, 1, EOS], 0.0867), (2, [1, EOS], 0.1793), (3, [EOS], 0.2207)):
    got, lp = seq2seq.beam_search(params, cfg, src, beam, 4)
    assert got == tokens and abs(np.exp(lp) - p) < 5e-5, (beam, got, np.exp(lp))
assert brute_force(params, cfg, src, 4)[0] == [EOS]

# Statement 5: the decoder runs on at most n * beam hypotheses in all.
rows, step = [0], seq2seq.decoder_step


def counting_step(params, tokens, state):
    rows[0] += len(tokens)
    return step(params, tokens, state)


seq2seq.decoder_step = counting_step
for beam in (1, 2, 3, 5):
    rows[0] = 0
    seq2seq.beam_search(params, cfg, src, beam, 4)
    assert rows[0] <= 4 * beam
assert sum(3 * 2 ** (t - 1) for t in range(1, 5)) == 45  # the extensions an exhaustive search scores, V = 3, n = 4
print("ok")
