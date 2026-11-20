# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://NEW-sequence-to-sequence-model).
"""Numeric check of the derivation factorization: the sum over all outputs, the loss as a sum, the logit gradient."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import seq2seq  # noqa: E402

EOS = seq2seq.EOS


def prefix_probs(params, cfg, src, n):
    """p of every prefix of up to n tokens, found by running the decoder one step at a time."""
    state, _ = seq2seq.encode(params, cfg, src[None])
    out = {(): (1.0, state)}
    frontier = [()]
    for _ in range(n):
        nxt = []
        for y in frontier:
            p, st = out[y]
            logp, st2 = seq2seq.decoder_step(params, np.array([y[-1] if y else EOS]), st)
            for tok in range(cfg["tgt_vocab"]):
                out[y + (tok,)] = (p * np.exp(logp[0, tok]), st2)
                if tok != EOS:
                    nxt.append(y + (tok,))
        frontier = nxt
    return {y: p for y, (p, _) in out.items()}


rng = np.random.default_rng(0)
for _ in range(20):
    V, n = int(rng.integers(2, 5)), int(rng.integers(1, 5))
    cfg = seq2seq.config(5, V, 2, 3, int(rng.integers(1, 3)))
    params, src = seq2seq.init(rng, cfg, scale=1.0), rng.integers(1, 5, 3)
    probs = prefix_probs(params, cfg, src, n)
    complete = sum(p for y, p in probs.items() if y and y[-1] == EOS)
    open_n = sum(p for y, p in probs.items() if len(y) == n and EOS not in y)
    assert abs(complete + open_n - 1.0) < 1e-12
    # The probability of a prefix never increases with its length (statement 4).
    for y, p in probs.items():
        if y:
            assert p <= probs[y[:-1]] + 1e-15
    # A complete sequence's probability from the teacher-forced pass is the product of the steps.
    for y, p in probs.items():
        if y and y[-1] == EOS:
            lp = seq2seq.sequence_log_prob(params, cfg, src[None], np.array([y]))[0]
            assert abs(np.exp(lp) - p) < 1e-12

# Statements 1 and 2 on a batch: the loss is the sum of per-step cross-entropies; its logit gradient is softmax - e_y.
cfg = seq2seq.config()
params = seq2seq.init(rng, cfg, scale=0.5)
src, tgt = rng.integers(1, 7, (3, 5)), rng.integers(1, 6, (3, 4))
tgt[:, -1] = EOS
logits, _ = seq2seq.forward(params, cfg, src, tgt)
B, T, V = logits.shape
p = np.exp(seq2seq.log_softmax(logits))
ce = -np.log(p[np.arange(B)[:, None], np.arange(T), tgt])
assert abs(ce.sum() / B - seq2seq.nll(logits, tgt)) < 1e-13
want = (p - np.eye(V)[tgt]) / B
for b, t, v in itertools.product(range(B), range(T), range(V)):
    z = logits.copy()
    z[b, t, v] += 1e-6
    up = seq2seq.nll(z, tgt)
    z[b, t, v] -= 2e-6
    down = seq2seq.nll(z, tgt)
    assert abs((up - down) / 2e-6 - want[b, t, v]) < 1e-8
print("ok")
