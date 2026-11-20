# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/t5/ (ai://NEW-t5).
"""Write the shared fixtures of T5 from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml: 13 tokens, width 8, 2 heads of 3,
a feed-forward width of 16, 2 layers in each stack and position tables of 8 buckets reaching 16, on a batch of two
pairs of 7 source and 5 target tokens. The expected values are the encoder output, the logits, eq-loss and its
gradients from the backward pass of the reference, the logits of the decoder computed in two pieces with the cache
of keys and values, and six greedy tokens.

The experiment trains small models with Adam to reverse a sequence: the source is x_1 ... x_6 and the target
x_6 ... x_1, with the x drawn uniformly from 8 symbols (ids 1 to 8; 0 is the start token of the decoder). Position
reaches T5 only through the bias tables, so the run with them and the run without them (pos = none) differ only
there. Each reports the training loss and, on 500 test pairs, the loss and the share of target tokens that the most
likely next token gets right under teacher forcing, and the share of whole sequences greedy decoding gets right.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 201910  # the month in which Exploring the Limits of Transfer Learning appeared, October 2019
CONFIG = {"vocab": 13, "d_model": 8, "n_heads": 2, "d_kv": 3, "d_ff": 16, "n_layers": 2, "buckets": 8,
          "max_distance": 16}  # fmt: skip
B, S, T, SPLIT, NEW = 2, 7, 5, 2, 6

N, SYMBOLS = 6, 8
EXP = {"vocab": 1 + SYMBOLS, "d_model": 32, "n_heads": 4, "d_kv": 8, "d_ff": 64, "n_layers": 2, "buckets": 8,
       "max_distance": 16}  # fmt: skip
RUNS = ("relative", "none")
STEPS, BATCH, LR, N_TEST, EVERY = 1500, 32, 3e-3, 500, 100


def load_reference():
    path = HERE.parent / "impl" / "numpy" / "t5.py"
    spec = importlib.util.spec_from_file_location("t5_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reverse_batch(rng, n):
    x = rng.integers(1, 1 + SYMBOLS, (n, N))
    return x, x[:, ::-1].copy()


def adam(params, grads, state, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for k, g in grads.items():
        m, v = state.get(k, (0.0, 0.0))
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        state[k] = (m, v)
        params[k] = params[k] - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)


def run(ref, pos, src, tgt):
    cfg = ref.config(**EXP, pos=pos)
    rng = np.random.default_rng(SEED + 1)
    params = ref.init(rng, cfg)
    state, hist = {}, []
    for t in range(1, STEPS + 1):
        x, y = reverse_batch(rng, BATCH)
        logits, cache = ref.forward(params, cfg, x, y)
        if t % EVERY == 0:
            hist.append(ref.nll(logits, y))
        adam(params, ref.backward(params, cfg, cache, logits, y), state, t, LR)
    logits = ref.forward(params, cfg, src, tgt)[0]
    token = float((logits.argmax(-1) == tgt).mean())
    whole = float((ref.generate(params, cfg, src, N) == tgt).all(1).mean())
    return np.array(hist), ref.nll(logits, tgt), token, whole


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg)
    src = rng.integers(0, CONFIG["vocab"], (B, S))
    tgt = rng.integers(0, CONFIG["vocab"], (B, T))
    logits, cache = ref.forward(params, cfg, src, tgt)
    grads = ref.backward(params, cfg, cache, logits, tgt)
    ids = ref.shift_right(tgt)
    first, c1 = ref.decode(params, cfg, cache["e"], ids[:, :SPLIT])
    second, _ = ref.decode(params, cfg, cache["e"], ids[:, SPLIT:], past=c1["kv"], start=SPLIT)
    expected = {
        "logits": logits,
        "loss": np.array(ref.nll(logits, tgt)),
        "e": cache["e"],
        "logits_cached": np.concatenate([first, second], axis=1),
        "greedy": ref.generate(params, cfg, src, NEW),
    }
    expected |= {"d" + k: v for k, v in grads.items()}

    test_src, test_tgt = reverse_batch(np.random.default_rng(SEED + 2), N_TEST)
    results = {}
    for pos in RUNS:
        hist, test_loss, token, whole = run(ref, pos, test_src, test_tgt)
        results[pos] = (test_loss, token, whole)
        expected |= {f"run_{pos}_hist": hist, f"run_{pos}_loss": np.array(test_loss)}
        expected |= {f"run_{pos}_token_acc": np.array(token), f"run_{pos}_seq_acc": np.array(whole)}

    files = {
        "params-small.npz": params,
        "input-small.npz": {"src": src, "tgt": tgt, "reverse_src": test_src, "reverse_tgt": test_tgt},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    runs = ", ".join(
        f"{k}: {{test_nats: {v[0]:.4f}, token_accuracy: {v[1]:.4f}, sequence_accuracy: {v[2]:.4f}}}"
        for k, v in results.items()
    )
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", pos: relative}",
        f"inputs: {{B: {B}, S: {S}, T: {T}, init_scale: 0.5, cache_split: {SPLIT}, greedy_tokens: {NEW}}}",
        f"experiment: {{task: reverse, n: {N}, symbols: {SYMBOLS}, d_model: {EXP['d_model']}, "
        f"n_heads: {EXP['n_heads']}, d_kv: {EXP['d_kv']}, d_ff: {EXP['d_ff']}, n_layers: {EXP['n_layers']}, "
        f"steps: {STEPS}, batch: {BATCH}, adam_lr: {LR}, test: {N_TEST}}}",
        f"runs: {{{runs}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {name}: {{sha256: {h}}}" for name, h in hashes.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
