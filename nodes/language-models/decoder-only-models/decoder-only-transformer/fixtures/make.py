# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/decoder-only-transformer/ (ai://NEW-decoder-only-transformer).
"""Write the shared fixtures of the decoder-only Transformer from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes params-small.npz, input-small.npz, expected-small.npz and
manifest.yaml next to itself. The network is the example config of shapes.yaml: 11 tokens, a context of 8, width 8
with 2 heads, an MLP of 16 and 2 layers, the embedding tied to the output layer and learned positions, on a batch of
two random sequences of 8 tokens. The expected values are the logits, eq-loss and its gradients from the backward
pass of the reference, the logits computed in two pieces with the cache of keys and values, and five greedy tokens
after a prompt of three.

The experiment trains four small models with Adam on a copy task: a sequence is [BOS] x_1 ... x_6 [SEP] x_1 ... x_6
with the x drawn uniformly from 8 symbols, 14 tokens in all. The six tokens after [SEP] can be predicted exactly by
looking back seven places, and the model has to know where it is to do so. The runs are 2 layers and 1 layer, each
with learned position embeddings and with none; each reports the training loss and, on 500 test sequences, the loss
and the share of the six copied tokens that the most likely next token gets right.
"""

import hashlib
import importlib.util
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 201806  # the month in which Improving Language Understanding by Generative Pre-Training appeared, June 2018
CONFIG = {"vocab": 11, "n_ctx": 8, "d_model": 8, "n_heads": 2, "d_ff": 16, "n_layers": 2}
B, SCALE, SPLIT, PROMPT, NEW = 2, 0.5, 5, 3, 5

COPY_K, SYMBOLS = 6, 8  # tokens 0 and 1 are [BOS] and [SEP], 2 to 9 the symbols
EXP = {"vocab": 2 + SYMBOLS, "n_ctx": 2 * COPY_K + 2, "d_model": 32, "n_heads": 4, "d_ff": 64}
RUNS = {
    "learned_2": {"n_layers": 2, "pos": "learned"},
    "none_2": {"n_layers": 2, "pos": "none"},
    "learned_1": {"n_layers": 1, "pos": "learned"},
    "none_1": {"n_layers": 1, "pos": "none"},
}
STEPS, BATCH, LR, INIT, N_TEST, EVERY = 1500, 32, 3e-3, 0.02, 500, 100


def load_reference():
    path = HERE.parent / "impl" / "numpy" / "decoder_lm.py"
    spec = importlib.util.spec_from_file_location("decoder_lm_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_batch(rng, n):
    x = rng.integers(2, 2 + SYMBOLS, (n, COPY_K))
    return np.concatenate([np.zeros((n, 1), np.int64), x, np.ones((n, 1), np.int64), x], axis=1)


def adam(params, grads, state, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for k, g in grads.items():
        m, v = state.get(k, (0.0, 0.0))
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        state[k] = (m, v)
        params[k] = params[k] - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)


def copy_accuracy(ref, params, cfg, ids):
    logits = ref.forward(params, cfg, ids)[0]
    return float((logits[:, COPY_K + 1 : -1].argmax(-1) == ids[:, COPY_K + 2 :]).mean())


def run(ref, name, test):
    cfg = ref.config(**EXP, **RUNS[name])
    rng = np.random.default_rng(SEED + 1)
    params = ref.init(rng, cfg, INIT, residual_scaled=True)
    state, hist = {}, []
    for t in range(1, STEPS + 1):
        ids = copy_batch(rng, BATCH)
        logits, cache = ref.forward(params, cfg, ids)
        if t % EVERY == 0:
            hist.append(ref.nll(logits, ids))
        adam(params, ref.backward(params, cfg, cache, ids), state, t, LR)
    return np.array(hist), ref.loss(params, cfg, test), copy_accuracy(ref, params, cfg, test)


def main():
    ref = load_reference()
    cfg = ref.config(**CONFIG)
    rng = np.random.default_rng(SEED)
    params = ref.init(rng, cfg, scale=SCALE)
    ids = rng.integers(0, CONFIG["vocab"], (B, CONFIG["n_ctx"]))
    logits, cache = ref.forward(params, cfg, ids)
    grads = ref.backward(params, cfg, cache, ids)
    first, c1 = ref.forward(params, cfg, ids[:, :SPLIT])
    second, _ = ref.forward(params, cfg, ids[:, SPLIT:], past=c1["kv"])
    expected = {
        "logits": logits,
        "loss": np.array(ref.nll(logits, ids)),
        "tokens": cache["tokens"],
        "final": cache["final"],
        "logits_cached": np.concatenate([first, second], axis=1),
        "greedy": ref.generate(params, cfg, ids[:, :PROMPT], NEW),
    }
    expected |= {"d" + k: v for k, v in grads.items()}

    test = copy_batch(np.random.default_rng(SEED + 2), N_TEST)
    floor = COPY_K * math.log(SYMBOLS) / (2 * COPY_K + 1)
    results = {}
    for name in RUNS:
        hist, test_loss, acc = run(ref, name, test)
        results[name] = (test_loss, acc)
        expected |= {f"run_{name}_hist": hist, f"run_{name}_loss": np.array(test_loss)}
        expected |= {f"run_{name}_acc": np.array(acc)}

    files = {
        "params-small.npz": params,
        "input-small.npz": {"ids": ids, "copy_test": test},
        "expected-small.npz": expected,
    }
    hashes = {}
    for name, arrays in files.items():
        out = HERE / name
        with open(out, "wb") as f:
            np.savez(f, **dict(sorted(arrays.items())))
        hashes[name] = hashlib.sha256(out.read_bytes()).hexdigest()
    runs = ", ".join(f"{k}: {{test_nats: {v[0]:.4f}, copy_accuracy: {v[1]:.4f}}}" for k, v in results.items())
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        "config: {" + ", ".join(f"{k}: {v}" for k, v in CONFIG.items()) + ", tied: true, pos: learned}",
        f"inputs: {{B: {B}, init_scale: {SCALE}, cache_split: {SPLIT}, prompt: {PROMPT}, greedy_tokens: {NEW}}}",
        f"experiment: {{task: copy, k: {COPY_K}, symbols: {SYMBOLS}, d_model: {EXP['d_model']}, "
        f"n_heads: {EXP['n_heads']}, d_ff: {EXP['d_ff']}, steps: {STEPS}, batch: {BATCH}, adam_lr: {LR}, "
        f"init_scale: {INIT}, test: {N_TEST}, floor_nats: {floor:.4f}}}",
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
