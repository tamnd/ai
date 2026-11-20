# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Write the shared fixtures of the neural probabilistic language model from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes input-small.npz, expected-small.npz and manifest.yaml next to
itself. The text comes from a class language: 61 words, word 0 a sentence boundary and words 1 to 60 in six classes
of ten. The class of each word depends on the classes of the two words before it, through transition rows drawn from
a Dirichlet(0.3) distribution, and the word is uniform within its class. So words of the same class are
interchangeable, which an n-gram model cannot see and a model with word feature vectors can learn. The input holds a
training stream of 4000 tokens, a test stream of 1000 tokens with the source's own log-probability of each, initial
parameters of an untied model with direct connections (|V| = 61, m = 8, h = 16, n = 3) and of a tied model (m = h = 8),
noise words for noise-contrastive estimation, two epoch permutations, and the counts, node parameters, hidden vectors
and targets of a Huffman-tree softmax.

The expected values are the logits, loss and gradients of both models at the initial parameters, the NCE loss and
gradients, the training loss of each epoch and the final parameters and test log-probabilities after two epochs of SGD,
and the log-probabilities, loss and gradients of the tree softmax."""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2003
K, PER = 6, 10
V, M, HID, N = K * PER + 1, 8, 16, 3
N_TRAIN, N_TEST, BATCH, NOISE = 4000, 1000, 16, 5
LR0, R, DECAY, EPOCHS = 0.3, 1e-3, 1e-4, 2


def load_reference():
    path = HERE.parent / "impl" / "numpy" / "neural_probabilistic_language_model.py"
    spec = importlib.util.spec_from_file_location("nplm_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def class_language(rng, trans, n):
    """n tokens of the class language and the natural log of the source probability of each."""
    out, lp = [], []
    u = v = K
    while len(out) < n:
        c = int(rng.choice(K + 1, p=trans[u, v]))
        if c == K:
            out.append(0)
            lp.append(np.log(trans[u, v, c]))
        else:
            out.append(1 + c * PER + int(rng.integers(PER)))
            lp.append(np.log(trans[u, v, c] / PER))
        u, v = v, c
    return np.array(out, dtype=np.int64), np.array(lp)


def grads(prefix, g):
    return {f"{prefix}_{k}": v for k, v in g.items()}


def main():
    nl = load_reference()
    rng = np.random.default_rng(SEED)
    trans = rng.dirichlet(np.full(K + 1, 0.3), size=(K + 1, K + 1))
    train, _ = class_language(rng, trans, N_TRAIN)
    test, test_lp = class_language(rng, trans, N_TEST)
    ctx, tgt = nl.windows(train, N, 0)
    cte, tte = nl.windows(test, N, 0)
    counts = np.bincount(train, minlength=V) + 1.0
    log_pn = np.log(counts / counts.sum())
    noise = rng.choice(V, size=(32, NOISE), p=counts / counts.sum())
    orders = np.stack([rng.permutation(N_TRAIN) for _ in range(EPOCHS)])

    model = nl.NPLM(V, M, HID, N, direct=True, seed=SEED)
    tied = nl.NPLM(V, M, M, N, direct=True, tie=True, seed=SEED + 1)
    hs = nl.HierarchicalSoftmax(counts, HID, seed=SEED + 2)
    hs_a = np.tanh(rng.normal(size=(8, HID)))
    hs_tgt = tgt[:8]
    inp = {"train": train, "test": test, "test_logprob": test_lp, "log_pn": log_pn, "noise": noise, "orders": orders}
    inp |= {f"init_{k}": v.copy() for k, v in model.p.items()}
    inp |= {f"tinit_{k}": v.copy() for k, v in tied.p.items()}
    inp |= {"hs_counts": counts, "hs_v": hs.v.copy(), "hs_c": hs.c.copy(), "hs_a": hs_a, "hs_tgt": hs_tgt}

    exp = {"logits0": model.logits(cte[:32])}
    loss, g = model.loss_and_grads(ctx[:32], tgt[:32], DECAY)
    exp |= {"loss0": np.array(loss)} | grads("grad0", g)
    loss, g = model.nce_loss_and_grads(ctx[:32], tgt[:32], noise, log_pn, DECAY)
    exp |= {"nce0": np.array(loss)} | grads("ncegrad0", g)
    exp |= {"tlogits0": tied.logits(cte[:32])}
    loss, g = tied.loss_and_grads(ctx[:32], tgt[:32], DECAY)
    exp |= {"tloss0": np.array(loss)} | grads("tgrad0", g)
    hist = model.train(ctx, tgt, EPOCHS, LR0, R, DECAY, BATCH, orders=orders)
    exp |= {"train_loss": np.array(hist), "final_test_logprob": model.token_logprobs(cte, tte)}
    exp |= {f"final_{k}": v for k, v in model.p.items()}
    exp |= {"hs_all_logprob": hs.all_log_probs(hs_a)}
    loss, g_a, g_v, g_c = hs.loss_and_grads(hs_a, hs_tgt)
    exp |= {"hs_loss": np.array(loss), "hs_grad_a": g_a, "hs_grad_v": g_v, "hs_grad_c": g_c}

    shas = {}
    for name, arrays in (("input-small.npz", inp), ("expected-small.npz", exp)):
        np.savez(HERE / name, **arrays)
        shas[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    ppl = float(np.exp(-exp["final_test_logprob"].mean()))
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        f"source: {{vocabulary: {V}, classes: {K}, per_class: {PER}, dirichlet: 0.3, train: {N_TRAIN}, test: {N_TEST}}}",
        f"model: {{V: {V}, m: {M}, h: {HID}, n: {N}, direct: true, tied_model_h: {M}}}",
        f"training: {{epochs: {EPOCHS}, batch: {BATCH}, lr0: {LR0}, r: {R}, decay: {DECAY}, noise_per_example: {NOISE}}}",
        f"results: {{source_test_perplexity: {np.exp(-test_lp.mean()):.4f}, model_test_perplexity: {ppl:.4f}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {k}: {{sha256: {v}}}" for k, v in shas.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
