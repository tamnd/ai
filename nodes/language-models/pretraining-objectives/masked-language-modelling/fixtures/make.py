# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Write the shared fixtures of masked language modelling from the NumPy reference (spec 08.7.2).

Run it from anywhere: python fixtures/make.py. It writes input-small.npz, expected-small.npz and manifest.yaml next to
itself. The text comes from a class language: 50 tokens, ids 0 and 1 for [PAD] and [MASK] and 48 words in six classes
of eight. The class follows a first-order Markov chain with rows drawn from a Dirichlet(0.3) distribution, and the
word is uniform within its class, so the source's own probability of a word given both its neighbours is known. The
input holds 400 training and 200 test sequences of 32 words, the chain, the initial parameters of a bidirectional
window model (offsets -2 to 2, m = 8, h = 32), of a left-to-right one (offsets -2 and -1), of an ELECTRA generator
(h = 8) and discriminator, and the random draws of one batch: BERT's chosen positions, uniform numbers and random
tokens, T5 noise masks, and ten corrupted batches for SGD.

The expected values are the corrupted batch, the masked LM and discriminator losses and gradients, the ELECTRA losses,
samples and gradients, the T5 inputs and targets, the parameters after ten SGD steps, and the test cloze losses after
40 epochs of five training runs: 80/10/10 with dynamic masking, the same with static masking, [MASK] only, a 40%
masking rate, and the left-to-right model."""

import hashlib
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SEED = 2019
K, PER = 6, 8
V, M, HID, GEN_HID = 2 + K * PER, 8, 32, 8
BI, LR_ONLY = (-2, -1, 0, 1, 2), (-2, -1)
N_TRAIN, N_TEST, L = 400, 200, 32
EPOCHS, LR, BATCH, LAMBDA, STEPS = 40, 0.5, 16, 50.0, 10
RUNS = {
    "dynamic": {"offsets": BI},
    "static": {"offsets": BI, "dynamic": False},
    "mask_only": {"offsets": BI, "p_mask": 1.0, "p_random": 0.0},
    "rate40": {"offsets": BI, "rate": 0.4},
    "causal": {"offsets": LR_ONLY},
}


def load_reference():
    path = HERE.parent / "impl" / "numpy" / "masked_language_modelling.py"
    spec = importlib.util.spec_from_file_location("mlm_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(ml, train, test, name):
    cfg = dict(RUNS[name])
    model = ml.WindowMLM(V, M, HID, cfg.pop("offsets"), seed=SEED + 1)
    hist = ml.train_mlm(model, train, EPOCHS, LR, BATCH, np.random.default_rng(SEED + 2), **cfg)
    return np.array(hist), -float(model.cloze_logprobs(test).mean())


def main():
    ml = load_reference()
    rng = np.random.default_rng(SEED)
    trans = ml.class_chain(rng, K, PER, 0.3)
    train = ml.sample_sequences(rng, trans, PER, N_TRAIN, L)
    test = ml.sample_sequences(rng, trans, PER, N_TEST, L)
    ids = train[:4]
    sel = np.stack([ml.select_positions(row, 0.15, rng) for row in ids])
    u = rng.random(ids.shape)
    rnd = rng.integers(ml.FIRST, V, size=ids.shape)
    noise = np.stack([ml.random_spans_noise_mask(L, 0.15, 3.0, rng) for _ in ids])
    sentinels = np.arange(V, V + L, dtype=np.int64)
    steps = [ml.bert_mask(train[4 * i + 4 : 4 * i + 8], V, rng) for i in range(STEPS)]

    model = ml.WindowMLM(V, M, HID, BI, seed=SEED + 1)
    gen = ml.WindowMLM(V, M, GEN_HID, BI, seed=SEED + 3)
    disc = ml.WindowMLM(V, M, HID, BI, seed=SEED + 4)
    inp = {"trans": trans, "train": train, "test": test, "sel": sel, "u": u, "rnd": rnd, "noise": noise}
    inp |= {"sentinels": sentinels, "step_inp": np.stack([s[0] for s in steps])}
    inp |= {"step_pred": np.stack([s[1] for s in steps])}
    for prefix, mod in (("init", model), ("gen", gen), ("disc", disc)):
        inp |= {f"{prefix}_{k}": v.copy() for k, v in mod.p.items()}

    corrupt = ml.bert_corrupt(ids, sel, u, rnd)
    exp = {"corrupt": corrupt, "logits0": model.logits(corrupt)}
    loss, g = model.mlm_loss_and_grads(corrupt, ids, sel)
    exp |= {"mlm0": np.array(loss)} | {f"mlmgrad0_{k}": v for k, v in g.items()}
    loss, g = model.disc_loss_and_grads(corrupt, corrupt == ids, ids >= ml.FIRST)
    exp |= {"disc0": np.array(loss)} | {f"discgrad0_{k}": v for k, v in g.items()}
    el = ml.electra_losses(gen, disc, ids, sel, u, LAMBDA)
    exp |= {"electra_corrupt": el["corrupt"], "electra_gen": np.array(el["gen_loss"])}
    exp |= {"electra_disc": np.array(el["disc_loss"])}
    exp |= {f"electra_ggrad_{k}": v for k, v in el["gen_grads"].items()}
    exp |= {f"electra_dgrad_{k}": v for k, v in el["disc_grads"].items()}
    t5 = [ml.span_corrupt(row, nz, sentinels) for row, nz in zip(ids, noise, strict=True)]
    exp |= {"t5_inp": np.concatenate([a for a, _ in t5]), "t5_inp_len": np.array([len(a) for a, _ in t5])}
    exp |= {"t5_tgt": np.concatenate([b for _, b in t5]), "t5_tgt_len": np.array([len(b) for _, b in t5])}
    step_losses = []
    for i, (s_inp, s_pred) in enumerate(steps):
        loss, g = model.mlm_loss_and_grads(s_inp, train[4 * i + 4 : 4 * i + 8], s_pred)
        model.sgd(g, LR)
        step_losses.append(loss)
    exp |= {"step_loss": np.array(step_losses)} | {f"step_{k}": v for k, v in model.p.items()}

    cloze = {}
    for name in RUNS:
        hist, cloze[name] = run(ml, train, test, name)
        exp |= {f"run_{name}_hist": hist, f"run_{name}_cloze": np.array(cloze[name])}
    src_bi = -float(ml.source_cloze_logprobs(test, trans, PER).mean())
    src_lr = -float(ml.source_cloze_logprobs(test, trans, PER, causal=True).mean())
    exp |= {"source_cloze": np.array(src_bi), "source_causal": np.array(src_lr)}

    shas = {}
    for name, arrays in (("input-small.npz", inp), ("expected-small.npz", exp)):
        np.savez(HERE / name, **arrays)
        shas[name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    runs = ", ".join(f"{k}: {v:.4f}" for k, v in cloze.items())
    lines = [
        "# Written by fixtures/make.py; do not edit.",
        f"seed: {SEED}",
        f"source: {{vocabulary: {V}, classes: {K}, per_class: {PER}, dirichlet: 0.3, train: {N_TRAIN}, "
        f"test: {N_TEST}, length: {L}}}",
        f"model: {{V: {V}, m: {M}, h: {HID}, offsets: [-2, 2], causal_offsets: [-2, -1], generator_h: {GEN_HID}}}",
        f"training: {{epochs: {EPOCHS}, batch: {BATCH}, lr: {LR}, electra_lambda: {LAMBDA}, sgd_steps: {STEPS}}}",
        f"results: {{source_cloze_nats: {src_bi:.4f}, source_left_to_right_nats: {src_lr:.4f}, "
        f"uniform_nats: {np.log(V - 2):.4f}}}",
        f"runs_test_nats: {{{runs}}}",
        "dtype: float64",
        f'numpy: "{np.__version__}"',
        "files:",
    ]
    lines += [f"  {k}: {{sha256: {v}}}" for k, v in shas.items()]
    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
