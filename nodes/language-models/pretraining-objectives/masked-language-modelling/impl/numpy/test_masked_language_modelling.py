# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Tests of the NumPy reference: the corruption rates of BERT, SpanBERT and T5, the T5 example of Raffel et al.,
the hand-written gradients against finite differences, the one-pass cloze scores, the source's exact conditionals,
ELECTRA's labels, the independence gap, and the shared fixture with its five training runs."""

import itertools
import math
import pathlib

import numpy as np
import pytest

import masked_language_modelling as ml

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())


def test_bert_selects_fifteen_percent_of_ordinary_tokens():
    rng = np.random.default_rng(0)
    ids = np.array([ml.PAD] * 3 + list(range(2, 102)) + [ml.PAD] * 5)
    for _ in range(20):
        sel = ml.select_positions(ids, 0.15, rng)
        assert sel.sum() == 15
        assert np.all(ids[sel] >= ml.FIRST)


def test_bert_corruption_rates():
    rng = np.random.default_rng(1)
    n, V = 400_000, 1002
    ids = rng.integers(ml.FIRST, V, size=n)
    sel = rng.random(n) < 0.15
    out = ml.bert_corrupt(ids, sel, rng.random(n), rng.integers(ml.FIRST, V, size=n))
    assert np.all(out[~sel] == ids[~sel])
    masked, changed = np.mean(out == ml.MASK), np.mean((out != ids) & (out != ml.MASK))
    unchanged_sel = np.mean(sel & (out == ids))
    assert masked == pytest.approx(0.15 * 0.8, abs=2e-3)
    assert changed == pytest.approx(0.15 * 0.1 * (1 - 1 / (V - 2)), abs=1e-3)
    assert unchanged_sel == pytest.approx(0.15 * 0.1 * (1 + 1 / (V - 2)), abs=1e-3)


def test_truncated_geometric_mean_is_spanbert_value():
    pmf = ml.truncated_geometric(0.2, 10)
    mean = float(np.arange(1, 11) @ pmf)
    assert round(mean, 1) == 3.8
    capped = sum(0.8**k for k in range(10))  # E[min(l, 10)] for an untruncated geometric
    assert capped == pytest.approx(4.4631, abs=1e-4)


def test_span_mask_budget_and_spans():
    rng = np.random.default_rng(2)
    lengths = []
    for _ in range(300):
        sel, ls = ml.span_mask(512, rng)
        assert sel.sum() >= round(0.15 * 512)
        assert sel.sum() - ls[-1] < round(0.15 * 512)  # the last span crossed the budget
        assert ml.run_count(sel) == len(ls)  # spans never touch, so each is its own run
        assert max(ls) <= 10
        lengths += ls
    # skipping spans that touch earlier ones skips long spans more often, so the kept spans are a little shorter
    assert 3.5 < np.mean(lengths) < 3.78


def test_t5_example():
    words = "Thank you for inviting me to your party last week .".split()
    vocab = {w: i + 2 for i, w in enumerate(words)}
    ids = np.array([vocab[w] for w in words])
    noise = np.isin(words, ["for", "inviting", "last"])
    X, Y, Z = 100, 101, 102
    inp, tgt = ml.span_corrupt(ids, noise, np.array([X, Y, Z]))
    inv = {i: w for w, i in vocab.items()} | {X: "<X>", Y: "<Y>", Z: "<Z>"}
    assert " ".join(inv[int(i)] for i in inp) == "Thank you <X> me to your party <Y> week ."
    assert " ".join(inv[int(i)] for i in tgt) == "<X> for inviting <Y> last <Z>"


@pytest.mark.parametrize("n,rate,mean_span", [(500, 0.15, 3.0), (512, 0.15, 3.0), (64, 0.5, 2.0), (100, 0.1, 1.0)])
def test_t5_noise_mask_counts_and_lengths(n, rate, mean_span):
    rng = np.random.default_rng(3)
    n_noise = round(rate * n)
    n_spans = round(n_noise / mean_span)
    for _ in range(50):
        noise = ml.random_spans_noise_mask(n, rate, mean_span, rng)
        assert noise.sum() == n_noise and ml.run_count(noise) == n_spans and not noise[0]
        inp, tgt = ml.span_corrupt(np.arange(2, n + 2), noise, np.arange(10_000, 10_000 + n_spans + 1))
        assert len(inp) == n - n_noise + n_spans
        assert len(tgt) == n_noise + n_spans + 1


def small_model(offsets=(-1, 0, 1), seed=4):
    return ml.WindowMLM(9, 3, 5, offsets, seed=seed, scale=0.8)


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        up = f()
        x[i] = old - eps
        down = f()
        x[i] = old
        g[i] = (up - down) / (2 * eps)
    return g


@pytest.mark.parametrize("offsets", [(-1, 0, 1), (-2, -1)])
def test_mlm_gradients(offsets):
    rng = np.random.default_rng(5)
    model = small_model(offsets)
    ids = rng.integers(ml.FIRST, 9, size=(3, 6))
    inp, pred = ml.bert_mask(ids, 9, rng, rate=0.4)
    loss, g = model.mlm_loss_and_grads(inp, ids, pred, decay=1e-2)
    for k in ("E", "H", "d", "U", "b"):
        num = numeric_grad(lambda: model.mlm_loss_and_grads(inp, ids, pred, decay=1e-2)[0], model.p[k])
        np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-9)
    assert loss > 0


def test_disc_gradients():
    rng = np.random.default_rng(6)
    model = small_model()
    ids = rng.integers(ml.FIRST, 9, size=(3, 6))
    corrupt = np.where(rng.random(ids.shape) < 0.3, rng.integers(ml.FIRST, 9, size=ids.shape), ids)
    valid = np.ones(ids.shape, dtype=bool)
    _, g = model.disc_loss_and_grads(corrupt, corrupt == ids, valid)
    for k in ("E", "H", "d", "g", "c"):
        num = numeric_grad(lambda: model.disc_loss_and_grads(corrupt, corrupt == ids, valid)[0], model.p[k])
        np.testing.assert_allclose(g[k], num, rtol=1e-6, atol=1e-9)


def test_unchosen_positions_carry_no_loss():
    rng = np.random.default_rng(7)
    model = small_model()
    ids = rng.integers(ml.FIRST, 9, size=(2, 8))
    inp, pred = ml.bert_mask(ids, 9, rng)
    other = np.where(pred, ids, rng.integers(ml.FIRST, 9, size=ids.shape))
    assert model.mlm_loss_and_grads(inp, ids, pred)[0] == model.mlm_loss_and_grads(inp, other, pred)[0]


def test_cloze_is_one_mask_at_a_time():
    rng = np.random.default_rng(8)
    model = small_model()
    ids = rng.integers(ml.FIRST, 9, size=(2, 7))
    got = model.cloze_logprobs(ids)
    for r, t in itertools.product(range(2), range(7)):
        one = ids.copy()
        one[r, t] = ml.MASK
        lg = model.logits(one)[r, t]
        want = lg[ids[r, t]] - np.log(np.exp(lg).sum())
        assert got[r, t] == pytest.approx(want, abs=1e-12)


def test_source_cloze_against_enumeration():
    rng = np.random.default_rng(9)
    K, per = 3, 2
    trans = ml.class_chain(rng, K, per, 1.0)
    seqs = ml.sample_sequences(rng, trans, per, 5, 4)

    def joint(seq):
        c, p = K, 1.0
        for x in seq:
            nxt = (x - ml.FIRST) // per
            p *= trans[c, nxt] / per
            c = nxt
        return p

    words = range(ml.FIRST, ml.FIRST + K * per)
    got = ml.source_cloze_logprobs(seqs, trans, per)
    got_lr = ml.source_cloze_logprobs(seqs, trans, per, causal=True)
    for r, t in itertools.product(range(5), range(4)):
        s = seqs[r].copy()
        total = 0.0
        for w in words:
            s[t] = w
            total += joint(s)
        assert got[r, t] == pytest.approx(math.log(joint(seqs[r]) / total), abs=1e-12)
        prefix = [joint(list(seqs[r, : t + 1])), sum(joint([*seqs[r, :t], w]) for w in words)]
        assert got_lr[r, t] == pytest.approx(math.log(prefix[0] / prefix[1]), abs=1e-12)


def test_electra_labels_and_no_gradient_through_samples():
    rng = np.random.default_rng(10)
    gen, disc = small_model(seed=11), small_model(seed=12)
    ids = rng.integers(ml.FIRST, 9, size=(4, 8))
    sel = np.stack([ml.select_positions(row, 0.3, rng) for row in ids])
    out = ml.electra_losses(gen, disc, ids, sel, rng.random(ids.shape), lam=50.0)
    assert np.all(out["corrupt"][~sel] == ids[~sel])
    # a sample that happens to equal the original counts as original
    masked = np.where(sel, ml.MASK, ids)
    _, g_alone = gen.mlm_loss_and_grads(masked, ids, sel)
    for k, v in g_alone.items():
        np.testing.assert_array_equal(out["gen_grads"][k], v)
    loss, g = disc.disc_loss_and_grads(out["corrupt"], out["corrupt"] == ids, ids >= ml.FIRST)
    assert out["disc_loss"] == loss
    np.testing.assert_allclose(out["disc_grads"]["g"], 50.0 * g["g"])


def test_sample_tokens_follows_the_softmax():
    rng = np.random.default_rng(13)
    logits = np.array([0.0, 0.0, 1.0, 2.0, -1.0])
    n = 200_000
    s = ml.sample_tokens(np.tile(logits, (n, 1)), rng.random(n))
    p = ml.softmax(logits[ml.FIRST :])
    freq = np.bincount(s - ml.FIRST, minlength=3) / n
    np.testing.assert_allclose(freq, p, atol=4e-3)


def test_independence_gap_new_york():
    # XLNet's example: the two masked tokens of "[MASK] [MASK] is a city" are (New, York) or (Los, Angeles)
    joint = np.array([[0.5, 0.0], [0.0, 0.5]])
    mlm, h, gap = ml.independence_gap(joint)
    assert mlm == pytest.approx(2 * math.log(2)) and h == pytest.approx(math.log(2))
    assert gap == pytest.approx(math.log(2))
    assert ml.independence_gap(np.outer([0.3, 0.7], [0.6, 0.4]))[2] == pytest.approx(0.0, abs=1e-15)


@pytest.fixture(scope="module")
def data():
    return dict(np.load(FIXTURES / "input-small.npz")), dict(np.load(FIXTURES / "expected-small.npz"))


def load(arrays, prefix, h, offsets=(-2, -1, 0, 1, 2)):
    model = ml.WindowMLM(50, 8, h, offsets)
    model.p = {k: arrays[f"{prefix}_{k}"].copy() for k in model.p}
    return model


def test_shared_fixture(data):
    inp, want = data
    ids, sel = inp["train"][:4], inp["sel"]
    corrupt = ml.bert_corrupt(ids, sel, inp["u"], inp["rnd"])
    np.testing.assert_array_equal(corrupt, want["corrupt"])
    model = load(inp, "init", 32)
    np.testing.assert_allclose(model.logits(corrupt), want["logits0"], rtol=1e-12)
    loss, g = model.mlm_loss_and_grads(corrupt, ids, sel)
    assert loss == pytest.approx(float(want["mlm0"]), rel=1e-12)
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"mlmgrad0_{k}"], rtol=1e-10, atol=1e-14)
    el = ml.electra_losses(load(inp, "gen", 8), load(inp, "disc", 32), ids, sel, inp["u"], 50.0)
    np.testing.assert_array_equal(el["corrupt"], want["electra_corrupt"])
    assert el["disc_loss"] == pytest.approx(float(want["electra_disc"]), rel=1e-12)
    for i in range(10):
        loss, g = model.mlm_loss_and_grads(inp["step_inp"][i], inp["train"][4 * i + 4 : 4 * i + 8], inp["step_pred"][i])
        model.sgd(g, 0.5)
        assert loss == pytest.approx(float(want["step_loss"][i]), rel=1e-12)
    for k, v in model.p.items():
        np.testing.assert_allclose(v, want[f"step_{k}"], rtol=1e-10, atol=1e-14)


def test_training_runs(data):
    """The five runs of the worked example, reproduced, and what they show."""
    inp, want = data
    train, test = inp["train"], inp["test"]
    runs = {
        "dynamic": {},
        "static": {"dynamic": False},
        "mask_only": {"p_mask": 1.0, "p_random": 0.0},
        "rate40": {"rate": 0.4},
    }
    got = {}
    for name, cfg in [*runs.items(), ("causal", None)]:
        offsets = (-2, -1) if cfg is None else (-2, -1, 0, 1, 2)
        model = ml.WindowMLM(50, 8, 32, offsets, seed=2020)
        hist = ml.train_mlm(model, train, 40, 0.5, 16, np.random.default_rng(2021), **(cfg or {}))
        np.testing.assert_allclose(hist, want[f"run_{name}_hist"], rtol=1e-9)
        got[name] = -float(model.cloze_logprobs(test).mean())
        assert got[name] == pytest.approx(float(want[f"run_{name}_cloze"]), rel=1e-9)
    src, src_lr = float(want["source_cloze"]), float(want["source_causal"])
    assert src < min(got[k] for k in runs) and src_lr < got["causal"]
    assert max(got[k] for k in runs if k != "static") < src_lr  # both sides beat the best left-only predictor
    assert got["static"] > got["dynamic"] + 0.3  # one fixed mask per sequence overfits this small corpus
    assert want["run_static_hist"][-1] < want["run_dynamic_hist"][-1]
