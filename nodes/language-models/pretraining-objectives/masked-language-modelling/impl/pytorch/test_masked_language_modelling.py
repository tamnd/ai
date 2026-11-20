# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Tests of the PyTorch implementation: given the same random draws, it reproduces the NumPy reference stored in the
shared fixture (BERT's corruption, the masked LM and discriminator losses and their hand-written gradients, ELECTRA's
samples and losses, T5's span corruption and ten SGD steps), and the T5 example of Raffel et al."""

import pathlib

import numpy as np
import pytest
import torch

from masked_language_modelling import FIRST, WindowMLM, bert_corrupt, electra_losses, span_corrupt

torch.set_default_dtype(torch.float64)
HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
BI = (-2, -1, 0, 1, 2)


@pytest.fixture(scope="module")
def data():
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return inp, want


def model(inp, prefix, h):
    arrays = {k[len(prefix) + 1 :]: v for k, v in inp.items() if k.startswith(prefix + "_")}
    return WindowMLM(50, 8, h, BI).load(arrays)


def grads(m, loss):
    m.zero_grad()
    loss.backward()
    return {name: p.grad for name, p in m.named_parameters() if p.grad is not None}


def close(got, want, rtol=1e-10):
    torch.testing.assert_close(got.detach(), want, rtol=rtol, atol=1e-14)


def test_bert_corruption_and_losses(data):
    inp, want = data
    ids, sel = inp["train"][:4], inp["sel"]
    corrupt = bert_corrupt(ids, sel, inp["u"], inp["rnd"])
    assert torch.equal(corrupt, want["corrupt"])
    m = model(inp, "init", 32)
    close(m.logits(corrupt), want["logits0"])
    loss = m.mlm_loss(corrupt, ids, sel)
    close(loss, want["mlm0"])
    for k, g in grads(m, loss).items():
        close(g, want[f"mlmgrad0_{k}"])
    loss = m.disc_loss(corrupt, corrupt == ids, ids >= FIRST)
    close(loss, want["disc0"])
    for k, g in grads(m, loss).items():
        close(g, want[f"discgrad0_{k}"])


def test_electra(data):
    inp, want = data
    ids, sel = inp["train"][:4], inp["sel"]
    gen, disc = model(inp, "gen", 8), model(inp, "disc", 32)
    gen_loss, disc_loss, corrupt = electra_losses(gen, disc, ids, sel, inp["u"])
    assert torch.equal(corrupt, want["electra_corrupt"])
    close(gen_loss, want["electra_gen"])
    close(disc_loss, want["electra_disc"])
    total = gen_loss + 50.0 * disc_loss
    gen.zero_grad()
    disc.zero_grad()
    total.backward()
    for k, p in gen.named_parameters():
        if p.grad is not None:
            close(p.grad, want[f"electra_ggrad_{k}"])
    for k, p in disc.named_parameters():
        if p.grad is not None:
            close(p.grad, want[f"electra_dgrad_{k}"])


def test_t5_span_corruption(data):
    inp, want = data
    starts_i = torch.cumsum(want["t5_inp_len"], 0) - want["t5_inp_len"]
    starts_t = torch.cumsum(want["t5_tgt_len"], 0) - want["t5_tgt_len"]
    for r in range(4):
        x, y = span_corrupt(inp["train"][r], inp["noise"][r], inp["sentinels"])
        assert torch.equal(x, want["t5_inp"][starts_i[r] : starts_i[r] + want["t5_inp_len"][r]])
        assert torch.equal(y, want["t5_tgt"][starts_t[r] : starts_t[r] + want["t5_tgt_len"][r]])


def test_t5_example():
    words = "Thank you for inviting me to your party last week .".split()
    vocab = {w: i + 2 for i, w in enumerate(words)}
    ids = torch.tensor([vocab[w] for w in words])
    noise = torch.tensor([w in ("for", "inviting", "last") for w in words])
    inp, tgt = span_corrupt(ids, noise, torch.tensor([100, 101, 102]))
    inv = {i: w for w, i in vocab.items()} | {100: "<X>", 101: "<Y>", 102: "<Z>"}
    assert " ".join(inv[int(i)] for i in inp) == "Thank you <X> me to your party <Y> week ."
    assert " ".join(inv[int(i)] for i in tgt) == "<X> for inviting <Y> last <Z>"


def test_sgd_steps(data):
    inp, want = data
    m = model(inp, "init", 32)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for i in range(10):
        loss = m.mlm_loss(inp["step_inp"][i], inp["train"][4 * i + 4 : 4 * i + 8], inp["step_pred"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        close(loss, want["step_loss"][i], rtol=1e-12)
    for k, p in m.named_parameters():
        close(p, want[f"step_{k}"])


def test_cloze_is_one_mask_at_a_time(data):
    inp, _ = data
    m = model(inp, "init", 32)
    ids = inp["test"][:2, :6]
    got = m.cloze_logprobs(ids)
    for r in range(2):
        for t in range(6):
            one = ids.clone()
            one[r, t] = 1
            close(got[r, t], torch.log_softmax(m.logits(one)[r, t], -1)[ids[r, t]], rtol=1e-12)
