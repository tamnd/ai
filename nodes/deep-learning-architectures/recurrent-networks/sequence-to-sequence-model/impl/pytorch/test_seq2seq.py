# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sequence-to-sequence-model/ (ai://N149H2).
"""Tests of the PyTorch version: the shared fixture, torch.nn.LSTM, the gradient, scheduled sampling and training."""

import json
import pathlib

import numpy as np
import pytest
import torch

from seq2seq import EOS, Seq2Seq, config, decoder_inputs, nll, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("src_vocab", "tgt_vocab", "embed_size", "hidden_size", "n_layers")


def cfg_of(c, reverse=True):
    return config(*(c["config"][k] for k in KEYS), reverse=reverse)


def sentences(n, length, vocab, generator=None):
    out = torch.randint(1, vocab, (n, length), generator=generator)
    out[:, -1] = EOS
    return out


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


def test_shared_fixture():
    params, inp, want = fixture()
    model = Seq2Seq.from_params(params, config(reverse=True))
    logits = model(inp["src"], inp["tgt"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    h, c = model.encode(inp["src"])
    torch.testing.assert_close(h, want["v_h"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(c, want["v_c"], rtol=1e-12, atol=1e-13)
    fwd = Seq2Seq.from_params(params, config(reverse=False))
    torch.testing.assert_close(fwd(inp["src"], inp["tgt"]), want["logits_forward"], rtol=1e-12, atol=1e-13)
    greedy = model.greedy(inp["src"], 6)
    for i in range(2):
        n = int((want["greedy"][i] >= 0).sum())
        assert greedy[i, :n].tolist() == want["greedy"][i, :n].tolist()


def test_shared_fixture_gradients():
    # Autograd of eq-loss against the backward pass of the NumPy reference.
    params, inp, want = fixture()
    model = Seq2Seq.from_params(params, config(reverse=True))
    loss = nll(model(inp["src"], inp["tgt"]), inp["tgt"])
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=0)
    loss.backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        model = Seq2Seq(cfg)
        assert sum(p.numel() for p in model.parameters()) == c["model"]["params"]
        k = c["config"]
        logits = model(sentences(k["B"], k["S"], k["src_vocab"]), sentences(k["B"], k["T"], k["tgt_vocab"]))
        assert list(logits.shape) == c["blocks"]["softmax"]["outputs"]["logits"]


def test_stacks_match_torch_nn_lstm():
    torch.manual_seed(0)
    cfg = config(9, 8, 5, 6, 3, reverse=True)
    model = Seq2Seq(cfg, scale=0.5)
    src, tgt = sentences(4, 7, 9), sentences(4, 5, 8)
    h, c = model.encode(src)
    enc = model.to_torch_lstm("enc")
    _, (h_lib, c_lib) = enc(model.src_embed[src.flip(1)])
    torch.testing.assert_close(h, h_lib, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(c, c_lib, rtol=1e-12, atol=1e-13)
    dec = model.to_torch_lstm("dec")
    top, _ = dec(model.tgt_embed[decoder_inputs(tgt)], (h, c))
    torch.testing.assert_close(model(src, tgt), top @ model.out_W + model.out_b, rtol=1e-12, atol=1e-13)


def test_gradcheck():
    torch.manual_seed(1)
    cfg = config(5, 4, 2, 3, 2)
    model = Seq2Seq(cfg, scale=0.5)
    names = [n for n, _ in model.named_parameters()]
    ps = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())
    src, tgt = sentences(2, 3, 5), sentences(2, 3, 4)

    def f(*q):
        return nll(torch.func.functional_call(model, dict(zip(names, q, strict=True)), (src, tgt)), tgt)

    assert torch.autograd.gradcheck(f, ps)


def test_scheduled_sampling_limits():
    # eps = 1 is teacher forcing; eps = 0 with the most likely token is the free-running greedy decoder.
    torch.manual_seed(2)
    model = Seq2Seq(config(), scale=0.5)
    src, tgt = sentences(3, 5, 7), sentences(3, 6, 6)
    g = torch.Generator().manual_seed(0)
    torch.testing.assert_close(model.forward_scheduled(src, tgt, 1.0, g), model(src, tgt))
    free = model.forward_scheduled(src, tgt, 0.0, g, sample=False)
    greedy = model.greedy(src, tgt.shape[1])
    assert torch.equal(free.argmax(-1)[:, :1], greedy[:, :1])
    # Fed its own greedy tokens, the decoder computes the logits of forward() on those tokens.
    own = torch.cat([torch.full_like(tgt[:, :1], EOS), free.argmax(-1)[:, :-1]], dim=1)
    torch.testing.assert_close(free, model(src, tgt, dec_in=own))


def test_scheduled_sampling_coin_rate():
    # With a model that always puts all its mass on one token that the target never uses, the decoder input at a
    # step is the true token exactly when the coin said so; the fraction is close to eps.
    torch.manual_seed(3)
    model = Seq2Seq(config(), scale=0.5)
    with torch.no_grad():
        model.out_W.zero_()
        model.out_b.copy_(torch.tensor([-50.0, -50, -50, -50, -50, 50]))  # the model's own token is always 5
    src, tgt = sentences(2000, 5, 7), sentences(2000, 4, 5)  # targets use 1..4 and <EOS>
    seen = []
    original = model.step

    def spy(prefix, x, state):
        if prefix == "dec":
            seen.append(x)
        return original(prefix, x, state)

    model.step = spy
    model.forward_scheduled(src, tgt, 0.3, torch.Generator().manual_seed(4))
    own = model.tgt_embed[5]
    fractions = [1.0 - torch.isclose(x, own).all(-1).double().mean().item() for x in seen[1:]]
    assert all(abs(f - 0.3) < 0.04 for f in fractions), fractions


def test_learns_to_reverse_a_sequence():
    # Read 5 tokens and write them back in reverse order; v must carry the whole sequence.
    torch.manual_seed(0)
    cfg = config(8, 8, 16, 64, 1, reverse=True)
    model = Seq2Seq(cfg, scale=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    def batch(n):
        x = torch.randint(1, 8, (n, 5))
        src = torch.cat([x, torch.zeros(n, 1, dtype=torch.long)], dim=1)
        return src, torch.cat([x.flip(1), torch.zeros(n, 1, dtype=torch.long)], dim=1)

    for _ in range(400):
        src, tgt = batch(64)
        loss = nll(model(src, tgt), tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # the threshold of Sutskever et al.
        opt.step()
    with torch.no_grad():
        src, tgt = batch(500)
        out = model.greedy(src, 6)
    acc = (out == tgt).all(1).double().mean().item()
    assert acc > 0.9, acc
