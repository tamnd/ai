# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Tests of the PyTorch implementation: autograd reproduces the hand-written gradients of the NumPy reference stored
in the shared fixture, for the untied and tied models, noise-contrastive estimation, two epochs of SGD and the
Huffman-tree softmax; the worked example; the parameter count; and the tree softmax sums to one."""

import math
import pathlib

import numpy as np
import pytest
import torch

from neural_probabilistic_language_model import NPLM, HierarchicalSoftmax, huffman

torch.set_default_dtype(torch.float64)
HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
V, M, HID, N, DECAY = 61, 8, 16, 3, 1e-4


@pytest.fixture(scope="module")
def data():
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return inp, want


def windows(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.cat([torch.zeros(N - 1, dtype=torch.long), ids.long()])
    idx = torch.arange(len(ids))[:, None] + torch.arange(N - 1)[None, :]
    return padded[idx], padded[N - 1 :]


def model(inp, prefix, h, tie):
    arrays = {k[len(prefix) + 1 :]: v for k, v in inp.items() if k.startswith(prefix + "_")}
    return NPLM(V, M, h, N, direct=True, tie=tie).load(arrays)


def close(got, want, rtol=1e-12):
    torch.testing.assert_close(got.detach(), want, rtol=rtol, atol=1e-14)


def grads(m, loss):
    m.zero_grad()
    loss.backward()
    return {name: p.grad for name, p in m.named_parameters() if p.grad is not None}


def test_shared_fixture(data):
    inp, want = data
    ctx, tgt = windows(inp["train"])
    cte, tte = windows(inp["test"])
    m = model(inp, "init", HID, False)
    close(m(cte[:32]), want["logits0"])
    loss = m.loss(ctx[:32], tgt[:32], DECAY)
    close(loss, want["loss0"])
    for k, g in grads(m, loss).items():
        close(g, want[f"grad0_{k}"])
    loss = m.nce_loss(ctx[:32], tgt[:32], inp["noise"].long(), inp["log_pn"], DECAY)
    close(loss, want["nce0"])
    for k, g in grads(m, loss).items():
        close(g, want[f"ncegrad0_{k}"])
    t = model(inp, "tinit", M, True)
    close(t(cte[:32]), want["tlogits0"])
    loss = t.loss(ctx[:32], tgt[:32], DECAY)
    close(loss, want["tloss0"])
    got = grads(t, loss)
    assert set(got) == {"C", "H", "d", "b", "W"}
    for k, g in got.items():
        close(g, want[f"tgrad0_{k}"])
    hist = m.fit(ctx, tgt, list(inp["orders"].long()), 0.3, 1e-3, DECAY, 16)
    close(torch.tensor(hist), want["train_loss"], rtol=1e-10)
    for name, p in m.named_parameters():
        close(p, want[f"final_{name}"], rtol=1e-9)
    close(m.token_logprobs(cte, tte), want["final_test_logprob"], rtol=1e-9)
    hs = HierarchicalSoftmax(inp["hs_counts"].tolist(), HID)
    with torch.no_grad():
        hs.v.copy_(inp["hs_v"])
        hs.c.copy_(inp["hs_c"])
    a = inp["hs_a"].clone().requires_grad_(True)
    close(hs.all_log_probs(a), want["hs_all_logprob"])
    loss = -hs.log_prob(a, inp["hs_tgt"].long()).mean()
    close(loss, want["hs_loss"])
    loss.backward()
    assert a.grad is not None and hs.v.grad is not None and hs.c.grad is not None
    close(a.grad, want["hs_grad_a"])
    close(hs.v.grad, want["hs_grad_v"])
    close(hs.c.grad, want["hs_grad_c"])


def test_worked_example():
    m = NPLM(3, 2, 1, 2, direct=False)
    ln2, ln3, ln4 = math.log(2), math.log(3), math.log(4)
    m.load(
        {
            "C": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "H": torch.tensor([[ln2, 0.0]]),
            "d": torch.zeros(1),
            "U": torch.tensor([[0.0], [5 / 3 * ln4], [5 / 3 * ln3]]),
            "b": torch.zeros(3),
        }
    )
    ctx, tgt = torch.tensor([[0]]), torch.tensor([1])
    close(torch.softmax(m(ctx), -1)[0], torch.tensor([1 / 8, 1 / 2, 3 / 8]), rtol=1e-14)
    loss = m.loss(ctx, tgt)
    assert loss.item() == pytest.approx(ln2, rel=1e-14)
    g = grads(m, loss)
    g_a = -(1 / 2) * 5 / 3 * ln4 + (3 / 8) * 5 / 3 * ln3
    close(g["b"], torch.tensor([1 / 8, -1 / 2, 3 / 8]), rtol=1e-14)
    assert float(g["d"][0]) == pytest.approx(0.64 * g_a, rel=1e-13)
    assert sum(p.numel() for p in m.parameters()) == 15


def test_parameter_count():
    for V_, m_, h_, n_ in ((17964, 100, 60, 6), (16383, 30, 100, 5), (7, 3, 2, 2)):
        want = V_ * (1 + n_ * m_ + h_) + h_ * (1 + (n_ - 1) * m_)
        assert sum(p.numel() for p in NPLM(V_, m_, h_, n_).parameters()) == want
        no_w = NPLM(V_, m_, h_, n_, direct=False)
        assert sum(p.numel() for p in no_w.parameters()) == want - V_ * (n_ - 1) * m_
    with pytest.raises(ValueError):
        NPLM(9, 5, 6, 3, tie=True)


def test_tree_softmax():
    gen = torch.Generator().manual_seed(0)
    counts = torch.randint(1, 100, (50,), generator=gen).double()
    hs = HierarchicalSoftmax(counts.tolist(), 5, seed=1)
    a = torch.randn(7, 5, generator=gen)
    lp = hs.all_log_probs(a)
    close(lp.exp().sum(dim=1), torch.ones(7))
    tgt = torch.randint(0, 50, (7,), generator=gen)
    close(hs.log_prob(a, tgt), lp[torch.arange(7), tgt])
    _, _, mask = huffman([1.0] * 33)
    depth = mask.sum(dim=1)
    assert int(depth.min()) == 5 and int(depth.max()) == 6
    q = counts / counts.sum()
    entropy = float(-(q * q.log2()).sum())
    mean_depth = float((q * hs.mask.sum(dim=1)).sum())
    assert entropy <= mean_depth < entropy + 1
