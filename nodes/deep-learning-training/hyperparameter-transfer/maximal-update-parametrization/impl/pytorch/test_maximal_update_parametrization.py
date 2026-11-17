# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/maximal-update-parametrization/ (ai://NEW-maximal-update-parametrization).
"""Tests of the PyTorch reference: the worked example of the page, the base-width agreement of muP and SP, the 1/d
attention scale, the initialization and the per-layer learning rates handed to torch.optim, one SGD and one Adam
step against a manual update, and the feature-update size that muP keeps constant across widths and SP does not."""

import math

import numpy as np
import pytest
import torch

from maximal_update_parametrization import (
    MuMLP,
    attention_logits,
    feature_change,
    layer_hparams,
    mse,
)

torch.set_default_dtype(torch.float64)


def test_worked_example():
    hp = layer_hparams(4096, 256, 0.01, 32, optimizer="adam")
    assert hp["input"] == {
        "std": pytest.approx(1 / math.sqrt(32)),
        "lr": pytest.approx(0.01),
    }
    assert hp["hidden"] == {
        "std": pytest.approx(0.015625),
        "lr": pytest.approx(0.000625),
    }
    assert hp["output"] == {
        "std": pytest.approx(0.00390625),
        "lr": pytest.approx(0.000625),
    }
    assert hp["bias"] == {"std": 0.0, "lr": pytest.approx(0.01)}
    sgd = layer_hparams(4096, 256, 0.01, 32, optimizer="sgd")
    assert [sgd[k]["lr"] for k in ("input", "hidden", "output", "bias")] == pytest.approx([0.16, 0.01, 0.000625, 0.16])
    sp = layer_hparams(4096, 256, 0.01, 32, optimizer="adam", param="sp")
    assert [sp[k]["lr"] for k in ("input", "hidden", "output", "bias")] == [0.01] * 4
    assert sp["output"]["std"] == pytest.approx(0.015625)


def test_mup_equals_sp_at_the_base_width():
    for opt in ("sgd", "adam"):
        assert layer_hparams(64, 64, 0.03, 5, 0.7, opt, "mup") == layer_hparams(64, 64, 0.03, 5, 0.7, opt, "sp")


def test_width_exponents():
    for opt, exps in (
        ("sgd", {"input": 1, "hidden": 0, "output": -1, "bias": 1}),
        ("adam", {"input": 0, "hidden": -1, "output": -1, "bias": 0}),
    ):
        a, b = (
            layer_hparams(300, 30, 0.1, 4, optimizer=opt),
            layer_hparams(1200, 30, 0.1, 4, optimizer=opt),
        )
        for k, e in exps.items():
            assert b[k]["lr"] / a[k]["lr"] == pytest.approx(4.0**e)
        assert b["output"]["std"] / a["output"]["std"] == pytest.approx(0.25)
        assert b["hidden"]["std"] / a["hidden"]["std"] == pytest.approx(0.5)
        assert b["input"]["std"] == a["input"]["std"]


def test_invalid_arguments():
    for args in ((0, 8, 0.1, 3), (8, 0, 0.1, 3), (8, 8, 0.1, 0)):
        with pytest.raises(ValueError):
            layer_hparams(*args)
    with pytest.raises(ValueError):
        layer_hparams(8, 8, 0.1, 3, optimizer="lion")
    with pytest.raises(ValueError):
        layer_hparams(8, 8, 0.1, 3, param="ntk")
    with pytest.raises(ValueError):
        attention_logits(torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(ValueError):
        attention_logits(torch.ones(2, 3), torch.ones(2, 3), param="ntk")


def test_attention_logits():
    g = torch.Generator().manual_seed(1)
    q, k = torch.randn(3, 5, 256, generator=g), torch.randn(3, 7, 256, generator=g)
    raw = q @ k.transpose(-1, -2)
    assert torch.allclose(attention_logits(q, k, base_d_head=64, alpha=2.0), raw * 2.0 * 8 / 256)
    assert torch.allclose(attention_logits(q, k, param="sp", alpha=0.5), raw * 0.5 / 16)
    assert torch.allclose(attention_logits(q, k), attention_logits(q, k, param="sp"))
    assert attention_logits(q, k).shape == (3, 5, 7)


def test_init_and_param_groups():
    net = MuMLP(6, 512, 3, 64, 0.01, sigma=1.5, optimizer="sgd", seed=2)
    hp = net.hp
    assert net.W.std().item() == pytest.approx(hp["hidden"]["std"], rel=0.02)
    assert net.V.std().item() == pytest.approx(1.5 / math.sqrt(512 * 8), rel=0.05)
    assert net.U.std().item() == pytest.approx(1.5 / math.sqrt(6), rel=0.1)
    assert torch.all(net.b1 == 0) and torch.all(net.b2 == 0)
    groups = {g["name"]: g["lr"] for g in net.make_optimizer().param_groups}
    assert groups == {
        "U": pytest.approx(0.08),
        "b1": pytest.approx(0.08),
        "W": pytest.approx(0.01),
        "b2": pytest.approx(0.08),
        "V": pytest.approx(0.01 / 8),
    }
    assert isinstance(net.make_optimizer(), torch.optim.SGD)
    assert isinstance(MuMLP(6, 64, 3, 64, 0.01).make_optimizer(), torch.optim.Adam)


def test_forward():
    net = MuMLP(4, 16, 2, 8, 0.01, seed=3)
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        net.b1.fill_(0.1)
        net.b2.fill_(-0.2)
    h2 = torch.tanh(x @ net.U.T + 0.1) @ net.W.T - 0.2
    f, h = net(x, return_features=True)
    assert torch.allclose(h, h2)
    assert torch.allclose(f, torch.tanh(h2) @ net.V.T)
    assert torch.equal(net(x), f)
    y = torch.ones(5, 2)
    assert mse(f, y).item() == pytest.approx(0.5 * ((f - y) ** 2).sum().item() / 5)


def _data(seed, n=8, d_in=4, d_out=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d_in, generator=g), torch.randn(n, d_out, generator=g)


def _grads(net, X, y):
    net.zero_grad()
    mse(net(X), y).backward()
    return {n: p.grad.clone() for n, p in net.named_parameters()}


def test_sgd_step_matches_manual_update():
    X, y = _data(5)
    net = MuMLP(4, 32, 2, 8, 0.05, optimizer="sgd", seed=6)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    g = _grads(net, X, y)
    opt = net.make_optimizer()
    opt.step()
    for n, p in net.named_parameters():
        lr = net.hp[MuMLP.KIND[n]]["lr"]
        assert torch.allclose(p.detach(), before[n] - lr * g[n])


def test_adam_steps_match_manual_update():
    X, y = _data(7)
    net = MuMLP(4, 32, 2, 8, 0.05, optimizer="adam", seed=8)
    opt = net.make_optimizer(betas=(0.8, 0.9), eps=1e-10)
    p0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    m = {n: torch.zeros_like(p) for n, p in p0.items()}
    v = {n: torch.zeros_like(p) for n, p in p0.items()}
    for t in (1, 2):
        g = _grads(net, X, y)
        opt.step()
        for n, p in net.named_parameters():
            m[n] = 0.8 * m[n] + 0.2 * g[n]
            v[n] = 0.9 * v[n] + 0.1 * g[n] ** 2
            upd = (m[n] / (1 - 0.8**t)) / ((v[n] / (1 - 0.9**t)).sqrt() + 1e-10)
            p0[n] = p0[n] - net.hp[MuMLP.KIND[n]]["lr"] * upd
            assert torch.allclose(p.detach(), p0[n])


def test_feature_change_is_the_rms_change():
    X, y = _data(9)
    net = MuMLP(4, 32, 2, 8, 0.05, optimizer="sgd", seed=10)
    with torch.no_grad():
        f0, h0 = net(X, return_features=True)
    dh, df = feature_change(net, X, y)
    with torch.no_grad():
        f1, h1 = net(X, return_features=True)
    assert dh == pytest.approx(((h1 - h0) ** 2).mean().sqrt().item())
    assert df == pytest.approx(((f1 - f0) ** 2).mean().sqrt().item())
    assert dh > 0 and df > 0


def _sizes(param, optimizer, lr, widths=(64, 256, 1024)):
    X = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))
    y = torch.ones(8, 1)
    rows = [
        np.mean(
            [
                feature_change(
                    MuMLP(16, n, 1, 64, lr, optimizer=optimizer, param=param, seed=s),
                    X,
                    y,
                )
                for s in range(3)
            ],
            axis=0,
        )
        for n in widths
    ]
    return np.array(rows)


@pytest.mark.parametrize(("optimizer", "lr"), [("sgd", 0.05), ("adam", 0.003)])
def test_mup_keeps_the_feature_update_constant_and_sp_does_not(optimizer, lr):
    mup = _sizes("mup", optimizer, lr)
    sp = _sizes("sp", optimizer, lr)
    np.testing.assert_allclose(mup[0], sp[0])  # identical at the base width
    assert 0.7 < mup[-1, 0] / mup[0, 0] < 1.4
    assert 0.7 < mup[-1, 1] / mup[0, 1] < 1.4
    # SP: the output change grows like the width; the hidden change like sqrt(n) under SGD and n under Adam.
    assert sp[-1, 1] / sp[0, 1] > 8
    assert sp[-1, 0] / sp[0, 0] > (2 if optimizer == "sgd" else 10)
