# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/fine-tuning/ (ai://NEW-fine-tuning).
import math

import pytest
import torch
from fine_tuning import (
    EWC,
    GradualUnfreezer,
    discriminative_param_groups,
    linear_probe_then_fine_tune,
    reinit_top_layers,
    stlr_lambda,
    train,
    warmup_linear_lambda,
)
from torch import nn

torch.set_default_dtype(torch.float64)
PHI = (1 + math.sqrt(5)) / 2


def example():
    body, head = nn.Linear(2, 1, bias=False), nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        body.weight.copy_(torch.tensor([[1.0, 1.0]]))
        head.weight.zero_()
    x, y = torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0]])
    return body, head, lambda: 0.5 * ((head(body(x)) - y) ** 2).mean()


def test_full_fine_tuning_vs_lp_ft():
    body, head, loss = example()
    train(list(body.parameters()) + list(head.parameters()), loss, 6000, 1e-2)
    ood = head(body(torch.tensor([[0.0, 1.0]]))).item()
    assert body.weight[0, 1].item() == 1.0 and abs(ood - 1 / math.sqrt(PHI)) < 2e-3
    body, head, loss = example()
    linear_probe_then_fine_tune(body, head, loss, 6000, 6000, 1e-2)
    assert abs(head(body(torch.tensor([[0.0, 1.0]]))).item() - 1.0) < 1e-6


def test_discriminative_groups_and_stlr():
    layers = [nn.Linear(3, 3) for _ in range(4)]
    groups = discriminative_param_groups(layers, 0.01)
    assert [g["lr"] for g in groups] == pytest.approx([0.01 / 2.6**3, 0.01 / 2.6**2, 0.01 / 2.6, 0.01])
    opt = torch.optim.SGD(groups, lr=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, stlr_lambda(1000))
    seen = []
    for _ in range(1000):
        seen.append(opt.param_groups[-1]["lr"])
        opt.step()
        sched.step()
    assert seen[0] == pytest.approx(0.01 / 32) and max(seen) == pytest.approx(0.01) and seen.index(max(seen)) == 100
    assert warmup_linear_lambda(1000)(100) == pytest.approx(1.0) and warmup_linear_lambda(1000)(1000) == 0


def test_gradual_unfreezer():
    layers = [nn.Linear(2, 2) for _ in range(3)]
    u = GradualUnfreezer(layers)
    flags = lambda: [all(p.requires_grad for p in m.parameters()) for m in layers]
    assert flags() == [False, False, True]
    u.step()
    assert flags() == [False, True, True]
    u.step()
    u.step()
    assert flags() == [True, True, True]


def test_torch_adam_corrects_bias():
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p], lr=1e-3)
    p.grad = torch.tensor([0.3, -2.0])
    opt.step()
    assert torch.allclose(p.detach().abs(), torch.full((2,), 1e-3), rtol=1e-4)


def test_reinit_top_layers():
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64)) for _ in range(4)])
    with torch.no_grad():
        for b in blocks:
            b[0].weight.fill_(1.0)
            b[1].weight.fill_(3.0)
    reinit_top_layers(blocks, 2)
    assert torch.all(blocks[0][0].weight == 1.0) and torch.all(blocks[1][1].weight == 3.0)
    for b in blocks[2:]:
        assert abs(b[0].weight.std().item() - 0.02) < 0.002 and torch.all(b[1].weight == 1.0)


def test_ewc_fisher_and_penalty():
    torch.manual_seed(1)
    model = nn.Linear(3, 4)
    xs = torch.randn(8, 3)
    ewc = EWC(model, xs, lam=2.0)
    probs = torch.softmax(model(xs), dim=-1).detach()
    want = (probs * (1 - probs)).T @ (xs**2) / len(xs)  # the exact Fisher of a softmax-regression weight
    assert torch.allclose(ewc.fisher["weight"], want)
    assert torch.allclose(ewc.fisher["bias"], (probs * (1 - probs)).mean(0))
    assert ewc.penalty(model).item() == 0.0
    with torch.no_grad():
        model.bias += 1.0
    assert ewc.penalty(model).item() == pytest.approx(ewc.fisher["bias"].sum().item())
