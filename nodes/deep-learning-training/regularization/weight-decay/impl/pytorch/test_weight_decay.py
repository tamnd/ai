# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/weight-decay/ (ai://DD5QVM).
"""Tests of the PyTorch code: the shared fixture against torch.optim.SGD, agreement with torch.optim.SGD (weight_decay,
momentum, dampening, Nesterov) and torch.optim.AdamW on a small model, the explicit penalty, the decay of torch.optim.AdamW
per step, the parameter groups, and the scale invariance of a layer followed by batch normalization."""

import pytest
import torch
from torch import nn

from weight_decay import SGD, AdamW, l2_penalty, param_groups

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
P = {"w": [[0.5, -1.5], [2.0, 0.25]], "b": [0.25, -0.5]}
G = [{"w": [[1.0, -2.0], [0.5, 0.0]], "b": [0.5, 1.0]},
     {"w": [[0.5, 1.0], [-0.25, 1.5]], "b": [-1.0, 0.0]},
     {"w": [[-1.5, 0.5], [0.0, -1.0]], "b": [2.0, -0.5]}]
CONFIGS = {"l2": {"lr": 0.1, "weight_decay": 0.1},
           "momentum": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1},
           "nesterov": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "nesterov": True},
           "sgdw": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1, "decoupled": True}}
AFTER3 = {
    "l2": {"w": [[0.4876395, -1.4084285], [1.916343, 0.19407474999999996]], "b": [0.09256975000000003, -0.5331595]},
    "momentum": {"w": [[0.2604795, -1.1211484999999999], [1.802403, 0.05259474999999999]],
                 "b": [0.09148975000000001, -0.6893994999999999]},
    "nesterov": {"w": [[0.27808138050000003, -1.0763897315], [1.741800087, 0.019138485250000017]],
                 "b": [-0.04922180975000001, -0.6991175605000001]},
    "sgdw": {"w": [[0.2725395, -1.1582284999999999], [1.8537929999999998, 0.05907474999999996]],
             "b": [0.09751974999999999, -0.7032594999999999]},
}


def run_fixture(make_opt):
    params = {k: torch.tensor(v, requires_grad=True) for k, v in P.items()}
    opt = make_opt(list(params.values()))
    for g in G:
        for k, p in params.items():
            p.grad = torch.tensor(g[k])
        opt.step()
    return {k: p.detach() for k, p in params.items()}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(name):
    out = run_fixture(lambda ps: SGD(ps, **CONFIGS[name]))
    for k in P:
        torch.testing.assert_close(out[k], torch.tensor(AFTER3[name][k]), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", ["l2", "momentum", "nesterov"])
def test_fixture_is_torch_sgd(name):
    out = run_fixture(lambda ps: torch.optim.SGD(ps, **CONFIGS[name]))
    for k in P:
        torch.testing.assert_close(out[k], torch.tensor(AFTER3[name][k]), rtol=1e-12, atol=1e-15)


def model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 3))


def train(net, opt, steps=12, seed=1, penalty=0.0):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(16, 5, generator=gen)
        y = torch.randn(16, 3, generator=gen)
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        if penalty:
            loss = loss + l2_penalty(net.parameters(), penalty)
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in net.parameters()]


@pytest.mark.parametrize("kw", [{}, {"momentum": 0.9}, {"momentum": 0.9, "dampening": 0.1},
                                {"momentum": 0.9, "nesterov": True}])
def test_matches_torch_sgd(kw):
    a = model()
    b = model()
    pa = train(a, SGD(a.parameters(), lr=0.05, weight_decay=0.01, **kw))
    pb = train(b, torch.optim.SGD(b.parameters(), lr=0.05, weight_decay=0.01, **kw))
    for x, y in zip(pa, pb):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-14)


def test_torch_sgd_weight_decay_is_l2_penalty():
    # torch.optim.SGD(weight_decay=wd) takes the steps of SGD without decay on the loss plus (wd / 2) |theta|^2.
    a = model()
    b = model()
    pa = train(a, torch.optim.SGD(a.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01))
    pb = train(b, torch.optim.SGD(b.parameters(), lr=0.05, momentum=0.9), penalty=0.01)
    for x, y in zip(pa, pb):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-13)


def test_plain_sgd_l2_equals_decoupled():
    a = model()
    b = model()
    pa = train(a, torch.optim.SGD(a.parameters(), lr=0.05, weight_decay=0.01))
    pb = train(b, SGD(b.parameters(), lr=0.05, weight_decay=0.01, decoupled=True))
    for x, y in zip(pa, pb):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-14)


def test_momentum_l2_differs_from_decoupled():
    a = model()
    b = model()
    pa = train(a, torch.optim.SGD(a.parameters(), lr=0.05, momentum=0.9, weight_decay=0.05))
    pb = train(b, SGD(b.parameters(), lr=0.05, momentum=0.9, weight_decay=0.05, decoupled=True))
    gap = max((x - y).abs().max().item() for x, y in zip(pa, pb))
    assert gap > 1e-4


@pytest.mark.parametrize("decoupled", [True, False])
def test_matches_torch_adamw_and_adam(decoupled):
    a = model()
    b = model()
    pa = train(a, AdamW(a.parameters(), lr=0.01, weight_decay=0.1, decoupled=decoupled))
    ref = torch.optim.AdamW if decoupled else torch.optim.Adam
    pb = train(b, ref(b.parameters(), lr=0.01, weight_decay=0.1))
    for x, y in zip(pa, pb):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)


def test_torch_adamw_decay_is_lr_times_weight_decay():
    # With a zero gradient, one step of torch.optim.AdamW only multiplies the parameters by 1 - lr * weight_decay,
    # so the decay per step changes with the learning rate.
    for lr in (0.1, 0.01):
        p = torch.tensor([1.0, -2.0], requires_grad=True)
        opt = torch.optim.AdamW([p], lr=lr, weight_decay=0.5)
        p.grad = torch.zeros(2)
        opt.step()
        torch.testing.assert_close(p.detach(), torch.tensor([1.0, -2.0]) * (1 - lr * 0.5), rtol=1e-14, atol=0)


def test_adam_l2_is_not_adamw():
    a = model()
    b = model()
    pa = train(a, torch.optim.Adam(a.parameters(), lr=0.01, weight_decay=0.1))
    pb = train(b, torch.optim.AdamW(b.parameters(), lr=0.01, weight_decay=0.1))
    assert max((x - y).abs().max().item() for x, y in zip(pa, pb)) > 1e-4


def test_param_groups():
    net = model()
    decay, no_decay = param_groups(net, 0.05)
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    assert {id(p) for p in decay["params"]} == {id(net[0].weight), id(net[3].weight)}
    assert {id(p) for p in no_decay["params"]} == {id(net[0].bias), id(net[1].weight), id(net[1].bias),
                                                   id(net[3].bias)}
    # With a zero gradient, torch.optim.SGD on these groups decays the weights and leaves the rest alone.
    opt = torch.optim.SGD(param_groups(net, 0.5), lr=0.1)
    before = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    after = list(net.parameters())
    torch.testing.assert_close(after[0].detach(), before[0] * 0.95)
    torch.testing.assert_close(after[1].detach(), before[1])
    torch.testing.assert_close(after[2].detach(), before[2])
    torch.testing.assert_close(after[4].detach(), before[4] * 0.95)


def invariant_net(seed=0):
    # A linear layer without bias followed by batch normalization without affine parameters: the output does not
    # change when the weight is scaled. eps is tiny so that the invariance is exact to rounding.
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 4, bias=False), nn.BatchNorm1d(4, affine=False, eps=1e-30))


def invariant_loss(net, x, y):
    return ((net(x) - y) ** 2).mean()


def test_batchnorm_scale_invariance():
    net = invariant_net()
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(32, 6, generator=gen)
    y = torch.randn(32, 4, generator=gen)
    w = net[0].weight
    loss = invariant_loss(net, x, y)
    (g,) = torch.autograd.grad(loss, w)
    assert abs((w * g).sum().item()) < 1e-12
    with torch.no_grad():
        w.mul_(5.0)
    loss5 = invariant_loss(net, x, y)
    (g5,) = torch.autograd.grad(loss5, w)
    torch.testing.assert_close(loss5, loss, rtol=1e-12, atol=0)
    torch.testing.assert_close(g5, g / 5.0, rtol=1e-9, atol=1e-14)


def test_norm_grows_without_decay_and_settles_with_it():
    gen = torch.Generator().manual_seed(3)
    data = [(torch.randn(32, 6, generator=gen), torch.randn(32, 4, generator=gen)) for _ in range(400)]
    norms = {}
    for wd in (0.0, 0.05):
        net = invariant_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.5, weight_decay=wd)
        start = net[0].weight.norm().item()
        for x, y in data:
            opt.zero_grad()
            invariant_loss(net, x, y).backward()
            opt.step()
        norms[wd] = (start, net[0].weight.norm().item())
    assert norms[0.0][1] > norms[0.0][0]
    assert norms[0.05][1] < norms[0.0][1]


def test_li_arora_exponential_schedule():
    # SGD with weight decay on a scale-invariant net equals SGD without decay and the learning rate
    # lr (1 - lr wd)^(-2t - 1), up to the scale (1 - lr wd)^t of the weights.
    gen = torch.Generator().manual_seed(4)
    data = [(torch.randn(32, 6, generator=gen), torch.randn(32, 4, generator=gen)) for _ in range(30)]
    lr, wd = 0.2, 0.05
    q = 1 - lr * wd
    a = invariant_net()
    b = invariant_net()
    opt_a = torch.optim.SGD(a.parameters(), lr=lr, weight_decay=wd)
    opt_b = torch.optim.SGD(b.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda t: q ** (-2 * t - 1))
    for t, (x, y) in enumerate(data):
        for net, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            invariant_loss(net, x, y).backward()
            opt.step()
        sched.step()
        torch.testing.assert_close(b[0].weight.detach() * q ** (t + 1), a[0].weight.detach(), rtol=1e-8, atol=1e-12)
