# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://88GHGT).
"""Tests of the PyTorch optimizers: the worked example, the shared fixture, agreement with torch.optim.Adam and
torch.optim.AdamW on a small model with and without AMSGrad, weight decay and its coupling to the learning rate, the
place of eps, and the epsilon hat of Keras."""

import math

import pytest
import torch

from adam import Adam, AdamW

torch.set_default_dtype(torch.float64)

# The worked example of the page.
WG = [2.0, -2.0]
WTHETA = [0.9, 14.0 / 15.0]

# The shared fixture of the NumPy tests.
P = {"w": [0.5, -1.5, 2.0], "b": [0.25]}
G = [{"w": [1.0, -2.0, 0.5], "b": [0.5]}, {"w": [0.5, 1.0, -0.25], "b": [-1.0]}, {"w": [-1.5, 0.5, 0.0], "b": [2.0]}]
CONFIGS = {"adam": {"lr": 0.1}, "l2": {"lr": 0.1, "weight_decay": 0.1},
           "adamw": {"lr": 0.1, "weight_decay": 0.1, "decoupled": True}, "amsgrad": {"lr": 0.1, "amsgrad": True}}
AFTER3 = {
    "adam": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527783689874684], "b": [0.14464479271810454]},
    "l2": {"w": [0.3108766333376067, -1.3489621467423747, 1.778235325837831], "b": [0.14103715505555778]},
    "adamw": {"w": [0.3030513976144773, -1.324932247497697, 1.7956327059805899], "b": [0.1388434391558989]},
    "amsgrad": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527886655267884], "b": [0.14464479271810454]},
}


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def make(cls, params, kw):
    kw = dict(kw)
    decoupled = kw.pop("decoupled", False)
    if cls is Adam:
        return Adam(params, decoupled_weight_decay=decoupled, **kw)
    return (torch.optim.AdamW if decoupled else torch.optim.Adam)(params, **kw)


def run_fixed(cls, kw, params, grads):
    ps = [t(p).requires_grad_() for p in params]
    opt = make(cls, ps, kw)
    for g in grads:
        for p, gi in zip(ps, g):
            p.grad = t(gi)
        opt.step()
    return [p.detach() for p in ps]


def test_worked_example():
    x = t(1.0).requires_grad_()
    opt = Adam([x], lr=0.1, betas=(0.5, 0.96), eps=0.0)
    for g, want in zip(WG, WTHETA):
        x.grad = t(g)
        opt.step()
        torch.testing.assert_close(x.detach(), t(want), rtol=1e-14, atol=0)
    x = t(1.0).requires_grad_()
    opt = AdamW([x], lr=0.1, betas=(0.5, 0.96), eps=0.0, weight_decay=0.1)
    x.grad = t(2.0)
    opt.step()
    torch.testing.assert_close(x.detach(), t(0.89), rtol=1e-14, atol=0)


@pytest.mark.parametrize("cls", [Adam, torch.optim.Adam])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(cls, name):
    params = [P["w"], P["b"]]
    grads = [[g["w"], g["b"]] for g in G]
    got = run_fixed(cls, CONFIGS[name], params, grads)
    torch.testing.assert_close(got[0], t(AFTER3[name]["w"]), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(got[1], t(AFTER3[name]["b"]), rtol=1e-13, atol=1e-15)


def train(opt_fn, steps=20):
    """Train a small two-layer network on fixed data and return its parameters."""
    torch.manual_seed(20261115)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    x, y = torch.randn(32, 4), torch.randn(32, 1)
    opt = opt_fn(model.parameters())
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


@pytest.mark.parametrize("kw", [{}, {"lr": 0.01}, {"lr": 0.01, "amsgrad": True}, {"lr": 0.01, "weight_decay": 0.1},
                                {"lr": 0.01, "betas": (0.9, 0.98), "eps": 1e-9}, {"lr": 0.05, "betas": (0.0, 0.5)}])
def test_matches_torch_adam(kw):
    for a, b in zip(train(lambda p: Adam(p, **kw)), train(lambda p: torch.optim.Adam(p, **kw))):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kw", [{}, {"lr": 0.01, "weight_decay": 0.1}, {"lr": 0.01, "weight_decay": 0.1, "amsgrad": True},
                                {"lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.1}])
def test_matches_torch_adamw(kw):
    for a, b in zip(train(lambda p: AdamW(p, **kw)), train(lambda p: torch.optim.AdamW(p, **kw))):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_adam_weight_decay_is_not_adamw():
    """torch.optim.Adam(weight_decay=...) is an L2 penalty, not the decoupled decay of AdamW."""
    a = train(lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1))
    b = train(lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1))
    assert max((x - y).abs().max().item() for x, y in zip(a, b)) > 1e-4


@pytest.mark.parametrize("cls", [AdamW, torch.optim.AdamW])
def test_adamw_decay_scales_with_lr(cls):
    """With a zero gradient, one step of AdamW multiplies the weights by 1 - lr * weight_decay: in PyTorch the decay
    per step is the product of the two, while the lambda of Loshchilov and Hutter does not depend on the step size."""
    for lr in [0.1, 0.01]:
        x = t([2.0, -4.0]).requires_grad_()
        opt = cls([x], lr=lr, weight_decay=0.5)
        x.grad = torch.zeros(2)
        opt.step()
        torch.testing.assert_close(x.detach(), t([2.0, -4.0]) * (1 - lr * 0.5), rtol=1e-14, atol=0)


def test_eps_outside_the_square_root():
    """eps is added to sqrt(v_hat), not to v_hat: with a gradient of 1e-4 the two places give different steps."""
    x = t([0.0]).requires_grad_()
    opt = torch.optim.Adam([x], lr=1.0, eps=1e-8)
    x.grad = t([1e-4])
    opt.step()
    torch.testing.assert_close(x.detach(), t([-1e-4 / (1e-4 + 1e-8)]), rtol=1e-14, atol=0)
    assert abs(x.item() + 1e-4 / math.sqrt(1e-8 + 1e-8)) > 0.1


def test_keras_epsilon_hat():
    """The Keras epsilon is the epsilon hat of Kingma and Ba: at step 1 it equals a PyTorch eps 1/sqrt(1 - beta2)
    times larger. Here one step of the efficient form is compared with torch.optim.Adam."""
    g, e_hat, b1, b2 = 3e-7, 1e-7, 0.9, 0.999
    m, v = (1 - b1) * g, (1 - b2) * g * g
    keras_step = math.sqrt(1 - b2) / (1 - b1) * m / (math.sqrt(v) + e_hat)
    x = t([0.0]).requires_grad_()
    opt = torch.optim.Adam([x], lr=1.0, eps=e_hat / math.sqrt(1 - b2))
    x.grad = t([g])
    opt.step()
    torch.testing.assert_close(-x.detach(), t([keras_step]), rtol=1e-12, atol=0)
    y = t([0.0]).requires_grad_()
    opt = torch.optim.Adam([y], lr=1.0, eps=e_hat)
    y.grad = t([g])
    opt.step()
    assert abs(y.item() + keras_step) > 0.1


def test_validation():
    x = [t(1.0).requires_grad_()]
    for kw in [{"lr": -1.0}, {"betas": (1.0, 0.999)}, {"eps": -1.0}, {"weight_decay": -1.0}]:
        with pytest.raises(ValueError):
            Adam(x, **kw)
