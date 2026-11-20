# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Tests of the PyTorch implementation: the shared fixture with autograd gradients, gradcheck through the model, the
boundary condition, the closed-form consistency function of a Gaussian by its zero total derivative along the ODE
(torch.func.jvp), the trained model's total derivative, the toy of Proposition 1 by autograd, and a model trained by
distillation and one trained from scratch."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

import consistency_model as cm

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def close(a, b, rtol=1e-12, atol=1e-13, msg=""):
    a = a.detach().numpy() if isinstance(a, torch.Tensor) else a
    b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def three_gaussians(gen, n=6000):
    ang = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    means = 0.8 * torch.stack([ang.cos(), ang.sin()], 1)
    x0 = means[torch.randint(0, 3, (n,), generator=gen)] + 0.1 * torch.randn(n, 2, generator=gen)
    return means, 0.01, torch.full((3,), 1 / 3), x0


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    D, H, E = (int(v) for v in inp["cfg"])
    model = cm.ConsistencyModel(D, H, E, params=P)
    target = cm.ConsistencyModel(D, H, E, params={k[6:]: v for k, v in P.items() if k.startswith("minus_")})
    ts, n = cm.karras_steps(18), inp["n"]
    t_lo, t_hi = ts[n], ts[n + 1]
    x_hi, x_lo = cm.cd_pair(
        inp["x0"], inp["z"], t_lo, t_hi, lambda x, t: cm.score(x, t, inp["means"], 0.1, inp["weights"])
    )
    close(x_lo, want["x_lo"])
    close(model(x_hi, t_hi), want["f"])
    loss = cm.consistency_loss(model, target, x_hi, t_hi, x_lo, t_lo)
    assert loss.item() == pytest.approx(float(want["cd_loss"]), rel=1e-12)
    loss.backward()
    for k, prm in model.named_parameters():
        assert prm.grad is not None
        close(prm.grad, want[f"cd_d{k}"], rtol=1e-10, msg=k)
        prm.grad = None
    xc_hi, xc_lo = cm.ct_pair(inp["x0"], inp["z"], t_lo, t_hi)
    loss = cm.consistency_loss(model, model, xc_hi, t_hi, xc_lo, t_lo, 1.0 / (t_hi - t_lo), 0.00054 * math.sqrt(3.0))
    assert loss.item() == pytest.approx(float(want["ct_loss"]), rel=1e-12)
    loss.backward()
    for k, prm in model.named_parameters():
        assert prm.grad is not None
        close(prm.grad, want[f"ct_d{k}"], rtol=1e-10, msg=k)
    cm.ema_(target, model, 0.95)
    close(target.W1, want["ema"])
    close(cm.sample(model, inp["xT"], [0.8], inp["noise"]), want["sample"])
    (a_t, s_t), (a_s, s_s), (a_s2, s_s2) = (cm.vp(v) for v in inp["lam"])
    z = inp["z_pd"]
    z_s2 = cm.ddim_step(cm.ddim_step(z, z.tanh(), a_t, s_t, a_s, s_s), 0.5 * z.tanh(), a_s, s_s, a_s2, s_s2)
    close(cm.pd_target(z, z_s2, a_t, s_t, a_s2, s_s2), want["pd_target"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    assert cm.param_count(k["dim"], k["hidden"], k["temb"]) == mod["params"]
    assert cm.flops_forward(k["dim"], k["hidden"], k["temb"], k["B"]) == mod["flops_forward"]
    model = cm.ConsistencyModel(k["dim"], k["hidden"], k["temb"])
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    with torch.no_grad():
        assert model(torch.zeros(k["B"], k["dim"]), 3.0).shape == (k["B"], k["dim"])


@pytest.mark.parametrize("c", [None, 0.3])
def test_gradcheck(c):
    """gradcheck of the loss through the student's input, levels and parameters, by torch.func.functional_call."""
    torch.manual_seed(1)
    model, target = cm.ConsistencyModel(3, 4, 5), cm.ConsistencyModel(3, 4, 5)
    t_hi, t_lo = torch.tensor([0.01, 0.5, 3.0, 40.0]), torch.tensor([0.005, 0.3, 2.0, 30.0])
    x_lo = torch.randn(4, 3)
    names = [k for k, _ in model.named_parameters()]

    def loss(x, t, *ps):
        out = torch.func.functional_call(model, dict(zip(names, ps, strict=True)), (x, t))
        return cm.metric(out, target(x_lo, t_lo).detach(), c).mean()

    args = (torch.randn(4, 3), t_hi, *model.parameters())
    assert torch.autograd.gradcheck(loss, tuple(v.detach().clone().requires_grad_(True) for v in args))


def test_boundary_condition():
    torch.manual_seed(2)
    model = cm.ConsistencyModel(4, 7, 6)
    with torch.no_grad():
        for prm in model.parameters():
            prm.mul_(5.0).add_(torch.randn_like(prm))
        x = torch.randn(5, 4)
        assert torch.equal(model(x, cm.EPS), x)


def test_gaussian_total_derivative_is_zero():
    """The exact consistency function is constant along every ODE path: its jvp with (dx/dt, 1) vanishes."""
    torch.manual_seed(3)
    mu, s2 = torch.tensor([[0.5, -1.0]]), 0.2

    def sf(x, t):
        return cm.score(x, t, mu, s2, torch.ones(1))

    def f(x, t):
        return mu + (x - mu) * torch.sqrt((s2 + cm.EPS**2) / (s2 + t[:, None] ** 2))

    for t in (0.01, 1.0, 50.0):
        x = t * torch.randn(6, 2)
        close(cm.total_derivative(f, x, t, sf), torch.zeros(6, 2), atol=1e-12)
        close(f(x, torch.full((6,), t)), cm.gaussian_consistency(x, t, mu, s2))


def test_toy_by_autograd():
    torch.manual_seed(5)
    ts, z, xi = torch.linspace(0.1, 1.0, 4001), torch.randn(1, 2000), 0.7
    smin, lo, hi = ts[0], ts[:-1], ts[1:]
    for theta0 in (-0.4, 1.3):
        theta = torch.tensor(theta0, requires_grad=True)
        cm.toy_ct_loss(theta, theta, xi, ts, z).backward()
        assert theta.grad is not None
        want = (2 * smin * (1 / lo - 1 / hi) * (1 - smin / hi) * (theta0 - xi)).mean()
        assert theta.grad.item() == pytest.approx(want.item(), rel=1e-9)
        theta.grad = None
        loss = cm.toy_ct_loss(theta, torch.tensor(0.0), xi, ts, z)
        assert loss.item() == pytest.approx(((1 - smin / hi) ** 2).mean().item() * theta0**2, rel=1e-2)


def near_props(x, means):
    d = torch.cdist(x, means)
    return float((d.min(1).values < 0.3).double().mean()), torch.bincount(d.argmin(1), minlength=3) / len(x)


def test_distillation_and_training():
    gen = torch.Generator().manual_seed(7)
    torch.manual_seed(7)
    means, s2, w, x0 = three_gaussians(gen)

    def sf(x, t):
        return cm.score(x, t, means, s2, w)

    xT = 80.0 * torch.randn(2000, 2, generator=torch.Generator().manual_seed(5))
    ref = cm.solve_ode(xT, cm.T_MAX, cm.EPS, sf, 200)
    model = cm.ConsistencyModel(2, 64, 8)
    target, _ = cm.train_cd(model, x0, sf, 3000, gen, lr=3e-3)
    one = cm.sample(target, xT)
    near, props = near_props(one, means)
    assert float((one - ref).norm(dim=1).mean()) < 0.2 and near > 0.65 and float((props - 1 / 3).abs().max()) < 0.06
    with torch.no_grad():
        t = torch.full((2000,), 1.0)
        x1 = cm.solve_ode(xT, cm.T_MAX, 1.0, sf, 100)
    d = cm.total_derivative(lambda x, tt: target(x, tt), x1, t, sf).detach()
    assert float(d.norm(dim=1).mean()) < 0.5

    model = cm.ConsistencyModel(2, 64, 8)
    cm.train_ct(model, x0, 3000, gen, lr=3e-3)
    noise = torch.randn(1, 2000, 2, generator=torch.Generator().manual_seed(6))
    (n1, props1), (n2, _) = (
        near_props(cm.sample(model, xT), means),
        near_props(cm.sample(model, xT, [0.8], noise), means),
    )
    assert n1 > 0.25 and n2 > max(n1, 0.6) and float((props1 - 1 / 3).abs().max()) < 0.1
