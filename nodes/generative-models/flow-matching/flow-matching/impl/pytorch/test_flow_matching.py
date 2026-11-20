# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Tests of the PyTorch version: the shared fixture and the autograd gradients of the loss, the counts of shapes.json,
gradcheck through the network, the target as the time derivative of the path and as the conditional field on the path,
the autograd divergence against the closed form, the log-likelihood of a Gaussian, minibatch optimal transport, and a
short training run on two Gaussian modes."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from flow_matching import (
    VelocityModel,
    assignment,
    cfm_loss,
    cond_field,
    config,
    divergence,
    euler_sample,
    flops_forward,
    log_likelihood,
    logit_normal,
    midpoint_sample,
    ot_pairs,
    param_count,
    path,
    target,
    time_features,
    train,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
MODES = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])


def two_modes(n, generator=None):
    k = torch.randint(0, 2, (n,), generator=generator)
    return MODES[k] + 0.3 * torch.randn(n, 2, generator=generator)


def random_model(cfg, seed, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    model = VelocityModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return model


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    cfg = config(sigma_min=float(inp["sigma_min"]))
    model = VelocityModel.from_params(params, cfg)
    x0, x1, t = inp["x0"], inp["x1"], inp["t"]
    torch.testing.assert_close(time_features(t, cfg["d_time"]), want["feat"], rtol=1e-12, atol=1e-14)
    xt = path(cfg, x0, x1, t)
    torch.testing.assert_close(xt, want["xt"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(target(cfg, x0, x1), want["u"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(model(xt, t), want["v"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(divergence(model, xt, t), want["div"], rtol=1e-10, atol=1e-12)
    loss = cfm_loss(model, x0, x1, t)
    loss.backward()
    torch.testing.assert_close(loss.detach(), want["loss"], rtol=1e-12, atol=1e-14)
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, want[f"d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    torch.testing.assert_close(euler_sample(model, x0, 4), want["x_euler"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(midpoint_sample(model, x0, 2), want["x_midpoint"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], k["d_hidden"], k["d_time"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = VelocityModel(cfg)
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    assert model(torch.zeros(k["B"], k["d_data"]), torch.zeros(k["B"])).shape == (k["B"], k["d_data"])


def test_gradcheck():
    model = random_model(config(), 3)
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(4), requires_grad=True)
    t = torch.tensor([0.1, 0.3, 0.6, 0.95])
    assert torch.autograd.gradcheck(lambda v: model(v, t), (x,))


@pytest.mark.parametrize("sigma_min", [0.0, 0.1])
def test_target_is_the_derivative_of_the_path_and_the_conditional_field(sigma_min):
    cfg = config(sigma_min=sigma_min)
    g = torch.Generator().manual_seed(5)
    x0, x1 = torch.randn(6, 3, generator=g), torch.randn(6, 3, generator=g)
    t = torch.rand(6, generator=g, dtype=torch.float64).requires_grad_(True)
    xt = path(cfg, x0, x1, t)
    dxdt = torch.stack([torch.autograd.grad(xt[:, i].sum(), t, retain_graph=True)[0] for i in range(3)], 1)
    torch.testing.assert_close(dxdt, target(cfg, x0, x1), rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(
        cond_field(cfg, xt.detach(), x1, t.detach()), target(cfg, x0, x1), rtol=1e-10, atol=1e-12
    )


def test_divergence_of_a_linear_field_is_its_trace():
    A = torch.randn(3, 3, generator=torch.Generator().manual_seed(6))
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(divergence(lambda z, s: z @ A.T, x, torch.zeros(5)), torch.full((5,), A.trace().item()))


def test_log_likelihood_of_a_gaussian_path():
    """The exact field of N(0, 1) -> N(0, s^2) along the straight path is ((s^2 t - (1 - t)) / (1 - t)^2 + t^2 s^2) x;
    its log-likelihood at t = 1 must be that of N(0, s^2)."""
    s = 0.5

    def v_fn(x, t):
        t = t[:, None]
        return (s**2 * t - (1 - t)) / ((1 - t) ** 2 + t**2 * s**2) * x

    def log_p0(x):
        return -0.5 * (x**2).sum(1) - 0.5 * x.shape[1] * math.log(2 * math.pi)

    x1 = torch.randn(7, 2, generator=torch.Generator().manual_seed(8))
    want = -0.5 * (x1**2).sum(1) / s**2 - x1.shape[1] * math.log(s * math.sqrt(2 * math.pi))
    torch.testing.assert_close(log_likelihood(v_fn, x1, 200, log_p0), want, rtol=0, atol=1e-4)


@pytest.mark.parametrize("n", [1, 2, 5, 7])
def test_hungarian_matches_brute_force(n):
    from itertools import permutations

    c = torch.rand(n, n, generator=torch.Generator().manual_seed(n)).numpy()
    best = min(sum(c[i, q[i]] for i in range(n)) for q in permutations(range(n)))
    col = assignment(c)
    assert sum(c[i, col[i]] for i in range(n)) == pytest.approx(best)
    assert sorted(col.tolist()) == list(range(n))


def test_ot_pairs_shorten_the_paths():
    g = torch.Generator().manual_seed(9)
    x0, x1 = torch.randn(64, 2, generator=g), two_modes(64, g)
    _, y1 = ot_pairs(x0, x1)
    assert ((y1 - x0) ** 2).sum() < 0.8 * ((x1 - x0) ** 2).sum()
    assert sorted(map(tuple, y1.tolist())) == sorted(map(tuple, x1.tolist()))


def test_logit_normal_is_symmetric_about_one_half():
    t = logit_normal(200_000, generator=torch.Generator().manual_seed(10))
    assert 0 < t.min() and t.max() < 1
    assert t.mean().item() == pytest.approx(0.5, abs=0.005)
    assert ((t > 0.25) & (t < 0.75)).double().mean().item() > 0.65


def test_learns_two_modes():
    torch.manual_seed(1)
    model = VelocityModel(config(2, 64, 16))
    g = torch.Generator().manual_seed(2)
    trace = train(model, two_modes, 1500, t_sampler=lambda n, gen: logit_normal(n, generator=gen), generator=g)
    assert np.mean(trace[-100:]) < 0.8 * np.mean(trace[:10])
    xs = midpoint_sample(model, torch.randn(2000, 2, generator=torch.Generator().manual_seed(3)), 20)
    dist = torch.cdist(xs, MODES).min(1).values
    assert 0.4 < (xs[:, 0] > 0).double().mean().item() < 0.6
    assert (dist < 1.0).double().mean().item() > 0.9
