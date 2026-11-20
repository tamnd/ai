# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Tests of the PyTorch version: the shared fixture with every gradient from autograd, the counts of shapes.json,
gradcheck through the free energy and the score-matching objective, the worked example, the autograd log-likelihood
gradient against the data-minus-model statistics of eq-ml-grad, the free energy against brute-force sums, Gibbs
sampling, CD-1 and PCD learning a binary mixture, the autograd score and Laplacian against closed forms, the stationary
variance of Langevin dynamics, an energy trained on Langevin samples, score matching on a Gaussian, noise-contrastive
estimation and the energy of a classifier."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from energy_based import (
    RBM,
    EnergyNet,
    all_binary,
    cd_loss,
    config,
    flops_forward,
    gaussian_energy,
    gaussian_natural,
    grad_x,
    jem_class_probs,
    jem_energy,
    langevin,
    langevin_step,
    laplacian,
    nce_fit,
    nce_objective,
    nll,
    param_count,
    sm_objective,
    train_ebm,
    train_rbm,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def random_rbm(D, H, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return RBM.from_params({"W": scale * torch.randn(D, H, generator=g), "b": scale * torch.randn(D, generator=g),
                            "c": scale * torch.randn(H, generator=g)})  # fmt: skip


def grads(loss, model):
    model.zero_grad()
    loss.backward()
    out = {}
    for k, p in model.named_params().items():
        assert p.grad is not None
        out[k] = p.grad.clone()
    return out


def test_shared_fixture():
    P = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    gamma, step = float(inp["gamma"]), float(inp["step"])
    rbm = RBM.from_params({k[4:]: v for k, v in P.items() if k.startswith("rbm_")})
    net = EnergyNet.from_params({k[4:]: v for k, v in P.items() if k.startswith("mlp_")}, gamma)
    v = inp["v"]
    with torch.no_grad():
        torch.testing.assert_close(rbm(v), want["free_energy"], rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(rbm.energy(v, inp["h"]), want["energy"], rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(rbm.p_h(v), want["p_h"], rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(rbm.log_partition(), want["log_z"], rtol=1e-12, atol=1e-13)
    h1, v1 = rbm.gibbs_step(v, inp["u_h"], inp["u_v"])
    torch.testing.assert_close(h1, want["h1"], rtol=0, atol=0)
    torch.testing.assert_close(v1, want["v1"], rtol=0, atol=0)
    for k, g in grads(cd_loss(rbm, v, v1), rbm).items():  # the ascent direction is minus the gradient of the loss
        torch.testing.assert_close(-g, want[f"cd_{k}"], rtol=1e-10, atol=1e-13, msg=k)
    for k, g in grads(nll(rbm, v), rbm).items():
        torch.testing.assert_close(-g, want[f"ml_{k}"], rtol=1e-10, atol=1e-13, msg=k)
    x = inp["x"]
    torch.testing.assert_close(net(x).detach(), want["mlp_energy"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(grad_x(net, x), want["mlp_grad_x"], rtol=1e-12, atol=1e-13)
    for k, g in grads(net(x).mean(), net).items():
        torch.testing.assert_close(g, want[f"mlp_d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    torch.testing.assert_close(sm_objective(net, x).detach(), want["mlp_sm"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(langevin_step(net, x, step, inp["z"]), want["langevin"], rtol=1e-12, atol=1e-13)
    theta = inp["theta"].clone().requires_grad_(True)
    J = nce_objective(theta, inp["nce_x"], inp["nce_y"], 0.0, 2.0)
    (gt,) = torch.autograd.grad(J, theta)
    torch.testing.assert_close(J.detach(), want["nce_j"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(gt, want["nce_grad"], rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(jem_energy(inp["logits"]), want["jem_energy"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_visible"], k["d_hidden"])
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = RBM(cfg)
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    v = torch.zeros(k["B"], k["d_visible"])
    h, v1 = model.gibbs_step(v, torch.ones(k["B"], k["d_hidden"]), torch.ones_like(v))
    assert h.shape == (k["B"], k["d_hidden"]) and v1.shape == v.shape
    assert model(v).shape == (k["B"],)


def test_gradcheck():
    rbm = random_rbm(4, 3, 1)
    v = all_binary(4)[[1, 6, 9, 15]]
    params = tuple(p.detach().clone().requires_grad_(True) for p in rbm.named_params().values())

    def free_energy(W, b, c):
        return torch.func.functional_call(rbm, {"W": W, "b": b, "c": c}, (v,))

    assert torch.autograd.gradcheck(free_energy, params)
    assert torch.autograd.gradcheck(lambda W, b, c: -torch.logsumexp(-free_energy(W, b, c), 0), params)
    g = torch.Generator().manual_seed(2)
    net = EnergyNet(3, 4).reset(g)
    with torch.no_grad():
        net.w2.copy_(torch.randn(4, generator=g))
    x = torch.randn(5, 3, generator=g)
    qs = tuple(p.detach().clone().requires_grad_(True) for p in net.named_params().values())

    def sm(W1, b1, w2):
        return sm_objective(lambda u: torch.func.functional_call(net, {"W1": W1, "b1": b1, "w2": w2}, (u,)), x)

    assert torch.autograd.gradcheck(sm, qs)


def test_worked_example():
    rbm = RBM.from_params({"W": torch.tensor([[2.0], [-1.0]]), "b": torch.zeros(2), "c": torch.tensor([-1.0])})
    X = all_binary(2)
    with torch.no_grad():
        torch.testing.assert_close((-rbm(X)).exp(), torch.tensor([1 + math.e**-1, 1 + math.e**-2, 1 + math.e, 2.0]))
        assert rbm.log_partition().exp().item() == pytest.approx(8.2215, abs=1e-4)
        probs = rbm.log_prob(X).exp()
        assert probs[2].item() == pytest.approx(0.4523, abs=1e-4)
        assert probs[3].item() == pytest.approx(0.2433, abs=1e-4)
        assert rbm.p_h(X)[[2, 3], 0].tolist() == pytest.approx([0.7311, 0.5], abs=1e-4)


def test_autograd_gradient_is_data_minus_model_statistics():
    rbm = random_rbm(4, 3, 3)
    X = all_binary(4)
    data = X[[1, 6, 7, 12, 15]]
    g = grads(nll(rbm, data), rbm)
    with torch.no_grad():
        w = rbm.log_prob(X).exp()
        ph_d, ph_m = rbm.p_h(data), rbm.p_h(X)
        want = {"W": data.T @ ph_d / len(data) - (X * w[:, None]).T @ ph_m, "b": data.mean(0) - w @ X,
                "c": ph_d.mean(0) - w @ ph_m}  # fmt: skip
    for k in want:
        torch.testing.assert_close(-g[k], want[k], rtol=1e-11, atol=1e-13, msg=k)


@pytest.mark.parametrize("D, H", [(3, 2), (4, 3), (5, 1)])
def test_free_energy_sums_out_the_hidden_units(D, H):
    rbm = random_rbm(D, H, 4)
    X, Hs = all_binary(D), all_binary(H)
    with torch.no_grad():
        E = torch.stack([rbm.energy(v.expand(len(Hs), D), Hs) for v in X])  # (2^D, 2^H)
        torch.testing.assert_close(-rbm(X), torch.logsumexp(-E, 1), rtol=1e-12, atol=1e-13)
        joint = torch.softmax(-E.flatten(), 0).reshape(E.shape)
        ph = rbm.p_h(X)
        fact = torch.where(Hs[None] == 1, ph[:, None], 1 - ph[:, None]).prod(-1)
        torch.testing.assert_close(joint / joint.sum(1, keepdim=True), fact, rtol=1e-12, atol=1e-14)


def test_gibbs_sampling_draws_from_the_model():
    rbm = random_rbm(3, 2, 5)
    g = torch.Generator().manual_seed(6)
    v = rbm.gibbs_chain(torch.randint(0, 2, (20000, 3), generator=g).double(), 30, g)
    X = all_binary(3)
    freq = torch.stack([(v == x).all(1).double().mean() for x in X])
    with torch.no_grad():
        pr = rbm.log_prob(X).exp()
    assert (freq - pr).abs().max() < 4 * torch.sqrt(pr * (1 - pr) / len(v)).max() + 1e-3


def mixture(D=8, eps=0.05):
    protos = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]])[:, :D]
    flips = (all_binary(D)[:, None, :] - protos[None]).abs().sum(-1)
    return torch.mean(eps**flips * (1 - eps) ** (D - flips), 1)


@pytest.mark.parametrize("persistent", [False, True], ids=["cd1", "pcd"])
def test_rbm_learns_a_binary_mixture(persistent):
    g = torch.Generator().manual_seed(7)
    prob = mixture()
    X = all_binary(8)
    data = X[torch.multinomial(prob, 4000, replacement=True, generator=g)]
    rbm = RBM(config(8, 8))
    with torch.no_grad():
        rbm.W.copy_(0.1 * torch.randn(8, 8, generator=g))
    trace = train_rbm(rbm, data, 800, g, k=1, persistent=persistent, lr=0.1)
    with torch.no_grad():
        kl = torch.sum(prob * (prob.log() - rbm.log_prob(X))).item()
    marg = data.mean(0)
    kl_indep = torch.sum(prob * (prob.log() - (X * marg.log() + (1 - X) * (1 - marg).log()).sum(1))).item()
    assert trace[-1] > trace[0] + 2.0
    assert kl < 0.1 < 1.0 < kl_indep


def test_autograd_score_and_laplacian_match_the_closed_forms():
    g = torch.Generator().manual_seed(10)
    net = EnergyNet(3, 5, 0.3).reset(g)
    with torch.no_grad():
        net.w2.copy_(torch.randn(5, generator=g))
    x = torch.randn(4, 3, generator=g)
    with torch.no_grad():
        t = torch.tanh(x @ net.W1 + net.b1)
        want_g = ((1 - t * t) * net.w2) @ net.W1.T + 0.3 * x
        want_lap = (-2 * t * (1 - t * t) * net.w2) @ (net.W1**2).sum(0) + 0.3 * 3
    torch.testing.assert_close(grad_x(net, x), want_g, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(laplacian(net, x), want_lap, rtol=1e-12, atol=1e-13)
    traces = []
    for r in x:
        h = torch.autograd.functional.hessian(lambda u: net(u[None])[0], r)
        assert isinstance(h, torch.Tensor)
        traces.append(h.trace())
    torch.testing.assert_close(laplacian(net, x), torch.stack(traces), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("step", [0.4, 0.1, 0.02])
def test_langevin_stationary_variance_on_a_quadratic_energy(step):
    s2 = 0.5
    net = EnergyNet(2, 1, 1 / s2)  # the network part is zero, so E(x) = |x|^2 / (2 s2)
    g = torch.Generator().manual_seed(11)
    x = langevin(net, torch.randn(20000, 2, generator=g), 300, step, g)
    a = 1 - step / (2 * s2)
    want = step / (1 - a * a)
    assert x.var().item() == pytest.approx(want, rel=0.03)
    assert abs(want - s2) <= 0.6 * step


def test_energy_trained_on_langevin_samples_finds_four_modes():
    g = torch.Generator().manual_seed(12)
    C = torch.tensor([[1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]])
    data = C[torch.randint(0, 4, (4000,), generator=g)] + 0.25 * torch.randn(4000, 2, generator=g)
    net = EnergyNet(2, 32).reset(g)
    train_ebm(net, data, 800, g)
    u = 6 * torch.rand(2000, 2, generator=g) - 3
    with torch.no_grad():
        assert net(data).mean() < net(u).mean() - 0.5
    s = langevin(net, u, 200, 0.02, g, 0.05)
    dist = torch.cdist(s, C)
    assert (dist.min(1).values < 0.75).double().mean() > 0.8
    assert torch.bincount(dist.argmin(1), minlength=4).min() / len(s) > 0.1


def test_score_matching_on_a_gaussian_drops_the_data_score_and_is_minimized_by_the_ml_estimate():
    g = torch.Generator().manual_seed(17)
    mu0 = torch.tensor([0.5, -1.0])
    L = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    z = torch.randn(5000, 2, generator=g)
    z = z - z.mean(0)
    z = z @ torch.linalg.inv(torch.linalg.cholesky(z.T @ z / len(z))).T
    x = mu0 + z @ L.T
    M0 = torch.linalg.inv(L @ L.T)
    for _ in range(3):
        A = torch.randn(2, 2, generator=g)
        mu, M = torch.randn(2, generator=g), A @ A.T + torch.eye(2)
        J = sm_objective(gaussian_energy(mu, M), x).item()
        assert J == pytest.approx((-torch.trace(M) + 0.5 * (((x - mu) @ M) ** 2).sum(1).mean()).item(), rel=1e-12)
        explicit = 0.5 * (((x - mu) @ M - (x - mu0) @ M0) ** 2).sum(1).mean().item()
        assert explicit - J == pytest.approx(0.5 * torch.trace(M0).item(), rel=1e-10)
    mu = x.mean(0)
    M = torch.linalg.inv((x - mu).T @ (x - mu) / len(x)).requires_grad_(True)
    m = mu.clone().requires_grad_(True)
    J = sm_objective(lambda u: 0.5 * (((u - m) @ M) * (u - m)).sum(1), x)
    gm, gM = torch.autograd.grad(J, (m, M))
    assert gm.abs().max() < 1e-10 and (gM + gM.T).abs().max() < 1e-10


def test_nce_recovers_the_parameters_and_the_normalizer():
    g = torch.Generator().manual_seed(18)
    x = 1.5 + 0.7 * torch.randn(40000, generator=g)
    y = 2.0 * torch.randn(40000, generator=g)
    theta = nce_fit(x, y, 0.0, 2.0)
    torch.testing.assert_close(theta, gaussian_natural(1.5, 0.7), rtol=0.03, atol=0.03)
    t = theta.clone().requires_grad_(True)
    (gt,) = torch.autograd.grad(nce_objective(t, x, y, 0.0, 2.0), t)
    assert gt.abs().max() < 1e-10
    a, m = theta[0].item(), theta[1].item()
    want_c = -0.5 * m * m / a + 0.5 * math.log(a) - 0.5 * math.log(2 * math.pi)
    assert theta[2].item() == pytest.approx(want_c, abs=0.03)


def test_classifier_energy_moves_with_a_logit_shift_and_the_class_probabilities_do_not():
    logits = torch.randn(6, 10, generator=torch.Generator().manual_seed(20))
    shift = torch.linspace(-2, 3, 6)[:, None]
    torch.testing.assert_close(jem_class_probs(logits + shift), jem_class_probs(logits), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(jem_energy(logits + shift), jem_energy(logits) - shift[:, 0], rtol=1e-12, atol=1e-13)
