# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Tests of the PyTorch version: the shared fixture and the autograd gradients of the loss, the counts of shapes.json,
gradcheck through MADE, the autoregressive property as a triangular autograd Jacobian, normalization over every
binary vector, ancestral sampling, the mask ensemble, NADE's cumulative sum against separate passes, the receptive
field of the Conv1d stack, the mu-law codes, and a MADE that learns a binary mixture."""

import json
import math
import pathlib

import numpy as np
import pytest
import torch

from autoregressive import (
    MADE,
    all_binary,
    config,
    degrees,
    ensemble_log_prob,
    flops_forward,
    mu_law_encode,
    nade_logits,
    nll,
    param_count,
    receptive_field,
    train,
    wavenet_log_prob,
    wavenet_logits,
    wavenet_sample,
    wavenet_shapes,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
CONFIGS = [((6,), True), ((5, 7), False), ((), True), ((8, 8), True)]


def random_model(cfg, seed, order=None, scale=0.7):
    g = torch.Generator().manual_seed(seed)
    model = MADE(cfg, degrees(cfg, order, g))
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return model


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    cfg = config()
    made = {k[5:]: v for k, v in params.items() if k.startswith("made_")}
    model = MADE.from_params(made, cfg, degrees(cfg, inp["order"]))
    x = inp["x"]
    torch.testing.assert_close(model(x), want["logits"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(model.log_prob(x), want["log_prob"], rtol=1e-12, atol=1e-14)
    loss = nll(model, x)
    loss.backward()
    torch.testing.assert_close(loss.detach(), want["loss"], rtol=1e-12, atol=1e-14)
    for k, p in model.named_params().items():
        assert p.grad is not None
        torch.testing.assert_close(p.grad, want[f"d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    nade = {k[5:]: torch.as_tensor(v) for k, v in params.items() if k.startswith("nade_")}
    torch.testing.assert_close(nade_logits(nade, x), want["nade_logits"], rtol=1e-12, atol=1e-14)
    wave = {k[5:]: torch.as_tensor(v) for k, v in params.items() if k.startswith("wave_")}
    dil = tuple(int(d) for d in inp["dilations"])
    torch.testing.assert_close(wavenet_logits(wave, inp["seq"], dil), want["wave_logits"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(mu_law_encode(inp["audio"]), want["mu_codes"].long())


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], (k["d_hidden"],))
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = MADE(cfg)
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    assert model(torch.zeros(k["B"], k["d_data"])).shape == (k["B"], k["d_data"])


def test_gradcheck():
    model = random_model(config(5, (7, 6)), 3, [2, 5, 1, 4, 3])
    x = torch.randn(4, 5, generator=torch.Generator().manual_seed(4), requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: model(v), (x,))
    assert torch.autograd.gradcheck(lambda v: nll(model, torch.sigmoid(v)), (x,))


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_autograd_jacobian_is_triangular_in_the_ordering(hidden, direct):
    order = [3, 5, 1, 4, 2]
    model = random_model(config(5, hidden, direct), 1, order)
    x = torch.randn(1, 5, generator=torch.Generator().manual_seed(2))
    full = torch.autograd.functional.jacobian(lambda v: model(v)[0], x)
    assert isinstance(full, torch.Tensor)
    jac = full[:, 0, :].T  # jac[i, d] = d a_d / d x_i
    m0 = torch.tensor(order)
    allowed = m0[:, None] < m0[None, :]
    assert jac[~allowed].abs().max() == 0.0
    assert jac[allowed].abs().max() > 1e-3


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_probabilities_sum_to_one(hidden, direct):
    model = random_model(config(6, hidden, direct), 5, [4, 6, 2, 1, 5, 3], 1.5)
    with torch.no_grad():
        total = model.log_prob(all_binary(6)).exp().sum()
    assert total.item() == pytest.approx(1.0, abs=1e-12)


def test_ancestral_sampling_draws_from_the_model():
    model = random_model(config(3, (6,)), 6, [2, 3, 1], 1.2)
    X = all_binary(3)
    with torch.no_grad():
        prob = model.log_prob(X).exp()
    n = 40000
    s = model.sample(n, torch.Generator().manual_seed(7))
    freq = torch.stack([(s == row).all(1).double().mean() for row in X])
    assert (freq - prob).abs().max() < 4 * torch.sqrt(prob * (1 - prob) / n).max() + 1e-3


def test_mask_ensemble_is_normalized_and_beats_the_mean_log_prob():
    cfg = config(5, (8, 8))
    model = random_model(cfg, 8)
    g = torch.Generator().manual_seed(9)
    mlist = [degrees(cfg, torch.randperm(5, generator=g) + 1, g) for _ in range(6)]
    X = all_binary(5)
    with torch.no_grad():
        ens = ensemble_log_prob(model, mlist, X)
        mean = torch.stack([ensemble_log_prob(model, [m], X) for m in mlist]).mean(0)
    assert ens.exp().sum().item() == pytest.approx(1.0, abs=1e-12)
    assert bool((ens >= mean - 1e-12).all())


def test_nade_cumsum_matches_separate_passes():
    g = torch.Generator().manual_seed(10)
    D, H = 6, 4
    q = {"W": torch.randn(D, H, generator=g), "c": torch.randn(H, generator=g)}
    q.update({"V": torch.randn(H, D, generator=g), "b": torch.randn(D, generator=g)})
    X = all_binary(D)
    a = nade_logits(q, X)
    cols = [q["b"][i] + torch.sigmoid(q["c"] + X[:, :i] @ q["W"][:i]) @ q["V"][:, i] for i in range(D)]
    torch.testing.assert_close(a, torch.stack(cols, 1), rtol=1e-12, atol=1e-13)
    assert (X * a - torch.nn.functional.softplus(a)).sum(1).exp().sum().item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kernel, dilations", [(2, (1, 2, 4)), (3, (1, 2)), (2, (1, 2, 4, 1, 2))])
def test_receptive_field_of_the_conv1d_stack(kernel, dilations):
    g = torch.Generator().manual_seed(11)
    K, T = 4, 24
    w = {k: 0.6 * torch.randn(s, generator=g) for k, s in wavenet_shapes(K, 16, kernel, dilations).items()}
    x = torch.randint(0, K, (1, T), generator=g)
    base = wavenet_logits(w, x, dilations)
    R = receptive_field(kernel, dilations)
    for t in range(T):
        y = x.clone()
        y[0, t] = (y[0, t] + 1) % K
        changed = (wavenet_logits(w, y, dilations) - base).abs().amax(dim=(0, 2)) > 1e-12
        assert set(torch.nonzero(changed)[:, 0].tolist()) == set(range(t + 1, min(t + R, T - 1) + 1))


def test_wavenet_probabilities_sum_to_one():
    g = torch.Generator().manual_seed(12)
    K, T, dil = 3, 4, (1, 2)
    w = {k: 0.8 * torch.randn(s, generator=g) for k, s in wavenet_shapes(K, 2, 2, dil).items()}
    X = torch.cartesian_prod(*[torch.arange(K)] * T)
    assert wavenet_log_prob(w, X, dil).exp().sum().item() == pytest.approx(1.0, abs=1e-12)
    s = wavenet_sample(w, dil, 5, T, g)
    assert s.shape == (5, T) and s.min() >= 0 and s.max() < K


def test_mu_law_codes():
    q = mu_law_encode(torch.linspace(-1.0, 1.0, 20001))
    assert q.min() == 0 and q.max() == 255 and bool((q.diff() >= 0).all())
    assert mu_law_encode(torch.tensor([0.0, 1e-4, -1e-4])).tolist() == [128, 128, 127]


def mixture(D=8, eps=0.05):
    protos = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]])[:, :D]
    X = all_binary(D)
    flips = (X[:, None, :] - protos[None]).abs().sum(-1)
    return torch.mean(eps**flips * (1 - eps) ** (D - flips), 1)


def test_learns_a_binary_mixture():
    torch.manual_seed(13)
    g = torch.Generator().manual_seed(13)
    prob = mixture()
    X = all_binary(8)
    data = X[torch.multinomial(prob, 4000, replacement=True, generator=g)]
    model = MADE(config(8, (48,)))
    trace = train(model, data, 800, g)
    with torch.no_grad():
        kl = torch.sum(prob * (prob.log() - model.log_prob(X))).item()
    marg = data.mean(0)
    indep = (X * marg.log() + (1 - X) * (1 - marg).log()).sum(1)
    kl_indep = torch.sum(prob * (prob.log() - indep)).item()
    assert np.mean(trace[-50:]) < 0.8 * trace[0]
    assert kl < 0.1 < 1.0 < kl_indep
    assert math.isfinite(kl)
