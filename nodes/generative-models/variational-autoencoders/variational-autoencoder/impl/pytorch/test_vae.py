# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Tests of the PyTorch version: the shared fixture and its gradients for both decoders and the three objectives, the
counts of shapes.json, the densities and the KL term against torch.distributions, gradcheck, and a short training."""

import json
import pathlib

import numpy as np
import pytest
import torch
from torch import distributions as dist

from vae import DECODERS, OBJECTIVES, VAE, config, kl, log_prior, log_q, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


@pytest.mark.parametrize("decoder", DECODERS)
def test_shared_fixture_and_gradients(decoder):
    params, inp, want = fixture()
    model = VAE.from_params(params, config(decoder=decoder))
    out = model(inp["x"], inp["eps"])
    for k in ("mu", "lv", "z", "a") + (("olv",) if decoder == "gaussian" else ()):
        torch.testing.assert_close(out[k], want[f"{decoder}/{k}"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.log_weights(inp["x"], out, inp["eps"]), want[f"{decoder}/log_w"], rtol=1e-12,
                               atol=1e-12)  # fmt: skip
    for kind in OBJECTIVES:
        model.zero_grad()
        value = model.objective(inp["x"], inp["eps"], kind)
        torch.testing.assert_close(value, want[f"{decoder}/{kind}/loss"], rtol=1e-12, atol=1e-13)
        value.backward()
        for name, p in model.p.items():
            torch.testing.assert_close(p.grad, want[f"{decoder}/{kind}/d{name}"], rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    for decoder, key in (("bernoulli", "params"), ("gaussian", "params_gaussian")):
        cfg = config(k["d_in"], k["d_hidden"], k["d_latent"], decoder)
        assert param_count(cfg) == mod[key]
        if k["d_in"] < 100:
            assert sum(p.numel() for p in VAE(cfg).parameters()) == mod[key]


def test_densities_and_kl_match_torch_distributions():
    g = torch.Generator().manual_seed(0)
    mu, lv = torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g)
    std, eps = (0.5 * lv).exp(), torch.randn(4, 5, 3, generator=g)
    q, prior = dist.Normal(mu, std), dist.Normal(torch.zeros(3), torch.ones(3))
    torch.testing.assert_close(kl(mu, lv), dist.kl_divergence(q, prior).sum(-1))
    z = mu + std * eps
    torch.testing.assert_close(log_q(lv, eps), q.log_prob(z).sum(-1))
    torch.testing.assert_close(log_prior(z), prior.log_prob(z).sum(-1))
    x = (torch.rand(5, 6, generator=g) > 0.5).double()
    for decoder in DECODERS:
        model = VAE(config(6, 4, 3, decoder), generator=g)
        out = model(x, eps)
        if decoder == "bernoulli":
            want = -dist.Bernoulli(logits=out["a"]).log_prob(x).sum(-1)
        else:
            want = -dist.Normal(out["a"], (0.5 * out["olv"]).exp()).log_prob(x).sum(-1)
        torch.testing.assert_close(model.nll(x, out), want)


class Objective(torch.nn.Module):
    """The objective as the forward of a module, so that torch.func.functional_call can swap the parameters."""

    def __init__(self, model, kind):
        super().__init__()
        self.model, self.kind = model, kind

    def forward(self, x, eps):
        return self.model.objective(x, eps, self.kind)


@pytest.mark.parametrize("decoder", DECODERS)
@pytest.mark.parametrize("kind", OBJECTIVES)
def test_gradcheck(decoder, kind):
    g = torch.Generator().manual_seed(1)
    model = VAE(config(4, 3, 2, decoder), scale=0.7, generator=g)
    x, eps = torch.rand(3, 4, generator=g), torch.randn(5, 3, 2, generator=g)
    obj, names = Objective(model, kind), list(model.p)

    def f(*ps):
        return torch.func.functional_call(obj, {f"model.p.{n}": v for n, v in zip(names, ps)}, (x, eps))

    ps = tuple(model.p[n].detach().clone().requires_grad_(True) for n in names)
    assert torch.autograd.gradcheck(f, ps)
    xs = x.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: obj(x, eps), (xs,))


def test_training_raises_the_bound():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(2)
    W = torch.randn(2, 8, generator=g)
    data = 0.8 * torch.tanh(torch.randn(1000, 2, generator=g) @ W) + 0.05 * torch.randn(1000, 8, generator=g)
    model = VAE(config(8, 16, 4, "gaussian"), generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    trace = []
    for _ in range(800):
        xb = data[torch.randint(0, 1000, (50,), generator=g)]
        opt.zero_grad()
        value = model.objective(xb, model.sample_eps(1, 50, g))
        value.backward()
        opt.step()
        trace.append(value.item())
    assert np.mean(trace[-100:]) < np.mean(trace[:20]) - 5.0
    with torch.no_grad():
        eps = model.sample_eps(50, 1000, g)
        elbo, iwae = -model.objective(data, eps, "elbo_mc"), -model.objective(data, eps, "iwae")
    assert iwae > elbo
    assert model.sample(7, g).shape == (7, 8)
