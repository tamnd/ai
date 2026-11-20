# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""Tests of the PyTorch version: the shared fixture and its gradients, the counts, a stack of torch.nn.Linear layers,
gradcheck with and without tied weights, the corruptions, the closed form of the contractive penalty against the
Jacobian from autograd, and a denoising autoencoder that fills in masked inputs."""

import json
import pathlib

import numpy as np
import pytest
import torch
from torch import nn

from autoencoder import (
    Autoencoder,
    config,
    contractive_penalty,
    corrupt,
    from_widths,
    jacobian_penalty,
    loss,
    param_count,
    train,
)

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["n_hidden"], k["d_code"], **kw)


def unflat(arrays):
    p = {}
    for key in sorted(arrays, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))):
        p.setdefault(key.split("/")[0], []).append(arrays[key])
    return p


def test_shared_fixture_and_gradients():
    params = unflat(dict(np.load(FIXTURES / "params-small.npz")))
    x = torch.as_tensor(np.load(FIXTURES / "input-small.npz")["x"]).requires_grad_(True)
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    model = Autoencoder.from_params(params, config())
    out = model(x)
    for k in ("y", "a", "z"):
        torch.testing.assert_close(out[k], want[k], rtol=1e-12, atol=1e-14)
    value = loss(model.cfg, x.detach(), out["a"])
    torch.testing.assert_close(value, want["loss"], rtol=1e-12, atol=0)
    value.backward()
    for name in params:
        for i, p in enumerate(getattr(model, name)):
            torch.testing.assert_close(p.grad, want[f"d{name}/{i}"], rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(x.grad, want["dx"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    for tied, key in ((False, "params"), (True, "params_tied")):
        assert param_count(cfg_of(c, tied=tied)) == c["model"][key]
        assert sum(p.numel() for p in Autoencoder(cfg_of(c, tied=tied)).parameters()) == c["model"][key]


def test_matches_a_stack_of_linear_layers():
    g = torch.Generator().manual_seed(0)
    cfg = from_widths([7, 5, 3, 2], act="tanh", code_act="linear", out="sigmoid")
    model = Autoencoder(cfg, generator=g)
    layers = []
    for W, b, act in zip(list(model.enc_W) + list(model.dec_W), list(model.enc_b) + list(model.dec_b),
                         [nn.Tanh(), nn.Tanh(), nn.Identity(), nn.Tanh(), nn.Tanh(), nn.Sigmoid()]):  # fmt: skip
        lin = nn.Linear(*W.shape)
        with torch.no_grad():
            lin.weight.copy_(W.T)
            lin.bias.copy_(b)
        layers += [lin, act]
    x = torch.rand(4, 7, generator=g)
    torch.testing.assert_close(nn.Sequential(*layers)(x), model(x)["z"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(nn.Sequential(*layers[:6])(x), model(x)["y"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("tied", [False, True])
def test_gradcheck(tied):
    g = torch.Generator().manual_seed(1)
    cfg = from_widths([4, 3, 2], act="tanh", code_act="tanh", out="sigmoid", tied=tied)
    model = Autoencoder(cfg, generator=g)
    target = torch.rand(3, 4, generator=g)  # a copy: gradcheck perturbs its inputs in place
    x = target.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: loss(cfg, target, model(v)["a"]), (x,))
    names = [n for n, _ in model.named_parameters()]

    def of(*ps):
        out = torch.func.functional_call(model, dict(zip(names, ps)), (target,))
        return loss(cfg, target, out["a"])

    assert torch.autograd.gradcheck(of, tuple(p.detach().clone().requires_grad_(True) for p in model.parameters()))


def test_corruptions():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(30, 20, generator=g) * 0.6 + 0.2
    for kind in ("mask", "salt_pepper"):
        xt, mask = corrupt(x, kind, 0.3, generator=g)
        assert (mask.sum(1) == 6).all()
        assert torch.equal(xt[~mask], x[~mask])
        assert set(xt[mask].unique().tolist()) <= {0.0, 1.0}


def test_contractive_penalty_is_the_jacobian_norm():
    g = torch.Generator().manual_seed(3)
    model = Autoencoder(from_widths([6, 4], code_act="sigmoid"), generator=g)
    with torch.no_grad():
        model.enc_b[0].normal_(generator=g)
    x = torch.randn(5, 6, generator=g)
    torch.testing.assert_close(contractive_penalty(model, x), jacobian_penalty(model, x), rtol=1e-12, atol=0)
    linear = Autoencoder(from_widths([6, 4], code_act="linear"), generator=g)  # Rifai et al. (2011): weight decay
    torch.testing.assert_close(jacobian_penalty(linear, x), (linear.enc_W[0] ** 2).sum(), rtol=1e-12, atol=0)


def test_contraction_shrinks_the_jacobian():
    torch.manual_seed(4)
    g = torch.Generator().manual_seed(4)
    X = torch.rand(400, 10, generator=g)
    norms = {}
    for lam in (0.0, 1.0):
        model = Autoencoder(from_widths([10, 20], code_act="sigmoid"), generator=torch.Generator().manual_seed(5))
        train(model, X, 600, lr=1e-2, contraction=lam, generator=torch.Generator().manual_seed(6))
        norms[lam] = jacobian_penalty(model, X).item()
    assert norms[1.0] < 0.5 * norms[0.0]


def test_denoising_fills_in_what_the_mask_removed():
    g = torch.Generator().manual_seed(7)
    protos = torch.rand(4, 16, generator=g) < 0.5

    def sample(n):
        x = protos[torch.randint(0, 4, (n,), generator=g)]
        return torch.where(torch.rand(x.shape, generator=g) < 0.03, ~x, x).double()

    X, test = sample(1000), sample(400)
    cfg = from_widths([16, 32], code_act="sigmoid", out="sigmoid")
    err = {}
    for noise in (None, ("mask", 0.25)):
        model = Autoencoder(cfg, generator=torch.Generator().manual_seed(8))
        train(model, X, 1500, lr=1e-2, noise=noise, generator=torch.Generator().manual_seed(9))
        xt, mask = corrupt(test, "mask", 0.25, generator=torch.Generator().manual_seed(10))
        with torch.no_grad():
            err[noise is None] = (model(xt)["z"] - test).abs()[mask].mean().item()
    assert err[False] < 0.15 and err[True] > 2 * err[False]
