# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/deep-sets/ (ai://NEW-deep-sets).
"""Tests of the PyTorch version: the shared fixture and its gradients, the NumPy reference, Lemma 3 as a dense layer,
torch.nn.MultiheadAttention without positions, gradcheck, and two small training tasks: the sum of a set of digits on
sets five times longer than in training, and an outlier that only the interaction between elements can find."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepsets import POOLS, DeepSets, config, nll, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d_in"], k["width"], k["n_layers"], k["n_classes"], **kw)


def test_shared_fixture_and_gradients():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    model = DeepSets.from_params(params, config())
    x = inp["x"].clone().requires_grad_(True)
    logits = model(x, inp["mask"])
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.features(inp["x"], inp["mask"])[0], want["features"], rtol=1e-12, atol=1e-13)
    loss = nll(logits, inp["labels"])
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=1e-13)
    loss.backward()
    for name, p in model.p.items():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)
    torch.testing.assert_close(x.grad, want["dx"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    assert param_count(cfg_of(c)) == c["model"]["params"]


def load_numpy_reference():
    spec = importlib.util.spec_from_file_location("deepsets_ref", HERE.parents[1] / "impl" / "numpy" / "deepsets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("head", ["set", "element"])
@pytest.mark.parametrize("p", POOLS)
def test_matches_the_numpy_reference(p, head):
    ref = load_numpy_reference()
    cfg = config(3, 5, 2, 4, pool=p, readout=p, act="relu", head=head)
    params = ref.init(np.random.default_rng(1), cfg, 0.5)
    rng = np.random.default_rng(2)
    mask = np.ones((3, 6))
    mask[0, 2:] = 0
    x = rng.normal(size=(3, 6, 3))
    want = ref.forward(params, cfg, x, mask)[0]
    got = DeepSets.from_params(params, cfg)(torch.as_tensor(x), torch.as_tensor(mask)).detach().numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_lemma3_is_a_dense_layer_with_tied_weights():
    """One feature per element: the equivariant layer is the linear layer of weight lambda I + gamma 1 1^T."""
    g = torch.Generator().manual_seed(3)
    M = 6
    model = DeepSets(config(1, 1, 1, 2, pool="sum"), scale=1.0, generator=g)
    lam, gam, beta = model.p["eq0_Lambda"].item(), -model.p["eq0_Gamma"].item(), model.p["eq0_beta"].item()
    dense = torch.nn.Linear(M, M)
    with torch.no_grad():
        dense.weight.copy_(lam * torch.eye(M) + gam * torch.ones(M, M))
        dense.bias.fill_(beta)
    x = torch.randn(4, M, 1, generator=g)
    mask = torch.ones(4, M)
    torch.testing.assert_close(model.equivariant(x, mask, "eq0_")[..., 0], dense(x[..., 0]), rtol=1e-12, atol=1e-13)
    # a dense layer with any untied weight is not equivariant
    with torch.no_grad():
        dense.weight[0, 1] += 0.5
    perm = torch.randperm(M, generator=g)
    assert not torch.allclose(dense(x[:, perm, 0]), dense(x[..., 0])[:, perm])


def test_attention_without_positions_is_equivariant_too():
    g = torch.Generator().manual_seed(4)
    attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    x = torch.randn(2, 7, 8, generator=g)
    perm = torch.randperm(7, generator=g)
    with torch.no_grad():
        y = attn(x, x, x, need_weights=False)[0]
        yp = attn(x[:, perm], x[:, perm], x[:, perm], need_weights=False)[0]
    torch.testing.assert_close(yp, y[:, perm], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kw", [{}, {"pool": "mean", "readout": "max", "act": "relu"}, {"head": "element"}],
                         ids=["max-sum", "mean-max-relu", "element"])  # fmt: skip
def test_gradcheck(kw):
    g = torch.Generator().manual_seed(5)
    cfg = config(2, 3, 2, 3, **kw)
    model = DeepSets(cfg, scale=0.7, generator=g)
    x = torch.randn(2, 4, 2, generator=g, requires_grad=True)
    mask = torch.ones(2, 4) if cfg["head"] == "element" else torch.tensor([[1.0, 1, 1, 1], [1, 1, 0, 0]])
    assert torch.autograd.gradcheck(lambda x: model(x, mask), (x,))
    names = list(model.p)

    def f(*ps):
        return torch.func.functional_call(model, {f"p.{n}": v for n, v in zip(names, ps)}, (x.detach(), mask))

    assert torch.autograd.gradcheck(f, tuple(model.p[n].detach().clone().requires_grad_(True) for n in names))


def digits(g, n, size_max, size=None):
    """Sets of 1 to size_max digits as one-hot vectors, padded to a common size, and their sums."""
    sizes = torch.randint(1, size_max + 1, (n,), generator=g) if size is None else torch.full((n,), size)
    M = int(sizes.max())
    d = torch.randint(0, 10, (n, M), generator=g)
    mask = (torch.arange(M)[None] < sizes[:, None]).double()
    return F.one_hot(d, 10).double() * mask[..., None], mask, (d * mask).sum(1)


def test_sum_of_digits_generalizes():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    cfg = config(10, 8, 1, 1, readout="sum", act="relu", interact=False)
    model = DeepSets(cfg, scale=0.3, generator=torch.Generator().manual_seed(2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(1500):
        x, mask, y = digits(g, 64, 10)
        opt.zero_grad()
        ((model(x, mask)[:, 0] - y) ** 2).mean().backward()
        opt.step()
    with torch.no_grad():
        x, mask, y = digits(g, 500, 50, size=50)
        acc = (model(x, mask)[:, 0].round() == y).double().mean().item()
    assert acc > 0.95, acc


def ring_with_centre(g, n, M=8):
    """M points in the plane around a random centre: M - 1 on a circle of radius about 2, one near the centre."""
    c = 20 * torch.rand(n, 1, 2, generator=g) - 10
    ang = 2 * torch.pi * torch.rand(n, M, generator=g)
    r = 2 + 0.1 * torch.randn(n, M, generator=g)
    x = c + torch.stack([r * ang.cos(), r * ang.sin()], -1)
    k = torch.randint(0, M, (n,), generator=g)
    x[torch.arange(n), k] = c[:, 0] + 0.2 * torch.randn(n, 2, generator=g)
    return x, k


def test_the_interaction_is_needed():
    torch.manual_seed(0)
    acc = {}
    for interact in (True, False):
        g = torch.Generator().manual_seed(3)
        cfg = config(2, 16, 2, pool="max", act="tanh", interact=interact, head="element")
        model = DeepSets(cfg, scale=0.3, generator=torch.Generator().manual_seed(4))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(500):
            x, k = ring_with_centre(g, 64)
            opt.zero_grad()
            nll(model(x), k).backward()
            opt.step()
        with torch.no_grad():
            x, k = ring_with_centre(g, 1000)
            acc[interact] = (model(x).argmax(1) == k).double().mean().item()
    assert acc[True] > 0.9 and acc[False] < 0.2, acc
