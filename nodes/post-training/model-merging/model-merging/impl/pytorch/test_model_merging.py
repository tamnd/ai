# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/model-merging/ (ai://NEW-model-merging).
"""Tests for model_merging.py. TAUS and TIES_TAU are the NumPy tests' fixture."""

import pytest
import torch
import torch.nn.functional as F
from model_merging import (
    add_task_vectors,
    dare,
    diagonal_fisher,
    fisher_merge,
    interpolate,
    task_vector,
    ties_merge,
    uniform_soup,
)
from torch import nn
from torch.optim.swa_utils import AveragedModel

torch.set_default_dtype(torch.float64)

TAUS = torch.tensor(
    [
        [0.5, -0.2, 0.05, 0.8, -0.6],
        [-0.45, -0.4, 0.02, 0.6, 0.1],
        [0.4, 0.1, -0.01, -0.05, -0.7],
    ]
)
TIES_TAU = torch.tensor([0.45, -0.4, 0.0, 0.7, -0.65])  # k = 0.6


def split(vec):
    """The five fixture parameters as a state dict of two tensors, of shapes (2,) and (3,)."""
    return {"a": vec[:2].clone(), "b": vec[2:].clone()}


def test_ties_fixture_trims_over_all_parameters():
    pre = split(torch.zeros(5))
    merged = ties_merge(pre, [split(t) for t in TAUS], k=0.6, lam=1.0)
    assert torch.allclose(torch.cat([merged["a"], merged["b"]]), TIES_TAU)
    shifted = {k: v + 1.0 for k, v in pre.items()}
    merged = ties_merge(shifted, [{k: v + 1.0 for k, v in split(t).items()} for t in TAUS], k=0.6, lam=0.5)
    assert torch.allclose(torch.cat([merged["a"], merged["b"]]), 1.0 + 0.5 * TIES_TAU)


def test_task_arithmetic_and_interpolation():
    torch.manual_seed(0)
    pre = nn.Linear(4, 3).state_dict()
    fts = [{k: v + 0.1 * torch.randn_like(v) for k, v in pre.items()} for _ in range(3)]
    taus = [task_vector(f, pre) for f in fts]
    one = add_task_vectors(pre, taus[:1], 1.0)
    assert all(torch.allclose(one[k], fts[0][k]) for k in pre)
    third = add_task_vectors(pre, taus, 1 / 3)
    soup = uniform_soup(fts)
    assert all(torch.allclose(third[k], soup[k]) for k in pre)
    for a in (0.0, 0.3, 1.0):
        wise = interpolate(pre, fts[0], a)
        ta = add_task_vectors(pre, taus[:1], a)
        assert all(torch.allclose(wise[k], ta[k]) for k in pre)


def test_soup_matches_swa_averaged_model():
    torch.manual_seed(1)
    nets = [nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)) for _ in range(4)]
    avg = AveragedModel(nets[0])  # a copy; the first update_parameters call overwrites it with its argument
    for m in nets:
        avg.update_parameters(m)
    soup = uniform_soup([m.state_dict() for m in nets])
    got = {k.removeprefix("module."): v for k, v in avg.state_dict().items() if k.startswith("module.")}
    assert all(torch.allclose(got[k], soup[k]) for k in soup)


def test_linear_head_weight_average_is_logit_average():
    torch.manual_seed(2)
    a, b = nn.Linear(6, 4), nn.Linear(6, 4)
    x = torch.randn(10, 6)
    mixed = nn.Linear(6, 4)
    mixed.load_state_dict(interpolate(a.state_dict(), b.state_dict(), 0.3))
    assert torch.allclose(mixed(x), 0.7 * a(x) + 0.3 * b(x))
    # Through a nonlinearity the two are different.
    deep = [nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4)) for _ in range(2)]
    mid = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4))
    mid.load_state_dict(interpolate(deep[0].state_dict(), deep[1].state_dict(), 0.5))
    assert not torch.allclose(mid(x), 0.5 * deep[0](x) + 0.5 * deep[1](x))


def test_dare_unbiased():
    pre = {"w": torch.zeros(200_000)}
    ft = {"w": torch.full((200_000,), 0.002)}
    g = torch.Generator().manual_seed(3)
    out = dare(ft, pre, 0.9, g)["w"]
    assert abs((out == 0).double().mean().item() - 0.9) < 0.005
    assert abs(out.mean().item() - 0.002) < 0.0001
    assert torch.allclose(out[out != 0], torch.full_like(out[out != 0], 0.02))
    assert torch.equal(dare(ft, pre, 0.0, g)["w"], ft["w"])
    with pytest.raises(ValueError):
        dare(ft, pre, 1.0, g)


def test_diagonal_fisher_of_softmax_regression():
    torch.manual_seed(4)
    model = nn.Linear(3, 4)
    x = torch.randn(20, 3)
    fisher = diagonal_fisher(model, x)
    p = F.softmax(model(x), dim=-1).detach()
    v = p * (1 - p)  # sum_y p_y (delta_yk - p_k)^2
    assert torch.allclose(fisher["weight"], v.T @ x**2 / len(x))
    assert torch.allclose(fisher["bias"], v.mean(dim=0))


def test_fisher_merge():
    s1 = {"w": torch.tensor([1.0, 0.0, 2.0]), "head": torch.tensor([7.0])}
    s2 = {"w": torch.tensor([0.0, 1.0, 4.0]), "head": torch.tensor([9.0])}
    f1, f2 = {"w": torch.tensor([3.0, 1.0, 0.0])}, {"w": torch.tensor([1.0, 1.0, 0.0])}
    out = fisher_merge([s1, s2], [f1, f2])
    assert torch.allclose(out["w"], torch.tensor([0.75, 0.5, 2.0])) and out["head"].item() == 7.0
    out = fisher_merge([s1, s2], [f1, f2], target=1)
    assert torch.allclose(out["w"], torch.tensor([0.75, 0.5, 4.0])) and out["head"].item() == 9.0
    eq = fisher_merge([s1, s2], [{"w": torch.ones(3)}, {"w": torch.ones(3)}], lambdas=[0.25, 0.75])
    assert torch.allclose(eq["w"], 0.25 * s1["w"] + 0.75 * s2["w"])
