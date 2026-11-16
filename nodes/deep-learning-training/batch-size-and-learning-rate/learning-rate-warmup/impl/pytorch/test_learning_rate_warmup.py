# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/learning-rate-warmup/ (ai://NEW-learning-rate-warmup).
"""Tests of the PyTorch code: the schedulers against the formulas of the page, Goyal et al.'s warmup and step decay,
the Transformer schedule, and the shared RAdam fixture against torch.optim.RAdam with its threshold rho_t > 5."""

import pytest
import torch

from learning_rate_warmup import (
    gradual_warmup,
    lrs,
    noam,
    noam_factor,
    radam_rectification,
    radam_rho,
    warmup_then_steps,
)

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
P0 = [0.5, -1.5, 2.0]
GRADS = [
    [1.0, -2.0, 0.5],
    [0.5, 1.0, -0.25],
    [-1.5, 0.5, 0.0],
    [2.0, -0.5, 1.0],
    [0.25, 0.75, -1.0],
    [1.0, 1.0, 1.0],
]
AFTER6_TORCH = [0.23675198784933882, -1.2330182831375551, 1.8971525192034477]


def sgd(lr):
    w = torch.zeros(2, requires_grad=True)
    w.grad = torch.zeros(2)
    return torch.optim.SGD([w], lr=lr)


def test_gradual_warmup_matches_eq_gradual():
    opt = sgd(3.2)
    got = lrs(gradual_warmup(opt, 10, 0.1), opt, 14)
    want = [0.1 + (3.2 - 0.1) * min(t, 10) / 10 for t in range(14)]
    assert got == pytest.approx(want)


def test_goyal_warmup_then_step_decay():
    opt = sgd(3.2)
    got = lrs(warmup_then_steps(opt, 5, 0.1, milestones=[30, 60, 80]), opt, 90)
    assert got[0] == pytest.approx(0.1)
    assert got[5:30] == pytest.approx([3.2] * 25)
    assert got[30:60] == pytest.approx([0.32] * 30)
    assert got[60:80] == pytest.approx([0.032] * 20)
    assert got[80:] == pytest.approx([0.0032] * 10)


def test_noam_schedule():
    opt = sgd(1.0)
    got = lrs(noam(opt, 512, 4000), opt, 16000)
    peak = (512 * 4000) ** -0.5
    assert got[0] == pytest.approx(peak / 4000)
    assert got[999] == pytest.approx(peak / 4)
    assert max(got) == pytest.approx(peak)
    assert got.index(max(got)) == 3999
    assert got[15999] == pytest.approx(peak / 2)
    assert noam_factor(512, 4000)(99999) == pytest.approx(0.2 * peak)


def test_bad_arguments():
    with pytest.raises(ValueError):
        gradual_warmup(sgd(1.0), 0, 0.1)
    with pytest.raises(ValueError):
        gradual_warmup(sgd(1.0), 10, 2.0)
    with pytest.raises(ValueError):
        gradual_warmup(sgd(1.0), 10, 0.0)


def test_edge_arguments():
    # One warmup step and a start equal to the lr are allowed; the second is a constant rate.
    opt = sgd(1.0)
    assert lrs(gradual_warmup(opt, 1, 0.5), opt, 3) == pytest.approx([0.5, 1.0, 1.0])
    opt = sgd(1.0)
    assert lrs(gradual_warmup(opt, 10, 1.0), opt, 3) == pytest.approx([1.0, 1.0, 1.0])


def test_lrs_steps_the_optimizer():
    opt = sgd(0.5)
    w = opt.param_groups[0]["params"][0]
    w.grad = torch.ones(2)
    lrs(gradual_warmup(opt, 2, 0.25), opt, 3)
    # SGD updates by 0.25, 0.375 and 0.5 times the gradient 1.
    assert w.detach().tolist() == pytest.approx([-1.125, -1.125])


def run_torch(grads, betas=(0.9, 0.999)):
    w = torch.tensor(P0, requires_grad=True)
    opt = torch.optim.RAdam([w], lr=0.1, betas=betas, eps=1e-8)
    for g in grads:
        w.grad = torch.tensor(g)
        opt.step()
    return w.detach()


def test_radam_fixture():
    assert run_torch(GRADS).tolist() == pytest.approx(AFTER6_TORCH, abs=1e-12)


def test_torch_radam_threshold_is_five():
    # Up to step 5, torch.optim.RAdam takes plain momentum steps: rho_5 = 4.996 is not above 5.
    m, q = torch.zeros(3), torch.tensor(P0)
    for t, g in enumerate(GRADS[:5], start=1):
        m = 0.9 * m + 0.1 * torch.tensor(g)
        q = q - 0.1 * m / (1 - 0.9**t)
    assert run_torch(GRADS[:5]).tolist() == pytest.approx(q.tolist(), abs=1e-12)
    assert 4.0 < radam_rho(5, 0.999) <= 5.0 < radam_rho(6, 0.999)
    assert radam_rectification(5, 0.999) is None
    assert radam_rectification(6, 0.999) == pytest.approx(0.026, abs=5e-4)
    assert radam_rectification(5, 0.999, threshold=4.0) == pytest.approx(0.017, abs=5e-4)


def test_rectification_values():
    # r_t of eq-rect with beta2 = 0.999, and the threshold compared with a strict >.
    assert radam_rectification(6, 0.999) == pytest.approx(0.02582111280185855, rel=1e-9)
    assert radam_rectification(100, 0.999) == pytest.approx(0.21533543653820142, rel=1e-9)
    assert radam_rectification(1000, 0.999) == pytest.approx(0.6453256523380656, rel=1e-9)
    assert radam_rectification(6, 0.999, threshold=radam_rho(6, 0.999)) is None


def test_radam_degenerates_to_momentum():
    # beta2 = 0.5 gives rho_inf = 3: no step is adaptive.
    m, q = torch.zeros(3), torch.tensor(P0)
    for t, g in enumerate(GRADS, start=1):
        m = 0.9 * m + 0.1 * torch.tensor(g)
        q = q - 0.1 * m / (1 - 0.9**t)
    assert run_torch(GRADS, betas=(0.9, 0.5)).tolist() == pytest.approx(q.tolist(), abs=1e-12)
