# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://17Q1PD).
"""Tests of the PyTorch code: gradient descent on a quadratic converges just below 2 / lambda_max and diverges just
above; the autograd HVP against the full Hessian; power iteration against eigvalsh, on a quadratic and on a small
tanh MLP; full-batch gradient descent against a hand-computed step; and the small experiment of the page, in which
the sharpness rises to 2 / eta and hovers there while the loss goes down non-monotonically."""

import pytest
import torch

from edge_of_stability import (
    TanhMLP,
    demo,
    hvp,
    make_regression,
    mode_factors,
    mse,
    power_iteration,
    quadratic_gd,
    sharpness,
    train_gd,
)


def random_psd(d, gen):
    A = torch.randn(d, d, generator=gen)
    return A @ A.T / d


def flat_hessian(model, X, Y):
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def loss_of(flat):
        chunks = torch.split(flat, sizes)
        h = X
        for k in range(0, len(chunks), 2):
            W = chunks[k].reshape(shapes[k])
            b = chunks[k + 1]
            h = h @ W.T + b
            if k < len(chunks) - 2:
                h = torch.tanh(h)
        return 0.5 * ((h - Y) ** 2).sum() / X.shape[0]

    flat = torch.cat([p.detach().reshape(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_of, flat)
    assert isinstance(H, torch.Tensor)
    return H


def test_mode_factors():
    assert torch.allclose(mode_factors([20.0, 1.0], 0.11), torch.tensor([-1.2, 0.89]))
    assert torch.allclose(mode_factors([20.0, 1.0], 0.09), torch.tensor([-0.8, 0.91]))


def test_quadratic_below_and_above_threshold():
    gen = torch.Generator().manual_seed(0)
    H = random_psd(5, gen)
    lam, Q = torch.linalg.eigh(H)
    x0 = torch.randn(5, generator=gen)
    below = quadratic_gd(H, x0, 0.99 * 2 / lam[-1], 500)
    above = quadratic_gd(H, x0, 1.01 * 2 / lam[-1], 500)
    assert below.shape == (501, 5)
    assert torch.allclose(below[0], x0)
    assert below[-1].norm() < 0.05 * x0.norm()
    assert above[-1].norm() > 10 * x0.norm()
    top = above @ Q[:, -1]
    expect = (Q[:, -1] @ x0) * mode_factors(lam[-1:], 1.01 * 2 / lam[-1])[0] ** torch.arange(501)
    assert torch.allclose(top, expect, rtol=1e-8, atol=1e-10)


def test_hvp_matches_full_hessian():
    gen = torch.Generator().manual_seed(1)
    X, Y = make_regression(12, 3, gen)
    torch.manual_seed(1)
    model = TanhMLP([3, 4, 2])
    H = flat_hessian(model, X, Y)
    params = list(model.parameters())
    v = torch.randn(H.shape[0], generator=gen)
    assert torch.allclose(hvp(lambda: mse(model, X, Y), params, v), H @ v, atol=1e-10)


def test_power_iteration_on_quadratic():
    gen = torch.Generator().manual_seed(2)
    for d in (4, 12):
        H = random_psd(d, gen)
        lam, q = power_iteration(
            lambda v, H=H: H @ v,
            d,
            iters=3000,
            generator=torch.Generator().manual_seed(d),
        )
        assert lam == pytest.approx(float(torch.linalg.eigvalsh(H)[-1]), rel=1e-6)
        assert float(q.norm()) == pytest.approx(1.0)
    lam, _ = power_iteration(lambda v: torch.diag(torch.tensor([2.0, -6.0])) @ v, 2, 300, torch.Generator())
    assert lam == pytest.approx(-6.0)
    lam, q = power_iteration(lambda v: 0 * v, 3, 4, torch.Generator())
    assert lam == 0.0 and float(q.norm()) == pytest.approx(1.0)
    lam, q = power_iteration(lambda v: 2 * v, 3, 0, torch.Generator())
    assert lam == 0.0 and float(q.norm()) == pytest.approx(1.0)
    one, _ = power_iteration(lambda v: torch.diag(torch.tensor([1.0, 0.99, 0.5])) @ v, 3, 1, torch.Generator())
    assert abs(one - 1.0) > 1e-3


def test_sharpness_of_mlp_matches_eigvalsh():
    gen = torch.Generator().manual_seed(3)
    X, Y = make_regression(20, 3, gen)
    torch.manual_seed(3)
    model = TanhMLP([3, 6, 1])
    train_gd(model, X, Y, 0.2, 200, 1000, 5)
    ev = torch.linalg.eigvalsh(flat_hessian(model, X, Y))
    assert ev[-1] > abs(ev[0])
    assert sharpness(model, X, Y, 3000, 0) == pytest.approx(float(ev[-1]), rel=1e-6)


def test_init_and_mse():
    torch.manual_seed(0)
    model = TanhMLP([5, 400, 3])
    l1, l2 = model.layers[0], model.layers[1]
    assert isinstance(l1, torch.nn.Linear) and isinstance(l2, torch.nn.Linear)
    w1, w2 = l1.weight.detach(), l2.weight.detach()
    assert float(w1.std()) == pytest.approx(5**-0.5, rel=0.05)
    assert float(w2.std()) == pytest.approx(400**-0.5, rel=0.05)
    assert bool((l1.bias == 0).all()) and bool((l2.bias == 0).all())
    X = torch.randn(4, 5)
    out = torch.tanh(X @ w1.T) @ w2.T
    assert torch.allclose(model(X).detach(), out)
    Y = torch.zeros(4, 3)
    assert float(mse(model, X, Y)) == pytest.approx(float(0.5 * (out**2).sum() / 4))


def test_train_gd_step_and_records():
    gen = torch.Generator().manual_seed(4)
    X, Y = make_regression(10, 3, gen)
    assert X.shape == (10, 3) and Y.shape == (10, 1) and float(Y.abs().max()) <= 1
    again = torch.Generator().manual_seed(4)
    X2, w = torch.randn(10, 3, generator=again), torch.randn(3, 1, generator=again)
    assert torch.equal(X, X2) and torch.allclose(Y, torch.sin(2.0 * X2 @ w))
    torch.manual_seed(4)
    model = TanhMLP([3, 5, 1])
    before = [p.detach().clone() for p in model.parameters()]
    grads = torch.autograd.grad(mse(model, X, Y), list(model.parameters()))
    l0 = float(mse(model, X, Y))
    s0 = sharpness(model, X, Y, 5, 0)
    losses, at, sharp = train_gd(model, X, Y, 0.1, 1, 1, 5)
    assert sharp[0] == s0
    for p, b, g in zip(model.parameters(), before, grads, strict=True):
        assert torch.allclose(p, b - 0.1 * g)
    assert at == [0, 1] and len(sharp) == 2 and len(losses) == 2
    assert float(losses[0]) == pytest.approx(l0)
    assert float(losses[1]) == pytest.approx(float(mse(model, X, Y)))
    losses, at, _ = train_gd(model, X, Y, 0.05, 30, 10, 5)
    assert at == [0, 10, 20, 30] and bool(torch.all(losses[1:] < losses[:-1]))


def test_sharpness_rises_to_two_over_eta_and_hovers():
    r = demo(0, 0.25, 1600)
    assert r["sizes"] == [5, 16, 16, 1] and r["examples"] == 64
    assert r["measured_at"] == list(range(0, 1601, 100)) and len(r["losses"]) == 1601
    thr = r["threshold"]
    assert thr == 8.0
    assert r["sharpness"][0] < 0.6 * thr
    assert len(late(r)) == 7
    assert all(abs(s - thr) < 0.05 * thr for s in late(r))
    assert int((r["losses"][1:] > r["losses"][:-1]).sum()) > 100  # the loss goes up on many steps
    assert float(r["losses"][-1]) < float(r["losses"][0]) / 100


def test_demo_is_seeded():
    a, b = demo(1, 0.25, 0), demo(1, 0.25, 0)
    assert a["sharpness"] == b["sharpness"] and len(a["losses"]) == 1


def late(r):
    """The sharpness measured at steps 1000 to 1600 of a demo run."""
    return [s for t, s in zip(r["measured_at"], r["sharpness"], strict=True) if t >= 1000]
