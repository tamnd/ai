# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://NEW-neural-tangent-kernel).
"""Tests of the PyTorch version: the worked example, autograd against the hand Jacobian, the limit and the dynamics."""

import math

import pytest
import torch

from neural_tangent_kernel import (
    TwoLayer,
    empirical_ntk,
    flat_params,
    jacobian,
    relative_change,
    relu_ntk,
    residual_flow,
    train_linearized,
    train_network,
)

torch.set_default_dtype(torch.float64)

# The worked example of the page, the same numbers as the NumPy tests.
X_A = torch.tensor([[1.0, 0.0]])
X_B = torch.tensor([[0.5, math.sqrt(3) / 2]])
THETA_AB = 1 / 6 + (math.sqrt(3) / 2 + math.pi / 3) / (2 * math.pi)

# The shared data set: five points on the unit sphere of R^3.
XS = torch.tensor(
    [
        [0.2672612419, 0.5345224838, 0.8017837257],
        [-0.7071067812, 0.0, 0.7071067812],
        [0.5773502692, -0.5773502692, 0.5773502692],
        [0.0, -1.0, 0.0],
        [0.9486832981, 0.3162277660, 0.0],
    ]
)
YS = torch.sin(3.0 * XS[:, 0])


def gen(seed):
    return torch.Generator().manual_seed(seed)


def hand_jacobian(model, X):
    """The hand formula of eq-jacobian, as in the NumPy reference."""
    W, a = model.W.detach(), model.a.detach()
    m = a.shape[0]
    U = X @ W.T
    dW = ((U > 0).double() * a)[:, :, None] * X[:, None, :] / math.sqrt(m)
    return torch.cat([dW.reshape(X.shape[0], -1), torch.relu(U) / math.sqrt(m)], dim=1)


def test_worked_example():
    assert relu_ntk(X_A, X_B)[0, 0].item() == pytest.approx(THETA_AB, abs=1e-12)
    assert relu_ntk(X_A, X_B)[0, 0].item() == pytest.approx(0.4711655572, abs=1e-9)


def test_special_angles_and_edges():
    e1, e2 = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    assert relu_ntk(e1, e1).item() == pytest.approx(1.0)
    assert relu_ntk(e1, e2).item() == pytest.approx(1 / (2 * math.pi))
    assert relu_ntk(e1, -e1).item() == pytest.approx(0.0, abs=1e-15)
    Z = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    K = relu_ntk(Z, Z)
    assert torch.isfinite(K).all() and torch.allclose(K[0], torch.zeros(2))
    big = torch.tensor([[1e8, 1.0]])
    assert relu_ntk(big, big).item() == pytest.approx((big**2).sum().item(), rel=1e-12)


def test_diagonal_and_homogeneity():
    X = torch.tensor([[3.0, -1.0, 2.0], [0.5, 0.5, -4.0]])
    assert torch.allclose(torch.diag(relu_ntk(X, X)), (X**2).sum(dim=1))
    assert torch.allclose(relu_ntk(2.5 * X, X), 2.5 * relu_ntk(X, X))


def test_invalid_sizes():
    with pytest.raises(ValueError):
        TwoLayer(0, 3)
    with pytest.raises(ValueError):
        TwoLayer(3, 0)


def test_forward_by_hand():
    model = TwoLayer(2, 4)
    with torch.no_grad():
        model.W.copy_(torch.tensor([[1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-2.0, 0.5]]))
        model.a.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0]))
    assert model(torch.tensor([[2.0, -1.0]])).item() == pytest.approx(0.25)


def test_autograd_jacobian_matches_hand_formula():
    model = TwoLayer(3, 40, gen(1))
    J = jacobian(model, XS)
    assert J.shape == (5, 40 * 3 + 40)
    assert torch.allclose(J, hand_jacobian(model, XS), atol=1e-12)
    K = empirical_ntk(model, XS, XS)
    assert torch.allclose(K, K.T) and torch.linalg.eigvalsh(K).min() > -1e-12
    assert torch.allclose(empirical_ntk(model, XS[:2], XS), K[:2])


def test_empirical_ntk_converges_to_limit():
    A = relu_ntk(XS, XS)
    widths = [256, 4096, 65536]
    errs = []
    for m in widths:
        sq = [((empirical_ntk(TwoLayer(3, m, gen(100 + s)), XS, XS) - A) ** 2).mean() for s in range(6)]
        errs.append(torch.stack(sq).mean().sqrt().item())
    slope = (math.log(errs[-1]) - math.log(errs[0])) / (math.log(widths[-1]) - math.log(widths[0]))
    assert -0.65 < slope < -0.35
    assert errs[-1] < 0.05


def test_residual_flow_by_hand():
    K = torch.diag(torch.tensor([2.0, 0.5]))
    r = residual_flow(K, torch.tensor([1.0, -3.0]), eta=0.5, t=4.0, n=2)
    assert torch.allclose(r, torch.tensor([math.exp(-2.0), -3.0 * math.exp(-0.5)]))


def test_linearized_gd_is_the_discrete_kernel_map():
    model = TwoLayer(3, 200, gen(3))
    K = empirical_ntk(model, XS, XS)
    lam, V = torch.linalg.eigh(K)
    rs, _ = train_linearized(model, XS, YS, eta=0.7, steps=30)
    for k in (0, 1, 30):
        exact = V @ ((1 - 0.7 * lam / 5) ** k * (V.T @ rs[0]))
        assert torch.allclose(rs[k], exact, atol=1e-12)


def test_linearized_gd_tends_to_the_flow():
    model = TwoLayer(3, 200, gen(4))
    K = empirical_ntk(model, XS, XS)
    with torch.no_grad():
        r0 = model(XS) - YS
    flow = residual_flow(K, r0, eta=1.0, t=30.0, n=5)
    gaps = [
        (train_linearized(model, XS, YS, eta, int(30 / eta))[0][-1] - flow).abs().max().item() for eta in (0.5, 0.25)
    ]
    assert gaps[1] < gaps[0] < 0.01
    assert gaps[0] / gaps[1] == pytest.approx(2.0, rel=0.1)


def test_network_follows_linearized_model_more_closely_when_wider():
    gaps = []
    for m in (64, 4096):
        model = TwoLayer(3, m, gen(20))
        lin, _ = train_linearized(model, XS, YS, eta=1.0, steps=20)
        train_network(model, XS, YS, eta=1.0, steps=20)
        with torch.no_grad():
            gaps.append((model(XS) - YS - lin[-1]).abs().max().item())
    assert gaps[1] < gaps[0] / 4


def test_relative_change_falls_like_inverse_sqrt_width():
    widths = [32, 256, 2048]
    rel = []
    for m in widths:
        vals = []
        for s in range(3):
            model = TwoLayer(3, m, gen(30 + s))
            t0 = flat_params(model)
            train_network(model, XS, YS, eta=1.0, steps=100)
            vals.append(relative_change(t0, flat_params(model)).item())
        rel.append(sum(vals) / len(vals))
    slope = (math.log(rel[-1]) - math.log(rel[0])) / (math.log(widths[-1]) - math.log(widths[0]))
    assert -0.65 < slope < -0.35


def test_relative_change_by_hand():
    assert relative_change(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.5])).item() == pytest.approx(0.1)


def test_linearized_weights_stay_in_the_kernel_bound():
    model = TwoLayer(3, 400, gen(9))
    K = empirical_ntk(model, XS, XS)
    with torch.no_grad():
        r0 = model(XS) - YS
    eta = 5 / torch.linalg.eigvalsh(K).max().item()
    _, omega = train_linearized(model, XS, YS, eta=eta, steps=3000)
    bound = torch.sqrt(r0 @ torch.linalg.solve(K, r0)).item()
    assert omega.norm().item() <= bound + 1e-9
    assert omega.norm().item() == pytest.approx(bound, rel=1e-3)
