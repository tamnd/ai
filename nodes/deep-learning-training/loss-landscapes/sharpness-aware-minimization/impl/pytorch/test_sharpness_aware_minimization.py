# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sharpness-aware-minimization/ (ai://8BNWST).
"""Tests of the PyTorch code: the perturbation against its closed forms and against brute force on the p-ball; the SAM
optimizer around SGD, which must give w - lr * grad L(w + eps_hat) with the parameters back at w before the base step,
with momentum, with several parameter tensors and through step(closure); training a small network; the
filter-normalized direction; and the worked example of the page."""

import math

import pytest
import torch

from sharpness_aware_minimization import SAM, filter_normalized_direction, sam_perturbation

torch.manual_seed(0)
DT = torch.float64


def test_perturbation_closed_forms():
    g = [torch.tensor([3.0, -4.0, 0.0], dtype=DT)]
    assert torch.allclose(sam_perturbation(g, 0.5)[0], torch.tensor([0.3, -0.4, 0.0], dtype=DT))
    assert torch.allclose(sam_perturbation(g, 0.5, math.inf)[0], torch.tensor([0.5, -0.5, 0.0], dtype=DT))
    e = sam_perturbation([torch.tensor([1.0, 8.0], dtype=DT)], 1.0, 4.0)[0]
    s = (1.0 + 8.0 ** (4 / 3)) ** 0.25
    assert torch.allclose(e, torch.tensor([1.0 / s, 2.0 / s], dtype=DT))
    # several tensors are one vector
    a, b = torch.randn(2, 3, dtype=DT), torch.randn(4, dtype=DT)
    flat = torch.cat([a.flatten(), b])
    for p in [2.0, 3.0, math.inf]:
        ea, eb = sam_perturbation([a, b], 0.3, p)
        assert ea.shape == a.shape
        assert torch.allclose(torch.cat([ea.flatten(), eb]), sam_perturbation([flat], 0.3, p)[0])


@pytest.mark.parametrize("p", [1.5, 2.0, 3.0, math.inf])
def test_perturbation_brute_force(p):
    q = 1.0 if math.isinf(p) else p / (p - 1.0)
    for _ in range(10):
        g = torch.randn(3, dtype=DT)
        e = sam_perturbation([g], 0.7, p)[0]
        assert torch.linalg.vector_norm(e, ord=p).item() == pytest.approx(0.7, rel=1e-9)
        assert (e @ g).item() == pytest.approx(0.7 * torch.linalg.vector_norm(g, ord=q).item(), rel=1e-9)
        z = torch.randn(4000, 3, dtype=DT)
        z = 0.7 * z / torch.linalg.vector_norm(z, ord=p, dim=1, keepdim=True)
        assert (z @ g).max().item() <= (e @ g).item() + 1e-9


def test_perturbation_edge_cases():
    assert torch.all(sam_perturbation([torch.zeros(3), torch.zeros(2, 2)], 0.5, 3.0)[1] == 0)
    assert torch.all(sam_perturbation([torch.zeros(3)], 0.5)[0] == 0)
    e = sam_perturbation([torch.tensor([0.5, -200.0, 100.0], dtype=DT)], 1.0, 1.001)[0]
    assert torch.isfinite(e).all() and e[1].item() == pytest.approx(-1.0, abs=1e-6)
    for c in [1e-6, 7.0, 1e6]:
        g = torch.tensor([1.0, -2.0, 0.5], dtype=DT)
        assert torch.allclose(sam_perturbation([c * g], 0.2, 3.0)[0], sam_perturbation([g], 0.2, 3.0)[0])
    with pytest.raises(ValueError):
        sam_perturbation([torch.ones(2)], -0.1)
    with pytest.raises(ValueError):
        sam_perturbation([torch.ones(2)], 0.1, 1.0)


def quad_setup(w0=(1.0, 2.0), h=(4.0, 1.0)):
    w = torch.nn.Parameter(torch.tensor(w0, dtype=DT))
    H = torch.diag(torch.tensor(h, dtype=DT))

    def loss():
        return 0.5 * w @ H @ w

    return w, H, loss


def test_two_passes_match_closed_form():
    w, H, loss = quad_setup()
    opt = SAM([w], torch.optim.SGD, rho=0.5, lr=0.1)
    w0 = w.detach().clone()
    loss().backward()
    opt.first_step(zero_grad=True)
    g = H @ w0
    e = 0.5 * g / g.norm()
    assert torch.allclose(w.detach(), w0 + e)
    assert w.grad is None or torch.all(w.grad == 0)
    loss().backward()
    assert torch.allclose(w.grad, H @ (w0 + e))
    opt.second_step(zero_grad=True)
    assert torch.allclose(w.detach(), w0 - 0.1 * H @ (w0 + e))
    assert torch.allclose(w.detach(), torch.tensor([0.42111456, 1.77763932], dtype=DT))
    assert "e_w" not in opt.state[w]


def test_step_with_closure_and_rho_zero():
    w, H, loss = quad_setup()

    def closure():
        lo = loss()
        lo.backward()
        return lo

    opt = SAM([w], torch.optim.SGD, rho=0.5, lr=0.1)
    out = opt.step(closure)
    assert out.item() == pytest.approx(4.0)
    assert torch.allclose(w.detach(), torch.tensor([0.42111456, 1.77763932], dtype=DT))
    with pytest.raises(ValueError):
        opt.step()
    w2, H2, loss2 = quad_setup()

    def closure2():
        lo = loss2()
        lo.backward()
        return lo

    SAM([w2], torch.optim.SGD, rho=0.0, lr=0.1).step(closure2)
    assert torch.allclose(w2.detach(), torch.tensor([0.6, 1.8], dtype=DT))
    with pytest.raises(ValueError):
        SAM([w2], torch.optim.SGD, rho=-1.0, lr=0.1)


def test_momentum_matches_numpy_rule():
    w, H, loss = quad_setup()
    opt = SAM([w], torch.optim.SGD, rho=0.5, lr=0.1, momentum=0.9)
    ref = w.detach().clone()
    buf = None
    for _ in range(3):
        g = H @ ref
        ga = H @ (ref + 0.5 * g / g.norm())
        buf = ga if buf is None else 0.9 * buf + ga
        ref = ref - 0.1 * buf
        loss().backward()
        opt.first_step(zero_grad=True)
        loss().backward()
        opt.second_step(zero_grad=True)
    assert torch.allclose(w.detach(), ref)


def test_global_norm_over_tensors_and_groups():
    a = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=DT))
    b = torch.nn.Parameter(torch.tensor([0.5], dtype=DT))
    c = torch.nn.Parameter(torch.tensor([2.0], dtype=DT))

    def loss():
        return a[0] ** 2 + 1.5 * a[1] ** 2 + 2.5 * b[0] ** 2 + 0.0 * c.sum()

    opt = SAM([a, b], torch.optim.SGD, rho=0.4, lr=0.1)
    loss().backward()
    opt.first_step()
    g = torch.tensor([2.0, -3.0, 2.5], dtype=DT)
    eh = 0.4 * g / g.norm()
    assert torch.allclose(torch.cat([a.detach(), b.detach()]), torch.tensor([1.0, -1.0, 0.5], dtype=DT) + eh)
    # a parameter with no gradient is not moved
    opt2 = SAM([c], torch.optim.SGD, rho=0.4, lr=0.1)
    opt2.first_step()
    opt2.second_step()
    assert c.item() == 2.0


def test_sharp_direction_shrinks_faster():
    w, H, loss = quad_setup((1.0, 1.0), (10.0, 0.1))
    v = torch.tensor([1.0, 1.0], dtype=DT)
    opt = SAM([w], torch.optim.SGD, rho=0.05, lr=0.01)
    for _ in range(100):
        v = v - 0.01 * H @ v
        loss().backward()
        opt.first_step(zero_grad=True)
        loss().backward()
        opt.second_step(zero_grad=True)
        assert abs(w[0].item() / w[1].item()) < abs(v[0].item() / v[1].item())


def test_trains_a_small_network():
    torch.manual_seed(1)
    X = torch.randn(64, 4, dtype=DT)
    y = (X[:, :1] * 2 - X[:, 1:2]).tanh()
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).to(DT)
    opt = SAM(net.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)

    def closure():
        lo = ((net(X) - y) ** 2).mean()
        lo.backward()
        return lo

    first = closure().item()
    net.zero_grad()
    for _ in range(200):
        last = opt.step(closure).item()
    assert last < 0.3 * first


def test_filter_normalized_direction():
    conv = torch.randn(4, 3, 3, 3, dtype=DT)
    fc = torch.randn(5, 6, dtype=DT) * 10
    bias = torch.randn(5, dtype=DT)
    d = filter_normalized_direction([conv, fc, bias], torch.Generator().manual_seed(0))
    for theta, dd in [(conv, d[0]), (fc, d[1])]:
        assert dd.shape == theta.shape
        assert torch.allclose(theta.flatten(1).norm(dim=1), dd.flatten(1).norm(dim=1))
    assert torch.all(d[2] == 0)
    d2 = filter_normalized_direction([bias], torch.Generator().manual_seed(1), ignore_1d=False)
    assert torch.allclose(d2[0].abs(), bias.abs())
    z = torch.ones(2, 3, dtype=DT)
    z[1] = 0
    dz = filter_normalized_direction([z], torch.Generator().manual_seed(2))[0]
    assert torch.all(dz[1] == 0) and dz[0].norm().item() == pytest.approx(math.sqrt(3))
    scaled = fc.clone()
    scaled[1] *= 10
    e1 = filter_normalized_direction([fc], torch.Generator().manual_seed(5))[0]
    e2 = filter_normalized_direction([scaled], torch.Generator().manual_seed(5))[0]
    assert torch.allclose(e2[1], 10 * e1[1]) and torch.allclose(e2[0], e1[0])
    g = torch.randn(conv.shape, generator=torch.Generator().manual_seed(0), dtype=DT)
    cos = torch.nn.functional.cosine_similarity(g.flatten(1), d[0].flatten(1), dim=1)
    assert torch.allclose(cos, torch.ones(4, dtype=DT))
