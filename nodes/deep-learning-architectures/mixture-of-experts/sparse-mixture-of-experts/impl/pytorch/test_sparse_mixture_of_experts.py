# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://1H67K0).
"""Tests of the PyTorch code: routing against softmax(KeepTopK), the capacity arithmetic, the dispatch tensor on the
worked example of the page and against a sequential loop, the Switch balance loss at uniform, collapsed and
adversarial routing with its gradient through P only, the z-loss and its gradient, the importance loss, and the layer
against a token-by-token loop, with gradients reaching the router and the experts and noise only in training."""

import math

import pytest
import torch
import torch.nn.functional as F

from sparse_mixture_of_experts import (
    SparseMoE,
    cv_squared,
    dispatch_mask,
    expert_capacity,
    importance_loss,
    switch_balance_loss,
    top_k_route,
    z_loss,
)

torch.set_default_dtype(torch.float64)


def grad_of(t):
    assert t.grad is not None
    return t.grad


def gen(seed):
    return torch.Generator().manual_seed(seed)


def test_top_k_route_matches_keep_top_k():
    g = gen(1)
    for k in (1, 2, 3):
        logits = torch.randn(9, 6, generator=g) * 3.0
        probs, idx, gates = top_k_route(logits, k)
        assert torch.allclose(probs, torch.softmax(logits, dim=-1))
        masked = torch.full_like(logits, -math.inf).scatter(1, idx, logits.gather(1, idx))
        ref = torch.softmax(masked, dim=-1).gather(1, idx)
        if k == 1:
            assert torch.allclose(gates, probs.gather(1, idx))
            assert (idx[:, 0] == logits.argmax(dim=-1)).all()
        else:
            assert torch.allclose(gates, ref)
            assert torch.allclose(gates.sum(dim=-1), torch.ones(9))
        assert (gates[:, :-1] >= gates[:, 1:]).all()
    _, _, g1 = top_k_route(torch.randn(4, 5, generator=g), 1, renormalize=True)
    assert torch.allclose(g1, torch.ones(4, 1))
    with pytest.raises(ValueError):
        top_k_route(torch.zeros(2, 3), 4)
    with pytest.raises(ValueError):
        top_k_route(torch.zeros(2, 3), 0)


def test_expert_capacity_arithmetic():
    assert expert_capacity(8, 4, 1.0) == 2
    assert expert_capacity(8, 4, 1.5) == 3
    assert expert_capacity(8, 4, 1.0, k=2) == 4
    assert expert_capacity(1024, 128, 1.25) == 10
    assert expert_capacity(10, 4, 1.0) == 2
    with pytest.raises(ValueError):
        expert_capacity(8, 0, 1.0)
    with pytest.raises(ValueError):
        expert_capacity(-3, 2, 1.0)
    with pytest.raises(ValueError):
        expert_capacity(8, 4, -1.0)


def _loop_dispatch(idx, n, cap):
    t, k = idx.shape
    count = [0] * n
    out = torch.zeros(t, k, n, max(cap, 0), dtype=torch.bool)
    for r in range(k):
        for s in range(t):
            e = int(idx[s, r])
            if count[e] < cap:
                out[s, r, e, count[e]] = True
            count[e] += 1
    return out


def test_dispatch_worked_example():
    idx = torch.tensor([[0], [1], [0], [2], [0], [1], [3], [0]])
    mask = dispatch_mask(idx, 4, 2)
    assert mask.shape == (8, 1, 4, 2)
    kept = mask.any(dim=(2, 3))[:, 0]
    assert kept.tolist() == [True, True, True, True, False, True, True, False]
    assert mask[2, 0, 0, 1] and mask[5, 0, 1, 1] and mask[6, 0, 3, 0]
    assert int(mask.sum()) == 6
    assert int(dispatch_mask(idx, 4, 3).sum()) == 7
    assert int(dispatch_mask(idx, 4, 4).sum()) == 8
    assert dispatch_mask(idx, 4, 0).shape == (8, 1, 4, 0)


def test_dispatch_first_choices_before_second():
    idx = torch.tensor([[1, 0], [0, 1]])
    mask = dispatch_mask(idx, 2, 1)
    assert mask.any(dim=(2, 3)).tolist() == [[True, False], [True, False]]


def test_dispatch_matches_loop():
    g = gen(2)
    for _ in range(40):
        n = int(torch.randint(2, 8, (1,), generator=g))
        k = int(torch.randint(1, min(n, 3) + 1, (1,), generator=g))
        t = int(torch.randint(1, 30, (1,), generator=g))
        cap = int(torch.randint(1, t + 1, (1,), generator=g))
        _, idx, _ = top_k_route(torch.randn(t, n, generator=g), k)
        mask = dispatch_mask(idx, n, cap)
        assert torch.equal(mask, _loop_dispatch(idx, n, cap))
        assert (mask.sum(dim=(0, 1)) <= 1).all()


def test_balance_loss_values():
    n = 4
    probs = torch.full((n * 3, n), 1.0 / n)
    probs[torch.arange(n * 3), torch.arange(n * 3) % n] += 1e-6
    probs = probs / probs.sum(dim=-1, keepdim=True)
    assert float(switch_balance_loss(probs, alpha=0.01)) == pytest.approx(0.01, rel=1e-4)
    collapsed = torch.zeros(10, n)
    collapsed[:, 2] = 1.0
    assert float(switch_balance_loss(collapsed, alpha=0.01)) == pytest.approx(0.04)
    adversarial = torch.tensor([[0.5 + 1e-9, 0.5 - 1e-9]] * 3 + [[0.0, 1.0]])
    assert float(switch_balance_loss(adversarial, alpha=1.0)) == pytest.approx(0.875, abs=1e-6)


def test_balance_loss_brute_force_and_default_alpha():
    g = gen(3)
    probs = torch.softmax(torch.randn(13, 5, generator=g) * 2.0, dim=-1)
    top = probs.argmax(dim=-1)
    total = sum(float((top == i).double().mean() * probs[:, i].mean()) for i in range(5))
    assert float(switch_balance_loss(probs, alpha=0.7)) == pytest.approx(0.7 * 5 * total)
    assert float(switch_balance_loss(probs)) == pytest.approx(0.01 * 5 * total)


def test_balance_loss_gradient_is_alpha_n_f_over_t():
    g = gen(4)
    logits = torch.randn(6, 3, generator=g)
    probs = torch.softmax(logits, dim=-1).detach().requires_grad_(True)
    loss = switch_balance_loss(probs, alpha=0.5)
    loss.backward()
    f = torch.bincount(probs.argmax(dim=-1), minlength=3).double() / 6
    assert torch.allclose(grad_of(probs), (0.5 * 3 * f / 6).expand(6, 3))


def test_z_loss_value_and_gradient():
    assert float(z_loss(torch.zeros(3, 4))) == pytest.approx(math.log(4.0) ** 2)
    x = torch.tensor([[1.0, 2.0], [0.0, -1.0]], requires_grad=True)
    expect = (math.log(math.e + math.e**2) ** 2 + math.log(1.0 + math.exp(-1.0)) ** 2) / 2
    assert float(z_loss(x.detach())) == pytest.approx(expect)
    assert torch.autograd.gradcheck(z_loss, (torch.randn(5, 3, generator=gen(5), requires_grad=True),))
    z_loss(x).backward()
    lse = torch.logsumexp(x.detach(), dim=-1, keepdim=True)
    assert torch.allclose(grad_of(x), lse * torch.softmax(x.detach(), dim=-1))


def test_cv_and_importance():
    assert float(cv_squared(torch.tensor([2.0, 2.0]))) == 0.0
    assert float(cv_squared(torch.tensor([0.0, 0.0]))) == 0.0
    assert float(cv_squared(torch.tensor([1.0, 3.0]))) == pytest.approx(0.25)
    skew = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert float(importance_loss(skew, 0.2)) == pytest.approx(0.05)
    assert float(importance_loss(skew)) == pytest.approx(0.025)
    even = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert float(importance_loss(even)) == 0.0


def _layer_loop(layer, x):
    t = x.shape[0]
    n, k = layer.n_experts, layer.k
    probs = torch.softmax(x @ layer.w_router, dim=-1)
    cap = int(math.floor(k * t * layer.capacity_factor / n + 1e-9))
    order = torch.argsort(probs, dim=-1, descending=True)[:, :k]
    count = [0] * n
    y = torch.zeros_like(x)
    for r in range(k):
        for s in range(t):
            e = int(order[s, r])
            if count[e] < cap:
                gate = probs[s, e] if k == 1 else probs[s, e] / probs[s, order[s]].sum()
                y[s] += gate * (torch.relu(x[s] @ layer.w_in[e]) @ layer.w_out[e])
            count[e] += 1
    return y


@pytest.mark.parametrize("k,cf", [(1, 1.0), (1, 0.5), (1, 2.0), (2, 1.0), (2, 1.25), (3, 0.75)])
def test_layer_matches_loop(k, cf):
    torch.manual_seed(10 * k + int(cf * 4))
    layer = SparseMoE(5, 7, 4, k=k, capacity_factor=cf)
    with torch.no_grad():
        layer.w_router.normal_()
    x = torch.randn(16, 5)
    y, info = layer(x)
    with torch.no_grad():
        assert torch.allclose(y, _layer_loop(layer, x))
    assert info["capacity"] == expert_capacity(16, 4, cf, k)
    assert info["dropped"] == int((~info["kept"]).sum())
    per_expert = torch.stack([((info["indices"] == e) & info["kept"]).sum() for e in range(4)])
    assert (per_expert <= info["capacity"]).all()
    assert float(info["balance_loss"]) == pytest.approx(float(switch_balance_loss(info["probs"])))
    assert float(info["z_loss"]) == pytest.approx(float(z_loss(x @ layer.w_router)))


def test_layer_drops_and_gradients():
    torch.manual_seed(0)
    layer = SparseMoE(3, 4, 2, k=1, capacity_factor=1.0)
    with torch.no_grad():
        layer.w_router.zero_()
        layer.w_router[:, 0] = 5.0
    x = torch.rand(6, 3) + 0.1
    y, info = layer(x)
    assert info["capacity"] == 3 and info["dropped"] == 3
    assert torch.allclose(y[3:], torch.zeros(3, 3))
    loss = y.pow(2).sum() + info["balance_loss"] + 1e-3 * info["z_loss"]
    loss.backward()
    assert grad_of(layer.w_router).abs().sum() > 0
    assert grad_of(layer.w_in)[0].abs().sum() > 0
    assert torch.allclose(grad_of(layer.w_in)[1], torch.zeros(3, 4))


def test_noise_only_in_training():
    torch.manual_seed(1)
    layer = SparseMoE(4, 5, 6, k=2, capacity_factor=4.0, noisy=True)
    with torch.no_grad():
        layer.w_router.normal_()
        assert layer.w_noise is not None
        layer.w_noise.fill_(1.0)
    x = torch.randn(8, 4)
    layer.eval()
    a, _ = layer(x)
    b, _ = layer(x)
    assert torch.equal(a, b)
    assert torch.allclose(layer.router_logits(x), x @ layer.w_router)
    layer.train()
    assert not torch.allclose(layer.router_logits(x), x @ layer.w_router)
    zero = SparseMoE(4, 5, 6, k=2, noisy=True)
    zero.train()
    assert not torch.allclose(zero.router_logits(x), x @ zero.w_router)
    plain = SparseMoE(4, 5, 6, k=2, noisy=False)
    plain.train()
    assert torch.equal(plain.router_logits(x), x @ plain.w_router)
    with pytest.raises(ValueError):
        SparseMoE(4, 5, 3, k=4)


def test_argument_edges():
    assert expert_capacity(0, 4, 1.0) == 0
    assert expert_capacity(8, 1, 1.0) == 8
    assert expert_capacity(4, 4, 1.0 - 5e-10) == 1
    assert expert_capacity(4, 4, 1.0 - 5e-9) == 0
    with pytest.raises(ValueError):
        expert_capacity(8, 4, 0.0)
    _, idx, gates = top_k_route(torch.zeros(2, 3), 3)
    assert idx.shape == (2, 3) and torch.allclose(gates, torch.full((2, 3), 1.0 / 3))
    assert SparseMoE(4, 5, 3).k == 1
    assert SparseMoE(4, 5, 3, k=3).k == 3
    with pytest.raises(ValueError):
        SparseMoE(4, 5, 3, k=0)


def test_initial_scales():
    torch.manual_seed(2)
    layer = SparseMoE(64, 256, 4)
    assert float(layer.w_in.std()) == pytest.approx(1.0 / 8.0, rel=0.05)
    assert float(layer.w_out.std()) == pytest.approx(1.0 / 16.0, rel=0.05)
    assert torch.equal(layer.w_router, torch.zeros(64, 4))


def test_noisy_logits_formula():
    layer = SparseMoE(3, 4, 5, k=2, noisy=True)
    with torch.no_grad():
        layer.w_router.normal_()
        assert layer.w_noise is not None
        layer.w_noise.normal_()
    layer.train()
    x = torch.randn(7, 3)
    torch.manual_seed(9)
    got = layer.router_logits(x)
    torch.manual_seed(9)
    clean = x @ layer.w_router
    noise = torch.randn_like(clean)
    assert torch.allclose(got, clean + noise * F.softplus(x @ layer.w_noise))
