# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://NEW-mode-connectivity).
"""Tests of the PyTorch code: the endpoints and midpoints of the paths; the loss barrier in both conventions, zero
for a convex loss, 1 on the segment through a ring-shaped loss and 1/16 on a Bezier curve around it; the invariance
of the MLP under a hidden-unit permutation and the barrier between a network and its permuted copy; the Hungarian
solver against brute force; weight matching that recovers a known permutation; curve fitting by autograd; and the
small experiment of the page."""

import itertools
import math

import pytest
import torch

from mode_connectivity import (
    DTYPE,
    bezier_path,
    demo,
    error,
    fit_bend,
    flatten,
    forward,
    init_mlp,
    linear_assignment,
    linear_path,
    loss,
    loss_barrier,
    make_data,
    permute_hidden,
    polychain_path,
    train_gd,
    unflatten,
    weight_matching,
)

W1 = torch.tensor([1.0, -2.0, 0.5], dtype=DTYPE)
TH = torch.tensor([3.0, 4.0, -1.0], dtype=DTYPE)
W2 = torch.tensor([-1.0, 0.0, 2.5], dtype=DTYPE)


def ring(w):
    return (w @ w - 1.0) ** 2


def gen(seed):
    return torch.Generator().manual_seed(seed)


def small_net(seed, d=3, h=5, k=4):
    g = gen(seed)
    p = init_mlp(d, h, k, g)
    p["b1"] = torch.randn(h, generator=g, dtype=DTYPE)
    p["b2"] = torch.randn(k, generator=g, dtype=DTYPE)
    return p


def test_paths():
    assert torch.allclose(linear_path(W1, W2, 0.25), 0.75 * W1 + 0.25 * W2)
    assert torch.allclose(bezier_path(W1, TH, W2, 0.0), W1)
    assert torch.allclose(bezier_path(W1, TH, W2, 1.0), W2)
    assert torch.allclose(bezier_path(W1, TH, W2, 0.5), (W1 + 2.0 * TH + W2) / 4.0)
    assert torch.allclose(bezier_path(W1, TH, W2, 0.25), (9.0 * W1 + 6.0 * TH + W2) / 16.0)
    assert torch.allclose(polychain_path(W1, TH, W2, 0.25), 0.5 * (W1 + TH))
    assert torch.allclose(polychain_path(W1, TH, W2, 0.5), TH)
    assert torch.allclose(polychain_path(W1, TH, W2, 0.75), 0.5 * (TH + W2))
    assert torch.allclose(polychain_path(W1, TH, W2, 1.0), W2)


def test_barrier_convex_and_ring():
    H = torch.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]], dtype=DTYPE)

    def f(w):
        return 0.5 * w @ H @ w + w.sum()

    ts = torch.linspace(0.0, 1.0, 51, dtype=DTYPE)
    b, t, losses = loss_barrier(f, lambda t: linear_path(W1, W2, t), ts, reference="linear")
    assert b == pytest.approx(0.0, abs=1e-12)
    assert t in (0.0, 1.0)
    assert losses.shape == (51,)
    bm, _, _ = loss_barrier(f, lambda t: linear_path(W1, W2, t), ts)
    assert bm == pytest.approx(0.5 * abs(float(f(W1) - f(W2))))
    a = torch.tensor([-1.0, 0.0], dtype=DTYPE)
    c = torch.tensor([1.0, 0.0], dtype=DTYPE)
    ts = torch.linspace(0.0, 1.0, 401, dtype=DTYPE)
    b_lin, t_lin, _ = loss_barrier(ring, lambda t: linear_path(a, c, t), ts)
    assert b_lin == pytest.approx(1.0)
    assert t_lin == pytest.approx(0.5)
    bend = torch.tensor([0.0, 2.0], dtype=DTYPE)
    b_bez, _, _ = loss_barrier(ring, lambda t: bezier_path(a, bend, c, t), ts)
    assert b_bez == pytest.approx(1.0 / 16.0, abs=1e-4)


def test_barrier_conventions_and_errors():
    ts = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)

    def f(w):
        return w[0] + 4.0 * w[0] * (1.0 - w[0])

    def path(t):
        return torch.tensor([t], dtype=DTYPE)

    assert loss_barrier(f, path, ts, reference="linear")[0] == pytest.approx(1.0)
    assert loss_barrier(f, path, ts, reference="mean")[0] == pytest.approx(25.0 / 16.0 - 0.5, abs=1e-3)
    with pytest.raises(ValueError):
        loss_barrier(f, path, [0.1, 1.0])
    with pytest.raises(ValueError):
        loss_barrier(f, path, [0.0, 0.9])
    with pytest.raises(ValueError):
        loss_barrier(f, path, [0.0, 1.0], reference="max")


def test_mlp_pieces():
    p = init_mlp(50, 400, 3, gen(0))
    assert p["W1"].shape == (400, 50) and p["W2"].shape == (3, 400)
    assert not p["b1"].any() and not p["b2"].any()
    assert float(p["W1"].std()) == pytest.approx(math.sqrt(2.0 / 50), rel=0.05)
    assert float(p["W2"].std()) == pytest.approx(math.sqrt(2.0 / 400), rel=0.1)
    q = small_net(1)
    v = flatten(q)
    assert v.numel() == 15 + 5 + 20 + 4
    assert torch.equal(v[15:20], q["b1"])
    r = unflatten(v, 3, 5, 4)
    assert all(torch.equal(q[n], r[n]) for n in q)
    with pytest.raises(ValueError):
        unflatten(torch.zeros(v.numel() + 1, dtype=DTYPE), 3, 5, 4)
    X = torch.randn(6, 3, generator=gen(2), dtype=DTYPE)
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    z = forward(q, X)
    for i in range(6):
        assert torch.allclose(z[i], q["W2"] @ torch.relu(q["W1"] @ X[i] + q["b1"]) + q["b2"])
    ce = sum(torch.logsumexp(z[i], 0) - z[i, y[i]] for i in range(6)) / 6
    assert float(loss(q, X, y)) == pytest.approx(float(ce))
    assert error(q, X, y) == pytest.approx(float((z.argmax(1) != y).double().mean()))


def test_train_gd_is_gradient_descent():
    p = small_net(4)
    X = torch.randn(9, 3, generator=gen(5), dtype=DTYPE)
    y = torch.randint(0, 4, (9,), generator=gen(6))
    before = {n: v.clone() for n, v in p.items()}
    q = train_gd(p, X, y, 0.1, 1)
    leaf = {n: v.clone().requires_grad_(True) for n, v in p.items()}
    grads = torch.autograd.grad(loss(leaf, X, y), list(leaf.values()))
    for (n, v), g in zip(leaf.items(), grads, strict=True):
        assert torch.allclose(q[n], v.detach() - 0.1 * g)
        assert torch.equal(p[n], before[n])
    assert float(loss(train_gd(p, X, y, 0.1, 50), X, y)) < float(loss(p, X, y))


def test_permutation_symmetry_and_barrier():
    p = small_net(7)
    perm = torch.tensor([2, 0, 4, 1, 3])
    q = permute_hidden(p, perm)
    X = torch.randn(4, 3, generator=gen(8), dtype=DTYPE)
    assert torch.allclose(forward(p, X), forward(q, X))
    for i, j in enumerate(perm.tolist()):
        assert torch.equal(q["W1"][i], p["W1"][j])
        assert torch.equal(q["W2"][:, i], p["W2"][:, j])
        assert q["b1"][i] == p["b1"][j]
    assert torch.equal(q["b2"], p["b2"])
    Xd, yd = make_data(128, gen(0))
    a = train_gd(init_mlp(2, 16, 2, gen(1)), Xd, yd, 0.5, 600)
    b = permute_hidden(a, torch.roll(torch.arange(16), 1))
    assert float(loss(b, Xd, yd)) == pytest.approx(float(loss(a, Xd, yd)))

    def f(w):
        return loss(unflatten(w, 2, 16, 2), Xd, yd)

    barrier, t, _ = loss_barrier(f, lambda t: linear_path(flatten(a), flatten(b), t), torch.linspace(0, 1, 21))
    assert barrier > 0.05
    assert 0.0 < t < 1.0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_linear_assignment_against_brute_force(n):
    g = gen(n)
    for _ in range(8):
        S = torch.randn(n, n, generator=g, dtype=DTYPE)
        perm = linear_assignment(S)
        best = max(sum(float(S[i, p[i]]) for i in range(n)) for p in itertools.permutations(range(n)))
        assert sorted(perm.tolist()) == list(range(n))
        assert float(S[torch.arange(n), perm].sum()) == pytest.approx(best)
    with pytest.raises(ValueError):
        linear_assignment(torch.zeros(2, 3))


def test_weight_matching_recovers_a_known_permutation():
    a = small_net(9, d=4, h=9, k=3)
    true = torch.randperm(9, generator=gen(10))
    b = permute_hidden(a, torch.argsort(true))
    perm, b_matched = weight_matching(a, b)
    assert torch.equal(perm, true)
    for n in a:
        assert torch.allclose(b_matched[n], a[n])
    for zeroed in (("W1", "b1"), ("W1", "W2"), ("b1", "W2")):
        c = {n: (torch.zeros_like(v) if n in zeroed else v) for n, v in a.items()}
        perm, _ = weight_matching(c, permute_hidden(c, torch.argsort(true)))
        assert torch.equal(perm, true)


def test_fit_bend():
    a = torch.tensor([-1.0, 0.0], dtype=DTYPE)
    c = torch.tensor([1.0, 0.0], dtype=DTYPE)
    assert torch.allclose(fit_bend(ring, a, c, 0, 0.1, gen(0)), torch.zeros(2, dtype=DTYPE))
    t = float(torch.rand((), generator=gen(3), dtype=DTYPE))
    th0 = torch.tensor([0.0, 0.3], dtype=DTYPE)
    one = fit_bend(ring, a, c, 1, 0.1, gen(3), theta0=th0)
    w = bezier_path(a, th0, c, t)
    assert torch.allclose(one, th0 - 0.1 * 2.0 * t * (1.0 - t) * 4.0 * (w @ w - 1.0) * w)
    theta = fit_bend(ring, a, c, 3000, 0.05, gen(0), theta0=th0)
    barrier, _, _ = loss_barrier(ring, lambda t: bezier_path(a, theta, c, t), torch.linspace(0, 1, 201))
    assert barrier < 0.07
    assert 1.8 < float(theta[1]) < 2.5


def test_make_data():
    X, y = make_data(400, gen(0))
    assert X.shape == (400, 2) and int((y == 0).sum()) == 200
    r0 = (X[y == 0]).norm(dim=1)
    r1 = (X[y == 1] - torch.tensor([1.0, 0.5], dtype=DTYPE)).norm(dim=1)
    assert abs(float(r0.mean()) - 1.0) < 0.05 and abs(float(r1.mean()) - 1.0) < 0.05
    assert float(X[y == 0, 1].mean()) > 0.4 and float(X[y == 1, 1].mean()) < 0.1


def test_demo_matching_and_curves_lower_the_barrier():
    r = demo(1, 2)
    assert max(r["loss_a"], r["loss_b"]) < 0.01
    assert max(r["error_a"], r["error_b"]) < 0.01
    assert r["barrier_naive"] > 0.04
    assert r["barrier_bezier"] < 0.3 * r["barrier_naive"]
    assert r["barrier_matched"] < 0.05 * r["barrier_naive"]
    assert sorted(r["perm"].tolist()) == list(range(64))
