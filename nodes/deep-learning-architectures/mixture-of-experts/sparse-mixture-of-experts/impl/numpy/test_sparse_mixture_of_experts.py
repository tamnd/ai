# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-mixture-of-experts/ (ai://NEW-sparse-mixture-of-experts).
"""Tests of the NumPy reference: the top-k gate against its two equal forms, the capacity arithmetic and the worked
example of the page, dispatch order and dropping, the Switch balance loss at uniform, collapsed and adversarial
routing and its gradient, the importance and load losses of Shazeer et al. with a Monte Carlo check of P(x, i), the
z-loss, and the full layer against a token-by-token loop."""

import math

import numpy as np
import pytest

from sparse_mixture_of_experts import (
    cv_squared,
    dispatch,
    dropped_choices,
    expert_capacity,
    expert_ffn,
    importance_loss,
    keep_top_k,
    load_loss,
    load_probability,
    logsumexp,
    moe_layer,
    noisy_logits,
    softmax,
    softplus,
    switch_balance_grad,
    switch_balance_loss,
    tokens_per_expert,
    top_k_gates,
    top_k_indices,
    z_loss,
)

RNG = np.random.default_rng(20261116)


def test_softmax_and_logsumexp():
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    p = softmax(x)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.allclose(p[0], np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum())
    assert np.allclose(p[1], 1.0 / 3.0)
    assert np.allclose(logsumexp(x), [math.log(math.exp(1) + math.exp(2) + math.exp(3)), 1000.0 + math.log(3.0)])
    assert np.allclose(softplus([0.0, 30.0, -30.0]), [math.log(2.0), 30.0, math.exp(-30.0)])


def test_top_k_indices_order_and_ties():
    x = np.array([[0.1, 0.5, 0.3, 0.5]])
    assert top_k_indices(x, 1).tolist() == [[1]]
    assert top_k_indices(x, 3).tolist() == [[1, 3, 2]]
    with pytest.raises(ValueError):
        top_k_indices(x, 0)
    with pytest.raises(ValueError):
        top_k_indices(x, 5)


def test_keep_top_k():
    v = np.array([[2.0, -1.0, 0.5, 3.0]])
    out = keep_top_k(v, 2)
    assert out[0, 0] == 2.0 and out[0, 3] == 3.0
    assert np.isneginf(out[0, 1]) and np.isneginf(out[0, 2])


def test_top_k_gate_equals_renormalized_softmax():
    for _ in range(50):
        n = int(RNG.integers(2, 12))
        k = int(RNG.integers(1, n + 1))
        h = RNG.normal(scale=3.0, size=(7, n))
        g = top_k_gates(h, k)
        assert ((g > 0).sum(axis=-1) == k).all()
        assert np.allclose(g.sum(axis=-1), 1.0)
        p = softmax(h)
        idx = top_k_indices(h, k)
        kept = np.take_along_axis(p, idx, axis=-1)
        assert np.allclose(np.take_along_axis(g, idx, axis=-1), kept / kept.sum(axis=-1, keepdims=True))


def test_top_1_gate_is_one():
    h = RNG.normal(size=(5, 4))
    g = top_k_gates(h, 1)
    assert np.allclose(g.max(axis=-1), 1.0)
    assert (g.argmax(axis=-1) == h.argmax(axis=-1)).all()


def test_noisy_logits():
    x = RNG.normal(size=(3, 4))
    wg = RNG.normal(size=(4, 5))
    wn = RNG.normal(size=(4, 5))
    noise = RNG.normal(size=(3, 5))
    h = noisy_logits(x, wg, wn, noise)
    assert np.allclose(h, x @ wg + noise * np.log1p(np.exp(x @ wn)))
    assert np.allclose(noisy_logits(x, wg, wn, np.zeros((3, 5))), x @ wg)


def test_expert_capacity_arithmetic():
    assert expert_capacity(8, 4, 1.0) == 2
    assert expert_capacity(8, 4, 1.5) == 3
    assert expert_capacity(8, 4, 1.25) == 2
    assert expert_capacity(8, 4, 1.0, k=2) == 4
    assert expert_capacity(1024, 128, 1.25) == 10
    assert expert_capacity(1024, 128, 1.0) == 8
    assert expert_capacity(1000, 3, 1.0) == 333
    assert expert_capacity(0, 3, 1.0) == 0
    with pytest.raises(ValueError):
        expert_capacity(8, 0, 1.0)
    with pytest.raises(ValueError):
        expert_capacity(-1, 2, 1.0)
    with pytest.raises(ValueError):
        expert_capacity(8, 4, 0.0)


def test_dispatch_worked_example():
    # Eight tokens over four experts, counts (4, 2, 1, 1): with capacity 2 the last two tokens of expert 0 are dropped.
    idx = np.array([[0], [1], [0], [2], [0], [1], [3], [0]])
    assert tokens_per_expert(idx, 4).tolist() == [4, 2, 1, 1]
    kept, pos = dispatch(idx, 4, 2)
    assert kept[:, 0].tolist() == [True, True, True, True, False, True, True, False]
    assert pos[:, 0].tolist() == [0, 0, 1, 0, -1, 1, 0, -1]
    assert dropped_choices(idx, 4, 2) == 2
    kept3, _ = dispatch(idx, 4, 3)
    assert (~kept3).sum() == 1 and not kept3[7, 0]
    assert dropped_choices(idx, 4, 3) == 1
    kept4, _ = dispatch(idx, 4, 4)
    assert kept4.all()
    kept0, pos0 = dispatch(idx, 4, 0)
    assert not kept0.any() and (pos0 == -1).all()


def test_dispatch_first_choices_before_second():
    # Expert 0 has room for one: token 1's first choice wins over token 0's second choice.
    idx = np.array([[1, 0], [0, 1]])
    kept, pos = dispatch(idx, 2, 1)
    assert kept.tolist() == [[True, False], [True, False]]
    assert pos.tolist() == [[0, -1], [0, -1]]


def test_dispatch_never_exceeds_capacity():
    for _ in range(50):
        n = int(RNG.integers(2, 9))
        k = int(RNG.integers(1, min(n, 3) + 1))
        t = int(RNG.integers(1, 40))
        idx = top_k_indices(RNG.normal(size=(t, n)), k)
        cap = int(RNG.integers(0, t + 1))
        kept, pos = dispatch(idx, n, cap)
        per = np.array([((idx == e) & kept).sum() for e in range(n)])
        assert (per <= cap).all()
        assert (per == np.minimum(tokens_per_expert(idx, n), cap)).all()
        assert (~kept).sum() == dropped_choices(idx, n, cap)
        for e in range(n):
            slots = sorted(pos[(idx == e) & kept].tolist())
            assert slots == list(range(len(slots)))


def test_balance_loss_uniform_is_alpha():
    for n in (2, 3, 8, 64):
        probs = np.full((n * 5, n), 1.0 / n)
        probs[np.arange(n * 5), np.arange(n * 5) % n] += 1e-6
        probs /= probs.sum(axis=-1, keepdims=True)
        assert switch_balance_loss(probs, alpha=0.01) == pytest.approx(0.01, rel=1e-4)


def test_balance_loss_collapse_is_alpha_n():
    n, t = 8, 32
    probs = np.zeros((t, n))
    probs[:, 3] = 1.0
    assert switch_balance_loss(probs, alpha=0.01) == pytest.approx(0.01 * n)


def test_balance_loss_one_hot_is_alpha_n_sum_f_squared():
    n, t = 5, 40
    choice = RNG.integers(0, n, size=t)
    probs = np.eye(n)[choice]
    f = np.bincount(choice, minlength=n) / t
    loss = switch_balance_loss(probs, alpha=0.5)
    assert loss == pytest.approx(0.5 * n * (f**2).sum())
    assert loss >= 0.5 - 1e-12


def test_balance_loss_brute_force():
    for _ in range(30):
        n = int(RNG.integers(2, 10))
        t = int(RNG.integers(1, 30))
        probs = softmax(RNG.normal(scale=2.0, size=(t, n)))
        total = 0.0
        for i in range(n):
            f_i = sum(1 for x in range(t) if probs[x].argmax() == i) / t
            p_i = sum(probs[x, i] for x in range(t)) / t
            total += f_i * p_i
        assert switch_balance_loss(probs, alpha=0.3) == pytest.approx(0.3 * n * total)


def test_balance_loss_can_dip_below_alpha():
    # Three quarters of the tokens pick expert 0 by a hair, a quarter pick expert 1 with certainty: 0.875 alpha.
    eps = 1e-9
    probs = np.array([[0.5 + eps, 0.5 - eps]] * 3 + [[0.0, 1.0]])
    assert switch_balance_loss(probs, alpha=1.0) == pytest.approx(0.875, abs=1e-6)


def test_balance_grad_matches_finite_differences():
    n, t = 4, 6
    probs = softmax(RNG.normal(size=(t, n)))
    g = switch_balance_grad(probs, alpha=0.2)
    h = 1e-7
    for x in range(t):
        for i in range(n):
            q = probs.copy()
            q[x, i] += h
            if q[x].argmax() != probs[x].argmax():
                continue
            num = (switch_balance_loss(q, alpha=0.2) - switch_balance_loss(probs, alpha=0.2)) / h
            assert g[x, i] == pytest.approx(num, rel=1e-5, abs=1e-9)


def test_cv_squared():
    assert cv_squared([2.0, 2.0, 2.0]) == 0.0
    assert cv_squared([0.0, 0.0]) == 0.0
    assert cv_squared([1.0, 3.0]) == pytest.approx(0.25)
    assert cv_squared([3.0, 0.0, 0.0]) == pytest.approx(2.0)


def test_importance_loss():
    gates = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    assert importance_loss(gates, 0.1) == 0.0
    skew = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert importance_loss(skew, 0.2) == pytest.approx(0.2 * 0.25)
    assert importance_loss(skew * 7.0, 0.2) == pytest.approx(0.2 * 0.25)


def test_load_probability_against_monte_carlo():
    rng = np.random.default_rng(7)
    n, k = 5, 2
    clean = rng.normal(size=(1, n))
    std = softplus(rng.normal(size=(1, n)))
    noisy = clean + rng.normal(size=(1, n)) * std
    p = load_probability(clean, noisy, std, k)
    draws = 40000
    for i in range(n):
        h = np.repeat(noisy, draws, axis=0)
        h[:, i] = clean[0, i] + rng.normal(size=draws) * std[0, i]
        in_top = (top_k_indices(h, k) == i).any(axis=-1)
        assert in_top.mean() == pytest.approx(p[0, i], abs=0.01)


def test_load_probability_threshold_cases():
    clean = np.array([[3.0, 1.0, 0.0]])
    noisy = np.array([[3.0, 1.0, 0.0]])
    std = np.ones((1, 3))
    p = load_probability(clean, noisy, std, 1)

    def phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    assert p[0].tolist() == pytest.approx([phi(2.0), phi(-2.0), phi(-3.0)])
    p2 = load_probability(clean, noisy, std, 2)
    assert p2[0].tolist() == pytest.approx([phi(3.0), phi(1.0), phi(-1.0)])
    with pytest.raises(ValueError, match="load estimate"):
        load_probability(clean, noisy, std, 3)  # k = n
    with pytest.raises(ValueError, match="load estimate"):
        load_probability(clean, noisy, std, 0)


def test_load_loss_balanced_and_skewed():
    clean = np.array([[1.0, 0.0], [0.0, 1.0]])
    std = np.ones((2, 2))
    assert load_loss(clean, clean, std, 1, 0.3) == pytest.approx(0.0)
    skew = np.array([[2.0, 0.0], [2.0, 0.0]])
    assert load_loss(skew, skew, std, 1, 0.3) > 0.1


def test_z_loss():
    assert z_loss(np.zeros((3, 4))) == pytest.approx(math.log(4.0) ** 2)
    x = np.array([[1.0, 2.0], [0.0, -1.0]])
    expect = np.mean([math.log(math.e + math.e**2) ** 2, math.log(1.0 + math.exp(-1.0)) ** 2])
    assert z_loss(x) == pytest.approx(expect)
    assert z_loss(x + 10.0) > z_loss(x)


def _layer_loop(x, wr, w_in, w_out, k, cf):
    t, n = x.shape[0], wr.shape[1]
    probs = softmax(x @ wr)
    cap = int(math.floor(k * t * cf / n + 1e-9))
    count = [0] * n
    y = np.zeros_like(x)
    order = [list(np.argsort(-probs[s], kind="stable")[:k]) for s in range(t)]
    for r in range(k):
        for s in range(t):
            e = order[s][r]
            if count[e] < cap:
                gate = probs[s, e] if k == 1 else probs[s, e] / sum(probs[s, j] for j in order[s])
                y[s] += gate * np.maximum(x[s] @ w_in[e], 0.0) @ w_out[e]
            count[e] += 1
    return y


@pytest.mark.parametrize("k,cf", [(1, 1.0), (1, 0.5), (1, 2.0), (2, 1.0), (2, 1.25), (3, 0.75)])
def test_layer_matches_loop(k, cf):
    rng = np.random.default_rng(k * 100 + int(cf * 10))
    t, d, h, n = 16, 6, 10, 4
    x = rng.normal(size=(t, d))
    wr = rng.normal(size=(d, n))
    w_in = rng.normal(size=(n, d, h)) / math.sqrt(d)
    w_out = rng.normal(size=(n, h, d)) / math.sqrt(h)
    y, info = moe_layer(x, wr, w_in, w_out, k=k, capacity_factor=cf)
    assert np.allclose(y, _layer_loop(x, wr, w_in, w_out, k, cf))
    assert info["capacity"] == expert_capacity(t, n, cf, k)
    assert info["dropped"] == dropped_choices(info["indices"], n, info["capacity"])
    assert info["balance_loss"] == pytest.approx(switch_balance_loss(info["probs"]))
    assert info["z_loss"] == pytest.approx(z_loss(x @ wr))


def test_layer_dropped_token_is_zero_and_kept_token_is_gated_expert():
    t, d, h, n = 6, 3, 4, 2
    x = RNG.normal(size=(t, d))
    wr = np.zeros((d, n))
    wr[:, 0] = 5.0
    x = np.abs(x)
    w_in = RNG.normal(size=(n, d, h))
    w_out = RNG.normal(size=(n, h, d))
    y, info = moe_layer(x, wr, w_in, w_out, k=1, capacity_factor=1.0)
    assert info["capacity"] == 3
    assert (info["indices"][:, 0] == 0).all()
    assert info["dropped"] == 3
    assert np.allclose(y[3:], 0.0)
    p = info["probs"][:3, 0]
    assert np.allclose(y[:3], p[:, None] * expert_ffn(x[:3], w_in[0], w_out[0]))


def test_layer_renormalize_switch():
    t, d, h, n = 8, 4, 5, 4
    x = RNG.normal(size=(t, d))
    wr = RNG.normal(size=(d, n))
    w_in = RNG.normal(size=(n, d, h))
    w_out = RNG.normal(size=(n, h, d))
    y1, info = moe_layer(x, wr, w_in, w_out, k=1, capacity_factor=8.0)
    y1n, info_n = moe_layer(x, wr, w_in, w_out, k=1, capacity_factor=8.0, renormalize=True)
    assert info["dropped"] == 0
    assert np.allclose(info_n["gates"], 1.0)
    assert np.allclose(y1n * info["gates"], y1)


def test_functions_accept_python_lists():
    assert softmax([0.0, 0.0]).tolist() == pytest.approx([0.5, 0.5])
    assert float(logsumexp([0.0, 0.0])) == pytest.approx(math.log(2.0))
    assert top_k_indices([1, 3, 2], 2).tolist() == [1, 2]
    assert keep_top_k([3, 1, 2], 1).tolist() == [3.0, -math.inf, -math.inf]
    assert noisy_logits([[1.0, 0.0]], [[1.0], [0.0]], [[0.0], [0.0]], [[0.0]]).tolist() == [[1.0]]
    kept, _ = dispatch([[0], [0], [1]], 2, 1)
    assert kept[:, 0].tolist() == [True, False, True]
    assert switch_balance_loss([[1.0, 0.0], [0.0, 1.0]], alpha=1.0) == pytest.approx(1.0)
    assert switch_balance_grad([[1.0, 0.0], [0.0, 1.0]], alpha=1.0).tolist() == [[0.5, 0.5], [0.5, 0.5]]
    p = load_probability([[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]], 1)
    assert p[0].tolist() == pytest.approx([0.5, 0.5])
    y, _ = moe_layer([[1.0, 1.0]], np.zeros((2, 2)), np.ones((2, 2, 1)), np.ones((2, 1, 2)), capacity_factor=2.0)
    assert y.tolist() == [[1.0, 1.0]]


def test_default_alpha_and_capacity_edges():
    probs = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert switch_balance_loss(probs) == pytest.approx(0.02)
    assert np.allclose(switch_balance_grad(probs), [[0.01, 0.0], [0.01, 0.0]])
    assert expert_capacity(0, 4, 1.0) == 0
    assert expert_capacity(8, 1, 1.0) == 8
    # Products within 1e-9 below an integer round up to it; anything further below rounds down.
    assert expert_capacity(4, 4, 1.0 - 5e-10) == 1
    assert expert_capacity(4, 4, 1.0 - 5e-9) == 0
    with pytest.raises(ValueError):
        expert_capacity(8, 4, 0.0)
