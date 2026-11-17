# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://NEW-linear-attention).
"""Tests of the NumPy reference: the fixtures, associativity (eq-linear against eq-general), the causal recurrence,
Algorithm 1 against the gradient of the quadratic form, the layer backward against finite differences, the RNN step,
the exact polynomial kernel, the worked example and the operation counts."""

import pathlib

import numpy as np
import pytest

import linear_attention as L

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"


def qkv(seed=0, B=2, N=7, D=3, M=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(B, N, D)), rng.normal(size=(B, N, D)), rng.normal(size=(B, N, M)), rng


def sim_of(phi):
    return lambda Q, K: phi(Q) @ np.swapaxes(phi(K), -1, -2)


def test_fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    inp, exp = np.load(FIX / "input-small.npz"), np.load(FIX / "expected-small.npz")
    for causal in (True, False):
        cfg = L.config(causal=causal)
        tag = "causal" if causal else "full"
        out, cache = L.forward(params, cfg, inp["x"])
        assert np.allclose(out, exp["out_" + tag], atol=1e-12)
        assert np.isclose(L.loss(out, inp["target"]), exp["loss_" + tag], atol=1e-12)
        grads, dx = L.backward(params, cfg, cache, (out - inp["target"]) / out.size)
        for k, g in grads.items():
            assert np.allclose(g, exp[f"d{k}_{tag}"], atol=1e-12), (tag, k)
        assert np.allclose(dx, exp["dx_" + tag], atol=1e-12)
    soft = L.softmax_attention(*(L.split(inp["x"] @ params[w], 2) for w in ("W_q", "W_k", "W_v")), causal=True)
    assert np.allclose(L.merge(soft) @ params["W_o"], exp["out_softmax_causal"], atol=1e-12)


@pytest.mark.parametrize("feature", ["elu", "relu", "poly2"])
def test_associativity_gives_the_same_numbers(feature):
    """eq-linear is eq-general: phi(Q)(phi(K)^T V) and (phi(Q) phi(K)^T) V are the same matrix."""
    Q, K, V, _ = qkv()
    phi = L.FEATURES[feature][0]
    if feature == "relu":  # keep every denominator away from 0
        Q, K = Q + 3, K + 3
    assert np.allclose(L.linear_attention(phi(Q), phi(K), V), L.kernel_attention(Q, K, V, sim_of(phi)), atol=1e-12)


def test_softmax_is_the_exponential_kernel():
    """eq-softmax is eq-general with sim(q, k) = exp(q . k / sqrt(D))."""
    Q, K, V, _ = qkv(1)
    sim = lambda Q, K: np.exp(Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1]))  # noqa: E731
    for causal in (False, True):
        assert np.allclose(L.softmax_attention(Q, K, V, causal), L.kernel_attention(Q, K, V, sim, causal))


def test_polynomial_feature_map_is_exact():
    """phi(q) . phi(k) = (q . k + c)^2 for the map of size 1 + D + D^2, so degree-2 polynomial attention is linear."""
    Q, K, V, _ = qkv(2)
    for c in (1.0, 0.5):
        f = lambda x: L.poly2(x, c)  # noqa: E731
        assert np.allclose(f(Q) @ np.swapaxes(f(K), -1, -2), (Q @ np.swapaxes(K, -1, -2) + c) ** 2)
    assert L.poly2(Q).shape[-1] == 1 + 3 + 9
    poly = lambda Q, K: (Q @ np.swapaxes(K, -1, -2) + 1) ** 2  # noqa: E731
    out = L.linear_attention(L.poly2(Q), L.poly2(K), V)
    assert np.allclose(out, L.kernel_attention(Q, K, V, poly))


def test_causal_recurrence_is_the_masked_quadratic_form():
    """eq-causal: S_i and z_i summed up to i give the attention with the future masked."""
    Q, K, V, _ = qkv(3)
    phi = L.elu_plus_one
    out, num, den = L.causal_linear_attention(phi(Q), phi(K), V)
    assert np.allclose(out, L.kernel_attention(Q, K, V, sim_of(phi), causal=True), atol=1e-12)
    A = np.tril(phi(Q) @ np.swapaxes(phi(K), -1, -2))
    assert np.allclose(num, A @ V) and np.allclose(den, A.sum(-1))
    assert np.allclose(out[:, -1], L.linear_attention(phi(Q), phi(K), V)[:, -1])  # the last step sees everything
    assert np.allclose(out[:, 0], V[:, 0])  # the first step sees only itself


def test_algorithm_1_is_the_gradient_of_the_numerator():
    """Algorithm 1 against the gradient of tril(phi(Q) phi(K)^T) V written out with the full N x N matrix."""
    fQ, fK, V, rng = qkv(4)
    fQ, fK = L.elu_plus_one(fQ), L.elu_plus_one(fK)
    G = rng.normal(size=V.shape)
    dQ, dK, dV = L.causal_numerator_backward(fQ, fK, V, G)
    T = np.tril(np.ones((7, 7)))
    dA = T * (G @ np.swapaxes(V, -1, -2))  # dL/dA for num = (T * fQ fK^T) V
    assert np.allclose(dQ, dA @ fK) and np.allclose(dK, np.swapaxes(dA, -1, -2) @ fQ)
    assert np.allclose(dV, np.swapaxes(T * (fQ @ np.swapaxes(fK, -1, -2)), -1, -2) @ G)


@pytest.mark.parametrize("causal", [True, False])
def test_backward_matches_finite_differences(causal):
    rng = np.random.default_rng(5)
    cfg = L.config(6, 2, causal=causal)
    P = L.init(rng, cfg)
    x, target = rng.normal(size=(2, 5, 6)), rng.normal(size=(2, 5, 6))
    out, cache = L.forward(P, cfg, x)
    grads, dx = L.backward(P, cfg, cache, (out - target) / out.size)

    def f():
        return L.loss(L.forward(P, cfg, x)[0], target)

    for k in P:
        for _ in range(5):
            i = tuple(rng.integers(0, d) for d in P[k].shape)
            o = P[k][i]
            P[k][i] = o + 1e-6
            up = f()
            P[k][i] = o - 1e-6
            down = f()
            P[k][i] = o
            assert abs((up - down) / 2e-6 - grads[k][i]) < 1e-7, (k, i)
    for _ in range(5):
        i = tuple(rng.integers(0, d) for d in x.shape)
        o = x[i]
        x[i] = o + 1e-6
        up = f()
        x[i] = o - 1e-6
        down = f()
        x[i] = o
        assert abs((up - down) / 2e-6 - dx[i]) < 1e-7, i


@pytest.mark.parametrize("feature", ["elu", "poly2"])
def test_the_layer_is_an_rnn(feature):
    """eq-rnn: feeding the tokens one at a time through step gives the causal forward pass, with a state whose size
    does not grow."""
    rng = np.random.default_rng(6)
    cfg = L.config(8, 2, feature=feature)
    P = L.init(rng, cfg)
    x = rng.normal(size=(3, 9, 8))
    state = L.initial_state(cfg, 3)
    sizes, outs = set(), []
    for i in range(9):
        y, state = L.step(P, cfg, state, x[:, i])
        outs.append(y)
        sizes.add(sum(a.size for a in state))
    if feature == "elu":
        assert np.allclose(np.stack(outs, 1), L.forward(P, cfg, x)[0], atol=1e-12)
    else:
        q, k, v = (L.split(x @ P[w], 2) for w in ("W_q", "W_k", "W_v"))
        want = L.kernel_attention(q, k, v, lambda Q, K: (Q @ np.swapaxes(K, -1, -2) + 1) ** 2, causal=True)
        assert np.allclose(np.stack(outs, 1), L.merge(want) @ P["W_o"])
    C = 4 if feature == "elu" else 21
    assert sizes == {3 * 2 * (C * 4 + C)}


def test_elu_feature_map():
    """eq-elu: positive everywhere, x + 1 for x > 0, exp(x) below; the gradient is 1 or exp(x), never 0."""
    x = np.linspace(-30, 5, 1001)
    f, g = L.elu_plus_one(x), L.elu_plus_one_grad(x)
    assert np.all(f > 0) and np.all(g > 0)
    assert np.allclose(f[x > 0], x[x > 0] + 1) and np.allclose(f[x <= 0], np.exp(x[x <= 0]))
    assert np.all(L.relu_grad(x[x < 0]) == 0)


def test_relu_features_can_divide_zero_by_zero():
    """The pitfall of a map that can be 0: a query with no positive entry gives phi(q) = 0, so its numerator and its
    denominator are both 0 and the output is NaN; with elu + 1 the denominator is a sum of positive terms."""
    Q, K, V, _ = qkv(7)
    dead = np.all(Q <= 0, -1)  # with D = 3 Gaussian entries, one query in 8 on average
    assert dead.sum() >= 2
    with np.errstate(invalid="ignore"):
        out = L.linear_attention(L.relu(Q), L.relu(K), V)
    assert np.array_equal(np.isnan(out).any(-1), dead)
    assert np.isfinite(L.linear_attention(L.elu_plus_one(Q), L.elu_plus_one(K), V)).all()


def test_worked_example():
    """The worked example of the page: three keys and values, one query, phi = elu + 1."""
    K = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    V = np.array([[1.0, 0.0], [0.0, 2.0], [4.0, 4.0]])
    q = np.array([[0.5, -0.5]])
    fK, fq = L.elu_plus_one(K), L.elu_plus_one(q)
    assert np.allclose(np.round(fK, 4), [[2.0, 1.0], [1.0, 2.0], [0.3679, 0.3679]])
    assert np.allclose(np.round(fq, 4), [[1.5, 0.6065]])
    KV, z = fK.T @ V, fK.sum(0)
    assert np.allclose(np.round(KV, 4), [[3.4715, 3.4715], [2.4715, 5.4715]])
    assert np.allclose(np.round(z, 4), [3.3679, 3.3679])
    assert np.allclose(np.round(fq @ KV, 4), [[6.7063, 8.5259]]) and np.isclose(round((fq @ z).item(), 4), 7.0945)
    out = L.linear_attention(fq, fK, V)
    assert np.allclose(np.round(out, 4), [[0.9453, 1.2018]])
    w = (fq @ fK.T) / (fq @ z)  # the weights that the linear form never builds
    assert np.allclose(np.round(w, 4), [[0.5084, 0.3824, 0.1092]]) and np.allclose(w @ V, out)
    soft = L.softmax_attention(q, K, V)  # the softmax weights, 0.4555, 0.2246 and 0.3199, rank the keys differently
    assert np.allclose(np.round(soft, 4), [[1.7350, 1.7287]])


def test_flops_and_the_crossover():
    """flops: softmax grows as N^2 and linear as N; with C = D = M they cross near N = D, and with the degree-2
    polynomial map (C = 1 + D + D^2) near N = D^2, the paper's 'favourable when N > D^2'."""
    D = M = 64
    assert L.flops(4096, D, M, kind="softmax") / L.flops(4096, D, M) == pytest.approx(4096 * 129 / (2 * 64 * 65))
    cross = min(N for N in range(1, 10**6) if L.flops(N, D, M) < L.flops(N, D, M, kind="softmax"))
    assert cross == 65
    C = 1 + D + D * D
    cross2 = min(N for N in range(1, 10**6) if L.flops(N, D, M, C) < L.flops(N, D, M, kind="softmax"))
    assert D * D < cross2 < D * D + 2 * D + 2


def test_dpfp():
    """Schlag et al.'s Figure 1: for D = 2 and nu = 1 the four features are the products r(+-k_1) r(+-k_2), one per
    quadrant, so a key off the axes has exactly one non-zero feature and keys in different quadrants are orthogonal.
    The size is 2 D nu, and the features are non-negative."""
    k = np.array([[1.0, 2.0], [-1.0, 2.0], [1.0, -2.0], [-1.0, -2.0]])
    f = L.dpfp(k)
    assert f.shape == (4, 4) and np.all((f > 0).sum(-1) == 1)
    assert np.allclose(f @ f.T, np.diag(np.diag(f @ f.T)))
    x = np.random.default_rng(8).normal(size=(5, 6))
    for nu in (1, 2, 3):
        assert L.dpfp(x, nu).shape == (5, 2 * 6 * nu) and np.all(L.dpfp(x, nu) >= 0)


def test_delta_rule_overwrites_and_the_sum_rule_superposes():
    """Write (k, v1) then (k, v2) with a one-hot key: the delta rule with beta = 1 reads back v2, the sum rule
    v1 + v2. With beta = 1 and orthonormal keys the delta rule stores each new key exactly as the sum rule does."""
    k = np.eye(4)[[1, 1]]
    v = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    _, S_sum = L.fast_weight_rule(k, k, v)
    _, S_delta = L.fast_weight_rule(k, k, v, beta=np.ones(2))
    assert np.allclose(np.eye(4)[1] @ S_sum, v[0] + v[1]) and np.allclose(np.eye(4)[1] @ S_delta, v[1])
    keys, vals = np.eye(4), np.random.default_rng(9).normal(size=(4, 3))
    assert np.allclose(L.fast_weight_rule(keys, keys, vals, np.ones(4))[1], L.fast_weight_rule(keys, keys, vals)[1])


def test_sum_rule_is_the_numerator_of_linear_attention():
    Q, K, V, _ = qkv(10)
    fQ, fK = L.elu_plus_one(Q), L.elu_plus_one(K)
    assert np.allclose(L.fast_weight_rule(fQ, fK, V)[0], L.causal_linear_attention(fQ, fK, V)[1])
