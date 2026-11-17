# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
"""Tests of the NumPy reference: the worked example, the logit bound, the z-loss gradient and the entropy."""

import numpy as np
import pytest

from qk_layernorm import (
    attention,
    attention_entropy,
    layer_norm,
    logit_bound,
    logsumexp,
    max_logit,
    merge_heads,
    softmax,
    split_heads,
    z_loss,
    z_loss_grad,
)

# The worked example of the page: three tokens, d = 4, one head, W_Q = W_K = W_V = s * I.
X = np.array([[1.0, 3.0, 1.0, 3.0], [5.0, 5.0, 1.0, 1.0], [3.0, 1.0, 3.0, 1.0]])
EYE = np.eye(4)
RAW_LOGITS = np.array([[10.0, 12.0, 6.0], [12.0, 26.0, 12.0], [6.0, 12.0, 10.0]])
QK_LOGITS = np.array([[2.0, 0.0, -2.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 2.0]])
Q_HAT = np.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
# The z-loss example: logits log(1, 2, 3, 2), so Z = 8.
Y = np.log(np.array([1.0, 2.0, 3.0, 2.0]))
P_Y = np.array([1.0, 2.0, 3.0, 2.0]) / 8.0


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def rows_softmax(z):
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_worked_example():
    np.testing.assert_allclose(layer_norm(X), Q_HAT, atol=1e-12)
    raw = attention(X, EYE, EYE, EYE)
    np.testing.assert_allclose(raw["logits"][0], RAW_LOGITS, atol=1e-12)
    np.testing.assert_allclose(raw["weights"][0], rows_softmax(RAW_LOGITS), atol=1e-12)
    qk = attention(X, EYE, EYE, EYE, qk_norm=True)
    np.testing.assert_allclose(qk["logits"][0], QK_LOGITS, atol=1e-12)
    w = qk["weights"][0]
    np.testing.assert_allclose(w[0], [0.8668, 0.1173, 0.0159], atol=1e-4)
    np.testing.assert_allclose(w[1], [0.1065, 0.7870, 0.1065], atol=1e-4)
    np.testing.assert_allclose(attention_entropy(w), [0.4411, 0.6656, 0.4411], atol=1e-4)
    np.testing.assert_allclose(attention_entropy(raw["weights"][0])[0], 0.3801, atol=1e-4)
    np.testing.assert_allclose(qk["out"], w @ X, atol=1e-12)
    # Scaling W_Q and W_K by 10 multiplies the raw logits by 100 and leaves the normalized ones alone.
    big = attention(X, 10 * EYE, 10 * EYE, EYE)
    np.testing.assert_allclose(big["logits"][0], 100 * RAW_LOGITS, atol=1e-9)
    np.testing.assert_allclose(big["weights"][0][:, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(attention_entropy(big["weights"][0]), 0.0, atol=1e-12)
    big_qk = attention(X, 10 * EYE, 10 * EYE, EYE, qk_norm=True)
    np.testing.assert_allclose(big_qk["logits"][0], QK_LOGITS, atol=1e-12)


def test_z_loss_worked_example():
    assert logsumexp(Y) == pytest.approx(np.log(8.0), abs=1e-12)
    assert z_loss(Y) == pytest.approx(1e-4 * np.log(8.0) ** 2, rel=1e-12)
    assert z_loss(Y, alpha=0.5) == pytest.approx(0.5 * np.log(8.0) ** 2, rel=1e-12)
    np.testing.assert_allclose(z_loss_grad(Y), 2e-4 * np.log(8.0) * P_Y, rtol=1e-12)
    # Shifting every logit by -log Z keeps the softmax and makes the z-loss 0.
    shifted = Y - np.log(8.0)
    np.testing.assert_allclose(softmax(shifted), P_Y, rtol=1e-12)
    assert z_loss(shifted) == pytest.approx(0.0, abs=1e-30)
    np.testing.assert_allclose(z_loss_grad(shifted), 0.0, atol=1e-18)


def test_layer_norm_moments_and_affine():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 8)) * 3 + 2
    y = layer_norm(x)
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose((y**2).mean(axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.sqrt(8), atol=1e-12)
    g, b = rng.normal(size=8), rng.normal(size=8)
    np.testing.assert_allclose(layer_norm(x, g, b), y * g + b, atol=1e-12)
    np.testing.assert_allclose(layer_norm(x, g), y * g, atol=1e-12)
    np.testing.assert_allclose(layer_norm(x, None, b), y + b, atol=1e-12)
    var = x.var(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        layer_norm(x, eps=0.5),
        (x - x.mean(-1, keepdims=True)) / np.sqrt(var + 0.5),
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", range(5))
def test_logit_bound_holds_for_any_weight_norm(seed):
    rng = np.random.default_rng(seed)
    d_model, heads, n = 12, 3, 7
    dh = d_model // heads
    x = rng.normal(size=(2, n, d_model))
    for scale in [1.0, 100.0, 1e4]:
        wq, wk, wv = (scale * rng.normal(size=(d_model, d_model)) for _ in range(3))
        r = attention(x, wq, wk, wv, heads=heads, qk_norm=True)
        assert max_logit(r["logits"]) <= logit_bound(dh) + 1e-9
        gq, gk = rng.normal(size=dh) * 3, rng.normal(size=dh) * 2
        rg = attention(x, wq, wk, wv, heads=heads, qk_norm=True, gain_q=gq, gain_k=gk)
        assert max_logit(rg["logits"]) <= logit_bound(dh, gq, gk) + 1e-9
    raw = attention(x, 1e4 * wq / scale, 1e4 * wk / scale, wv, heads=heads)
    assert max_logit(raw["logits"]) > 100 * logit_bound(dh)


def test_logit_bound_is_tight():
    rng = np.random.default_rng(1)
    d_model = 6
    x = rng.normal(size=(4, d_model))
    w = rng.normal(size=(d_model, d_model)) * 50
    r = attention(x, w, w, w, qk_norm=True)
    np.testing.assert_allclose(np.diag(r["logits"][0]), np.sqrt(d_model), rtol=1e-12)
    r2 = attention(x, w, -w, w, qk_norm=True)
    np.testing.assert_allclose(np.diag(r2["logits"][0]), -np.sqrt(d_model), rtol=1e-12)
    g = np.full(d_model, 1.5)
    r3 = attention(x, w, w, w, qk_norm=True, gain_q=g, gain_k=2 * g)
    np.testing.assert_allclose(np.diag(r3["logits"][0]), logit_bound(d_model, g, 2 * g), rtol=1e-12)


def test_logit_bound_values():
    assert logit_bound(4) == pytest.approx(2.0)
    assert logit_bound(64) == pytest.approx(8.0)
    assert logit_bound(16, [1.0, -3.0], [0.5, 2.0]) == pytest.approx(4.0 * 3.0 * 2.0)
    assert logit_bound(16, None, [0.5, -2.5]) == pytest.approx(10.0)
    assert logit_bound(16, [0.5, -2.5], None) == pytest.approx(10.0)


def test_eps_shrinks_the_norm():
    x = np.array([[1.0, 3.0, 1.0, 3.0]])
    r = attention(x, EYE, EYE, EYE, qk_norm=True, eps=1.0)
    assert r["logits"][0, 0, 0] == pytest.approx(2.0 / 2.0)
    np.testing.assert_allclose(np.linalg.norm(r["q"], axis=-1), np.sqrt(2.0), rtol=1e-12)


def test_weight_scale_invariance_and_quadratic_growth():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 8))
    wq, wk, wv = (rng.normal(size=(8, 8)) for _ in range(3))
    base = attention(x, wq, wk, wv, heads=2)
    base_qk = attention(x, wq, wk, wv, heads=2, qk_norm=True)
    for c in [0.1, 3.0, 30.0]:
        r = attention(x, c * wq, c * wk, wv, heads=2)
        np.testing.assert_allclose(r["logits"], c**2 * base["logits"], rtol=1e-10)
        rq = attention(x, c * wq, c * wk, wv, heads=2, qk_norm=True)
        np.testing.assert_allclose(rq["logits"], base_qk["logits"], atol=1e-10)


def test_multi_head_is_per_head():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 5, 8))
    wq, wk, wv = (rng.normal(size=(8, 8)) for _ in range(3))
    gq, gk = rng.normal(size=4), rng.normal(size=4)
    full = attention(x, wq, wk, wv, heads=2, qk_norm=True, gain_q=gq, gain_k=gk)
    assert full["out"].shape == (2, 5, 8)
    assert full["weights"].shape == (2, 2, 5, 5)
    for h in range(2):
        cols = slice(4 * h, 4 * h + 4)
        one = attention(x, wq[:, cols], wk[:, cols], wv[:, cols], qk_norm=True, gain_q=gq, gain_k=gk)
        np.testing.assert_allclose(full["logits"][:, h], one["logits"][:, 0], atol=1e-12)
        np.testing.assert_allclose(full["out"][..., cols], one["out"], atol=1e-12)


def test_split_and_merge_heads():
    x = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6)
    s = split_heads(x, 3)
    assert s.shape == (2, 3, 3, 2)
    np.testing.assert_array_equal(s[0, 1, 2], x[0, 2, 2:4])
    np.testing.assert_array_equal(merge_heads(s), x)
    with pytest.raises(ValueError):
        split_heads(x, 4)


def test_causal_mask():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 4))
    r = attention(x, EYE, EYE, EYE, causal=True, qk_norm=True)
    w = r["weights"][0]
    np.testing.assert_array_equal(np.triu(w, 1), 0.0)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    assert w[0, 0] == pytest.approx(1.0)
    full = attention(x, EYE, EYE, EYE, qk_norm=True)
    np.testing.assert_allclose(np.tril(r["logits"][0]), np.tril(full["logits"][0]), atol=1e-12)
    assert max_logit(r["logits"]) <= 2.0 + 1e-12


def test_softmax_and_logsumexp_are_stable():
    z = np.array([[1000.0, 1000.0, 999.0], [-1000.0, -1001.0, -1002.0]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=-1), 1.0)
    np.testing.assert_allclose(p[0], np.exp([0, 0, -1]) / np.exp([0, 0, -1]).sum(), rtol=1e-12)
    np.testing.assert_allclose(
        logsumexp(z),
        [1000 + np.log(2 + np.exp(-1)), -1000 + np.log(1 + np.exp(-1) + np.exp(-2))],
        rtol=1e-12,
    )
    small = np.array([[0.5, -1.0, 2.0], [0.0, 0.1, 0.2]])
    np.testing.assert_allclose(logsumexp(small), np.log(np.exp(small).sum(axis=-1)), rtol=1e-12)
    np.testing.assert_allclose(logsumexp(small.T, axis=0), np.log(np.exp(small).sum(axis=-1)), rtol=1e-12)
    np.testing.assert_allclose(softmax(small.T, axis=0), rows_softmax(small).T, rtol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_z_loss_gradient_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(3, 5)) * 4 + rng.normal() * 3
    alpha = 1e-2
    num = num_grad(lambda t: z_loss(t, alpha), z)
    np.testing.assert_allclose(z_loss_grad(z, alpha), num, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(z_loss_grad(z, alpha).sum(axis=-1), 2 * alpha * logsumexp(z) / 3, rtol=1e-10)


def test_z_loss_is_a_mean_over_positions():
    z = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, -1.0]])
    expected = 0.1 * np.mean(logsumexp(z) ** 2)
    assert z_loss(z, alpha=0.1) == pytest.approx(expected, rel=1e-12)
    assert z_loss(z[:1], alpha=0.1) == pytest.approx(0.1 * np.log(2.0) ** 2, rel=1e-12)


def test_z_loss_gradient_step_pulls_log_z_toward_zero():
    for start in [-6.0, -0.5, 0.7, 9.0]:
        z = np.array([0.3, -1.2, 2.0, 0.1])
        z = z - logsumexp(z) + start
        lz = [logsumexp(z)]
        for _ in range(200):
            z = z - 50.0 * z_loss_grad(z, alpha=1e-2)
            lz.append(logsumexp(z))
        lz = np.abs(np.array(lz))
        assert np.all(np.diff(lz) <= 1e-12)
        assert lz[-1] < 0.05 * lz[0]


def test_attention_entropy():
    np.testing.assert_allclose(attention_entropy(np.full(8, 1 / 8)), np.log(8), rtol=1e-12)
    assert attention_entropy(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-15)
    p = np.array([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    np.testing.assert_allclose(attention_entropy(p), [np.log(2), 1.5 * np.log(2)], rtol=1e-12)
    np.testing.assert_allclose(attention_entropy(p.T, axis=0), [np.log(2), 1.5 * np.log(2)], rtol=1e-12)
    assert np.all(np.isfinite(attention_entropy(p)))


def test_entropy_collapses_as_logits_scale():
    z = np.array([2.0, 0.0, -2.0, 1.0])
    hs = [attention_entropy(softmax(c * z)) for c in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]]
    assert hs[0] == pytest.approx(np.log(4))
    assert np.all(np.diff(hs) < 0)
    assert hs[-1] < 1e-20
    np.testing.assert_allclose(softmax(64.0 * z), [1.0, 0.0, 0.0, 0.0], atol=1e-20)


def test_max_logit_ignores_masked_entries():
    assert max_logit(np.array([[-np.inf, 3.0], [-5.0, 1.0]])) == 5.0
    assert max_logit(np.array([2.0, -1.0])) == 2.0
