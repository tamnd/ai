# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Tests of the NumPy reference: the fixtures, the gradient against finite differences, the padding, the
alignment weights, RNNencdec as the model with c_i fixed, the parameter count at the paper's sizes and the
relations between the scores of Luong et al."""

import pathlib

import numpy as np
import pytest

import attention as A

FIX = pathlib.Path(__file__).resolve().parents[2] / "fixtures"


def small(attention=True, seed=0, scale=0.5):
    cfg = A.config(attention=attention)
    rng = np.random.default_rng(seed)
    params = A.init(rng, cfg, scale)
    src, tgt = rng.integers(1, 7, (2, 5)), rng.integers(0, 6, (2, 4))
    src_mask, tgt_mask = np.ones((2, 5), bool), np.ones((2, 4), bool)
    src_mask[1, 3:], tgt_mask[1, 3:] = False, False
    return cfg, params, src, tgt, src_mask, tgt_mask


def test_fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    inp, exp = np.load(FIX / "input-small.npz"), np.load(FIX / "expected-small.npz")
    cfg = A.config()
    logits, alpha, cache = A.forward(params, cfg, inp["src"], inp["tgt"], inp["src_mask"])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.allclose(alpha, exp["alpha"], atol=1e-12)
    assert np.isclose(A.nll(logits, inp["tgt"], inp["tgt_mask"]), exp["loss"], atol=1e-12)
    g = A.backward(params, cfg, cache, inp["tgt_mask"])
    for k, v in g.items():
        assert np.allclose(v, exp["d" + k], atol=1e-12), k
    plain = A.forward(params, A.config(attention=False), inp["src"], inp["tgt"], inp["src_mask"])[0]
    assert np.allclose(plain, exp["logits_encdec"], atol=1e-12)


@pytest.mark.parametrize("attention", [True, False])
def test_gradient_matches_finite_differences(attention):
    cfg, P, src, tgt, sm, tm = small(attention)
    g = A.backward(P, cfg, A.forward(P, cfg, src, tgt, sm)[2], tm)
    rng = np.random.default_rng(1)
    for k in P:
        for _ in range(4):
            i = tuple(rng.integers(0, d) for d in P[k].shape)
            o = P[k][i]
            P[k][i] = o + 1e-6
            up = A.nll(A.forward(P, cfg, src, tgt, sm)[0], tgt, tm)
            P[k][i] = o - 1e-6
            down = A.nll(A.forward(P, cfg, src, tgt, sm)[0], tgt, tm)
            P[k][i] = o
            assert abs((up - down) / 2e-6 - g[k][i]) < 1e-7, (k, i)


def test_padding_changes_nothing():
    """A sentence padded to a longer batch gets the logits and alignments it gets alone, and zero weight on the pads."""
    cfg, P, src, tgt, sm, tm = small()
    logits, alpha, _ = A.forward(P, cfg, src, tgt, sm)
    alone, alpha1, _ = A.forward(P, cfg, src[1:, :3], tgt[1:], None)
    assert np.allclose(logits[1], alone[0], atol=1e-13)
    assert np.allclose(alpha[1, :, :3], alpha1[0], atol=1e-13)
    assert np.all(alpha[1, :, 3:] == 0.0)


def test_weights_are_a_distribution_and_the_context_an_average():
    """eq-weights and eq-context: alpha_i is a probability vector, so c_i lies in the convex hull of the
    annotations; with all the scores equal (v_a = 0, as Bahdanau et al. initialize it) it is their mean."""
    cfg, P, src, tgt, sm, _ = small()
    _, alpha, cache = A.forward(P, cfg, src, tgt, sm)
    assert np.all(alpha >= 0) and np.allclose(alpha.sum(-1), 1.0)
    H = cache["H"]
    for i, (s, c, *_rest) in enumerate(cache["steps"]):
        assert np.allclose(c, np.einsum("bs,bsk->bk", alpha[:, i], H))
        assert np.all(c <= H.max(1) + 1e-12) and np.all(c >= H.min(1) - 1e-12)
    P["v_a"][:] = 0.0
    _, _, cache = A.forward(P, cfg, src, tgt, None)
    assert all(np.allclose(st[1], cache["H"].mean(1)) for st in cache["steps"])


def test_encdec_is_attention_with_a_fixed_context():
    """Bahdanau et al.: the model becomes the RNN Encoder-Decoder if c_i is fixed. With attention=False every step
    gets the same context, the last forward and the first backward annotation, whatever the decoder state."""
    cfg, P, src, tgt, sm, _ = small(attention=False)
    _, _, cache = A.forward(P, cfg, src, tgt, None)
    cs = [st[1] for st in cache["steps"]]
    n = cfg["hidden"]
    for c in cs:
        assert np.array_equal(c, np.concatenate([cache["H"][:, -1, :n], cache["H"][:, 0, n:]], -1))


def test_annotations_see_both_sides():
    """eq-annot: changing the last source word changes the forward half of h_S only, and the backward half of every
    h_j; the forward half of h_j depends on x_1..x_j and the backward half on x_j..x_S."""
    cfg, P, src, *_ = small()
    H, _ = A.encode(P, src, np.ones(src.shape, bool))
    src2 = src.copy()
    src2[:, -1] = src[:, -1] % 6 + 1
    H2, _ = A.encode(P, src2, np.ones(src.shape, bool))
    n = cfg["hidden"]
    assert np.allclose(H[:, :-1, :n], H2[:, :-1, :n]) and not np.allclose(H[:, -1, :n], H2[:, -1, :n])
    assert all(not np.allclose(H[:, j, n:], H2[:, j, n:]) for j in range(src.shape[1]))


def test_parameter_count_at_the_paper_sizes():
    """Appendix A.2.3: n = 1000, m = 620, l = 500, n' = 1000, with the shortlists of 30,000 words of Section 4.1."""
    cfg = A.config(30000, 30000, 620, 1000, 1000, 500)
    assert A.param_count(cfg) == 80_401_000
    sh = A.shapes(cfg)
    assert sum(int(np.prod(sh[k])) for k in ("W_a", "U_a", "v_a")) == 3_001_000
    assert A.param_count(A.config()) == sum(p.size for p in A.init(np.random.default_rng(0), A.config()).values())


def test_greedy_decoding_reproduces_the_teacher_forced_logits():
    cfg, P, src, *_ = small(scale=1.5)
    for x, (ys, rows) in zip(src, A.greedy_decode(P, cfg, src, 6)):
        tgt = ys[None]
        logits, alpha, _ = A.forward(P, cfg, x[None], tgt)
        assert np.array_equal(logits[0].argmax(-1), ys)
        assert np.allclose(alpha[0], rows)


def luong_inputs(seed=0, B=3, S=6, n=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(B, n)), rng.normal(size=(B, S, n)), rng


def test_general_with_the_identity_is_dot():
    h, Hs, _ = luong_inputs()
    assert np.allclose(A.luong_score("general", h, Hs, np.eye(4)), A.luong_score("dot", h, Hs))


def test_concat_is_the_additive_score():
    """Luong et al.'s concat, v_a . tanh(W_a [h_t; hs]), is Bahdanau et al.'s v_a . tanh(W_a s + U_a h) with the two
    matrices stacked; only which state is the query differs (h_t against s_{i-1})."""
    h, Hs, rng = luong_inputs()
    Wa, Ua, va = rng.normal(size=(4, 5)), rng.normal(size=(4, 5)), rng.normal(size=5)
    bahdanau = np.tanh((h @ Wa)[:, None] + Hs @ Ua) @ va
    assert np.allclose(A.luong_score("concat", h, Hs, np.vstack([Wa, Ua]), va), bahdanau)


def test_location_ignores_the_source():
    h, Hs, rng = luong_inputs()
    Wa = rng.normal(size=(4, 6))
    assert np.array_equal(A.luong_score("location", h, Hs, Wa), A.luong_score("location", h, -3 * Hs, Wa))


def test_local_p_weights():
    """eq-local-p: zero outside [p_t - D, p_t + D] and past the sentence, the global weights times the Gaussian inside,
    and p_t in [0, S]; a window that covers the sentence with an infinite sigma would give global attention."""
    h, Hs, rng = luong_inputs(B=4, S=9)
    lengths = np.array([9, 9, 5, 2])
    Wp, vp, Wa = rng.normal(size=(4, 4)), rng.normal(size=4) * 3, rng.normal(size=(4, 4))
    D = 2
    p_t, a, c = A.local_p_attention("general", h, Hs, lengths, D, Wp, vp, Wa)
    assert np.all((p_t >= 0) & (p_t <= lengths))
    pos = np.arange(9)
    inside = (np.abs(pos - p_t[:, None]) <= D) & (pos < lengths[:, None])
    assert np.all(a[~inside] == 0.0) and np.all(a[inside] > 0.0)
    assert np.all(a.sum(-1) <= 1.0 + 1e-12)
    g, _ = A.global_attention("general", h, Hs, inside, Wa)
    assert np.allclose(a, g * np.exp(-((pos - p_t[:, None]) ** 2) / (2 * (D / 2) ** 2)))
    assert np.allclose(c, np.einsum("bs,bsk->bk", a, Hs))


def test_attentional_state_shape_and_range():
    h, Hs, rng = luong_inputs()
    a, c = A.global_attention("dot", h, Hs, np.ones((3, 6), bool))
    out = A.attentional_state(c, h, rng.normal(size=(8, 4)))
    assert out.shape == (3, 4) and np.all(np.abs(out) < 1)


def test_worked_example():
    """The worked example of the page: s = (0.5, -0.5), annotations (1, 0), (0, 1) and (1, 1), W_a = U_a = I and
    v_a = (1, 1). The additive score puts the most weight on h_3 and the dot score on h_1."""
    s, H = np.array([[0.5, -0.5]]), np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    P = {"W_a": np.eye(2), "U_a": np.eye(2), "v_a": np.ones(2)}
    c, alpha, _ = A.align(P, s, H, H @ P["U_a"], np.ones((1, 3), bool))
    e = np.tanh(s[:, None] + H) @ P["v_a"]
    assert np.allclose(np.round(e, 4), [[0.4430, 0.9242, 1.3673]])
    assert np.allclose(np.round(alpha, 4), [[0.1946, 0.3149, 0.4905]])
    assert np.allclose(np.round(c, 4), [[0.6851, 0.8054]])
    a_dot, _ = A.global_attention("dot", s, H, np.ones((1, 3), bool))
    assert np.allclose(np.round(a_dot, 4), [[0.5065, 0.1863, 0.3072]])
    assert alpha.argmax() == 2 and a_dot.argmax() == 0
