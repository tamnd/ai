# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the gradients against finite
differences, the corruptions, the contractive penalty, the identity an over-complete autoencoder can learn, the linear
autoencoder and PCA with and without weight decay, a curve that PCA cannot follow, and denoising."""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from autoencoder import (
    backward,
    config,
    contractive_penalty,
    corrupt,
    emphasis,
    flops,
    forward,
    from_widths,
    init,
    jacobian,
    loss,
    objective,
    param_count,
    pca,
    pca_reconstruct,
    reconstruct,
    sigmoid,
    train,
)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["d_in"], k["d_hidden"], k["n_hidden"], k["d_code"], **kw)


def unflat(arrays):
    p = {}
    for key in sorted(arrays, key=lambda s: (s.split("/")[0], int(s.split("/")[1]))):
        p.setdefault(key.split("/")[0], []).append(arrays[key])
    return p


def random_params(rng, cfg):
    p = init(rng, cfg)
    for k in p:
        if k.endswith("_b"):
            p[k] = [rng.normal(0.0, 0.3, b.shape) for b in p[k]]
    return p


def test_shared_fixture():
    params = unflat(dict(np.load(FIXTURES / "params-small.npz")))
    x = np.load(FIXTURES / "input-small.npz")["x"]
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    out, cache = forward(params, cfg, x)
    for k in ("y", "a", "z"):
        np.testing.assert_allclose(out[k], want[k], rtol=1e-12, atol=1e-14)
    value, da = loss(cfg, x, out["a"])
    np.testing.assert_allclose(value, want["loss"], rtol=1e-12)
    grads, dx = backward(params, cfg, cache, da)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-10, atol=1e-14)
    for k, vs in grads.items():
        for i, v in enumerate(vs):
            np.testing.assert_allclose(v, want[f"d{k}/{i}"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    m, B = c["model"], c["config"]["B"]
    assert param_count(cfg_of(c)) == m["params"]
    assert param_count(cfg_of(c, tied=True)) == m["params_tied"]
    assert flops(cfg_of(c), B) == (m["flops_forward"], m["flops_backward"])
    if c["kind"] != "named":
        for tied in (False, True):
            p = init(np.random.default_rng(0), cfg_of(c, tied=tied))
            n = sum(a.size for vs in p.values() for a in vs)
            assert n == m["params_tied" if tied else "params"]
        out, cache = forward(p, cfg_of(c, tied=True), np.zeros((B, c["config"]["d_in"])))
        for name, blk in c["blocks"].items():
            if name == "code":
                assert list(out["y"].shape) == blk["outputs"]["y"]
            if name == "recon":
                assert list(out["z"].shape) == blk["outputs"]["z"]


def numeric_grads(f, p, eps=1e-6):
    g = {}
    for k, vs in p.items():
        g[k] = []
        for W in vs:
            G = np.zeros_like(W)
            for idx in np.ndindex(W.shape):
                old = W[idx]
                W[idx] = old + eps
                up = f()
                W[idx] = old - eps
                down = f()
                W[idx] = old
                G[idx] = (up - down) / (2 * eps)
            g[k].append(G)
    return g


CASES = [
    dict(widths=[5, 4, 2], act="sigmoid", code_act="linear", out="sigmoid", tied=False),
    dict(widths=[5, 4, 3, 2], act="tanh", code_act="tanh", out="linear", tied=True),
    dict(widths=[4, 6], act="sigmoid", code_act="sigmoid", out="sigmoid", tied=True),
    dict(widths=[6, 3, 3], act="relu", code_act="linear", out="linear", tied=False),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c["widths"])) + ("-tied" if c["tied"] else ""))
def test_gradients_match_finite_differences(case):
    rng = np.random.default_rng(1)
    cfg = from_widths(**case)
    p = random_params(rng, cfg)
    D = cfg["widths"][0]
    x = rng.uniform(0.0, 1.0, (3, D))
    weight = rng.uniform(0.0, 2.0, (3, D))

    def value():
        return loss(cfg, x, forward(p, cfg, x)[0]["a"], weight)[0]

    out, cache = forward(p, cfg, x)
    grads, dx = backward(p, cfg, cache, loss(cfg, x, out["a"], weight)[1])
    want = numeric_grads(value, p)
    for k in p:
        for G, N in zip(grads[k], want[k]):
            np.testing.assert_allclose(G, N, rtol=1e-6, atol=1e-8)
    # the gradient in the input, with x held fixed as the target
    xin = x.copy()

    def value_in():
        return loss(cfg, x, forward(p, cfg, xin)[0]["a"], weight)[0]

    num = numeric_grads(value_in, {"x": [xin]})["x"][0]
    np.testing.assert_allclose(dx, num, rtol=1e-6, atol=1e-8)


def test_cross_entropy_from_logits():
    rng = np.random.default_rng(2)
    x, a = rng.uniform(0, 1, (4, 5)), rng.normal(0, 3, (4, 5))
    z = sigmoid(a)
    want = -np.sum(x * np.log(z) + (1 - x) * np.log(1 - z)) / 4
    np.testing.assert_allclose(loss(config(), x, a)[0], want, rtol=1e-12)
    assert np.isfinite(loss(config(), np.ones((1, 2)), np.array([[-800.0, 800.0]]))[0])


def test_corruptions():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.8, (50, 20))
    for kind in ("mask", "salt_pepper"):
        xt, mask = corrupt(rng, x, kind, 0.25)
        assert (mask.sum(1) == 5).all()  # exactly round(nu D) components of each example
        np.testing.assert_array_equal(xt[~mask], x[~mask])
        if kind == "mask":
            assert (xt[mask] == 0.0).all()
        else:
            assert set(np.unique(xt[mask])) == {0.0, 1.0}
    xt, mask = corrupt(rng, np.zeros((400, 50)), "gaussian", 0.5)
    assert mask.all() and abs(xt.std() - 0.5) < 0.01
    w = emphasis(mask[:2, :3] & np.array([True, False, True]), 1.0, 0.0)
    np.testing.assert_array_equal(w, [[1, 0, 1], [1, 0, 1]])


def test_full_emphasis_ignores_the_uncorrupted_components():
    rng = np.random.default_rng(4)
    cfg = config(6, 5, 1, 2)
    p = random_params(rng, cfg)
    x = rng.uniform(0, 1, (3, 6))
    _, mask = corrupt(rng, x, "mask", 0.5)
    a = forward(p, cfg, x)[0]["a"]
    a2 = np.where(mask, a, a + rng.normal(size=a.shape))  # change only the components that are not corrupted
    w = emphasis(mask, 1.0, 0.0)
    assert loss(cfg, x, a, w)[0] == pytest.approx(loss(cfg, x, a2, w)[0], rel=1e-14)
    assert (loss(cfg, x, a, w)[1][~mask] == 0).all()


def test_contractive_penalty():
    rng = np.random.default_rng(5)
    cfg = from_widths([5, 4], code_act="sigmoid")
    p = random_params(rng, cfg)
    x = rng.normal(0, 1, (3, 5))
    value, g = contractive_penalty(p, cfg, x)
    want = np.mean([np.sum(jacobian(p, cfg, xi) ** 2) for xi in x])
    np.testing.assert_allclose(value, want, rtol=1e-12)
    num = numeric_grads(lambda: contractive_penalty(p, cfg, x)[0], {"W": p["enc_W"], "b": p["enc_b"]})
    np.testing.assert_allclose(g["W"], num["W"][0], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(g["b"], num["b"][0], rtol=1e-6, atol=1e-9)


def test_contractive_gradient_in_the_objective():
    rng = np.random.default_rng(6)
    cfg = from_widths([5, 4], code_act="sigmoid", out="sigmoid")
    p = random_params(rng, cfg)
    x = rng.uniform(0, 1, (3, 5))
    _, grads = objective(p, cfg, x, contraction=0.3, weight_decay=0.1)
    num = numeric_grads(lambda: objective(p, cfg, x, contraction=0.3, weight_decay=0.1)[0], p)
    for k in p:
        for G, N in zip(grads[k], num[k]):
            np.testing.assert_allclose(G, N, rtol=1e-6, atol=1e-8)


def test_a_linear_encoder_makes_the_contractive_penalty_weight_decay():
    # Rifai et al. (2011), p. 3: with a linear encoder ||J_f(x)||_F^2 = ||W||_F^2 at every x
    rng = np.random.default_rng(7)
    cfg = from_widths([5, 3], code_act="linear")
    p = random_params(rng, cfg)
    for xi in rng.normal(0, 3, (4, 5)):
        np.testing.assert_allclose(np.sum(jacobian(p, cfg, xi) ** 2), np.sum(p["enc_W"][0] ** 2), rtol=1e-12)


def test_an_over_complete_autoencoder_can_learn_the_identity():
    # Vincent et al. (2010), p. 3377, footnote 3: with d' >= d, W' = W^{-1} reconstructs every input exactly
    rng = np.random.default_rng(8)
    cfg = from_widths([4, 4], code_act="linear", out="linear")
    p = init(rng, cfg)
    p["enc_b"] = [rng.normal(size=4)]
    p["dec_W"] = [np.linalg.inv(p["enc_W"][0])]
    p["dec_b"] = [-p["enc_b"][0] @ p["dec_W"][0]]
    x = rng.normal(0, 10, (20, 4))
    np.testing.assert_allclose(reconstruct(p, cfg, x), x, rtol=1e-9, atol=1e-9)


def gaussian_data(rng, variances, n):
    """n points with mean 1 whose sample covariance has exactly the given eigenvalues, along a random rotation."""
    D = len(variances)
    Q = np.linalg.qr(rng.normal(size=(D, D)))[0]
    Z = rng.normal(size=(n, D))
    Z -= Z.mean(0)
    Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z / n)).T  # whitened: Z.T Z / n = I
    return 1.0 + Z * np.sqrt(variances) @ Q.T, Q


def gd(p, cfg, X, steps, lr, **kw):
    for _ in range(steps):
        _, g = objective(p, cfg, X, **kw)
        p = {k: [W - lr * G for W, G in zip(p[k], g[k])] for k in p}
    return p


def test_linear_autoencoder_finds_the_principal_subspace_but_not_the_directions():
    rng = np.random.default_rng(9)
    variances = np.array([4.0, 2.0, 0.5, 0.1])
    X, Q = gaussian_data(rng, variances, 200)
    cfg = from_widths([4, 2], code_act="linear", out="linear")
    p = gd(init(rng, cfg), cfg, X, 4000, 0.05)
    err = np.sum((reconstruct(p, cfg, X) - X) ** 2) / len(X)
    assert err == pytest.approx(variances[2:].sum(), rel=1e-6)  # Eckart-Young: the discarded variance
    np.testing.assert_allclose(reconstruct(p, cfg, X), pca_reconstruct(X, 2), atol=1e-4)
    P = Q[:, :2] @ Q[:, :2].T
    Wd = p["dec_W"][0].T  # (D, C), the columns span the decoder's range
    np.testing.assert_allclose(P @ Wd, Wd, atol=1e-4)  # in the principal subspace
    # ... but any invertible G gives the same reconstruction, so the columns need not be the principal directions
    G = np.array([[2.0, 1.0], [0.5, 1.0]])
    q = {"enc_W": [p["enc_W"][0] @ G], "enc_b": [p["enc_b"][0] @ G], "dec_W": [np.linalg.inv(G) @ p["dec_W"][0]],
         "dec_b": p["dec_b"]}  # fmt: skip
    np.testing.assert_allclose(reconstruct(q, cfg, X), reconstruct(p, cfg, X), atol=1e-10)
    cos = np.abs(Q[:, :2].T @ (q["dec_W"][0].T / np.linalg.norm(q["dec_W"][0], axis=1)))
    assert cos.max(0).min() < 0.99  # a column of the decoder that is not a principal direction


def test_weight_decay_recovers_the_principal_directions():
    # Kunin et al. (2019), Theorem 2.1 and p. 4: the encoder is the decoder's transpose, the decoder's left singular
    # vectors are the principal directions, and its squared singular values are 1 - lambda / sigma_i^2
    rng = np.random.default_rng(10)
    variances, lam = np.array([4.0, 2.0, 0.5, 0.1]), 0.3
    X, Q = gaussian_data(rng, variances, 200)
    cfg = from_widths([4, 2], code_act="linear", out="linear")
    p = gd(init(rng, cfg), cfg, X, 6000, 0.05, weight_decay=lam)
    np.testing.assert_allclose(p["enc_W"][0], p["dec_W"][0].T, atol=1e-6)
    U, s, _ = np.linalg.svd(p["dec_W"][0].T, full_matrices=False)
    np.testing.assert_allclose(s**2, 1.0 - lam / variances[:2], atol=1e-6)
    np.testing.assert_allclose(np.abs(U.T @ Q[:, :2]), np.eye(2), atol=1e-5)
    _, P, s2 = pca(X, 2)
    np.testing.assert_allclose(s2 / len(X), variances, rtol=1e-10)
    # a direction with sigma^2 <= lambda collapses to zero
    p = gd(init(rng, cfg), cfg, X, 6000, 0.05, weight_decay=2.5)
    s = np.linalg.svd(p["dec_W"][0], compute_uv=False)
    np.testing.assert_allclose(s**2, [1 - 2.5 / 4.0, 0.0], atol=1e-4)


def test_worked_example():
    # 2-D data with variances 4 and 1 and one linear code unit: the error is the discarded variance 1 without decay;
    # with lambda = 1 the encoder and decoder are both sqrt(0.75) times the first principal direction, the
    # reconstruction is 0.75 times the projection and the error 0.25^2 * 4 + 1 = 1.25; at lambda >= 4 the code dies.
    rng = np.random.default_rng(12)
    X, Q = gaussian_data(rng, np.array([4.0, 1.0]), 100)
    cfg = from_widths([2, 1], code_act="linear", out="linear")
    for lam, m2, want in ((0.0, None, 1.0), (1.0, 0.75, 1.25), (5.0, 0.0, 5.0)):
        p = gd(init(rng, cfg), cfg, X, 3000, 0.05, weight_decay=lam)
        err = np.sum((reconstruct(p, cfg, X) - X) ** 2) / len(X)
        assert err == pytest.approx(want, abs=1e-6)
        if m2 is not None:
            np.testing.assert_allclose(p["enc_W"][0][:, 0], p["dec_W"][0][0], atol=1e-6)
            np.testing.assert_allclose(np.abs(Q[:, 0] @ p["dec_W"][0][0]), np.sqrt(m2), atol=1e-6)


def curve(t, D=8):
    """Points of a 1-D curve in R^D, open, so that one number can name each point."""
    k = np.arange(1, D // 2 + 1)
    return np.hstack([np.cos(np.outer(t, k)), np.sin(np.outer(t, k))]) / np.sqrt(np.concatenate([k, k]))


def test_a_nonlinear_code_follows_a_curve_that_pca_cannot():
    rng = np.random.default_rng(11)
    X = curve(rng.uniform(0.0, 0.8 * np.pi, 300))
    cfg = from_widths([8, 16, 1], act="tanh", code_act="linear", out="linear")
    p, _ = train(rng, cfg, X, 4000, lr=3e-3, batch=64)
    ae = np.mean(np.sum((reconstruct(p, cfg, X) - X) ** 2, 1))
    lin = np.mean(np.sum((pca_reconstruct(X, 1) - X) ** 2, 1))
    assert ae < 0.01 * lin


def test_denoising_fills_in_what_the_mask_removed():
    rng = np.random.default_rng(12)
    protos = rng.random((4, 16)) < 0.5

    def sample(n):
        x = protos[rng.integers(0, 4, n)]
        return np.where(rng.random(x.shape) < 0.03, ~x, x).astype(float)

    X, test = sample(1000), sample(400)
    cfg = from_widths([16, 32], code_act="sigmoid", out="sigmoid")  # over-complete, so the plain one can copy
    plain, _ = train(np.random.default_rng(13), cfg, X, 1500, lr=1e-2)
    dae, _ = train(np.random.default_rng(13), cfg, X, 1500, lr=1e-2, noise=("mask", 0.25))
    xt, mask = corrupt(rng, test, "mask", 0.25)

    def fill_error(p):
        return np.mean(np.abs(reconstruct(p, cfg, xt) - test)[mask])

    assert fill_error(dae) < 0.15 and fill_error(plain) > 2 * fill_error(dae)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.integers(1, 6), min_size=2, max_size=4), st.integers(1, 4), st.booleans())
def test_shapes(widths, B, tied):
    cfg = from_widths(widths, tied=tied)
    p = init(np.random.default_rng(0), cfg)
    out, cache = forward(p, cfg, np.zeros((B, widths[0])))
    assert out["y"].shape == (B, widths[-1]) and out["z"].shape == (B, widths[0])
    grads, dx = backward(p, cfg, cache, np.ones((B, widths[0])))
    assert {k: [g.shape for g in v] for k, v in grads.items()} == {k: [w.shape for w in v] for k, v in p.items()}
    assert dx.shape == (B, widths[0])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg, k = cfg_of(c), c["config"]
    rng = np.random.default_rng(0)
    return forward, (init(rng, cfg), cfg, rng.uniform(size=(k["B"], k["d_in"])))
