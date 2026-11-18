# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
"""Tests of the NumPy reference: TopK as an exact L0 constraint and as the projection onto nonnegative k-sparse
vectors, the soft threshold against a brute-force minimum and a ReLU encoder, the shrinkage of the one-latent example,
the scale symmetry that the unit-norm decoder removes, the gradient projection, FVU, the relative reconstruction bias,
dead latents, the AuxK loss, and the worked example of the page."""

import itertools

import numpy as np
import pytest

from sparse_autoencoders import (
    auxk_loss,
    dead_fraction,
    decode,
    encode_relu,
    encode_topk,
    fire_counts,
    fvu,
    l1_loss,
    mse_loss,
    normalize_decoder,
    pre_activations,
    project_decoder_grad,
    relative_reconstruction_bias,
    relu,
    soft_threshold,
    topk,
)

RNG = np.random.default_rng(20261116)


def random_sae(d, n, rng=RNG):
    W_dec = normalize_decoder(rng.normal(size=(d, n)))
    return W_dec.T.copy(), rng.normal(size=n) * 0.1, W_dec, rng.normal(size=d) * 0.1


def test_relu_and_topk_basics():
    a = np.array([[3.0, -1.0, 2.0, 2.0, 0.5]])
    assert np.array_equal(relu(a), [[3.0, 0.0, 2.0, 2.0, 0.5]])
    assert np.array_equal(topk(a, 1), [[3.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.array_equal(
        topk(a, 2), [[3.0, 0.0, 2.0, 0.0, 0.0]]
    )  # tie between indices 2 and 3 goes to 2
    assert np.array_equal(topk(a, 5), a)
    with pytest.raises(ValueError):
        topk(a, 0)
    with pytest.raises(ValueError):
        topk(a, 6)


def test_topk_is_an_exact_l0_constraint():
    for _ in range(50):
        n = int(RNG.integers(2, 12))
        k = int(RNG.integers(1, n + 1))
        a = RNG.normal(size=(7, n))
        t = topk(a, k)
        assert np.all(np.count_nonzero(t, axis=-1) == k)
        kept = t != 0
        assert np.allclose(t[kept], a[kept])
        # every kept value is at least every dropped value in its row
        for row_a, row_k in zip(a, kept):
            if k < n:
                assert row_a[row_k].min() >= row_a[~row_k].max()


def test_encode_topk_never_exceeds_k_and_is_nonnegative():
    W_enc, _, _W_dec, b_pre = random_sae(6, 20)
    x = RNG.normal(size=(40, 6))
    for k in (1, 3, 8, 20):
        z = encode_topk(x, W_enc, b_pre, k)
        assert np.all(z >= 0)
        assert np.all(np.count_nonzero(z, axis=-1) <= k)
        pre = pre_activations(x, W_enc, b_pre)
        positive = np.sum(pre > 0, axis=-1)
        assert np.array_equal(np.count_nonzero(z, axis=-1), np.minimum(positive, k))


def brute_force_projection(a, k):
    """The nonnegative vector with at most k nonzeros closest to a, by trying every support."""
    best, best_cost = None, np.inf
    for size in range(k + 1):
        for s in itertools.combinations(range(a.size), size):
            z = np.zeros_like(a)
            idx = list(s)
            z[idx] = np.maximum(a[idx], 0.0)
            cost = np.sum((a - z) ** 2)
            if cost < best_cost - 1e-12:
                best, best_cost = z, cost
    return best, best_cost


def test_topk_relu_is_the_projection_onto_nonnegative_k_sparse_vectors():
    for _ in range(60):
        n = int(RNG.integers(2, 8))
        k = int(RNG.integers(1, n + 1))
        a = RNG.normal(size=n) * 2.0
        _, cost = brute_force_projection(a, k)
        mine = relu(topk(a[None, :], k))[0]
        assert np.sum((a - mine) ** 2) == pytest.approx(cost, abs=1e-12)


def test_topk_code_is_optimal_for_an_orthonormal_decoder():
    for _ in range(40):
        d = int(RNG.integers(3, 7))
        n = int(RNG.integers(2, d + 1))
        k = int(RNG.integers(1, n + 1))
        Q, _ = np.linalg.qr(RNG.normal(size=(d, d)))
        W_dec = Q[:, :n]
        x = RNG.normal(size=d)
        z = encode_topk(x[None, :], W_dec.T, np.zeros(d), k)[0]
        mine = np.sum((x - W_dec @ z) ** 2)
        best = np.inf
        for s in itertools.combinations(range(n), k):
            idx = list(s)
            # best nonnegative code on this support, found by brute force over a fine grid would be slow: since the
            # columns are orthonormal the problem separates, and each coefficient is max(0, <d_i, x>)
            zz = np.zeros(n)
            zz[idx] = np.maximum(W_dec[:, idx].T @ x, 0.0)
            best = min(best, np.sum((x - W_dec @ zz) ** 2))
        assert mine == pytest.approx(best, abs=1e-12)
        # and the Pythagorean split of the derivation topk-l0
        a = W_dec.T @ x
        split = np.sum((x - W_dec @ a) ** 2) + np.sum((a - z) ** 2)
        assert mine == pytest.approx(split, abs=1e-12)


def test_soft_threshold_matches_a_brute_force_minimum():
    grid = np.linspace(0.0, 6.0, 600001)
    for _ in range(30):
        d = int(RNG.integers(2, 6))
        u = RNG.normal(size=d)
        u /= np.linalg.norm(u)
        x = RNG.normal(size=d) * 2.0
        lam = float(RNG.uniform(0.0, 3.0))
        obj = (
            np.sum((x[None, :] - grid[:, None] * u[None, :]) ** 2, axis=1) + lam * grid
        )
        f = soft_threshold(u @ x, lam)
        assert f == pytest.approx(grid[np.argmin(obj)], abs=2e-5)
        # the unpenalized optimum is max(0, <u, x>), so the penalty lowers an active latent by exactly lam / 2
        if u @ x > lam / 2:
            assert max(u @ x, 0.0) - f == pytest.approx(lam / 2)


def test_tied_relu_encoder_with_bias_minus_half_lambda_is_the_soft_threshold():
    W_dec = normalize_decoder(RNG.normal(size=(5, 9)))
    x = RNG.normal(size=(30, 5))
    lam = 0.7
    z = encode_relu(x, W_dec.T, np.full(9, -lam / 2), np.zeros(5))
    assert np.allclose(z, soft_threshold(x @ W_dec, lam))


def test_one_latent_example_of_figure_2():
    # x = d with ||d|| = 1 and lam = 1: the optimum activation is 1/2, not 1
    d = np.array([[1.0]])
    for lam, expect in ((1.0, 0.5), (0.4, 0.8), (2.0, 0.0), (3.0, 0.0)):
        f = soft_threshold(1.0, lam)
        assert f == pytest.approx(expect)
        if f > 0:
            x_hat = decode(np.array([[f]]), d, np.zeros(1))
            assert relative_reconstruction_bias(
                np.array([[1.0]]), x_hat
            ) == pytest.approx(expect)


def test_scaling_the_decoder_lowers_the_l1_term_without_changing_the_output():
    W_enc, b_enc, W_dec, b_pre = random_sae(4, 10)
    x = RNG.normal(size=(25, 4))
    z = encode_relu(x, W_enc, b_enc, b_pre)
    x_hat = decode(z, W_dec, b_pre)
    for s in (2.0, 10.0, 1e3):
        z_s = encode_relu(x, W_enc / s, b_enc / s, b_pre)
        x_hat_s = decode(z_s, W_dec * s, b_pre)
        assert np.allclose(x_hat_s, x_hat)
        assert np.abs(z_s).sum() == pytest.approx(np.abs(z).sum() / s)
        assert (
            l1_loss(x, x_hat_s, z_s, 1.0) < l1_loss(x, x_hat, z, 1.0)
            or np.abs(z).sum() == 0
        )


def test_normalize_decoder_and_project_gradient():
    W = RNG.normal(size=(6, 11)) * 3.0
    Wn = normalize_decoder(W)
    assert np.allclose(np.linalg.norm(Wn, axis=0), 1.0)
    assert np.allclose(Wn * np.linalg.norm(W, axis=0), W)
    g = RNG.normal(size=W.shape)
    gp = project_decoder_grad(Wn, g)
    assert np.allclose(np.sum(gp * Wn, axis=0), 0.0)
    # the removed part is parallel to the column
    assert np.allclose(g - gp, Wn * np.sum(g * Wn, axis=0))
    # a small step along the projected gradient keeps the norms at 1 to first order
    eps = 1e-6
    assert np.allclose(np.linalg.norm(Wn - eps * gp, axis=0), 1.0, atol=1e-10)


def test_fvu():
    x = RNG.normal(size=(50, 4)) + 3.0
    assert fvu(x, x) == 0.0
    assert fvu(x, np.broadcast_to(x.mean(axis=0), x.shape)) == pytest.approx(1.0)
    x_hat = x + RNG.normal(size=x.shape) * 0.1
    expect = np.sum((x - x_hat) ** 2) / np.sum((x - x.mean(axis=0)) ** 2)
    assert fvu(x, x_hat) == pytest.approx(expect)
    with pytest.raises(ValueError):
        fvu(np.ones((3, 2)), np.ones((3, 2)))


def test_relative_reconstruction_bias():
    x = RNG.normal(size=(200, 5))
    x_hat = 0.7 * x + RNG.normal(size=x.shape) * 0.2
    g = relative_reconstruction_bias(x, x_hat)
    # 1/g is the scale that minimizes ||x_hat * c - x||^2
    errs = [np.sum((x_hat * c - x) ** 2) for c in (1 / g * 0.999, 1 / g, 1 / g * 1.001)]
    assert errs[1] < errs[0] and errs[1] < errs[2]
    # the second form of eq-gamma
    sq_hat, sq_x, sq_err = np.sum(x_hat**2), np.sum(x**2), np.sum((x_hat - x) ** 2)
    assert g == pytest.approx(2 * sq_hat / (sq_hat + sq_x - sq_err))
    assert relative_reconstruction_bias(x, x) == pytest.approx(1.0)
    assert relative_reconstruction_bias(x, 0.5 * x) == pytest.approx(0.5)


def test_fire_counts_and_dead_fraction():
    z = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.5, 0.0], [0.0, 0.0, 0.1, 0.0]])
    assert np.array_equal(fire_counts(z), [0, 2, 2, 0])
    assert dead_fraction(fire_counts(z)) == 0.5
    assert dead_fraction([1, 1, 1]) == 0.0


def test_auxk_loss():
    W_enc, _b_enc, W_dec, b_pre = random_sae(5, 8)
    x = RNG.normal(size=(12, 5))
    pre = pre_activations(x, W_enc, b_pre)
    z = relu(topk(pre, 2))
    x_hat = decode(z, W_dec, b_pre)
    assert auxk_loss(x, x_hat, pre, np.zeros(8, bool), W_dec, 3) == 0.0
    dead = np.array([True, False, True, False, False, True, False, False])
    got = auxk_loss(x, x_hat, pre, dead, W_dec, 2)
    e = x - x_hat
    expect = 0.0
    for row_pre, row_e in zip(pre, e):
        cand = [(row_pre[i], i) for i in range(8) if dead[i]]
        cand.sort(key=lambda t: (-t[0], t[1]))
        zz = np.zeros(8)
        for v, i in cand[:2]:
            zz[i] = max(v, 0.0)
        expect += np.sum((row_e - W_dec @ zz) ** 2)
    assert got == pytest.approx(expect / len(x))
    # live latents never enter: changing their pre-activations leaves the loss alone
    pre2 = pre.copy()
    pre2[:, ~dead] += 100.0
    assert auxk_loss(x, x_hat, pre2, dead, W_dec, 2) == pytest.approx(got)
    # k_aux larger than the number of dead latents uses all of them
    assert np.isfinite(auxk_loss(x, x_hat, pre, dead, W_dec, 50))


def test_l1_loss():
    x = np.array([[1.0, 2.0]])
    assert l1_loss(x, x, np.array([[1.0, -2.0]]), 0.5) == pytest.approx(1.5)
    assert mse_loss(x, np.zeros_like(x)) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        l1_loss(x, x, x, -1.0)


def test_worked_example():
    # two tokens, an orthonormal decoder of two latents, encoder tied to the decoder, no biases
    x = np.array([[3.0, 1.0], [1.0, 3.0]])
    W_dec = np.eye(2)
    zero = np.zeros(2)
    rows = {}
    for name, z in (
        ("l1 lam=1", encode_relu(x, W_dec.T, np.full(2, -0.5), zero)),
        ("l1 lam=2", encode_relu(x, W_dec.T, np.full(2, -1.0), zero)),
        ("topk k=1", encode_topk(x, W_dec.T, zero, 1)),
    ):
        x_hat = decode(z, W_dec, zero)
        l0 = float(np.mean(np.count_nonzero(z, axis=-1)))
        rows[name] = (
            z,
            l0,
            np.sum((x - x_hat) ** 2),
            fvu(x, x_hat),
            relative_reconstruction_bias(x, x_hat),
        )
    z, l0, err, f, g = rows["l1 lam=1"]
    assert (
        np.allclose(z, [[2.5, 0.5], [0.5, 2.5]])
        and l0 == 2
        and err == pytest.approx(1.0)
    )
    assert f == pytest.approx(0.25) and g == pytest.approx(13 / 16)
    z, l0, err, f, g = rows["l1 lam=2"]
    assert (
        np.allclose(z, [[2.0, 0.0], [0.0, 2.0]])
        and l0 == 1
        and err == pytest.approx(4.0)
    )
    assert f == pytest.approx(1.0) and g == pytest.approx(2 / 3)
    z, l0, err, f, g = rows["topk k=1"]
    assert (
        np.allclose(z, [[3.0, 0.0], [0.0, 3.0]])
        and l0 == 1
        and err == pytest.approx(2.0)
    )
    assert f == pytest.approx(0.5) and g == pytest.approx(1.0)


def test_biases_and_plain_list_inputs():
    x = [[1.0, 2.0], [0.0, -1.0]]
    W_enc = [[1.0, 0.5], [-0.5, 2.0], [0.0, 1.0]]
    b_pre = np.array([0.5, -0.25])
    b_enc = np.array([0.1, -0.2, 0.3])
    xc = np.array(x) - b_pre
    pre = pre_activations(x, W_enc, b_pre, b_enc)
    assert np.allclose(pre, xc @ np.array(W_enc).T + b_enc)
    assert np.allclose(
        encode_relu(x, W_enc, b_enc, b_pre),
        np.maximum(xc @ np.array(W_enc).T + b_enc, 0.0),
    )
    W_dec = [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]
    z = [[1.0, 0.0, 2.0]]
    assert np.allclose(decode(z, W_dec, b_pre), np.array([[5.0, -2.0]]) + b_pre)
    assert np.array_equal(topk([[1, 3, 2]], 1), np.array([[0.0, 3.0, 0.0]]))
    assert l1_loss(x, x, z, 0.0) == 0.0
    assert l1_loss(x, np.zeros((2, 2)), z, 0.0) == mse_loss(x, np.zeros((2, 2)))
    g = [[1.0, 2.0], [3.0, 4.0]]
    W = np.eye(2)
    assert np.allclose(project_decoder_grad(W, g), np.array([[0.0, 2.0], [3.0, 0.0]]))
    assert fvu([[1.0], [3.0]], [[2.0], [2.0]]) == 1.0
    assert relative_reconstruction_bias([[2.0, 0.0]], [[1.0, 0.0]]) == 0.5
    assert dead_fraction([0, 3, 0, 1]) == 0.5
    e = auxk_loss(
        [[1.0, 0.0]], [[0.0, 0.0]], [[2.0, 1.0]], [1, 0], [[1.0, 0.0], [0.0, 1.0]], 1
    )
    assert e == 1.0
