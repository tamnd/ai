# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the worked example, the free energy and
the conditionals of the RBM against brute-force sums, the exact gradient against finite differences, the Gibbs
transition and its fixed point, CD-k against the exact gradient as k grows, CD and PCD learning a binary mixture, the
network energy's derivatives, the stationary variance of Langevin dynamics, an energy trained with Langevin samples,
score matching, noise-contrastive estimation and the energy of a classifier."""

import json
import pathlib

import numpy as np
import pytest

import energy_based as eb

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def random_rbm(D, H, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return {k: scale * rng.standard_normal(s) for k, s in eb.param_shapes(eb.config(D, H)).items()}


def mixture(D=8, eps=0.05):
    protos = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]], float)[:, :D]
    X = eb.all_binary(D)
    flips = np.abs(X[:, None, :] - protos[None]).sum(-1)
    return np.mean(eps**flips * (1 - eps) ** (D - flips), 1)


def test_shared_fixture():
    P = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    p = {k[4:]: v for k, v in P.items() if k.startswith("rbm_")}
    q = {k[4:]: v for k, v in P.items() if k.startswith("mlp_")}
    v, gamma = inp["v"], float(inp["gamma"])
    np.testing.assert_allclose(eb.free_energy(p, v), want["free_energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.energy(p, v, inp["h"]), want["energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.p_h(p, v), want["p_h"], rtol=1e-12, atol=1e-13)
    h1, v1 = eb.gibbs_step(p, v, inp["u_h"], inp["u_v"])
    np.testing.assert_array_equal(h1, want["h1"])
    np.testing.assert_array_equal(v1, want["v1"])
    for k, g in eb.grad_from_samples(p, v, v1).items():
        np.testing.assert_allclose(g, want[f"cd_{k}"], err_msg=k, rtol=1e-12, atol=1e-13)
    for k, g in eb.exact_grad(p, v).items():
        np.testing.assert_allclose(g, want[f"ml_{k}"], err_msg=k, rtol=1e-12, atol=1e-13)
    assert eb.log_partition(p) == pytest.approx(float(want["log_z"]), rel=1e-12)
    x = inp["x"]
    np.testing.assert_allclose(eb.mlp_energy(q, x, gamma), want["mlp_energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.mlp_grad_x(q, x, gamma), want["mlp_grad_x"], rtol=1e-12, atol=1e-13)
    for k, g in eb.mlp_grad_params(q, x).items():
        np.testing.assert_allclose(g, want[f"mlp_d{k}"], err_msg=k, rtol=1e-12, atol=1e-13)
    assert eb.mlp_sm_objective(q, x, gamma) == pytest.approx(float(want["mlp_sm"]), rel=1e-12)
    step = float(inp["step"])
    np.testing.assert_allclose(
        eb.langevin_step(q, x, step, inp["z"], None, gamma), want["langevin"], rtol=1e-12, atol=1e-13
    )
    theta, nx, ny = inp["theta"], inp["nce_x"], inp["nce_y"]
    assert eb.nce_objective(theta, nx, ny, 0.0, 2.0) == pytest.approx(float(want["nce_j"]), rel=1e-12)
    np.testing.assert_allclose(eb.nce_grad(theta, nx, ny, 0.0, 2.0), want["nce_grad"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.jem_energy(inp["logits"]), want["jem_energy"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = eb.config(k["d_visible"], k["d_hidden"])
    assert eb.param_count(cfg) == mod["params"]
    assert eb.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = eb.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    v = np.zeros((k["B"], k["d_visible"]))
    h, v1 = eb.gibbs_step(p, v, np.ones((k["B"], k["d_hidden"])), np.ones_like(v))
    assert h.shape == (k["B"], k["d_hidden"]) and v1.shape == v.shape


def test_worked_example():
    p = {"W": np.array([[2.0], [-1.0]]), "b": np.zeros(2), "c": np.array([-1.0])}
    X = eb.all_binary(2)  # (0,0), (0,1), (1,0), (1,1)
    np.testing.assert_allclose(np.exp(-eb.free_energy(p, X)), [1 + np.e**-1, 1 + np.e**-2, 1 + np.e, 2.0])
    assert np.exp(eb.log_partition(p)) == pytest.approx(8.2215, abs=1e-4)
    probs = eb.model_probs(p)
    assert probs[2] == pytest.approx(0.4523, abs=1e-4) and probs[3] == pytest.approx(0.2433, abs=1e-4)
    assert eb.p_h(p, X)[[2, 3], 0] == pytest.approx([0.7311, 0.5], abs=1e-4)


@pytest.mark.parametrize("D, H", [(3, 2), (4, 3), (5, 1)])
def test_free_energy_sums_out_the_hidden_units(D, H):
    p = random_rbm(D, H, 1)
    X, Hs = eb.all_binary(D), eb.all_binary(H)
    brute = np.array([np.log(np.sum(np.exp(-eb.energy(p, np.repeat(v[None], len(Hs), 0), Hs)))) for v in X])
    np.testing.assert_allclose(-eb.free_energy(p, X), brute, rtol=1e-12)
    joint = np.exp(-np.array([eb.energy(p, np.repeat(v[None], len(Hs), 0), Hs) for v in X]))
    joint /= joint.sum()
    np.testing.assert_allclose(joint.sum(1), eb.model_probs(p), rtol=1e-12)
    # p(h | v) is the product of the sigmoids of eq-rbm-conditionals
    ph = eb.p_h(p, X)
    fact = np.prod(np.where(Hs[None] == 1, ph[:, None], 1 - ph[:, None]), -1)
    np.testing.assert_allclose(joint / joint.sum(1, keepdims=True), fact, rtol=1e-12)
    pv = eb.p_v(p, Hs)
    fact_v = np.prod(np.where(X[None] == 1, pv[:, None], 1 - pv[:, None]), -1)
    np.testing.assert_allclose((joint / joint.sum(0, keepdims=True)).T, fact_v, rtol=1e-12)


def test_exact_gradient_matches_finite_differences():
    p = random_rbm(4, 3, 2)
    data = eb.all_binary(4)[[1, 6, 7, 12, 15]]
    g = eb.exact_grad(p, data)
    eps = 1e-6
    for k in p:
        for idx in np.ndindex(p[k].shape):
            a = {key: v.copy() for key, v in p.items()}
            b = {key: v.copy() for key, v in p.items()}
            a[k][idx] += eps
            b[k][idx] -= eps
            fd = (eb.log_prob(a, data).mean() - eb.log_prob(b, data).mean()) / (2 * eps)
            assert g[k][idx] == pytest.approx(fd, abs=1e-8), (k, idx)


def test_gibbs_transition_is_stochastic_and_keeps_the_model_distribution():
    p = random_rbm(4, 3, 3)
    T = eb.gibbs_transition(p)
    np.testing.assert_allclose(T.sum(1), 1.0, rtol=1e-12)
    pi = eb.model_probs(p)
    np.testing.assert_allclose(pi @ T, pi, rtol=1e-10, atol=1e-14)
    assert (T > 0).all()  # every transition has non-zero probability, so the chain is ergodic


def test_cd_k_approaches_the_exact_gradient_as_k_grows():
    p = random_rbm(4, 3, 4, 1.5)
    X = eb.all_binary(4)
    data = X[[0, 3, 5, 6, 9, 15, 15]]
    T = eb.gibbs_transition(p)
    emp = np.array([np.mean(np.all(data == v, 1)) for v in X])
    exact = eb.exact_grad(p, data)
    ph = eb.p_h(p, X)

    def expected_cd(k):
        w = emp @ np.linalg.matrix_power(T, k)  # the distribution of the chains after k steps from the data
        sp = eb.stats(p, data)
        return {"W": sp["W"] - (X * w[:, None]).T @ ph, "b": sp["b"] - w @ X, "c": sp["c"] - w @ ph}

    err = [max(np.abs(expected_cd(k)[key] - exact[key]).max() for key in p) for k in (1, 3, 10, 60)]
    assert err[0] > 1e-3
    assert err[0] > err[1] > err[2] > err[3]
    assert err[3] < 1e-8


def test_gibbs_sampling_draws_from_the_model():
    p = random_rbm(3, 2, 5)
    rng = np.random.default_rng(6)
    v = eb.gibbs_chain(p, rng.integers(0, 2, (20000, 3)).astype(float), 30, rng)
    X = eb.all_binary(3)
    freq = np.array([np.mean(np.all(v == x, 1)) for x in X])
    pr = eb.model_probs(p)
    assert np.abs(freq - pr).max() < 4 * np.sqrt(pr * (1 - pr) / len(v)).max() + 1e-3


@pytest.mark.parametrize("persistent", [False, True], ids=["cd1", "pcd"])
def test_rbm_learns_a_binary_mixture(persistent):
    rng = np.random.default_rng(7)
    prob = mixture()
    X = eb.all_binary(8)
    data = X[rng.choice(len(X), 4000, p=prob / prob.sum())]
    p0 = eb.init(np.random.default_rng(8), eb.config(8, 8), 0.1)
    p, trace = eb.train_rbm(p0, data, 800, np.random.default_rng(9), k=1, persistent=persistent, lr=0.1)
    kl = float(np.sum(prob * (np.log(prob) - eb.log_prob(p, X))))
    marg = data.mean(0)
    indep = (X * np.log(marg) + (1 - X) * np.log(1 - marg)).sum(1)
    kl_indep = float(np.sum(prob * (np.log(prob) - indep)))
    assert trace[-1] > trace[0] + 2.0
    assert kl < 0.1 < 1.0 < kl_indep


def test_network_energy_derivatives_match_finite_differences():
    rng = np.random.default_rng(10)
    q = {k: rng.standard_normal(s) for k, s in eb.mlp_shapes(3, 5).items()}
    x = rng.standard_normal((4, 3))
    eps = 1e-6
    g = eb.mlp_grad_x(q, x)
    lap = np.zeros(4)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        fd = (eb.mlp_energy(q, x + e) - eb.mlp_energy(q, x - e)) / (2 * eps)
        np.testing.assert_allclose(g[:, i], fd, atol=1e-8)
        e[i] = 1e-4
        lap += (eb.mlp_energy(q, x + e) - 2 * eb.mlp_energy(q, x) + eb.mlp_energy(q, x - e)) / 1e-8
    np.testing.assert_allclose(eb.mlp_laplacian(q, x), lap, atol=1e-5)
    gp = eb.mlp_grad_params(q, x)
    for k in q:
        for idx in np.ndindex(q[k].shape):
            a = {key: v.copy() for key, v in q.items()}
            b = {key: v.copy() for key, v in q.items()}
            a[k][idx] += eps
            b[k][idx] -= eps
            fd = (eb.mlp_energy(a, x).mean() - eb.mlp_energy(b, x).mean()) / (2 * eps)
            assert gp[k][idx] == pytest.approx(fd, abs=1e-8), (k, idx)


@pytest.mark.parametrize("step", [0.4, 0.1, 0.02])
def test_langevin_stationary_variance_on_a_quadratic_energy(step):
    s2 = 0.5  # E(x) = |x|^2 / (2 s2): the network part off, gamma = 1 / s2
    q = {"W1": np.zeros((2, 1)), "b1": np.zeros(1), "w2": np.zeros(1)}
    rng = np.random.default_rng(11)
    x = eb.langevin(q, rng.standard_normal((20000, 2)), 300, step, rng, None, 1 / s2)
    a = 1 - step / (2 * s2)
    want = step / (1 - a * a)  # the exact stationary variance of the discretized chain; s2 as step -> 0
    assert x.var() == pytest.approx(want, rel=0.03)
    assert abs(want - s2) <= 0.6 * step


def test_energy_trained_on_langevin_samples_finds_four_modes():
    rng = np.random.default_rng(12)
    C = np.array([[1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]])
    data = C[rng.integers(0, 4, 4000)] + 0.25 * rng.standard_normal((4000, 2))
    q, _, _ = eb.train_ebm(eb.mlp_init(np.random.default_rng(13), 2, 32), data, 800, np.random.default_rng(14))
    u = np.random.default_rng(15).uniform(-3, 3, (2000, 2))
    assert eb.mlp_energy(q, data).mean() < eb.mlp_energy(q, u).mean() - 0.5
    s = eb.langevin(q, u, 200, 0.02, np.random.default_rng(16), 0.05)
    dist = np.linalg.norm(s[:, None] - C[None], axis=-1)
    assert np.mean(dist.min(1) < 0.75) > 0.8
    share = np.bincount(dist.argmin(1), minlength=4) / len(s)
    assert share.min() > 0.1


def test_score_matching_drops_the_data_score_and_matches_maximum_likelihood():
    rng = np.random.default_rng(17)
    mu0 = np.array([0.5, -1.0])
    L = np.array([[1.0, 0.0], [0.6, 0.8]])
    z = rng.standard_normal((5000, 2))
    z -= z.mean(0)
    z = z @ np.linalg.inv(np.linalg.cholesky(z.T @ z / len(z))).T  # sample mean 0 and covariance I exactly
    x = mu0 + z @ L.T
    M0 = np.linalg.inv(L @ L.T)
    # Theorem 1: the explicit objective, with the true data score, differs from the implicit one by a constant. Both
    # are quadratic in x, so with the sample's moments equal to the data distribution's the identity holds exactly.
    diffs = []
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        A = r.standard_normal((2, 2))
        mu, M = r.standard_normal(2), A @ A.T + np.eye(2)
        explicit = 0.5 * np.mean(np.sum(((x - mu) @ M - (x - mu0) @ M0) ** 2, 1))
        diffs.append(explicit - eb.gaussian_sm_objective(mu, M, x))
    np.testing.assert_allclose(diffs, 0.5 * np.trace(M0), rtol=1e-10)
    mu, M = eb.gaussian_sm_fit(x)
    r = x - x.mean(0)
    np.testing.assert_allclose(M, np.linalg.inv(r.T @ r / len(x)), rtol=1e-12)  # the ML estimate, for any sample
    best = eb.gaussian_sm_objective(mu, M, x)
    for d in (np.array([[0.05, 0], [0, 0]]), np.array([[0, 0.03], [0.03, 0]]), np.array([[0, 0], [0, -0.05]])):
        assert eb.gaussian_sm_objective(mu, M + d, x) > best
    assert eb.gaussian_sm_objective(mu + 0.05, M, x) > best


def test_nce_recovers_the_parameters_and_the_normalizer():
    rng = np.random.default_rng(18)
    x = rng.normal(1.5, 0.7, 40000)
    y = rng.normal(0.0, 2.0, 40000)
    theta = eb.nce_fit(x, y, 0.0, 2.0)
    want = eb.gaussian_natural(1.5, 0.7)
    np.testing.assert_allclose(theta, want, rtol=0.03, atol=0.03)
    assert np.abs(eb.nce_grad(theta, x, y, 0.0, 2.0)).max() < 1e-10
    # the fitted log-normalizer is close to the true one, which maximum likelihood cannot estimate
    a, m = theta[0], theta[1]
    assert theta[2] == pytest.approx(-0.5 * m * m / a + 0.5 * np.log(a) - 0.5 * np.log(2 * np.pi), abs=0.03)


def test_nce_gradient_matches_finite_differences():
    rng = np.random.default_rng(19)
    x, y = rng.normal(1.0, 0.8, 50), rng.normal(0.0, 2.0, 50)
    theta = np.array([1.3, 0.4, -1.2])
    g = eb.nce_grad(theta, x, y, 0.0, 2.0)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1e-6
        fd = (eb.nce_objective(theta + e, x, y, 0.0, 2.0) - eb.nce_objective(theta - e, x, y, 0.0, 2.0)) / 2e-6
        assert g[i] == pytest.approx(fd, abs=1e-8)


def test_classifier_energy_moves_with_a_logit_shift_and_the_class_probabilities_do_not():
    logits = np.random.default_rng(20).standard_normal((6, 10))
    shift = np.linspace(-2, 3, 6)[:, None]
    np.testing.assert_allclose(eb.jem_class_probs(logits + shift), eb.jem_class_probs(logits), rtol=1e-12)
    np.testing.assert_allclose(eb.jem_energy(logits + shift), eb.jem_energy(logits) - shift[:, 0], rtol=1e-12)
    np.testing.assert_allclose(eb.jem_energy(logits), -np.log(np.exp(logits).sum(1)), rtol=1e-12)


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: one full Gibbs step v -> h -> v."""
    k = c["config"]
    cfg = eb.config(k["d_visible"], k["d_hidden"])
    rng = np.random.default_rng(0)
    v = (rng.uniform(size=(k["B"], k["d_visible"])) < 0.5).astype(float)
    u_h, u_v = rng.uniform(size=(k["B"], k["d_hidden"])), rng.uniform(size=v.shape)
    return eb.gibbs_step, (eb.init(rng, cfg), v, u_h, u_v)
