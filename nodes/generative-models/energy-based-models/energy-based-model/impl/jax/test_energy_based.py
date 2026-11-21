# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Tests of the JAX version: the shared fixture with every gradient from jax.grad, the counts of shapes.json, the
worked example, jax.grad of the log-likelihood against the data-minus-model statistics of eq-ml-grad, the Gibbs
fori_loop drawing from the model, CD-1 and PCD learning a binary mixture, the vmapped score and the Hessian-trace
Laplacian against closed forms, the gradient of the score-matching objective against finite differences, the
stationary variance of Langevin dynamics, an energy trained on Langevin samples, score matching on a Gaussian,
noise-contrastive estimation and the energy of a classifier."""

import json
import math
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import energy_based as eb  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def random_rbm(D, H, seed, scale=1.0):
    ks = jax.random.split(jax.random.key(seed), 3)
    return {
        k: scale * jax.random.normal(kk, s)
        for kk, (k, s) in zip(ks, eb.param_shapes(eb.config(D, H)).items(), strict=True)
    }


def test_shared_fixture():
    P = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    p = {k[4:]: v for k, v in P.items() if k.startswith("rbm_")}
    q = {k[4:]: v for k, v in P.items() if k.startswith("mlp_")}
    v, gamma, step = inp["v"], float(inp["gamma"]), float(inp["step"])
    np.testing.assert_allclose(eb.free_energy(p, v), want["free_energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.energy(p, v, inp["h"]), want["energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.p_h(p, v), want["p_h"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.log_partition(p), want["log_z"], rtol=1e-12, atol=1e-13)
    h1, v1 = eb.gibbs_step(p, v, inp["u_h"], inp["u_v"])
    np.testing.assert_array_equal(h1, want["h1"])
    np.testing.assert_array_equal(v1, want["v1"])
    for k, g in jax.grad(eb.cd_objective)(p, v, v1).items():
        np.testing.assert_allclose(g, want[f"cd_{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    for k, g in jax.grad(eb.mean_log_lik)(p, v).items():
        np.testing.assert_allclose(g, want[f"ml_{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    x = inp["x"]
    np.testing.assert_allclose(eb.mlp_energy(q, x, gamma), want["mlp_energy"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(eb.grad_x(q, x, gamma), want["mlp_grad_x"], rtol=1e-12, atol=1e-13)
    for k, g in jax.grad(lambda a: jnp.mean(eb.mlp_energy(a, x, gamma)))(q).items():
        np.testing.assert_allclose(g, want[f"mlp_d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    np.testing.assert_allclose(eb.sm_objective(q, x, gamma), want["mlp_sm"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        eb.langevin_step(q, x, step, inp["z"], None, gamma), want["langevin"], rtol=1e-12, atol=1e-13
    )
    args = (inp["theta"], inp["nce_x"], inp["nce_y"], 0.0, 2.0)
    np.testing.assert_allclose(eb.nce_objective(*args), want["nce_j"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(jax.grad(eb.nce_objective)(*args), want["nce_grad"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(eb.jem_energy(inp["logits"]), want["jem_energy"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = eb.config(k["d_visible"], k["d_hidden"])
    assert eb.param_count(cfg) == mod["params"]
    assert eb.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = eb.init(jax.random.key(0), cfg)
    assert sum(a.size for a in p.values()) == mod["params"]
    v = jnp.zeros((k["B"], k["d_visible"]))
    h, v1 = eb.gibbs_step(p, v, jnp.ones((k["B"], k["d_hidden"])), jnp.ones_like(v))
    assert h.shape == (k["B"], k["d_hidden"]) and v1.shape == v.shape


def test_worked_example():
    p = {"W": jnp.array([[2.0], [-1.0]]), "b": jnp.zeros(2), "c": jnp.array([-1.0])}
    X = eb.all_binary(2)
    np.testing.assert_allclose(jnp.exp(-eb.free_energy(p, X)), [1 + math.e**-1, 1 + math.e**-2, 1 + math.e, 2.0])
    assert float(jnp.exp(eb.log_partition(p))) == pytest.approx(8.2215, abs=1e-4)
    probs = jnp.exp(eb.log_prob(p, X))
    assert float(probs[2]) == pytest.approx(0.4523, abs=1e-4) and float(probs[3]) == pytest.approx(0.2433, abs=1e-4)


def test_grad_of_the_log_likelihood_is_data_minus_model_statistics():
    p = random_rbm(4, 3, 3)
    X = eb.all_binary(4)
    data = X[jnp.array([1, 6, 7, 12, 15])]
    g = jax.grad(eb.mean_log_lik)(p, data)
    w = jnp.exp(eb.log_prob(p, X))
    ph_d, ph_m = eb.p_h(p, data), eb.p_h(p, X)
    want = {"W": data.T @ ph_d / len(data) - (X * w[:, None]).T @ ph_m, "b": data.mean(0) - w @ X,
            "c": ph_d.mean(0) - w @ ph_m}  # fmt: skip
    for k in want:
        np.testing.assert_allclose(g[k], want[k], rtol=1e-11, atol=1e-13, err_msg=k)


def test_gibbs_chain_draws_from_the_model():
    p = random_rbm(3, 2, 5)
    key = jax.random.key(6)
    v0 = jax.random.bernoulli(key, 0.5, (20000, 3)).astype(jnp.float64)
    v = jax.jit(eb.gibbs_chain, static_argnums=2)(p, v0, 30, jax.random.fold_in(key, 1))
    X = eb.all_binary(3)
    freq = jnp.stack([jnp.mean(jnp.all(v == x, 1)) for x in X])
    pr = jnp.exp(eb.log_prob(p, X))
    assert float(jnp.abs(freq - pr).max()) < 4 * float(jnp.sqrt(pr * (1 - pr) / len(v)).max()) + 1e-3


def mixture(D=8, eps=0.05):
    protos = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]])[:, :D]
    flips = jnp.abs(eb.all_binary(D)[:, None, :] - protos[None]).sum(-1)
    return jnp.mean(eps**flips * (1 - eps) ** (D - flips), 1)


@pytest.mark.parametrize("persistent", [False, True], ids=["cd1", "pcd"])
def test_rbm_learns_a_binary_mixture(persistent):
    key = jax.random.key(7)
    prob = mixture()
    X = eb.all_binary(8)
    data = X[jax.random.choice(key, len(X), (4000,), p=prob)]
    p = eb.init(jax.random.fold_in(key, 1), eb.config(8, 8), 0.1)
    chains = data[:400]  # PCD's estimate of the model term is an average over the chains, so more chains cut noise
    step = eb.make_rbm_step(k=1, persistent=persistent, lr=0.1)
    start = float(eb.mean_log_lik(p, data))
    for i in range(800):
        p, chains = step(p, chains, data, jax.random.fold_in(key, 100 + i))
    kl = float(jnp.sum(prob * (jnp.log(prob) - eb.log_prob(p, X))))
    marg = data.mean(0)
    kl_indep = float(jnp.sum(prob * (jnp.log(prob) - (X * jnp.log(marg) + (1 - X) * jnp.log(1 - marg)).sum(1))))
    assert float(eb.mean_log_lik(p, data)) > start + 2.0
    assert kl < 0.1 < 1.0 < kl_indep


def test_score_and_laplacian_match_the_closed_forms_and_the_sm_gradient_finite_differences():
    ks = jax.random.split(jax.random.key(10), 4)
    q = {
        "W1": jax.random.normal(ks[0], (3, 5)),
        "b1": jax.random.normal(ks[1], (5,)),
        "w2": jax.random.normal(ks[2], (5,)),
    }
    x = jax.random.normal(ks[3], (4, 3))
    t = jnp.tanh(x @ q["W1"] + q["b1"])
    np.testing.assert_allclose(eb.grad_x(q, x, 0.3), ((1 - t * t) * q["w2"]) @ q["W1"].T + 0.3 * x, rtol=1e-12)
    lap = (-2 * t * (1 - t * t) * q["w2"]) @ jnp.sum(q["W1"] ** 2, 0) + 0.3 * 3
    np.testing.assert_allclose(eb.laplacian(q, x, 0.3), lap, rtol=1e-12, atol=1e-13)
    g = jax.grad(eb.sm_objective)(q, x, 0.3)
    sm = jax.jit(eb.sm_objective)
    eps = 1e-6
    for k in q:
        for idx in np.ndindex(q[k].shape):
            e = jnp.zeros(q[k].shape).at[idx].set(eps)
            hi = sm({**q, k: q[k] + e}, x, 0.3)
            lo = sm({**q, k: q[k] - e}, x, 0.3)
            assert float(g[k][idx]) == pytest.approx(float(hi - lo) / (2 * eps), abs=1e-7), (k, idx)


@pytest.mark.parametrize("step", [0.4, 0.1, 0.02])
def test_langevin_stationary_variance_on_a_quadratic_energy(step):
    s2 = 0.5
    q = {"W1": jnp.zeros((2, 1)), "b1": jnp.zeros(1), "w2": jnp.zeros(1)}
    key = jax.random.key(11)
    x = eb.langevin(q, jax.random.normal(key, (20000, 2)), 300, step, jax.random.fold_in(key, 1), None, 1 / s2)
    a = 1 - step / (2 * s2)
    want = step / (1 - a * a)
    assert float(x.var()) == pytest.approx(want, rel=0.03)
    assert abs(want - s2) <= 0.6 * step


def test_energy_trained_on_langevin_samples_finds_four_modes():
    key = jax.random.key(12)
    C = jnp.array([[1.5, 1.5], [-1.5, 1.5], [1.5, -1.5], [-1.5, -1.5]])
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    data = C[jax.random.randint(k1, (4000,), 0, 4)] + 0.25 * jax.random.normal(k2, (4000, 2))
    q = eb.mlp_init(k3, 2, 32)
    st = eb.adam_init(q)
    buf = jax.random.uniform(k4, (4096, 2), minval=-3.0, maxval=3.0)
    update = eb.make_ebm_step()
    for i in range(800):
        q, st, buf, _ = update(q, st, buf, data, jax.random.fold_in(k5, i))
    u = jax.random.uniform(jax.random.fold_in(key, 1), (2000, 2), minval=-3.0, maxval=3.0)
    assert float(eb.mlp_energy(q, data).mean()) < float(eb.mlp_energy(q, u).mean()) - 0.5
    s = eb.langevin(q, u, 200, 0.02, jax.random.fold_in(key, 2), 0.05)
    dist = jnp.linalg.norm(s[:, None] - C[None], axis=-1)
    assert float(jnp.mean(dist.min(1) < 0.75)) > 0.8
    assert float(jnp.bincount(dist.argmin(1), length=4).min()) / len(s) > 0.1


def test_score_matching_on_a_gaussian_drops_the_data_score_and_is_minimized_by_the_ml_estimate():
    key = jax.random.key(17)
    mu0 = jnp.array([0.5, -1.0])
    L = jnp.array([[1.0, 0.0], [0.6, 0.8]])
    z = jax.random.normal(key, (5000, 2))
    z = z - z.mean(0)
    z = z @ jnp.linalg.inv(jnp.linalg.cholesky(z.T @ z / len(z))).T
    x = mu0 + z @ L.T
    M0 = jnp.linalg.inv(L @ L.T)
    for i in range(3):
        ka, kb = jax.random.split(jax.random.fold_in(key, i))
        A = jax.random.normal(ka, (2, 2))
        mu, M = jax.random.normal(kb, (2,)), A @ A.T + jnp.eye(2)
        J = float(eb.gaussian_sm_objective(mu, M, x))
        assert J == pytest.approx(float(-jnp.trace(M) + 0.5 * jnp.mean(jnp.sum(((x - mu) @ M) ** 2, 1))), rel=1e-12)
        explicit = float(0.5 * jnp.mean(jnp.sum(((x - mu) @ M - (x - mu0) @ M0) ** 2, 1)))
        assert explicit - J == pytest.approx(0.5 * float(jnp.trace(M0)), rel=1e-10)
    mu = x.mean(0)
    M = jnp.linalg.inv((x - mu).T @ (x - mu) / len(x))
    gm, gM = jax.grad(eb.gaussian_sm_objective, (0, 1))(mu, M, x)
    assert float(jnp.abs(gm).max()) < 1e-10 and float(jnp.abs(gM + gM.T).max()) < 1e-10


def test_nce_recovers_the_parameters_and_the_normalizer():
    kx, ky = jax.random.split(jax.random.key(18))
    x = 1.5 + 0.7 * jax.random.normal(kx, (40000,))
    y = 2.0 * jax.random.normal(ky, (40000,))
    theta = eb.nce_fit(x, y, 0.0, 2.0)
    np.testing.assert_allclose(theta, eb.gaussian_natural(1.5, 0.7), rtol=0.03, atol=0.03)
    assert float(jnp.abs(jax.grad(eb.nce_objective)(theta, x, y, 0.0, 2.0)).max()) < 1e-10
    a, m = float(theta[0]), float(theta[1])
    want_c = -0.5 * m * m / a + 0.5 * math.log(a) - 0.5 * math.log(2 * math.pi)
    assert float(theta[2]) == pytest.approx(want_c, abs=0.03)


def test_classifier_energy_moves_with_a_logit_shift_and_the_class_probabilities_do_not():
    logits = jax.random.normal(jax.random.key(20), (6, 10))
    shift = jnp.linspace(-2, 3, 6)[:, None]
    np.testing.assert_allclose(eb.jem_class_probs(logits + shift), eb.jem_class_probs(logits), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(eb.jem_energy(logits + shift), eb.jem_energy(logits) - shift[:, 0], rtol=1e-12)
