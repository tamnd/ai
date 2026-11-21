# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Tests of the JAX version: the shared fixture and jax.grad of the loss, the counts of shapes.json, the
autoregressive property as a triangular jax.jacfwd, normalization over every binary vector, the fori_loop sampler, the
mask ensemble, NADE's cumsum against separate passes, the receptive field of the dilated stack, the WaveNet sampler,
the mu-law codes, and a MADE that learns a binary mixture with the jitted Adam step."""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from autoregressive import (
    adam_init,
    all_binary,
    config,
    degrees,
    ensemble_log_prob,
    flops_forward,
    init,
    log_prob,
    logits,
    make_step,
    masks,
    mu_law_encode,
    nade_logits,
    nll,
    param_count,
    param_shapes,
    receptive_field,
    sample,
    wavenet_log_prob,
    wavenet_logits,
    wavenet_sample,
    wavenet_shapes,
)

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
CONFIGS = [((6,), True), ((5, 7), False), ((), True), ((8, 8), True)]


def random_params(key, cfg, scale=0.7):
    p = {}
    for k, s in param_shapes(cfg).items():
        key, sub = jax.random.split(key)
        p[k] = scale * jax.random.normal(sub, s)
    return p


def test_shared_fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    mk = masks(cfg, degrees(cfg, inp["order"]))
    made = {k[5:]: jnp.asarray(v) for k, v in params.items() if k.startswith("made_")}
    x = jnp.asarray(inp["x"])
    np.testing.assert_allclose(logits(made, mk, x), want["logits"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(log_prob(made, mk, x), want["log_prob"], rtol=1e-12)
    loss, g = jax.value_and_grad(nll)(made, mk, x)
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-12)
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    nade = {k[5:]: jnp.asarray(v) for k, v in params.items() if k.startswith("nade_")}
    np.testing.assert_allclose(nade_logits(nade, x), want["nade_logits"], rtol=1e-12, atol=1e-14)
    wave = {k[5:]: jnp.asarray(v) for k, v in params.items() if k.startswith("wave_")}
    dil = tuple(int(d) for d in inp["dilations"])
    got = wavenet_logits(wave, jnp.asarray(inp["seq"]), dil)
    np.testing.assert_allclose(got, want["wave_logits"], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(mu_law_encode(jnp.asarray(inp["audio"])), want["mu_codes"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = config(k["d_data"], (k["d_hidden"],))
    assert param_count(cfg) == mod["params"]
    assert flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = init(jax.random.PRNGKey(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    mk = masks(cfg, degrees(cfg))
    assert logits(p, mk, jnp.zeros((k["B"], k["d_data"]))).shape == (k["B"], k["d_data"])


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_jacfwd_is_triangular_in_the_ordering(hidden, direct):
    cfg = config(5, hidden, direct)
    p = random_params(jax.random.PRNGKey(1), cfg)
    m0 = np.array([3, 5, 1, 4, 2])
    mk = masks(cfg, degrees(cfg, m0, np.random.default_rng(1)))
    x = jax.random.normal(jax.random.PRNGKey(2), (5,))
    jac = jax.jacfwd(lambda v: logits(p, mk, v[None])[0])(x).T  # jac[i, d] = d a_d / d x_i
    allowed = m0[:, None] < m0[None, :]
    assert float(jnp.abs(jac[~allowed]).max()) == 0.0
    assert float(jnp.abs(jac[allowed]).max()) > 1e-3


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_probabilities_sum_to_one(hidden, direct):
    cfg = config(6, hidden, direct)
    p = random_params(jax.random.PRNGKey(3), cfg, 1.5)
    mk = masks(cfg, degrees(cfg, np.array([4, 6, 2, 1, 5, 3]), np.random.default_rng(3)))
    assert float(jnp.exp(log_prob(p, mk, all_binary(6))).sum()) == pytest.approx(1.0, abs=1e-12)


def test_fori_loop_sampler_draws_from_the_model():
    cfg = config(3, (6,))
    p = random_params(jax.random.PRNGKey(4), cfg, 1.2)
    m0 = np.array([2, 3, 1])
    mk = masks(cfg, degrees(cfg, m0))
    X = all_binary(3)
    prob = jnp.exp(log_prob(p, mk, X))
    n = 40000
    s = sample(p, mk, m0, jax.random.PRNGKey(5), n)
    freq = jnp.stack([jnp.mean(jnp.all(s == row, 1)) for row in X])
    assert float(jnp.abs(freq - prob).max()) < 4 * float(jnp.sqrt(prob * (1 - prob) / n).max()) + 1e-3


def test_mask_ensemble_is_normalized_and_beats_the_mean_log_prob():
    cfg = config(5, (8, 8))
    p = random_params(jax.random.PRNGKey(6), cfg)
    rng = np.random.default_rng(6)
    mks = [masks(cfg, degrees(cfg, rng.permutation(5) + 1, rng)) for _ in range(6)]
    X = all_binary(5)
    ens = ensemble_log_prob(p, mks, X)
    mean = jnp.mean(jnp.stack([log_prob(p, mk, X) for mk in mks]), 0)
    assert float(jnp.exp(ens).sum()) == pytest.approx(1.0, abs=1e-12)
    assert bool(jnp.all(ens >= mean - 1e-12))


def test_nade_cumsum_matches_separate_passes():
    D, H = 6, 4
    k = jax.random.split(jax.random.PRNGKey(7), 4)
    q = {"W": jax.random.normal(k[0], (D, H)), "c": jax.random.normal(k[1], (H,))}
    q.update({"V": jax.random.normal(k[2], (H, D)), "b": jax.random.normal(k[3], (D,))})
    X = all_binary(D)
    a = nade_logits(q, X)
    cols = [q["b"][i] + jax.nn.sigmoid(q["c"] + X[:, :i] @ q["W"][:i]) @ q["V"][:, i] for i in range(D)]
    np.testing.assert_allclose(a, jnp.stack(cols, 1), rtol=1e-12, atol=1e-13)
    assert float(jnp.exp(jnp.sum(X * a - jax.nn.softplus(a), 1)).sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kernel, dilations", [(2, (1, 2, 4)), (3, (1, 2))])
def test_receptive_field_of_the_dilated_stack(kernel, dilations):
    K, T = 4, 20
    keys = jax.random.split(jax.random.PRNGKey(8), 20)
    w = {
        n: 0.6 * jax.random.normal(keys[i], s)
        for i, (n, s) in enumerate(wavenet_shapes(K, 16, kernel, dilations).items())
    }
    x = jax.random.randint(keys[-1], (1, T), 0, K)
    base = wavenet_logits(w, x, dilations)
    R = receptive_field(kernel, dilations)
    for t in range(T):
        y = x.at[0, t].set((x[0, t] + 1) % K)
        changed = np.nonzero(np.abs(np.asarray(wavenet_logits(w, y, dilations) - base)).max(axis=(0, 2)) > 1e-12)[0]
        assert set(changed.tolist()) == set(range(t + 1, min(t + R, T - 1) + 1))


def test_wavenet_probabilities_sum_to_one_and_sampler():
    K, T, dil = 3, 4, (1, 2)
    keys = jax.random.split(jax.random.PRNGKey(9), 12)
    w = {n: 0.8 * jax.random.normal(keys[i], s) for i, (n, s) in enumerate(wavenet_shapes(K, 2, 2, dil).items())}
    X = jnp.array(np.array(np.meshgrid(*[range(K)] * T, indexing="ij")).reshape(T, -1).T)
    assert float(jnp.exp(wavenet_log_prob(w, X, dil)).sum()) == pytest.approx(1.0, abs=1e-12)
    s = wavenet_sample(w, dil, keys[-1], 5, T)
    assert s.shape == (5, T) and int(s.min()) >= 0 and int(s.max()) < K


def test_mu_law_codes():
    q = mu_law_encode(jnp.linspace(-1.0, 1.0, 20001))
    assert int(q.min()) == 0 and int(q.max()) == 255 and bool(jnp.all(jnp.diff(q) >= 0))
    assert mu_law_encode(jnp.array([0.0, 1e-4, -1e-4])).tolist() == [128, 128, 127]


def mixture(D=8, eps=0.05):
    protos = jnp.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]], float)[:, :D]
    X = all_binary(D)
    flips = jnp.abs(X[:, None, :] - protos[None]).sum(-1)
    return jnp.mean(eps**flips * (1 - eps) ** (D - flips), 1)


def test_learns_a_binary_mixture():
    rng = np.random.default_rng(10)
    prob = np.asarray(mixture())
    X = all_binary(8)
    data = X[rng.choice(len(X), 4000, p=prob / prob.sum())]
    cfg = config(8, (48,))
    p = init(jax.random.PRNGKey(10), cfg)
    mk = masks(cfg, degrees(cfg))
    step, st, trace = make_step(), adam_init(p), []
    for _ in range(800):
        p, st, loss = step(p, st, mk, data[rng.integers(0, len(data), 100)])
        trace.append(float(loss))
    kl = float(jnp.sum(prob * (jnp.log(prob) - log_prob(p, mk, X))))
    marg = data.mean(0)
    indep = jnp.sum(X * jnp.log(marg) + (1 - X) * jnp.log(1 - marg), 1)
    kl_indep = float(jnp.sum(prob * (jnp.log(prob) - indep)))
    assert np.mean(trace[-50:]) < 0.8 * trace[0]
    assert kl < 0.1 < 1.0 < kl_indep
