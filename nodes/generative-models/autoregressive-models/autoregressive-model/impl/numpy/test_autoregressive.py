# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Tests of the NumPy reference: the shared fixture, the counts of shapes.json, the autoregressive property of the
MADE masks as a triangular Jacobian, normalization over every binary vector, the hand-written gradients against finite
differences, exact ancestral sampling, the mask ensemble, NADE's O(DH) recursion against the naive one, the receptive
field of dilated causal convolutions, the mu-law codes, the blind spot of masked convolutions and its absence in the
two-stack design, the worked example of the page, and a MADE that learns a binary mixture."""

import json
import math
import pathlib

import numpy as np
import pytest

import autoregressive as ar

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
CONFIGS = [((6,), True), ((5, 7), False), ((), True), ((8, 8), True)]


def random_params(cfg, rng, scale=0.7):
    return {k: rng.normal(0.0, scale, s) for k, s in ar.param_shapes(cfg).items()}


def test_shared_fixture():
    p = dict(np.load(FIXTURES / "params-small.npz"))
    inp = dict(np.load(FIXTURES / "input-small.npz"))
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = ar.config()
    mk = ar.masks(cfg, ar.degrees(cfg, inp["order"]))
    made = {k[5:]: v for k, v in p.items() if k.startswith("made_")}
    a, _ = ar.logits(made, cfg, mk, inp["x"])
    np.testing.assert_allclose(a, want["logits"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ar.log_prob(made, cfg, mk, inp["x"]), want["log_prob"], rtol=1e-12)
    loss, g = ar.loss_and_grads(made, cfg, mk, inp["x"])
    assert loss == pytest.approx(float(want["loss"]), rel=1e-12)
    assert set(g) == set(ar.param_shapes(cfg))
    for k, v in g.items():
        np.testing.assert_allclose(v, want[f"d{k}"], rtol=1e-10, atol=1e-13, err_msg=k)
    nade = {k[5:]: v for k, v in p.items() if k.startswith("nade_")}
    np.testing.assert_allclose(ar.nade_logits(nade, inp["x"]), want["nade_logits"], rtol=1e-12, atol=1e-14)
    wave = {k[5:]: v for k, v in p.items() if k.startswith("wave_")}
    dil = tuple(int(d) for d in inp["dilations"])
    np.testing.assert_allclose(ar.wavenet_logits(wave, inp["seq"], dil), want["wave_logits"], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(ar.mu_law_encode(inp["audio"]), want["mu_codes"])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = ar.config(k["d_data"], (k["d_hidden"],))
    assert ar.param_count(cfg) == mod["params"]
    assert ar.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    p = ar.init(np.random.default_rng(0), cfg)
    assert sum(v.size for v in p.values()) == mod["params"]
    mk = ar.masks(cfg, ar.degrees(cfg))
    assert ar.logits(p, cfg, mk, np.zeros((k["B"], k["d_data"])))[0].shape == (k["B"], k["d_data"])


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ar.config(d_data=1)
    with pytest.raises(ValueError):
        ar.config(hidden=(4, 0))
    with pytest.raises(ValueError):
        ar.config(hidden=(), direct=False)


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_masks_make_the_jacobian_triangular_in_the_ordering(hidden, direct):
    rng = np.random.default_rng(1)
    cfg = ar.config(5, hidden, direct)
    p = random_params(cfg, rng)
    m0 = rng.permutation(5) + 1
    mk = ar.masks(cfg, ar.degrees(cfg, m0, rng))
    x = rng.standard_normal((1, 5))
    jac = np.zeros((5, 5))
    for i in range(5):
        e = np.zeros((1, 5))
        e[0, i] = 1e-6
        jac[i] = (ar.logits(p, cfg, mk, x + e)[0] - ar.logits(p, cfg, mk, x - e)[0])[0] / 2e-6
    allowed = m0[:, None] < m0[None, :]  # input i may reach output d only if i comes before d
    assert np.all(np.abs(jac[~allowed]) < 1e-9)
    assert np.abs(jac[allowed]).max() > 1e-3


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_probabilities_sum_to_one(hidden, direct):
    rng = np.random.default_rng(2)
    cfg = ar.config(6, hidden, direct)
    p = random_params(cfg, rng, 1.5)
    mk = ar.masks(cfg, ar.degrees(cfg, rng.permutation(6) + 1, rng))
    assert np.exp(ar.log_prob(p, cfg, mk, ar.all_binary(6))).sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("hidden, direct", CONFIGS)
def test_gradients_match_finite_differences(hidden, direct):
    rng = np.random.default_rng(3)
    cfg = ar.config(4, hidden, direct)
    p = random_params(cfg, rng)
    mk = ar.masks(cfg, ar.degrees(cfg, [2, 4, 1, 3]))
    x = ar.all_binary(4)[rng.integers(0, 16, 6)]
    _, g = ar.loss_and_grads(p, cfg, mk, x)
    for k, arr in p.items():
        for idx in np.ndindex(arr.shape):
            up, dn = {a: b.copy() for a, b in p.items()}, {a: b.copy() for a, b in p.items()}
            up[k][idx] += 1e-6
            dn[k][idx] -= 1e-6
            fd = (ar.loss_and_grads(up, cfg, mk, x)[0] - ar.loss_and_grads(dn, cfg, mk, x)[0]) / 2e-6
            assert g[k][idx] == pytest.approx(fd, abs=1e-7), (k, idx)


def test_ancestral_sampling_draws_from_the_model():
    rng = np.random.default_rng(4)
    cfg = ar.config(3, (6,))
    p = random_params(cfg, rng, 1.2)
    m0 = np.array([2, 3, 1])
    mk = ar.masks(cfg, ar.degrees(cfg, m0))
    X = ar.all_binary(3)
    prob = np.exp(ar.log_prob(p, cfg, mk, X))
    s = ar.sample(p, cfg, mk, m0, rng, 40000)
    freq = np.array([np.mean(np.all(s == row, 1)) for row in X])
    assert np.abs(freq - prob).max() < 4 * np.sqrt(prob * (1 - prob) / 40000).max() + 1e-3


def test_mask_ensemble_is_normalized_and_beats_the_mean_log_prob():
    rng = np.random.default_rng(5)
    cfg = ar.config(5, (8, 8))
    p = random_params(cfg, rng)
    mks = ar.mask_list(cfg, rng, 6)
    X = ar.all_binary(5)
    ens = ar.ensemble_log_prob(p, cfg, mks, X)
    assert np.exp(ens).sum() == pytest.approx(1.0, abs=1e-12)
    mean = np.mean([ar.log_prob(p, cfg, mk, X) for mk in mks], 0)
    assert np.all(ens >= mean - 1e-12)


def test_nade_recursion_matches_the_naive_passes_and_normalizes():
    rng = np.random.default_rng(6)
    q = {k: rng.standard_normal(s) for k, s in ar.nade_shapes(6, 4).items()}
    X = ar.all_binary(6)
    a = ar.nade_logits(q, X)
    np.testing.assert_allclose(a, ar.nade_logits_naive(q, X), rtol=1e-12, atol=1e-13)
    assert np.exp(np.sum(X * a - ar.softplus(a), 1)).sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kernel, dilations", [(2, (1, 2, 4)), (3, (1, 2)), (2, (1, 1, 1)), (2, (1, 2, 4, 1, 2))])
def test_receptive_field_of_the_dilated_causal_stack(kernel, dilations):
    rng = np.random.default_rng(7)
    K, T = 4, 24
    w = {k: rng.normal(0.0, 0.6, s) for k, s in ar.wavenet_shapes(K, 16, kernel, dilations).items()}
    x = rng.integers(0, K, (1, T))
    base = ar.wavenet_logits(w, x, dilations)
    R = ar.receptive_field(kernel, dilations)
    for t in range(T):
        y = x.copy()
        y[0, t] = (y[0, t] + 1) % K
        changed = np.nonzero(np.abs(ar.wavenet_logits(w, y, dilations) - base).max(axis=(0, 2)) > 1e-12)[0]
        assert set(changed) == set(range(t + 1, min(t + R, T - 1) + 1))


def test_wavenet_probabilities_sum_to_one():
    rng = np.random.default_rng(8)
    K, T, dil = 3, 4, (1, 2)
    w = {k: rng.normal(0.0, 0.8, s) for k, s in ar.wavenet_shapes(K, 2, 2, dil).items()}
    X = np.array(np.meshgrid(*[range(K)] * T, indexing="ij")).reshape(T, -1).T
    assert np.exp(ar.wavenet_log_prob(w, X, dil)).sum() == pytest.approx(1.0, abs=1e-12)
    s = ar.wavenet_sample(w, dil, rng, 5, T, K)
    assert s.shape == (5, T) and s.min() >= 0 and s.max() < K


def test_doubling_dilations_cover_1024_samples():
    assert ar.receptive_field(2, tuple(2**i for i in range(10))) == 1024
    assert ar.receptive_field(2, tuple(2**i for i in range(10)) * 3) == 3070


def test_mu_law_codes():
    x = np.linspace(-1.0, 1.0, 20001)
    q = ar.mu_law_encode(x)
    assert q.min() == 0 and q.max() == 255 and np.all(np.diff(q) >= 0)
    err = np.abs(ar.mu_law_decode(q) - x)
    small = np.abs(x) < 0.01
    assert err[small].max() < 3e-4 and err.max() < 0.022
    # nearly a quarter of the codes, 0.23 x 256, go to the one percent of amplitudes nearest zero
    assert len(np.unique(q[small])) >= 58


def test_masked_convolutions_have_a_blind_spot():
    rng = np.random.default_rng(9)
    n_layers, size, c = 3, 11, 5
    ws = [rng.standard_normal((3, 3, 2 if i else 1, 2)) for i in range(n_layers)]

    def net(img):
        h = img
        for i, w in enumerate(ws):
            h = np.tanh(ar.masked_conv2d(h, w, ar.conv_mask(3, "A" if i == 0 else "B")))
        return h

    img = rng.standard_normal((1, size, size, 1))
    base = net(img)[0, c, c]
    seen = set()
    for i in range(size):
        for j in range(size):
            other = img.copy()
            other[0, i, j, 0] += 1.0
            if np.abs(net(other)[0, c, c] - base).max() > 1e-12:
                seen.add((i - c, j - c))
    field = ar.masked_stack_field(n_layers)
    assert seen == field
    assert (-1, 2) not in field and (-1, 1) in field and (0, -3) in field


@pytest.mark.parametrize("L", [1, 2, 3, 5, 8])
def test_blind_spot_size_and_the_two_stack_fix(L):
    box = {(di, dj) for di in range(-L, 0) for dj in range(-L, L + 1)} | {(0, dj) for dj in range(-L, 0)}
    field = ar.masked_stack_field(L)
    assert field <= box and len(box) == 2 * L * (L + 1)
    assert box - field == {(-u, dj) for u in range(1, L + 1) for dj in range(u + 1, L + 1)}
    assert len(box - field) == L * (L - 1) // 2
    gated = ar.gated_stack_field(L)
    assert box <= gated and all(di < 0 or (di == 0 and dj < 0) for di, dj in gated)


def test_worked_example():
    cfg = ar.config(3, (), True)
    p = {"c": np.array([0.0, -1.0, 1.0]), "A": np.array([[0.0, 2.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])}
    mk = ar.masks(cfg, ar.degrees(cfg))
    lp = ar.log_prob(p, cfg, mk, np.array([[1.0, 0.0, 1.0]]))[0]
    s1 = 1 / (1 + math.exp(-1.0))
    assert math.exp(lp) == pytest.approx(0.5 * (1 - s1) * 0.5, rel=1e-12)
    assert math.exp(lp) == pytest.approx(0.0672, abs=5e-5)
    assert -lp / (3 * math.log(2)) == pytest.approx(1.29821, abs=1e-5)
    assert np.exp(ar.log_prob(p, cfg, mk, ar.all_binary(3))).sum() == pytest.approx(1.0, abs=1e-12)


def mixture(D=8, eps=0.05):
    protos = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]], float)[:, :D]
    X = ar.all_binary(D)
    flips = np.abs(X[:, None, :] - protos[None]).sum(-1)
    return protos, np.mean(eps**flips * (1 - eps) ** (D - flips), 1)


def test_learns_a_binary_mixture():
    rng = np.random.default_rng(10)
    _, prob = mixture()
    X = ar.all_binary(8)
    data = X[rng.choice(len(X), 4000, p=prob / prob.sum())]
    cfg = ar.config(8, (48,))
    p = ar.init(rng, cfg)
    mk = ar.masks(cfg, ar.degrees(cfg))
    trace = ar.train(p, cfg, data, 800, rng, [mk])
    kl = float(np.sum(prob * (np.log(prob) - ar.log_prob(p, cfg, mk, X))))
    marg = data.mean(0)
    indep = np.sum(X * np.log(marg) + (1 - X) * np.log(1 - marg), 1)
    kl_indep = float(np.sum(prob * (np.log(prob) - indep)))
    assert np.mean(trace[-50:]) < 0.8 * trace[0]
    assert kl < 0.1 < 1.0 < kl_indep


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: the logits of one masked pass on B vectors."""
    k = c["config"]
    cfg = ar.config(k["d_data"], (k["d_hidden"],))
    rng = np.random.default_rng(0)
    x = (rng.uniform(size=(k["B"], k["d_data"])) < 0.5).astype(float)
    return ar.logits, (ar.init(rng, cfg), cfg, ar.masks(cfg, ar.degrees(cfg)), x)
