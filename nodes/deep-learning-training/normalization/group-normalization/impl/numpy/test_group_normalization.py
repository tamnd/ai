# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-normalization/ (ai://QFEJMP).
"""Tests of the NumPy reference: the worked example, a loop over the sets of the definitions, the special cases
G = 1 (layer normalization) and G = C (instance normalization), independence from the batch, the group moments, the
gradient against finite differences, and agreement with torch.nn.functional.group_norm when PyTorch is installed."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from group_normalization import KINDS, backward, config, forward, init, normalize, normalize_over

# The worked example of the page: one example, C = 4 channels of 2 positions, G = 2 groups, eps = 0.
WX = np.array([[[0.0, 0.0], [4.0, 4.0], [0.0, 6.0], [6.0, 8.0]]])
WY = [[[-1.0, -1.0], [1.0, 1.0], [-5 / 3, 1 / 3], [1 / 3, 1.0]]]
WDY = np.array([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
WDX = np.array([[[0.0, 0.0], [0.0, 0.0], [1.0, -2.0], [-2.0, 3.0]]]) / 54

# The shared fixture: N = 2, C = 4, three positions, G = 2, eps = 1e-5. The PyTorch tests hold the same numbers.
X = ((np.arange(24.0) * 7) % 11 - 5).reshape(2, 4, 3) / 2
DY = ((np.arange(24.0) * 5) % 13 - 6).reshape(2, 4, 3) / 3
GAMMA = np.array([1.5, 0.5, 2.0, -1.0])
BETA = np.array([0.25, -0.5, 0.0, 1.0])
Y = [
    [
        [-1.83957437545, 1.294787187725, -0.496276562661],
        [0.296028333505, -0.300992916624, -0.898014166752],
        [2.723934680039, 0.302659408893, -2.118615862252],
        [-0.059307931126, 1.151329704447, 2.361967340019],
    ],
    [
        [0.679447736958, -1.381901400442, 2.225459590008],
        [-0.528629849131, -1.215746228264, -0.013292564781],
        [0.572596982611, -2.175868533922, 2.633946120011],
        [1.057259698261, 2.431492456528, 0.026585129561],
    ],
]
DX = [
    [
        [-1.546738738476, -0.271417818487, 1.345267995596],
        [-0.464020876747, 0.157629520456, 0.779279917658],
        [-0.595563434834, 1.533063655466, -1.584405675048],
        [0.239252746634, -0.658714251997, 1.066366959779],
    ],
    [
        [0.824164796071, -1.299515880715, -0.675098402439],
        [0.751322212964, 0.459951880533, -0.060824606413],
        [2.098897024156, -1.013029383241, 0.310143554903],
        [0.633799043173, 0.041299359266, -2.071109598257],
    ],
]
DGAMMA = [3.818310466865, -1.425345532937, 3.623724250235, 2.827542113775]
DBETA = [-8 / 3, 4 / 3, 1.0, 2 / 3]


def reference(x, kind, groups, eps):
    """Normalize x by looping over the sets S_i of eq-sets and eq-gn-set, one entry at a time."""
    n, c = x.shape[0], x.shape[1]
    out = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        i_n, i_c = idx[0], idx[1]
        members = []
        for k in np.ndindex(*x.shape):
            k_n, k_c = k[0], k[1]
            if kind == "batch":
                inside = k_c == i_c
            elif kind == "layer":
                inside = k_n == i_n
            elif kind == "instance":
                inside = k_n == i_n and k_c == i_c
            else:
                inside = k_n == i_n and k_c // (c // groups) == i_c // (c // groups)
            if inside:
                members.append(x[k])
        mu, var = np.mean(members), np.var(members)
        out[idx] = (x[idx] - mu) / np.sqrt(var + eps)
    assert n == x.shape[0]
    return out


def numeric_grad(fun, a, h=1e-6):
    g = np.zeros_like(a)
    for idx in np.ndindex(*a.shape):
        e = np.zeros_like(a)
        e[idx] = h
        g[idx] = (fun(a + e) - fun(a - e)) / (2 * h)
    return g


def test_worked_example():
    cfg = config(4, 2, eps=0.0, affine=False)
    assert init(cfg) == {}
    y, cache = forward({}, WX, cfg)
    np.testing.assert_allclose(y, WY, atol=1e-12)
    np.testing.assert_allclose(cache["std"].ravel(), [2.0, 3.0], atol=1e-12)
    dx, grads = backward({}, cache, WDY)
    assert grads == {}
    np.testing.assert_allclose(dx, WDX, atol=1e-12)
    np.testing.assert_allclose(normalize(WX, "layer", eps=0.0) * np.sqrt(35) / 2, WX - 3.5, atol=1e-12)
    inst = normalize(WX, "instance", eps=1e-5)
    np.testing.assert_allclose(inst[0, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(inst[0, 2:], [[-1.0, 1.0], [-1.0, 1.0]], atol=1e-5)


def test_shared_fixture():
    cfg = config(4, 2, eps=1e-5)
    p = {"gamma": GAMMA.copy(), "beta": BETA.copy()}
    y, cache = forward(p, X, cfg)
    np.testing.assert_allclose(y, Y, atol=1e-11)
    dx, grads = backward(p, cache, DY)
    np.testing.assert_allclose(dx, DX, atol=1e-11)
    np.testing.assert_allclose(grads["gamma"], DGAMMA, atol=1e-11)
    np.testing.assert_allclose(grads["beta"], DBETA, atol=1e-11)


def test_init():
    p = init(config(6, 3))
    np.testing.assert_array_equal(p["gamma"], np.ones(6))
    np.testing.assert_array_equal(p["beta"], np.zeros(6))
    assert config(6, 3) == {"C": 6, "G": 3, "eps": 1e-5, "affine": True}
    assert config(32)["G"] == 32


@pytest.mark.parametrize(("c", "g", "eps"), [(6, 4, 1e-5), (0, 1, 1e-5), (4, 0, 1e-5), (4, 2, -1.0)])
def test_config_rejects(c, g, eps):
    with pytest.raises(ValueError):
        config(c, g, eps=eps)


def test_bad_inputs():
    with pytest.raises(ValueError):
        forward({}, np.zeros((2, 3, 4)), config(4, 2, affine=False))
    with pytest.raises(ValueError):
        forward({}, np.zeros(4), config(4, 2, affine=False))
    with pytest.raises(ValueError):
        normalize_over(np.zeros((2, 6)), 4)
    with pytest.raises(ValueError):
        normalize_over(np.zeros((2, 6)), 0)
    with pytest.raises(ValueError):
        normalize_over(np.zeros(6), 1)
    with pytest.raises(ValueError):
        normalize(X, "weight")
    with pytest.raises(ValueError):
        normalize(X, "group")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(2, 4, 3), (3, 4, 2, 2), (2, 4)])
def test_sets_match_loop(kind, shape):
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape) * 2 + 1
    np.testing.assert_allclose(normalize(x, kind, num_groups=2, eps=1e-3), reference(x, kind, 2, 1e-3), atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    st.integers(1, 3),
    st.sampled_from([(4, 1), (4, 2), (4, 4), (6, 3), (8, 2)]),
    st.integers(1, 3),
    st.integers(1, 3),
    st.integers(0, 2**31 - 1),
)
def test_special_cases(n, cg, h, w, seed):
    c, g = cg
    x = np.random.default_rng(seed).normal(size=(n, c, h, w)) * 3 - 1
    gn = forward({}, x, config(c, g, eps=1e-5, affine=False))[0]
    np.testing.assert_allclose(gn, normalize(x, "group", num_groups=g), atol=1e-12)
    flat = x.reshape(n, -1)
    ln = (flat - flat.mean(1, keepdims=True)) / np.sqrt(flat.var(1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(normalize(x, "layer").reshape(n, -1), ln, atol=1e-10)
    np.testing.assert_allclose(forward({}, x, config(c, 1, affine=False))[0], normalize(x, "layer"), atol=1e-12)
    m, v = x.mean((2, 3), keepdims=True), x.var((2, 3), keepdims=True)
    np.testing.assert_allclose(normalize(x, "instance"), (x - m) / np.sqrt(v + 1e-5), atol=1e-10)
    np.testing.assert_allclose(forward({}, x, config(c, c, affine=False))[0], normalize(x, "instance"), atol=1e-12)
    m, v = x.mean((0, 2, 3), keepdims=True), x.var((0, 2, 3), keepdims=True)
    np.testing.assert_allclose(normalize(x, "batch"), (x - m) / np.sqrt(v + 1e-5), atol=1e-10)
    np.testing.assert_allclose(normalize(x[:1], "batch"), normalize(x[:1], "instance"), atol=1e-12)


def test_batch_independence():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 4, 3, 3))
    other = rng.normal(size=(5, 4, 3, 3)) * 10 + 4
    cfg = config(4, 2, affine=False)
    for k in range(5):
        alone = forward({}, x[k : k + 1], cfg)[0]
        np.testing.assert_allclose(forward({}, x, cfg)[0][k : k + 1], alone, atol=1e-12)
        mixed = x.copy()
        mixed[np.arange(5) != k] = other[np.arange(5) != k]
        np.testing.assert_allclose(forward({}, mixed, cfg)[0][k : k + 1], alone, atol=1e-12)
    changed = x.copy()
    changed[1:] = other[1:]
    assert np.max(np.abs(normalize(changed, "batch")[0] - normalize(x, "batch")[0])) > 1e-3


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**31 - 1), st.sampled_from([0.0, 1e-5, 0.1, 2.0]))
def test_group_moments(seed, eps):
    x = np.random.default_rng(seed).normal(size=(2, 6, 2, 3)) * 2 + 0.5
    _, mean, var = normalize_over(x, 3)
    xhat = normalize_over(x, 3, eps=eps)[0].reshape(2, 3, 2, 2, 3)
    np.testing.assert_allclose(xhat.mean(axis=(2, 3, 4)), 0.0, atol=1e-12)
    np.testing.assert_allclose(xhat.var(axis=(2, 3, 4)), (var / (var + eps)).reshape(2, 3), rtol=1e-10)
    np.testing.assert_allclose(mean.reshape(2, 3), x.reshape(2, 3, -1).mean(-1), atol=1e-12)


@pytest.mark.parametrize("g", [1, 2, 4])
@pytest.mark.parametrize("eps", [0.0, 1e-3, 0.5])
@pytest.mark.parametrize("shape", [(2, 4, 3), (2, 4, 2, 2)])
def test_gradient_finite_differences(g, eps, shape):
    rng = np.random.default_rng(20261116)
    x, dy = rng.normal(size=shape), rng.normal(size=shape)
    cfg = config(4, g, eps=eps)
    p = {"gamma": rng.normal(size=4), "beta": rng.normal(size=4)}
    _, cache = forward(p, x, cfg)
    dx, grads = backward(p, cache, dy)
    np.testing.assert_allclose(
        dx, numeric_grad(lambda a: float(np.sum(dy * forward(p, a, cfg)[0])), x), rtol=1e-6, atol=1e-8
    )
    for name in ("gamma", "beta"):

        def loss(v, name=name):
            return float(np.sum(dy * forward({**p, name: v}, x, cfg)[0]))

        np.testing.assert_allclose(grads[name], numeric_grad(loss, p[name]), rtol=1e-6, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**31 - 1), st.sampled_from([0.0, 1e-3, 1.0]))
def test_gradient_group_sums(seed, eps):
    rng = np.random.default_rng(seed)
    x, dy = rng.normal(size=(2, 6, 4)), rng.normal(size=(2, 6, 4))
    cfg = config(6, 3, eps=eps, affine=False)
    _, cache = forward({}, x, cfg)
    dx = backward({}, cache, dy)[0].reshape(2, 3, 8)
    xhat = cache["xhat"].reshape(2, 3, 8)
    u = dy.reshape(2, 3, 8)
    np.testing.assert_allclose(dx.sum(-1), 0.0, atol=1e-10)
    std = cache["std"].reshape(2, 3)
    var = std**2 - eps
    leak = (u * xhat).mean(-1) / std * eps / (var + eps)
    np.testing.assert_allclose((xhat * dx).mean(-1), leak, atol=1e-10)


def test_against_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, 8, 4, 5))
    gamma, beta = rng.normal(size=8), rng.normal(size=8)
    xt = torch.tensor(x)
    for g in (1, 2, 4, 8):
        y = forward({"gamma": gamma, "beta": beta}, x, config(8, g))[0]
        ref = torch.nn.functional.group_norm(xt, g, torch.tensor(gamma), torch.tensor(beta), eps=1e-5)
        np.testing.assert_allclose(y, ref.numpy(), atol=1e-10)
    inst = torch.nn.functional.instance_norm(xt, eps=1e-5).numpy()
    np.testing.assert_allclose(normalize(x, "instance"), inst, atol=1e-10)
    lay = torch.nn.functional.layer_norm(xt, (8, 4, 5), eps=1e-5).numpy()
    np.testing.assert_allclose(normalize(x, "layer"), lay, atol=1e-10)
    bat = torch.nn.functional.batch_norm(xt, None, None, training=True, eps=1e-5).numpy()
    np.testing.assert_allclose(normalize(x, "batch"), bat, atol=1e-10)
