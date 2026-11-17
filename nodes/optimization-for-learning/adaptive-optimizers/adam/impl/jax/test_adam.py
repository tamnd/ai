# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/adam/ (ai://88GHGT).
"""Tests of the JAX reference: the worked example, the shared fixture, jit and pytrees, the constant gradient, the
bound on the step, and the configuration. optax is not installed, so the tests compare with fixed numbers that the
PyTorch tests check against torch.optim.Adam and torch.optim.AdamW."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adam import config, init, update

jax.config.update("jax_enable_x64", True)

# The worked example of the page.
WG = [2.0, -2.0]
WM = [1.0, -0.5]
WV = [0.16, 0.3136]
WTHETA = [0.9, 14.0 / 15.0]

# The shared fixture of the NumPy tests.
P = {"w": [0.5, -1.5, 2.0], "b": [0.25]}
G = [{"w": [1.0, -2.0, 0.5], "b": [0.5]}, {"w": [0.5, 1.0, -0.25], "b": [-1.0]}, {"w": [-1.5, 0.5, 0.0], "b": [2.0]}]
CONFIGS = {"adam": {"lr": 0.1}, "l2": {"lr": 0.1, "weight_decay": 0.1},
           "adamw": {"lr": 0.1, "weight_decay": 0.1, "decoupled": True}, "amsgrad": {"lr": 0.1, "amsgrad": True}}
AFTER3 = {
    "adam": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527783689874684], "b": [0.14464479271810454]},
    "l2": {"w": [0.3108766333376067, -1.3489621467423747, 1.778235325837831], "b": [0.14103715505555778]},
    "adamw": {"w": [0.3030513976144773, -1.324932247497697, 1.7956327059805899], "b": [0.1388434391558989]},
    "amsgrad": {"w": [0.3149797180073589, -1.3672274104696713, 1.8527886655267884], "b": [0.14464479271810454]},
}


def arr(d):
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in d.items()}


def test_worked_example():
    cfg = config(lr=0.1, betas=(0.5, 0.96), eps=0.0)
    x = {"x": jnp.asarray(1.0)}
    s = init(x)
    for i in range(2):
        x, s = update(cfg, {"x": jnp.asarray(WG[i])}, s, x)
        np.testing.assert_allclose(s["m"]["x"], WM[i], rtol=1e-14)
        np.testing.assert_allclose(s["v"]["x"], WV[i], rtol=1e-14)
        np.testing.assert_allclose(x["x"], WTHETA[i], rtol=1e-14)
    cfg = config(lr=0.1, betas=(0.5, 0.96), eps=0.0, weight_decay=0.1, decoupled=True)
    x0 = {"x": jnp.asarray(1.0)}
    x, _ = update(cfg, {"x": jnp.asarray(2.0)}, init(x0), x0)
    np.testing.assert_allclose(x["x"], 0.89, rtol=1e-14)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_fixture(name):
    cfg = config(**CONFIGS[name])
    p = arr(P)
    s = init(p)
    for g in G:
        p, s = update(cfg, arr(g), s, p)
    for k in P:
        np.testing.assert_allclose(p[k], AFTER3[name][k], rtol=1e-13)


def test_jit_and_pytree():
    cfg = config(lr=0.1, weight_decay=0.1, decoupled=True)
    step = jax.jit(functools.partial(update, cfg))
    p = {"layer": arr(P)}
    s = init(p)
    for g in G:
        p, s = step({"layer": arr(g)}, s, p)
    assert int(s["t"]) == 3
    for k in P:
        np.testing.assert_allclose(p["layer"][k], AFTER3["adamw"][k], rtol=1e-13)


def test_gradient_of_a_loss():
    """One step on the loss sum((w - 3)^2) with jax.grad: the first step is lr times the sign of the gradient."""
    cfg = config(lr=0.1, eps=0.0)
    w = {"w": jnp.asarray([0.0, 5.0, 3.5])}
    g = jax.grad(lambda p: jnp.sum((p["w"] - 3.0) ** 2))(w)
    new, _ = update(cfg, g, init(w), w)
    np.testing.assert_allclose(new["w"], [0.1, 4.9, 3.4], rtol=1e-14)


@pytest.mark.parametrize("b1,b2", [(0.9, 0.999), (0.5, 0.96), (0.0, 0.5)])
def test_constant_gradient_step_is_lr(b1, b2):
    cfg = config(lr=0.01, betas=(b1, b2), eps=0.0)
    x = {"x": jnp.asarray(0.0)}
    s = init(x)
    for _ in range(25):
        new, s = update(cfg, {"x": jnp.asarray(-7.0)}, s, x)
        np.testing.assert_allclose(new["x"] - x["x"], 0.01, rtol=1e-9)
        x = new


def test_step_bound():
    """Random gradients stay under the bound of derivations/step-bound.md, and g_{t-k} = (b1/b2)^k reaches it."""
    b1, b2, n = 0.9, 0.999, 60
    gamma = b1**2 / b2
    bound = (1 - b1) / np.sqrt(1 - b2) * np.sqrt(1 - b2**n) / (1 - b1**n) * np.sqrt((1 - gamma**n) / (1 - gamma))
    cfg = config(lr=1.0, betas=(b1, b2), eps=0.0)
    rng = np.random.default_rng(20261115)
    cases = [rng.standard_normal(n) * np.exp(rng.normal(0, 2, n)) for _ in range(20)]
    for g in cases + [(b1 / b2) ** np.arange(n)[::-1]]:
        x = {"x": jnp.asarray(0.0)}
        s = init(x)
        for gi in g:
            new, s = update(cfg, {"x": jnp.asarray(gi)}, s, x)
            delta, x = abs(float(new["x"] - x["x"])), new
        assert delta <= bound * (1 + 1e-9)
    np.testing.assert_allclose(delta, bound, rtol=1e-9)


@pytest.mark.parametrize("kw", [{"lr": -1.0}, {"betas": (1.0, 0.999)}, {"eps": -1e-8}, {"weight_decay": -0.1}])
def test_config_rejects(kw):
    with pytest.raises(ValueError):
        config(**kw)
