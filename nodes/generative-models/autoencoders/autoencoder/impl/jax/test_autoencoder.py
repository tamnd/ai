# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://JC1Q9N).
"""Tests of the JAX version: the shared fixture and its gradients by jax.grad, the counts, the corruptions, the
contractive penalty, and the linear autoencoder with weight decay, whose encoder becomes the decoder's transpose."""

import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from autoencoder import (  # noqa: E402
    apply,
    config,
    contractive_penalty,
    corrupt,
    from_widths,
    init,
    jacobian_penalty,
    loss,
    objective,
    param_count,
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
        p.setdefault(key.split("/")[0], []).append(jnp.asarray(arrays[key]))
    return p


def test_shared_fixture_and_gradients():
    p = unflat(dict(np.load(FIXTURES / "params-small.npz")))
    x = jnp.asarray(np.load(FIXTURES / "input-small.npz")["x"])
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    cfg = config()
    out = apply(p, cfg, x)
    for k in ("y", "a", "z"):
        np.testing.assert_allclose(out[k], want[k], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(loss(cfg, x, out["a"]), want["loss"], rtol=1e-12)
    grads = jax.grad(lambda q: loss(cfg, x, apply(q, cfg, x)["a"]))(p)
    for k, vs in grads.items():
        for i, v in enumerate(vs):
            np.testing.assert_allclose(v, want[f"d{k}/{i}"], rtol=1e-10, atol=1e-14)
    dx = jax.grad(lambda v: loss(cfg, x, apply(p, cfg, v)["a"]))(x)
    np.testing.assert_allclose(dx, want["dx"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    for tied, key in ((False, "params"), (True, "params_tied")):
        cfg = cfg_of(c, tied=tied)
        assert param_count(cfg) == c["model"][key]
        assert sum(a.size for a in jax.tree_util.tree_leaves(init(jax.random.key(0), cfg))) == c["model"][key]


def test_corruptions():
    x = jax.random.uniform(jax.random.key(1), (30, 20), minval=0.2, maxval=0.8)
    for kind in ("mask", "salt_pepper"):
        xt, mask = corrupt(jax.random.key(2), x, kind, 0.3)
        assert (mask.sum(1) == 6).all()
        np.testing.assert_array_equal(xt[~mask], x[~mask])
        assert set(np.unique(np.asarray(xt[mask])).tolist()) <= {0.0, 1.0}


def test_contractive_penalty():
    cfg = from_widths([6, 4], code_act="sigmoid")
    p = init(jax.random.key(3), cfg)
    p["enc_b"] = [jax.random.normal(jax.random.key(4), (4,))]
    x = jax.random.normal(jax.random.key(5), (5, 6))
    np.testing.assert_allclose(contractive_penalty(p, x), jacobian_penalty(p, cfg, x), rtol=1e-12)
    lin = from_widths([6, 4], code_act="linear")  # Rifai et al. (2011): a linear encoder makes it weight decay
    np.testing.assert_allclose(jacobian_penalty(p, lin, x), jnp.sum(p["enc_W"][0] ** 2), rtol=1e-12)


def test_weight_decay_ties_the_linear_autoencoder():
    # Kunin et al. (2019), Theorem 2.1: at a critical point of the regularized loss, the encoder is the decoder's
    # transpose; here gradient descent reaches the minimum and the decoder's squared singular values are 1 - lam/s2
    rng = np.random.default_rng(6)
    variances, lam = np.array([3.0, 1.5, 0.4]), 0.2
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    Z = rng.normal(size=(300, 3))
    Z -= Z.mean(0)
    Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z / 300)).T
    X = jnp.asarray(Z * np.sqrt(variances) @ Q.T)
    cfg = from_widths([3, 2], code_act="linear", out="linear")
    p = init(jax.random.key(7), cfg)
    step = jax.jit(lambda q: jax.tree_util.tree_map(lambda w, g: w - 0.05 * g, q,
                                                    jax.grad(objective)(q, cfg, X, weight_decay=lam)))  # fmt: skip
    for _ in range(5000):
        p = step(p)
    np.testing.assert_allclose(p["enc_W"][0], p["dec_W"][0].T, atol=1e-6)
    s = np.linalg.svd(np.asarray(p["dec_W"][0]), compute_uv=False)
    np.testing.assert_allclose(s**2, 1 - lam / variances[:2], atol=1e-6)
