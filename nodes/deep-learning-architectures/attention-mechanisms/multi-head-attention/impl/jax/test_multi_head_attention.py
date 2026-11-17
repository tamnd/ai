# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://ND4EXH).
"""Tests of the JAX reference: the shared fixture with jax.grad, the masks, the heads, and jit."""

import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from multi_head_attention import apply, attention, causal_mask, config, init, padding_mask, param_count

jax.config.update("jax_enable_x64", True)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())


def fixture():
    params = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp = {k: jnp.asarray(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = dict(np.load(FIXTURES / "expected-small.npz"))
    return params, inp, want, config(int(inp["d_model"]), int(inp["n_heads"]))


@pytest.mark.parametrize("case", ["self", "cross"])
def test_shared_fixture_forward_and_grad(case):
    params, inp, want, cfg = fixture()
    mask = inp["causal"] if case == "self" else inp["padding"]

    def loss(p, x, mem):
        return jnp.sum(apply(p, x, cfg, mem, mask)[0] * inp["dy"])

    mem = None if case == "self" else inp["memory"]
    y, a = apply(params, inp["x"], cfg, mem, mask)
    np.testing.assert_allclose(y, want[f"{case}/y"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a, want[f"{case}/a"], rtol=1e-10, atol=1e-12)
    argnums = (0, 1) if mem is None else (0, 1, 2)
    grads = jax.grad(loss, argnums=argnums)(params, inp["x"], mem)
    np.testing.assert_allclose(grads[1], want[f"{case}/dx"], rtol=1e-10, atol=1e-12)
    if mem is not None:
        np.testing.assert_allclose(grads[2], want["cross/dmemory"], rtol=1e-10, atol=1e-12)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        np.testing.assert_allclose(grads[0][name], want[f"{case}/d{name}"], rtol=1e-10, atol=1e-12)


def test_worked_example():
    x = jnp.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    eye = {n: jnp.eye(2) for n in ("W_q", "W_k", "W_v", "W_o")}
    y, _ = apply(eye, x, config(2, 2))
    np.testing.assert_allclose(y[0, 0], [2 * math.e / (2 * math.e + 1), 2 / 3], rtol=1e-12)


def test_rows_sum_to_one_and_the_causal_mask_hides_the_future():
    cfg = config(8, 2)
    kp, kx = jax.random.split(jax.random.PRNGKey(1))
    params = init(kp, cfg)
    x = jax.random.normal(kx, (2, 6, 8))
    y, a = jax.jit(lambda p, x: apply(p, x, cfg, mask=causal_mask(6)))(params, x)
    np.testing.assert_allclose(a.sum(-1), 1.0, atol=1e-12)
    assert np.all(np.asarray(a)[..., ~np.asarray(causal_mask(6))] == 0)
    y2, _ = apply(params, x.at[:, 3:].add(1.0), cfg, mask=causal_mask(6))
    np.testing.assert_allclose(y2[:, :3], y[:, :3], rtol=1e-12, atol=1e-12)


def test_permutation_equivariance():
    cfg = config(8, 4)
    kp, kx = jax.random.split(jax.random.PRNGKey(2))
    params, x = init(kp, cfg), jax.random.normal(kx, (2, 5, 8))
    perm = np.array([3, 0, 4, 1, 2])
    np.testing.assert_allclose(apply(params, x[:, perm], cfg)[0], apply(params, x, cfg)[0][:, perm], rtol=1e-10)


def test_padding_and_all_masked_rows():
    cfg = config(8, 2)
    kp, kx, km = jax.random.split(jax.random.PRNGKey(3), 3)
    params, x, mem = init(kp, cfg), jax.random.normal(kx, (2, 3, 8)), jax.random.normal(km, (2, 4, 8))
    y, _ = apply(params, x, cfg, mem, padding_mask([4, 2], 4))
    np.testing.assert_allclose(y[1], apply(params, x[1:], cfg, mem[1:, :2])[0][0], rtol=1e-10, atol=1e-12)
    q = jax.random.normal(kx, (1, 1, 2, 4))
    mask = jnp.array([[True, False, True], [False, False, False]])
    g = jax.grad(lambda q: attention(q, mem[None, :1, :3, :4], mem[None, :1, :3, 4:], mask)[0].sum())(q)
    assert np.all(np.isfinite(g))
    np.testing.assert_array_equal(attention(q, mem[None, :1, :3, :4], mem[None, :1, :3, 4:], mask)[0][0, 0, 1], 0.0)


def test_param_count():
    cfg = config(12, 3, 5, 7)
    assert param_count(cfg) == sum(p.size for p in init(jax.random.PRNGKey(0), cfg).values())


def test_config_defaults_and_errors():
    assert config() == {"d_model": 512, "n_heads": 8, "d_k": 64, "d_v": 64}
    assert config(12, 3)["d_k"] == 4 and config(12, 3)["d_v"] == 4
    with pytest.raises(ValueError):
        config(8, 2, d_k=0, d_v=4)
    with pytest.raises(ValueError):
        config(10, 3)


def test_init_shapes_and_scale():
    p = init(jax.random.PRNGKey(0), config(256, 4, d_k=32, d_v=48))
    assert p["W_q"].shape == p["W_k"].shape == (256, 128)
    assert p["W_v"].shape == (256, 192) and p["W_o"].shape == (192, 256)
    for name, fan_in in (("W_q", 256), ("W_k", 256), ("W_v", 256), ("W_o", 192)):
        std = float(jnp.std(p[name]))
        assert abs(std - fan_in**-0.5) < 0.1 * fan_in**-0.5, name
