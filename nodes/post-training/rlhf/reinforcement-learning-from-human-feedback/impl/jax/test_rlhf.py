# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
"""Tests of the JAX functions: the shared fixture, the gradients of jax.grad against the closed forms, jit, the
KL-regularized optimum and the gradient that vanishes there, and DPO. The numbers are those of the NumPy reference,
which the PyTorch tests check against torch.nn.functional."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rlhf import bt_loss, dpo_loss, kl_objective, kl_optimal_policy, kl_shaped_rewards, ppo_clip_loss, ranked_loss

jax.config.update("jax_enable_x64", True)

# The shared fixture of the NumPy tests.
RW = [1.0, -0.5, 2.0, 0.3]
RL = [0.0, 0.5, -1.0, 0.3]
BT_LOSS = 0.5920644767925333
BT_GRAD_W = [-0.06723535534249878, -0.18276464465750122, -0.011856468294391695, -0.125]
RANKED = [[2.0, 1.0, 0.0, -1.0], [0.5, 1.5, -0.5, 0.0]]
RANKED_LOSS = 0.38710392238639496
LOGP = [[-1.0, -2.0, -0.5], [-0.3, -0.7, 0.0]]
LOGP_REF = [[-1.2, -1.5, -0.5], [-0.4, -0.2, 0.0]]
MASK = [[1, 1, 1], [1, 1, 0]]
SCORE = [1.0, -0.5]
KL_REWARDS = [[-0.02, 0.05, 1.0], [-0.01, -0.45, 0.0]]
PPO_LOGP = [0.0, 0.5, -0.5, 0.1]
PPO_OLD = [0.0, 0.0, 0.0, 0.0]
PPO_ADV = [1.0, 1.0, -1.0, -2.0]
PPO_OBJ = -0.20258545903782382
DPO_ARGS = ([-1.0, -2.0], [-1.5, -1.0], [-1.2, -1.8], [-1.2, -1.4], 0.1)
DPO_LOSS = 0.6960283805447136


def a(v):
    return jnp.asarray(v, dtype=jnp.float64)


def test_fixture():
    assert float(bt_loss(a(RW), a(RL))) == pytest.approx(BT_LOSS, rel=1e-12)
    gw, gl = jax.grad(bt_loss, argnums=(0, 1))(a(RW), a(RL))
    np.testing.assert_allclose(gw, BT_GRAD_W, rtol=1e-12)
    np.testing.assert_allclose(gl, -np.array(BT_GRAD_W), rtol=1e-12)
    assert float(ranked_loss(a(RANKED))) == pytest.approx(RANKED_LOSS, rel=1e-12)
    out = kl_shaped_rewards(a(SCORE), a(LOGP), a(LOGP_REF), 0.1, a(MASK))
    np.testing.assert_allclose(out, KL_REWARDS, atol=1e-12)
    assert float(ppo_clip_loss(a(PPO_LOGP), a(PPO_OLD), a(PPO_ADV))) == pytest.approx(-PPO_OBJ, rel=1e-12)
    assert float(dpo_loss(*[a(x) for x in DPO_ARGS[:4]], DPO_ARGS[4])) == pytest.approx(DPO_LOSS, rel=1e-12)


def test_jit_agrees():
    f = jax.jit(bt_loss)
    assert float(f(a(RW), a(RL))) == pytest.approx(BT_LOSS, rel=1e-12)
    g = jax.jit(dpo_loss, static_argnums=4)
    assert float(g(*[a(x) for x in DPO_ARGS[:4]], DPO_ARGS[4])) == pytest.approx(DPO_LOSS, rel=1e-12)
    h = jax.jit(ppo_clip_loss)
    assert float(h(a(PPO_LOGP), a(PPO_OLD), a(PPO_ADV))) == pytest.approx(-PPO_OBJ, rel=1e-12)


def test_bt_gradient_closed_form():
    key = jax.random.PRNGKey(20261115)
    k1, k2 = jax.random.split(key)
    rw, rl = 3 * jax.random.normal(k1, (16,)), 3 * jax.random.normal(k2, (16,))
    gw = jax.grad(bt_loss)(rw, rl)
    np.testing.assert_allclose(gw, -jax.nn.sigmoid(-(rw - rl)) / 16, rtol=1e-12)
    # The loss and its gradient do not change when a constant is added to both rewards.
    assert float(bt_loss(rw + 7.0, rl + 7.0)) == pytest.approx(float(bt_loss(rw, rl)), rel=1e-12)


def test_kl_shaped_rewards_stop_gradient():
    g = jax.grad(lambda lp: jnp.sum(kl_shaped_rewards(a([1.0]), lp, a([[0.0, 0.0]]), 0.1, a([[1, 1]]))))
    np.testing.assert_allclose(g(a([[-1.0, -2.0]])), [[0.0, 0.0]])


def test_ppo_clip_gradient():
    logp = a([math.log(1.5), math.log(1.1), math.log(0.5), math.log(0.5)])
    adv = a([1.0, 1.0, -1.0, 1.0])
    g = jax.grad(ppo_clip_loss)(logp, jnp.zeros(4), adv)
    np.testing.assert_allclose(g, -np.array([0.0, 1.1, 0.0, 0.5]) / 4, rtol=1e-12, atol=1e-15)


def test_kl_optimum_is_stationary():
    # On the simplex, pi_star is where the gradient of the objective in the logits of pi vanishes, and the maximum is
    # beta log Z.
    ref = jnp.log(a([0.5, 0.25, 0.25]))
    r = a([0.0, math.log(2), math.log(4)])
    pi, log_z = kl_optimal_policy(ref, r, 1.0)
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.5], rtol=1e-12)
    assert float(log_z) == pytest.approx(math.log(2), rel=1e-12)
    obj = lambda z, beta: kl_objective(jax.nn.softmax(z), ref, r, beta)  # noqa: E731
    np.testing.assert_allclose(jax.grad(obj)(jnp.log(pi), 1.0), [0.0, 0.0, 0.0], atol=1e-12)
    assert float(obj(jnp.log(pi), 1.0)) == pytest.approx(math.log(2), rel=1e-12)
    # Gradient ascent from pi_ref reaches it.
    z = ref
    for _ in range(2000):
        z = z + 0.5 * jax.grad(obj)(z, 1.0)
    np.testing.assert_allclose(jax.nn.softmax(z), pi, atol=1e-6)


def test_dpo_gradient_and_frozen_reference():
    gw, gl, grw, grl = jax.grad(dpo_loss, argnums=(0, 1, 2, 3))(a([-1.0]), a([-1.5]), a([-1.2]), a([-1.2]), 0.1)
    weight = jax.nn.sigmoid(0.1 * ((-1.5 + 1.2) - (-1.0 + 1.2)))
    assert float(gw[0]) == pytest.approx(-0.1 * float(weight), rel=1e-12)
    assert float(gl[0]) == pytest.approx(0.1 * float(weight), rel=1e-12)
    assert float(grw[0]) == 0.0 and float(grl[0]) == 0.0


def test_dpo_at_optimum_is_bradley_terry():
    key = jax.random.PRNGKey(7)
    ref = jax.nn.log_softmax(jax.random.normal(key, (5,)))
    r = a([0.3, -1.0, 2.0, 0.0, 1.5])
    for beta in [0.1, 0.5, 2.0]:
        pi, _ = kl_optimal_policy(ref, r, beta)
        lp = jnp.log(pi)
        w, lo = jnp.array([0, 1, 2]), jnp.array([3, 4, 0])
        assert float(dpo_loss(lp[w], lp[lo], ref[w], ref[lo], beta)) == pytest.approx(float(bt_loss(r[w], r[lo])),
                                                                                     rel=1e-9)


def test_ranked_loss_shape():
    with pytest.raises(ValueError):
        ranked_loss(a([1.0, 2.0]))
