# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Flow matching in JAX: pure functions of a parameter dictionary, the conditional optimal-transport path
(eq-ot-path) and its target (eq-cfm-target), the velocity network, jax.grad of the conditional flow matching loss
(eq-cfm-loss), Euler and midpoint samplers of the ODE (eq-ode) as jax.lax.fori_loop, the divergence as the trace of
jax.jacfwd, the log-likelihood by the instantaneous change of variables (eq-change-of-variables), logit-normal times
(eq-logit-normal) and a jitted Adam step. The parameters have the names and shapes of the NumPy reference; time runs
from noise at t = 0 to data at t = 1.
"""

import math

import jax
import jax.numpy as jnp


def config(d_data=3, d_hidden=5, d_time=4, sigma_min=0.0):
    if min(d_data, d_hidden, d_time) < 1:
        raise ValueError("every size must be positive")
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("sigma_min must lie in [0, 1)")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "d_time": int(d_time), "sigma_min": float(sigma_min)}


def param_shapes(cfg):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return {"W1": (D + E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return 2 * batch * H * (2 * D + E)


def init(key, cfg):
    p = {k: jnp.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W1"] = jax.random.normal(key, p["W1"].shape) / math.sqrt(p["W1"].shape[0])
    return p


def time_features(t, d_time):
    """eq-time-features: entry 2k is sin((k + 1) pi t), entry 2k + 1 is cos((k + 1) pi t)."""
    j = jnp.arange(d_time)
    a = jnp.asarray(t)[:, None] * (j // 2 + 1) * math.pi
    return jnp.where(j % 2 == 0, jnp.sin(a), jnp.cos(a))


def velocity(p, cfg, x, t):
    """v_theta(x, t) = tanh([x, e(t)] W1 + b1) W2 + b2 for a batch x (B, D) and times t (B,)."""
    inp = jnp.concatenate([x, time_features(t, cfg["d_time"])], 1)
    return jnp.tanh(inp @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def divergence(v_fn, x, t):
    """The trace of the Jacobian of v_fn in x, row by row with jax.jacfwd under vmap."""

    def one(xi, ti):
        return jnp.trace(jax.jacfwd(lambda z: v_fn(z[None], ti[None])[0])(xi))

    return jax.vmap(one)(x, t)


def path(cfg, x0, x1, t):
    """eq-ot-path: x_t = (1 - (1 - sigma_min) t) x_0 + t x_1."""
    t = jnp.asarray(t)[:, None]
    return (1.0 - (1.0 - cfg["sigma_min"]) * t) * x0 + t * x1


def target(cfg, x0, x1):
    """eq-cfm-target: x_1 - (1 - sigma_min) x_0."""
    return x1 - (1.0 - cfg["sigma_min"]) * x0


def cond_field(cfg, x, x1, t):
    """eq-cond-field: (x_1 - (1 - sigma_min) x) / (1 - (1 - sigma_min) t)."""
    c = 1.0 - cfg["sigma_min"]
    return (x1 - c * x) / (1.0 - c * jnp.asarray(t)[:, None])


def cfm_loss(p, cfg, x0, x1, t):
    """eq-cfm-loss: the mean over rows of ||v_theta(x_t, t) - (x_1 - (1 - sigma_min) x_0)||^2."""
    return jnp.mean(jnp.sum((velocity(p, cfg, path(cfg, x0, x1, t), t) - target(cfg, x0, x1)) ** 2, 1))


def euler_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    h = (t1 - t0) / n_steps

    def body(i, x):
        return x + h * v_fn(x, jnp.full(len(x), t0 + i * h))

    return jax.lax.fori_loop(0, n_steps, body, x)


def midpoint_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    h = (t1 - t0) / n_steps

    def body(i, x):
        t = jnp.full(len(x), t0 + i * h)
        return x + h * v_fn(x + 0.5 * h * v_fn(x, t), t + 0.5 * h)

    return jax.lax.fori_loop(0, n_steps, body, x)


def log_likelihood(v_fn, x1, n_steps, log_p0):
    """eq-change-of-variables: back from t = 1 to t = 0 with midpoint steps, integrating the divergence alongside."""
    h = 1.0 / n_steps

    def body(i, carry):
        x, acc = carry
        t = jnp.full(len(x), 1.0 - i * h)
        xm = x - 0.5 * h * v_fn(x, t)
        return x - h * v_fn(xm, t - 0.5 * h), acc + h * divergence(v_fn, xm, t - 0.5 * h)

    x0, acc = jax.lax.fori_loop(0, n_steps, body, (x1, jnp.zeros(len(x1))))
    return log_p0(x0) - acc


def logit_normal(key, n, m=0.0, s=1.0):
    """eq-logit-normal: t = sigmoid(m + s z), z ~ N(0, 1)."""
    return jax.nn.sigmoid(m + s * jax.random.normal(key, (n,)))


def adam_init(p):
    return {"m": jax.tree.map(jnp.zeros_like, p), "v": jax.tree.map(jnp.zeros_like, p), "t": 0}


def make_step(cfg, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One jitted step of conditional flow matching with Adam, on a batch (x0, x1, t)."""

    @jax.jit
    def step(p, st, x0, x1, t):
        loss, g = jax.value_and_grad(cfm_loss)(p, cfg, x0, x1, t)
        k = st["t"] + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, st["m"], g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b**2, st["v"], g)
        p = jax.tree.map(lambda w, mi, vi: w - lr * (mi / (1 - b1**k)) / (jnp.sqrt(vi / (1 - b2**k)) + eps), p, m, v)
        return p, {"m": m, "v": v, "t": k}, loss

    return step
