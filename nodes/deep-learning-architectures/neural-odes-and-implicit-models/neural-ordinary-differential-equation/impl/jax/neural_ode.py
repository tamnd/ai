# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://WNCNQE).
"""The neural ordinary differential equation, the JAX reference: eq-field, eq-augment and RK4 steps of eq-ode.

The parameters are a dict with the names and shapes of the NumPy reference. jax.grad of forward differentiates the
discrete RK4 map (discretize, then optimize). odeint_adjoint is a jax.custom_vjp whose backward rule solves the
adjoint system eq-adjoint and eq-adjoint-theta from T back to 0 with the same steps (optimize, then discretize).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

NAMES = ("W1", "u", "b1", "W2", "b2")


def config(d=2, d_hidden=8, d_state=None, n_steps=4, T=1.0):
    d_state = d if d_state is None else d_state
    if min(d, d_hidden, n_steps) < 1 or d_state < d or T <= 0:
        raise ValueError("d, d_hidden and n_steps must be at least 1, d_state at least d and T positive")
    return {"d": d, "d_hidden": d_hidden, "d_state": d_state, "n_steps": n_steps, "T": float(T)}


def param_count(cfg):
    D, H = cfg["d_state"], cfg["d_hidden"]
    return 2 * D * H + 2 * H + D  # eq: eq-params


def init(key, cfg):
    D, H = cfg["d_state"], cfg["d_hidden"]
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W1": jax.random.normal(k1, (H, D)) / jnp.sqrt(D),
        "u": jax.random.normal(k2, (H,)),
        "b1": jnp.zeros(H),
        "W2": jax.random.normal(k3, (D, H)) / jnp.sqrt(H),
        "b2": jnp.zeros(D),
    }


def field(params, h, t):
    return jnp.tanh(h @ params["W1"].T + t * params["u"] + params["b1"]) @ params["W2"].T + params["b2"]  # eq: eq-field


def rk4(f, z, t0, t1, n):
    """n classical Runge-Kutta steps from t0 to t1 of dz/dt = f(z, t), for any pytree z."""
    dt = (t1 - t0) / n

    def add(a, b, s):
        return jax.tree_util.tree_map(lambda x, y: x + s * y, a, b)

    def step(k, z):
        t = t0 + k * dt
        k1 = f(z, t)
        k2 = f(add(z, k1, 0.5 * dt), t + 0.5 * dt)
        k3 = f(add(z, k2, 0.5 * dt), t + 0.5 * dt)
        k4 = f(add(z, k3, dt), t + dt)
        return jax.tree_util.tree_map(
            lambda x, a, b, c, e: x + dt / 6.0 * (a + 2.0 * b + 2.0 * c + e), z, k1, k2, k3, k4
        )

    return jax.lax.fori_loop(0, n, step, z)


def augment(x, cfg):
    return jnp.concatenate([x, jnp.zeros((x.shape[0], cfg["d_state"] - cfg["d"]))], 1)  # eq: eq-augment


def forward(params, x, cfg):
    """h(T) of shape (B, D) from x of shape (B, d), differentiated through the steps."""
    return rk4(lambda h, t: field(params, h, t), augment(x, cfg), 0.0, cfg["T"], cfg["n_steps"])


def odeint_adjoint(params, h0, T, n):
    @jax.custom_vjp
    def solve(params, h0):
        return rk4(lambda h, t: field(params, h, t), h0, 0.0, T, n)

    def solve_fwd(params, h0):
        y = solve(params, h0)
        return y, (params, y)  # the residual is the output only, not the steps

    def solve_bwd(res, dy):
        params, y = res

        def aug(z, t):
            h, a, _ = z
            f, vjp = jax.vjp(lambda p, hh: field(p, hh, t), params, h)
            gp, gh = vjp(a)
            return (f, -gh, jax.tree_util.tree_map(jnp.negative, gp))  # eq: eq-adjoint and eq-adjoint-theta

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        _, a0, g = rk4(aug, (y, dy, zeros), T, 0.0, n)
        return g, a0

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(params, h0)


def forward_adjoint(params, x, cfg):
    """The same h(T), with gradients from the adjoint system instead of the steps."""
    return odeint_adjoint(params, augment(x, cfg), cfg["T"], cfg["n_steps"])


def mse_loss(y, t):
    return 0.5 * jnp.sum((y - t) ** 2) / y.shape[0]
