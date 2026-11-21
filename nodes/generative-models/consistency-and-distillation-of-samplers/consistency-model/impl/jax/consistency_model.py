# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Consistency models in JAX, as pure functions.

- The noise levels of Karras and colleagues, the boundary-preserving scales c_skip, c_out and c_in (eq-param), the
  exact score and probability-flow ODE of a mixture of Gaussians, the ODE solved by a jax.lax.scan of Heun steps, and
  the closed-form consistency function of one Gaussian (eq-gauss-cf).
- The model as a parameter dictionary, and its total derivative along the ODE by jax.jvp.
- The distillation and training losses (eq-cd, eq-ct) with the squared l2 or Pseudo-Huber metric (eq-huber), the
  target held out of the gradient by jax.lax.stop_gradient, and their gradients by jax.grad.
- Consistency distillation as one jax.lax.scan of Adam steps with the EMA target (eq-ema) in the carry; improved
  consistency training as one scan per stage of the doubling curriculum, since each stage has its own number of levels.
- The sampler of Algorithm 1, the toy of Proposition 1 of Song and Dhariwal, and the target of progressive
  distillation (eq-pd).
"""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SIGMA_DATA, EPS, T_MAX, RHO = 0.5, 0.002, 80.0, 7.0


def karras_steps(N, eps=EPS, T=T_MAX, rho=RHO):
    i = jnp.arange(N)
    return (eps ** (1 / rho) + i / (N - 1) * (T ** (1 / rho) - eps ** (1 / rho))) ** rho


def c_skip(t, eps=EPS, sd=SIGMA_DATA):
    return sd**2 / ((t - eps) ** 2 + sd**2)


def c_out(t, eps=EPS, sd=SIGMA_DATA):
    return sd * (t - eps) / jnp.sqrt(sd**2 + t**2)


def c_in(t, sd=SIGMA_DATA):
    return 1.0 / jnp.sqrt(t**2 + sd**2)


def _col(t, n):
    return jnp.broadcast_to(jnp.asarray(t, jnp.float64), (n,))[:, None]


def score(x, t, means, s2, weights):
    v = s2 + _col(t, len(x)) ** 2
    r = x[:, None, :] - means[None]
    post = jax.nn.softmax(jnp.log(weights)[None] - 0.5 * (r**2).sum(-1) / v, 1)
    return -jnp.einsum("bk,bkd->bd", post, r) / v


def ode_drift(x, t, score_fn):
    return -_col(t, len(x)) * score_fn(x, t)


def teacher_step(x, t_from, t_to, score_fn, solver="heun"):
    h = _col(t_to, len(x)) - _col(t_from, len(x))
    d1 = ode_drift(x, t_from, score_fn)
    x1 = x + h * d1
    if solver == "euler":
        return x1
    return x + 0.5 * h * (d1 + ode_drift(x1, t_to, score_fn))


def solve_ode(x, t_from, t_to, score_fn, n=200):
    ts = karras_steps(n + 1, eps=t_to, T=t_from)[::-1]
    x, _ = jax.lax.scan(lambda xx, ab: (teacher_step(xx, ab[0], ab[1], score_fn), None), x, (ts[:-1], ts[1:]))
    return x


def gaussian_consistency(x, t, mu, s2, eps=EPS):
    return mu + (x - mu) * jnp.sqrt((s2 + eps**2) / (s2 + _col(t, len(x)) ** 2))


def config(D, H, E):
    return {"dim": D, "hidden": H, "temb": E}


def param_shapes(cfg):
    D, H, E = cfg["dim"], cfg["hidden"], cfg["temb"]
    return {"W1": (D, H), "Wt": (E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    return 2 * B * (cfg["dim"] * cfg["hidden"] + cfg["temb"] * cfg["hidden"] + cfg["hidden"] * cfg["dim"])


def init(key, cfg):
    shapes = param_shapes(cfg)
    keys = jax.random.split(key, len(shapes))
    return {k: jnp.zeros(s) if k.startswith("b") else jax.random.normal(kk, s) / math.sqrt(s[0])
            for (k, s), kk in zip(shapes.items(), keys, strict=True)}  # fmt: skip


def time_embedding(t, E):
    n = (E + 1) // 2
    a = (jnp.log(t) / 4.0)[:, None] * 2.0 ** jnp.arange(n)
    return jnp.concatenate([jnp.sin(a), jnp.cos(a[:, : E // 2])], -1)


def forward(p, cfg, x, t):
    """f(x, t) = c_skip(t) x + c_out(t) (tanh(c_in(t) x W1 + emb(t) Wt + b1) W2 + b2)."""
    t = jnp.broadcast_to(jnp.asarray(t, jnp.float64), (len(x),))
    h = jnp.tanh(c_in(t)[:, None] * x @ p["W1"] + time_embedding(t, cfg["temb"]) @ p["Wt"] + p["b1"])
    return c_skip(t)[:, None] * x + c_out(t)[:, None] * (h @ p["W2"] + p["b2"])


def total_derivative(f, x, t, score_fn):
    """d/dt f(x_t, t) along the ODE through (x, t), by jax.jvp with the tangent (dx/dt, 1)."""
    t = jnp.broadcast_to(jnp.asarray(t, jnp.float64), (len(x),))
    return jax.jvp(f, (x, t), (ode_drift(x, t, score_fn), jnp.ones_like(t)))[1]


def metric(a, b, c=None):
    d2 = ((a - b) ** 2).sum(1)
    return d2 if c is None else jnp.sqrt(d2 + c**2) - c


def consistency_loss(p, p_minus, cfg, x_hi, t_hi, x_lo, t_lo, weight=1.0, c=None):
    """mean lambda d(f_theta(x_hi, t_hi), f_theta-(x_lo, t_lo)); the target is held out of the gradient."""
    tgt = jax.lax.stop_gradient(forward(p_minus, cfg, x_lo, t_lo))
    return (weight * metric(forward(p, cfg, x_hi, t_hi), tgt, c)).mean()


consistency_loss_grad = jax.grad(consistency_loss)


def cd_pair(x0, z, t_lo, t_hi, score_fn, solver="heun"):
    x_hi = x0 + _col(t_hi, len(x0)) * z
    return x_hi, teacher_step(x_hi, t_hi, t_lo, score_fn, solver)


def ct_pair(x0, z, t_lo, t_hi):
    return x0 + _col(t_hi, len(x0)) * z, x0 + _col(t_lo, len(x0)) * z


def ema(p_minus, p, mu):
    return jax.tree.map(lambda a, b: mu * a + (1.0 - mu) * b, p_minus, p)


def cm_schedule(k, K, s0=2, s1=150, mu0=0.9):
    N = math.ceil(math.sqrt(k / K * ((s1 + 1) ** 2 - s0**2) + s0**2) - 1) + 1
    return N, math.exp(s0 * math.log(mu0) / N)


def ict_steps(k, K, s0=10, s1=1280):
    Kp = math.floor(K / (math.log2(math.floor(s1 / s0)) + 1))
    return min(s0 * 2 ** math.floor(k / Kp), s1) + 1


def lognormal_probs(ts, p_mean=-1.1, p_std=2.0):
    e = jax.scipy.special.erf((jnp.log(ts) - p_mean) / (math.sqrt(2.0) * p_std))
    q = e[1:] - e[:-1]
    return q / q.sum()


def _adam(q, g, m, v, n, lr, b1=0.9, b2=0.999, ad_eps=1e-8):
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    q = jax.tree.map(lambda w, mm, vv: w - lr * (mm / (1 - b1**n)) / (jnp.sqrt(vv / (1 - b2**n)) + ad_eps), q, m, v)
    return q, m, v


def train_cd(p, cfg, x0, score_fn, steps, key, N=18, batch=256, lr=2e-3, mu=0.95, c=None, solver="heun"):
    """Consistency distillation as one scan; the carry holds the student, the EMA target and Adam's moments."""
    ts = karras_steps(N)

    def step(carry, k):
        q, qm, m, v, n = carry
        ki, kn, kz = jax.random.split(k, 3)
        x = x0[jax.random.randint(ki, (batch,), 0, len(x0))]
        i = jax.random.randint(kn, (batch,), 0, N - 1)
        x_hi, x_lo = cd_pair(x, jax.random.normal(kz, x.shape), ts[i], ts[i + 1], score_fn, solver)
        loss, g = jax.value_and_grad(consistency_loss)(q, qm, cfg, x_hi, ts[i + 1], x_lo, ts[i], 1.0, c)
        q, m, v = _adam(q, g, m, v, n + 1, lr)
        return (q, ema(qm, q, mu), m, v, n + 1), loss

    zeros = jax.tree.map(jnp.zeros_like, p)
    (p, pm, *_), trace = jax.lax.scan(step, (p, p, zeros, zeros, jnp.zeros(())), jax.random.split(key, steps))
    return p, pm, trace


def train_ct(p, cfg, x0, steps, key, batch=256, lr=2e-3, s0=10, s1=160, c=None):
    """Improved consistency training: the target is stop_gradient of the student, lognormal levels, the weight
    1 / (t_{i+1} - t_i) and the Pseudo-Huber metric; one jitted scan per stage of the curriculum."""
    if c is None:
        c = 0.00054 * math.sqrt(x0.shape[1])
    stages = []
    for k in range(steps):
        N = ict_steps(k, steps, s0, s1)
        if stages and stages[-1][0] == N:
            stages[-1][1] += 1
        else:
            stages.append([N, 1])

    def run(carry, keys, N):
        ts = karras_steps(N)
        probs = lognormal_probs(ts)

        def step(cr, k):
            q, m, v, n = cr
            ki, kn, kz = jax.random.split(k, 3)
            x = x0[jax.random.randint(ki, (batch,), 0, len(x0))]
            i = jax.random.choice(kn, N - 1, (batch,), p=probs)
            x_hi, x_lo = ct_pair(x, jax.random.normal(kz, x.shape), ts[i], ts[i + 1])
            loss, g = jax.value_and_grad(consistency_loss)(q, q, cfg, x_hi, ts[i + 1], x_lo, ts[i],
                                                           1.0 / (ts[i + 1] - ts[i]), c)  # fmt: skip
            q, m, v = _adam(q, g, m, v, n + 1, lr)
            return (q, m, v, n + 1), loss

        return jax.lax.scan(step, carry, keys)

    run_jit = jax.jit(run, static_argnums=2)
    zeros = jax.tree.map(jnp.zeros_like, p)
    carry, traces = (p, zeros, zeros, jnp.zeros(())), []
    for (N, count), k in zip(stages, jax.random.split(key, len(stages)), strict=True):
        carry, tr = run_jit(carry, jax.random.split(k, count), N)
        traces.append(tr)
    return carry[0], jnp.concatenate(traces)


def sample(f, xT, taus=(), noise=None, eps=EPS, T=T_MAX):
    """Algorithm 1 with the noise given, shape (len(taus),) + x_T.shape."""
    x = f(xT, T)
    for k, tau in enumerate(taus):
        assert noise is not None
        x = f(x + math.sqrt(tau**2 - eps**2) * noise[k], tau)
    return x


def toy_model(x, t, theta, smin):
    return smin / t * x + (1 - smin / t) * theta


def toy_ct_loss(theta, theta_minus, xi, ts, z):
    smin, lo, hi = ts[0], ts[:-1, None], ts[1:, None]
    tgt = jax.lax.stop_gradient(toy_model(xi + lo * z, lo, theta_minus, smin))
    return ((toy_model(xi + hi * z, hi, theta, smin) - tgt) ** 2).mean()


def vp(lam):
    a = 1.0 / jnp.sqrt(1.0 + jnp.exp(-jnp.asarray(lam, jnp.float64)))
    return a, jnp.sqrt(1.0 - a**2)


def ddim_step(z, x_pred, a_t, s_t, a_s, s_s):
    return a_s * x_pred + s_s * (z - a_t * x_pred) / s_t


def pd_target(z_t, z_s2, a_t, s_t, a_s2, s_s2):
    r = s_s2 / s_t
    return (z_s2 - r * z_t) / (a_s2 - r * a_t)
