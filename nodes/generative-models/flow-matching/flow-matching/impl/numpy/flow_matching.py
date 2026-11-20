# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/flow-matching/ (ai://NEW-flow-matching).
"""Flow matching in NumPy: the conditional optimal-transport path of Lipman et al. (eq-ot-path), its velocity
(eq-cond-field, eq-cfm-target), a velocity network trained on the conditional flow matching loss (eq-cfm-loss) with
every backward pass written by hand, the Euler and midpoint samplers of the ODE (eq-ode), the exact divergence of the
network and the log-likelihood by the instantaneous change of variables (eq-change-of-variables), minibatch optimal
transport pairing (eq-ot-coupling), the reflow of rectified flow, and logit-normal sampling of t (eq-logit-normal).

Time runs from noise at t = 0 to data at t = 1, as in Lipman et al.; sigma_min = 0 gives the straight path of
rectified flow and of independent conditional flow matching. For data that is a mixture of isotropic Gaussians the
marginal velocity E[x_1 - (1 - sigma_min) x_0 | x_t] is known in closed form; mixture_field gives it, and the tests
drive the samplers with it.
"""

import math

import numpy as np


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
    """eq-flops: the two products of one evaluation of the network, a multiply-add as 2."""
    D, H, E = cfg["d_data"], cfg["d_hidden"], cfg["d_time"]
    return 2 * batch * H * (2 * D + E)


def init(rng, cfg):
    """W1 random, the rest 0: the initial network predicts the velocity 0 everywhere."""
    p = {k: np.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W1"] = rng.standard_normal(p["W1"].shape) / math.sqrt(p["W1"].shape[0])
    return p


def time_features(t, d_time):
    """eq-time-features: entry 2k is sin((k + 1) pi t), entry 2k + 1 is cos((k + 1) pi t), for t (B,) in [0, 1]."""
    j = np.arange(d_time)
    a = np.asarray(t, float)[:, None] * (j // 2 + 1) * math.pi
    return np.where(j % 2 == 0, np.sin(a), np.cos(a))


def velocity(p, cfg, x, t):
    """v_theta(x, t) = tanh([x, e(t)] W1 + b1) W2 + b2. Returns the velocity and the cache of the backward pass."""
    inp = np.concatenate([x, time_features(t, cfg["d_time"])], 1)
    h = np.tanh(inp @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"], (inp, h)


def divergence(p, cfg, x, t):
    """The trace of the Jacobian of v_theta in x, exactly: sum_j (1 - h_j^2) sum_{i < D} W1[i, j] W2[j, i]."""
    _, (_, h) = velocity(p, cfg, x, t)
    D = cfg["d_data"]
    return (1.0 - h**2) @ np.einsum("ij,ji->j", p["W1"][:D], p["W2"])


def _c(t):
    return np.asarray(t, float).reshape(-1, 1)


def path(cfg, x0, x1, t):
    """eq-ot-path: x_t = (1 - (1 - sigma_min) t) x_0 + t x_1, noise x_0 and data x_1."""
    return (1.0 - (1.0 - cfg["sigma_min"]) * _c(t)) * x0 + _c(t) * x1


def target(cfg, x0, x1):
    """eq-cfm-target: the time derivative of the path, x_1 - (1 - sigma_min) x_0, the same at every t."""
    return x1 - (1.0 - cfg["sigma_min"]) * x0


def cond_field(cfg, x, x1, t):
    """eq-cond-field: u_t(x | x_1) = (x_1 - (1 - sigma_min) x) / (1 - (1 - sigma_min) t), Lipman et al. eq. 21."""
    c = 1.0 - cfg["sigma_min"]
    return (x1 - c * x) / (1.0 - c * _c(t))


def regress(p, cfg, x, t, u):
    """The mean over rows of ||v_theta(x, t) - u||^2 and its gradients: the loss of any regression on a velocity."""
    B = len(x)
    out, (inp, h) = velocity(p, cfg, x, t)
    r = out - u
    go = 2.0 * r / B
    gh = (go @ p["W2"].T) * (1.0 - h**2)
    return float(np.sum(r**2) / B), {"W1": inp.T @ gh, "b1": gh.sum(0), "W2": h.T @ go, "b2": go.sum(0)}


def loss_and_grads(p, cfg, x0, x1, t):
    """eq-cfm-loss on one batch of pairs (x_0, x_1) and times t."""
    return regress(p, cfg, path(cfg, x0, x1, t), t, target(cfg, x0, x1))


def euler_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    """Integrate eq-ode from t0 to t1 with n_steps Euler steps x <- x + h v(x, t)."""
    h = (t1 - t0) / n_steps
    for i in range(n_steps):
        x = x + h * v_fn(x, np.full(len(x), t0 + i * h))
    return x


def midpoint_sample(v_fn, x, n_steps, t0=0.0, t1=1.0):
    """Integrate eq-ode with n_steps explicit midpoint steps, two evaluations of v per step."""
    h = (t1 - t0) / n_steps
    for i in range(n_steps):
        t = np.full(len(x), t0 + i * h)
        x = x + h * v_fn(x + 0.5 * h * v_fn(x, t), t + 0.5 * h)
    return x


def log_likelihood(v_fn, div_fn, x1, n_steps, log_p0):
    """eq-change-of-variables: run the ODE back from x_1 at t = 1 to t = 0 with midpoint steps, carrying the integral
    of the divergence along; log p_1(x_1) = log p_0(x_0) - int_0^1 div v_t(x_t) dt."""
    h = 1.0 / n_steps
    x, acc = x1, np.zeros(len(x1))
    for i in range(n_steps):
        t = np.full(len(x), 1.0 - i * h)
        xm = x - 0.5 * h * v_fn(x, t)
        acc = acc + h * div_fn(xm, t - 0.5 * h)
        x = x - h * v_fn(xm, t - 0.5 * h)
    return log_p0(x) - acc


def log_normal(x, m, v):
    """log N(x; m, v I) per row; v is a scalar or a (B, 1) column."""
    D = x.shape[-1]
    v = np.broadcast_to(v, (len(x), 1))[:, 0]
    return -0.5 * (np.sum((x - m) ** 2, -1) / v + D * np.log(2 * math.pi * v))


def _mixture_parts(x, t, means, s, weights, sigma_min):
    t = np.broadcast_to(_c(t), (len(x), 1))
    c = 1.0 - sigma_min
    a = 1.0 - c * t
    var = a**2 + t**2 * s**2
    means = np.atleast_2d(means)
    comp = np.stack([log_normal(x, t * m, var) for m in means], 1) + np.log(weights)
    return t, c, a, var, means, comp


def mixture_log_density(x, t, means, s, weights, sigma_min=0.0):
    """log p_t(x) of the marginal path when x_0 ~ N(0, I) and x_1 ~ sum_k w_k N(m_k, s^2 I): component k becomes
    N(t m_k, ((1 - (1 - sigma_min) t)^2 + t^2 s^2) I)."""
    return np.logaddexp.reduce(_mixture_parts(x, t, means, s, weights, sigma_min)[-1], 1)


def mixture_field(x, t, means, s, weights, sigma_min=0.0):
    """The exact marginal velocity u_t(x) = E[x_1 - (1 - sigma_min) x_0 | x_t = x] for the mixture of
    mixture_log_density: a responsibility-weighted sum of each component's linear regression of the target on x_t."""
    t, c, a, var, means, comp = _mixture_parts(x, t, means, s, weights, sigma_min)
    r = np.exp(comp - np.logaddexp.reduce(comp, 1)[:, None])
    k = (t * s**2 - c * a) / var
    return np.sum([r[:, [i]] * (m + k * (x - t * m)) for i, m in enumerate(means)], 0)


def assignment(cost):
    """The permutation col minimizing sum_i cost[i, col[i]] for a square cost matrix: the Hungarian algorithm with
    potentials, O(n^3)."""
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    row, way = np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        row[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            cur = cost[row[j0] - 1] - u[row[j0]] - v[1:]
            better = ~used[1:] & (cur < minv[1:])
            minv[1:][better], way[1:][better] = cur[better], j0
            free = np.nonzero(~used[1:])[0] + 1
            j1 = free[np.argmin(minv[free])]
            delta = minv[j1]
            u[row[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if row[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            row[j0], j0 = row[j1], j1
    col = np.zeros(n, int)
    col[row[1:] - 1] = np.arange(n)
    return col


def ot_pairs(x0, x1):
    """eq-ot-coupling on a minibatch: reorder x_1 so that the pairs (x_0[i], x_1[i]) minimize the summed squared
    distance, the exact optimal transport plan between the two empirical batches."""
    cost = np.sum((x0[:, None, :] - x1[None, :, :]) ** 2, -1)
    return x0, x1[assignment(cost)]


def reflow_pairs(v_fn, x0, n_steps):
    """One reflow step of rectified flow: the coupling (Z_0, Z_1) made by running the learned ODE from Z_0 = x_0."""
    return x0, midpoint_sample(v_fn, x0, n_steps)


def logit_normal(rng, n, m=0.0, s=1.0):
    """eq-logit-normal: t = sigmoid(m + s z) with z ~ N(0, 1), which favors the middle of [0, 1] for m = 0."""
    return 1.0 / (1.0 + np.exp(-(m + s * rng.standard_normal(n))))


def adam(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] = st.get("t", 0) + 1
    for k in g:
        st[k + "m"] = b1 * st.get(k + "m", 0.0) + (1 - b1) * g[k]
        st[k + "v"] = b2 * st.get(k + "v", 0.0) + (1 - b2) * g[k] ** 2
        mh, vh = st[k + "m"] / (1 - b1 ** st["t"]), st[k + "v"] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)


def train(p, cfg, sample_data, steps, rng, batch=256, lr=3e-3, coupling="independent", t_sampler=None):
    """Conditional flow matching with Adam: x_0 ~ N(0, I), x_1 from the data, paired independently or by minibatch
    optimal transport, t uniform unless t_sampler(rng, n) is given; returns the losses."""
    st, trace = {}, []
    for _ in range(steps):
        x1 = sample_data(batch, rng)
        x0 = rng.standard_normal(x1.shape)
        if coupling == "ot":
            x0, x1 = ot_pairs(x0, x1)
        t = rng.uniform(0.0, 1.0, batch) if t_sampler is None else t_sampler(rng, batch)
        loss, g = loss_and_grads(p, cfg, x0, x1, t)
        adam(p, g, st, lr)
        trace.append(loss)
    return trace
