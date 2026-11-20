# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/consistency-model/ (ai://NEW-consistency-model).
"""Consistency models in NumPy, every gradient written by hand.

- The noise levels: the variance-exploding process x_t = x + t z of Karras and colleagues on [eps, T], their
  discretization t_i of rho = 7, and the boundary-preserving skip and output scales c_skip and c_out (eq-param).
- Exact references: for data drawn from a mixture of Gaussians N(means[k], s2 I), the score of the noisy marginal and
  its probability-flow ODE dx/dt = -t score(x, t); for one Gaussian the consistency function in closed form
  (eq-gauss-cf); for a mixture the consistency function by many Heun steps of the ODE.
- The model: f(x, t) = c_skip(t) x + c_out(t) F(x, t), with F a tanh layer of H units on the scaled input
  c_in(t) x and a Fourier embedding of log(t) / 4 (eq-param).
- Training: one Euler or Heun step of the teacher ODE (eq-teacher), the consistency distillation loss (eq-cd), the
  consistency training loss (eq-ct), the squared l2 and Pseudo-Huber metrics (eq-huber), the gradient of each loss in
  the student's parameters, the EMA of the target network (eq-ema), Adam, and the schedules of N and mu of Song and
  colleagues and of Song and Dhariwal, with the lognormal choice of the level and its weight 1 / (t_{i+1} - t_i).
- Sampling: one step, x = f(x_T, T), and the multistep sampler of Algorithm 1, which adds noise back between steps.
- The toy of Song and Dhariwal's Proposition 1: a point mass, a one-parameter model and the CT loss in closed form.
- Progressive distillation: the DDIM step of a variance-preserving process and the target of Salimans and Ho
  (eq-pd), with which one student step equals two teacher steps.

A batch of samples is a (B, D) array; noise levels t are positive floats, one per row or one for the batch.
"""

import math

import numpy as np

SIGMA_DATA, EPS, T_MAX, RHO = 0.5, 0.002, 80.0, 7.0

# ---------------------------------------------------------------------------------------------------------------------
# Noise levels and the parameterization


def karras_steps(N, eps=EPS, T=T_MAX, rho=RHO):
    """t_1 = eps < t_2 < ... < t_N = T, evenly spaced in t^(1 / rho)."""
    i = np.arange(N)
    return (eps ** (1 / rho) + i / (N - 1) * (T ** (1 / rho) - eps ** (1 / rho))) ** rho


def c_skip(t, eps=EPS, sd=SIGMA_DATA):
    """sd^2 / ((t - eps)^2 + sd^2): 1 at t = eps."""
    return sd**2 / ((t - eps) ** 2 + sd**2)


def c_out(t, eps=EPS, sd=SIGMA_DATA):
    """sd (t - eps) / sqrt(sd^2 + t^2): 0 at t = eps."""
    return sd * (t - eps) / np.sqrt(sd**2 + t**2)


def c_in(t, sd=SIGMA_DATA):
    return 1.0 / np.sqrt(t**2 + sd**2)


def _col(t, n):
    return np.broadcast_to(np.asarray(t, float), (n,))[:, None]


# ---------------------------------------------------------------------------------------------------------------------
# Exact references for a mixture of Gaussians


def score(x, t, means, s2, weights):
    """grad log p_t(x) for x0 drawn from the mixture of N(means[k], s2 I) and x = x0 + t z."""
    v = s2 + _col(t, len(x)) ** 2
    r = x[:, None, :] - means[None]
    logw = np.log(weights)[None] - 0.5 * (r**2).sum(-1) / v
    post = np.exp(logw - logw.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    return -np.einsum("bk,bkd->bd", post, r) / v


def denoise(x, t, means, s2, weights):
    """E[x0 | x_t = x] = x + t^2 score(x, t), Tweedie's formula."""
    return x + _col(t, len(x)) ** 2 * score(x, t, means, s2, weights)


def ode_drift(x, t, score_fn):
    """dx/dt of the probability-flow ODE of x_t = x0 + t z: -t score(x, t)."""
    return -_col(t, len(x)) * score_fn(x, t)


def teacher_step(x, t_from, t_to, score_fn, solver="heun"):
    """One step of the ODE from t_from to t_to: Euler, or Heun's second-order method of Karras and colleagues."""
    h = _col(t_to, len(x)) - _col(t_from, len(x))
    d1 = ode_drift(x, t_from, score_fn)
    x1 = x + h * d1
    if solver == "euler":
        return x1
    return x + 0.5 * h * (d1 + ode_drift(x1, t_to, score_fn))


def solve_ode(x, t_from, t_to, score_fn, n=200):
    """Many Heun steps from t_from down to t_to, evenly spaced in t^(1 / 7)."""
    ts = karras_steps(n + 1, eps=t_to, T=t_from)[::-1]
    for a, b in zip(ts[:-1], ts[1:], strict=True):
        x = teacher_step(x, a, b, score_fn)
    return x


def gaussian_consistency(x, t, mu, s2, eps=EPS):
    """eq-gauss-cf: the exact consistency function for data N(mu, s2 I), mu + (x - mu) sqrt((s2 + eps^2) / (s2 + t^2))."""
    return mu + (x - mu) * np.sqrt((s2 + eps**2) / (s2 + _col(t, len(x)) ** 2))


# ---------------------------------------------------------------------------------------------------------------------
# The model


def config(D, H, E):
    return {"dim": D, "hidden": H, "temb": E}


def param_shapes(cfg):
    D, H, E = cfg["dim"], cfg["hidden"], cfg["temb"]
    return {"W1": (D, H), "Wt": (E, H), "b1": (H,), "W2": (H, D), "b2": (D,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    """Multiply-adds of the three matrix products, counted as 2; scales, biases, the embedding and tanh are left out."""
    return 2 * B * (cfg["dim"] * cfg["hidden"] + cfg["temb"] * cfg["hidden"] + cfg["hidden"] * cfg["dim"])


def init(rng, cfg):
    return {k: np.zeros(s) if k.startswith("b") else rng.standard_normal(s) / math.sqrt(s[0])
            for k, s in param_shapes(cfg).items()}  # fmt: skip


def time_embedding(t, E):
    """(len(t), E): sines of c = log(t) / 4 at the frequencies 1, 2, 4, ... ((E + 1) // 2 of them), then cosines."""
    n = (E + 1) // 2
    a = (np.log(np.asarray(t, float)) / 4.0)[:, None] * 2.0 ** np.arange(n)
    return np.concatenate([np.sin(a), np.cos(a[:, : E // 2])], -1)


def _forward(p, cfg, x, t):
    tt = np.broadcast_to(np.asarray(t, float), (len(x),))
    te = time_embedding(tt, cfg["temb"])
    xin = c_in(tt)[:, None] * x
    h = np.tanh(xin @ p["W1"] + te @ p["Wt"] + p["b1"])
    F = h @ p["W2"] + p["b2"]
    return c_skip(tt)[:, None] * x + c_out(tt)[:, None] * F, (tt, te, xin, h)


def forward(p, cfg, x, t):
    """f(x, t) = c_skip(t) x + c_out(t) F(c_in(t) x, t): the model's estimate of the start of the ODE path through x."""
    return _forward(p, cfg, x, t)[0]


def _backward(p, go, cache):
    """The gradient in p of sum(go * f(x, t)), the input fixed."""
    tt, te, xin, h = cache
    gF = c_out(tt)[:, None] * go
    da = (gF @ p["W2"].T) * (1.0 - h**2)
    return {"W1": xin.T @ da, "Wt": te.T @ da, "b1": da.sum(0), "W2": h.T @ gF, "b2": gF.sum(0)}


# ---------------------------------------------------------------------------------------------------------------------
# Metrics and losses


def metric(a, b, c=None):
    """Per row: the squared l2 distance, or the Pseudo-Huber metric sqrt(|a - b|^2 + c^2) - c of eq-huber."""
    d2 = ((a - b) ** 2).sum(1)
    return d2 if c is None else np.sqrt(d2 + c**2) - c


def metric_grad(a, b, c=None):
    """The gradient of metric(a, b) in a, per row."""
    return 2.0 * (a - b) if c is None else (a - b) / np.sqrt(((a - b) ** 2).sum(1, keepdims=True) + c**2)


def consistency_loss(p, p_minus, cfg, x_hi, t_hi, x_lo, t_lo, weight: float | np.ndarray = 1.0, c=None):
    """mean lambda d(f_theta(x_hi, t_hi), f_theta-(x_lo, t_lo)): the student at the higher level against the target
    network at the lower one, with its gradient in the student's parameters p (the target is a constant)."""
    out, cache = _forward(p, cfg, x_hi, t_hi)
    tgt = forward(p_minus, cfg, x_lo, t_lo)
    w = np.broadcast_to(np.asarray(weight, float), (len(x_hi),))
    loss = float((w * metric(out, tgt, c)).mean())
    return loss, _backward(p, w[:, None] * metric_grad(out, tgt, c) / len(x_hi), cache)


def cd_pair(x0, z, t_lo, t_hi, score_fn, solver="heun"):
    """eq-teacher: x_hi = x0 + t_hi z, and x_lo one teacher ODE step from it down to t_lo."""
    x_hi = x0 + _col(t_hi, len(x0)) * z
    return x_hi, teacher_step(x_hi, t_hi, t_lo, score_fn, solver)


def ct_pair(x0, z, t_lo, t_hi):
    """eq-ct: the same data point and noise at two levels, x0 + t_hi z and x0 + t_lo z."""
    return x0 + _col(t_hi, len(x0)) * z, x0 + _col(t_lo, len(x0)) * z


def ema(p_minus, p, mu):
    """eq-ema: theta- <- mu theta- + (1 - mu) theta."""
    return {k: mu * p_minus[k] + (1.0 - mu) * p[k] for k in p}


# ---------------------------------------------------------------------------------------------------------------------
# Schedules


def cm_schedule(k, K, s0=2, s1=150, mu0=0.9):
    """N(k) and mu(k) of consistency training in Song and colleagues' Appendix C."""
    N = math.ceil(math.sqrt(k / K * ((s1 + 1) ** 2 - s0**2) + s0**2) - 1) + 1
    return N, math.exp(s0 * math.log(mu0) / N)


def ict_steps(k, K, s0=10, s1=1280):
    """N(k) of Song and Dhariwal: s0 doubled every K' iterations up to s1, plus one."""
    Kp = math.floor(K / (math.log2(math.floor(s1 / s0)) + 1))
    return min(s0 * 2 ** math.floor(k / Kp), s1) + 1


def lognormal_probs(ts, p_mean=-1.1, p_std=2.0):
    """p(i) proportional to erf((log t_{i+1} - P_mean) / (sqrt 2 P_std)) - erf((log t_i - P_mean) / (sqrt 2 P_std))."""
    e = np.array([math.erf((math.log(v) - p_mean) / (math.sqrt(2.0) * p_std)) for v in ts])
    q = e[1:] - e[:-1]
    return q / q.sum()


def ict_weight(t_lo, t_hi):
    """lambda = 1 / (t_{i+1} - t_i)."""
    return 1.0 / (np.asarray(t_hi) - np.asarray(t_lo))


# ---------------------------------------------------------------------------------------------------------------------
# Training


def adam_init(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()}, "v": {k: np.zeros_like(v) for k, v in p.items()}, "t": 0}


def adam_step(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] += 1
    for k in p:
        st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
        st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
        mh, vh = st["m"][k] / (1 - b1 ** st["t"]), st["v"][k] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
    return p


def train_cd(p, cfg, x0, score_fn, steps, rng, N=18, batch=256, lr=2e-3, mu=0.95, c=None, solver="heun"):
    """Consistency distillation, Algorithm 2: a pair from one teacher step, the student against the EMA target."""
    ts, st, pm, trace = karras_steps(N), adam_init(p), dict(p), []
    for _ in range(steps):
        x = x0[rng.integers(0, len(x0), batch)]
        n = rng.integers(0, N - 1, batch)
        x_hi, x_lo = cd_pair(x, rng.standard_normal(x.shape), ts[n], ts[n + 1], score_fn, solver)
        loss, g = consistency_loss(p, pm, cfg, x_hi, ts[n + 1], x_lo, ts[n], 1.0, c)
        trace.append(loss)
        p = adam_step(p, g, st, lr)
        pm = ema(pm, p, mu)
    return p, pm, np.array(trace)


def train_ct(p, cfg, x0, steps, rng, batch=256, lr=2e-3, s0=10, s1=160, c=None):
    """Improved consistency training of Song and Dhariwal: no EMA in the target (theta- = stopgrad theta), the doubling
    curriculum of N, lognormal levels, the weight 1 / (t_{i+1} - t_i) and the Pseudo-Huber metric."""
    st, trace = adam_init(p), []
    if c is None:
        c = 0.00054 * math.sqrt(x0.shape[1])
    for k in range(steps):
        ts = karras_steps(ict_steps(k, steps, s0, s1))
        x = x0[rng.integers(0, len(x0), batch)]
        n = rng.choice(len(ts) - 1, batch, p=lognormal_probs(ts))
        x_hi, x_lo = ct_pair(x, rng.standard_normal(x.shape), ts[n], ts[n + 1])
        loss, g = consistency_loss(p, p, cfg, x_hi, ts[n + 1], x_lo, ts[n], ict_weight(ts[n], ts[n + 1]), c)
        trace.append(loss)
        p = adam_step(p, g, st, lr)
    return p, np.array(trace)


# ---------------------------------------------------------------------------------------------------------------------
# Sampling


def sample(f, xT, taus=(), noise=None, eps=EPS, T=T_MAX):
    """Algorithm 1: x = f(x_T, T), then for each tau_k, x <- f(x + sqrt(tau_k^2 - eps^2) noise[k], tau_k), with noise
    standard normal of shape (len(taus),) + x_T.shape. One evaluation of f per level."""
    x = f(xT, T)
    for k, tau in enumerate(taus):
        assert noise is not None
        x = f(x + math.sqrt(tau**2 - eps**2) * noise[k], tau)
    return x


# ---------------------------------------------------------------------------------------------------------------------
# The toy of Proposition 1 of Song and Dhariwal


def toy_model(x, t, theta, smin):
    """f_theta(x, t) = (smin / t) x + (1 - smin / t) theta: the exact consistency function of a point mass at theta."""
    return smin / t * x + (1 - smin / t) * theta


def toy_ct_loss(theta, theta_minus, xi, ts, z):
    """The CT loss with the squared l2 metric and lambda = 1 for data at xi, averaged over the levels and the noises z,
    with ts[0] = smin; and its derivative in theta through the student only."""
    smin, lo, hi = ts[0], ts[:-1, None], ts[1:, None]
    diff = toy_model(xi + hi * z, hi, theta, smin) - toy_model(xi + lo * z, lo, theta_minus, smin)
    return float((diff**2).mean()), float((2 * diff * (1 - smin / hi)).mean())


# ---------------------------------------------------------------------------------------------------------------------
# Progressive distillation on a variance-preserving process


def vp(lam):
    """alpha and sigma of log-SNR lam, alpha^2 + sigma^2 = 1."""
    a = 1.0 / np.sqrt(1.0 + np.exp(-lam))
    return a, np.sqrt(1.0 - a**2)


def ddim_step(z, x_pred, a_t, s_t, a_s, s_s):
    """z_s = a_s x_pred + s_s (z_t - a_t x_pred) / s_t."""
    return a_s * x_pred + s_s * (z - a_t * x_pred) / s_t


def pd_target(z_t, z_s2, a_t, s_t, a_s2, s_s2):
    """eq-pd: the x-prediction at t whose single DDIM step to t'' lands on z_t'', the end of two teacher steps."""
    r = s_s2 / s_t
    return (z_s2 - r * z_t) / (a_s2 - r * a_t)
