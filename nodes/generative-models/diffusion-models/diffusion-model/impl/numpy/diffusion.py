# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/diffusion-model/ (ai://NEW-diffusion-model).
"""The denoising diffusion model of the page in NumPy: the forward process with a linear variance schedule
(eq-forward-step, eq-forward-marginal), its posterior (eq-posterior), a noise-prediction network trained on the simple
loss of eq-simple-loss with every backward pass written by hand, the ancestral sampler of eq-ddpm-step, the
deterministic and strided sampler of eq-ddim-step, the variational bound of eq-elbo, the score of eq-tweedie and
classifier-free guidance (eq-guidance).

The network is one tanh hidden layer on the concatenation of x_t and a sinusoidal embedding of t: small enough to
check by hand and to train in seconds on two-dimensional data. Time steps are integers t = 1..T; arrays indexed by
time hold step t at index t - 1, and alpha_bar_0 = 1. For data that is a mixture of isotropic Gaussians the exact
noise prediction E[eps | x_t] is known in closed form; mixture_eps gives it, and the tests drive the samplers with it.
"""

import math

import numpy as np

PARTS = ("W1", "b1", "W2", "b2")


def config(d_data=3, d_hidden=5, d_time=4, n_steps=10, beta_1=0.02, beta_T=0.5):
    if min(d_data, d_hidden, d_time, n_steps) < 1:
        raise ValueError("every size must be positive")
    if not 0 < beta_1 <= beta_T < 1:
        raise ValueError("the variances must satisfy 0 < beta_1 <= beta_T < 1")
    return {"d_data": int(d_data), "d_hidden": int(d_hidden), "d_time": int(d_time), "n_steps": int(n_steps),
            "beta_1": float(beta_1), "beta_T": float(beta_T)}  # fmt: skip


def schedule(cfg):
    """The linear schedule of Ho et al. (2020): beta_t from beta_1 to beta_T, alpha_t = 1 - beta_t, alpha_bar_t their
    running product, alpha_bar_prev_t = alpha_bar_{t-1} with alpha_bar_0 = 1, and the posterior variance beta_tilde_t
    of eq-posterior, which is 0 at t = 1."""
    betas = np.linspace(cfg["beta_1"], cfg["beta_T"], cfg["n_steps"])
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    return {"beta": betas, "alpha": alphas, "abar": abar, "abar_prev": abar_prev,
            "beta_tilde": (1.0 - abar_prev) / (1.0 - abar) * betas}  # fmt: skip


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
    """W1 random, the rest 0: the initial network predicts eps = 0 everywhere."""
    p = {k: np.zeros(s) for k, s in param_shapes(cfg).items()}
    p["W1"] = rng.standard_normal(p["W1"].shape) / math.sqrt(p["W1"].shape[0])
    return p


def time_embedding(t, d_time):
    """eq-time-embedding, the sinusoids of the Transformer: entry 2i is sin(t w_i), entry 2i + 1 is cos(t w_i), with
    w_i = 10000^(-2i / d_time). t is an integer array (B,); the result is (B, d_time)."""
    i = np.arange(d_time)
    w = 10000.0 ** (-2.0 * (i // 2) / d_time)
    a = np.asarray(t, float)[:, None] * w
    return np.where(i % 2 == 0, np.sin(a), np.cos(a))


def eps_model(p, cfg, x, t):
    """eps_theta(x_t, t): h = tanh([x_t, e(t)] W1 + b1), eps = h W2 + b2. Returns the prediction and the cache."""
    inp = np.concatenate([x, time_embedding(t, cfg["d_time"])], 1)
    h = np.tanh(inp @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"], (inp, h)


def _col(v, t):
    """Values of a per-step array at integer steps t (B,), as a (B, 1) column."""
    return v[np.asarray(t) - 1][:, None]


def forward_step(cfg, x_prev, t, z):
    """eq-forward-step: x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) z."""
    b = _col(schedule(cfg)["beta"], t)
    return np.sqrt(1.0 - b) * x_prev + np.sqrt(b) * z


def q_sample(cfg, x0, t, eps):
    """eq-forward-marginal: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps in one step."""
    ab = _col(schedule(cfg)["abar"], t)
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def posterior(cfg, xt, x0, t):
    """eq-posterior: the mean and variance of q(x_{t-1} | x_t, x_0)."""
    s = schedule(cfg)
    b, a, ab, abp = (_col(s[k], t) for k in ("beta", "alpha", "abar", "abar_prev"))
    mean = np.sqrt(abp) * b / (1.0 - ab) * x0 + np.sqrt(a) * (1.0 - abp) / (1.0 - ab) * xt
    return mean, _col(s["beta_tilde"], t)


def predict_x0(cfg, xt, t, eps):
    """x_0 = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t), eq-forward-marginal solved for x_0."""
    ab = _col(schedule(cfg)["abar"], t)
    return (xt - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)


def score_from_eps(cfg, eps, t):
    """eq-tweedie: the score of the noisy marginal is -eps / sqrt(1 - alpha_bar_t) at the exact eps."""
    return -eps / np.sqrt(1.0 - _col(schedule(cfg)["abar"], t))


def loss_and_grads(p, cfg, x0, t, eps):
    """eq-simple-loss on one batch: the mean over rows of ||eps - eps_theta(x_t, t)||^2, and its gradients."""
    B = len(x0)
    out, (inp, h) = eps_model(p, cfg, q_sample(cfg, x0, t, eps), t)
    r = out - eps
    loss = float(np.sum(r**2) / B)
    go = 2.0 * r / B
    gh = (go @ p["W2"].T) * (1.0 - h**2)
    g = {"W2": h.T @ go, "b2": go.sum(0), "W1": inp.T @ gh, "b1": gh.sum(0)}
    return loss, g


def sigma2(cfg, t, var="beta"):
    """The reverse-process variance sigma_t^2: beta_t, or beta_tilde_t for t > 1 (and beta_1 at t = 1)."""
    s = schedule(cfg)
    if var == "beta":
        return _col(s["beta"], t)
    if var == "beta_tilde":
        return np.where(np.asarray(t)[:, None] > 1, _col(s["beta_tilde"], t), _col(s["beta"], t))
    raise ValueError(var)


def reverse_mean(cfg, xt, t, eps):
    """eq-ddpm-step without the noise: mu_theta = (x_t - beta_t / sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_t)."""
    s = schedule(cfg)
    b, a, ab = (_col(s[k], t) for k in ("beta", "alpha", "abar"))
    return (xt - b / np.sqrt(1.0 - ab) * eps) / np.sqrt(a)


def ddpm_step(cfg, xt, t, eps, z, var="beta"):
    """eq-ddpm-step: x_{t-1} = mu_theta(x_t, t) + sigma_t z."""
    return reverse_mean(cfg, xt, t, eps) + np.sqrt(sigma2(cfg, t, var)) * z


def ddpm_sample(eps_fn, cfg, n, rng, var="beta", x_T=None):
    """Algorithm 2 of Ho et al.: x_T ~ N(0, I), then T ancestral steps, without noise at t = 1."""
    x = rng.standard_normal((n, cfg["d_data"])) if x_T is None else x_T
    for t in range(cfg["n_steps"], 0, -1):
        tt = np.full(len(x), t)
        z = rng.standard_normal(x.shape) if t > 1 else np.zeros_like(x)
        x = ddpm_step(cfg, x, tt, eps_fn(x, tt), z, var)
    return x


def _abar_at(cfg, s):
    """alpha_bar at integer steps s (B,), with alpha_bar_0 = 1, as a (B, 1) column."""
    ab = np.concatenate([[1.0], schedule(cfg)["abar"]])
    return ab[np.asarray(s)][:, None]


def ddim_sigma(cfg, t, s, eta):
    """eq-ddim-sigma: the standard deviation of a step from t to s < t; eta = 1 is the DDPM posterior's, eta = 0 none."""
    at, as_ = _abar_at(cfg, t), _abar_at(cfg, s)
    return eta * np.sqrt((1.0 - as_) / (1.0 - at) * (1.0 - at / as_))


def ddim_step(cfg, xt, t, s, eps, z, eta=0.0):
    """eq-ddim-step from step t to step s < t: predicted x_0, the direction pointing to x_t, and fresh noise."""
    at, as_ = _abar_at(cfg, t), _abar_at(cfg, s)
    sig = ddim_sigma(cfg, t, s, eta)
    x0 = (xt - np.sqrt(1.0 - at) * eps) / np.sqrt(at)
    return np.sqrt(as_) * x0 + np.sqrt(1.0 - as_ - sig**2) * eps + sig * z


def ddim_steps(cfg, n_sample):
    """The increasing sub-sequence tau of n_sample steps out of 1..T, evenly spaced and ending at T."""
    return np.unique(np.round(np.linspace(1, cfg["n_steps"], n_sample)).astype(int))


def ddim_sample(eps_fn, cfg, n, rng, n_sample=None, eta=0.0, x_T=None):
    """Sample along tau from x_T ~ N(0, I) to x_0; with eta = 0 the map from x_T to x_0 is deterministic."""
    tau = ddim_steps(cfg, cfg["n_steps"] if n_sample is None else n_sample)
    x = rng.standard_normal((n, cfg["d_data"])) if x_T is None else x_T
    for t, s in zip(tau[::-1], np.concatenate([[0], tau[:-1]])[::-1], strict=True):
        tt, ss = np.full(len(x), t), np.full(len(x), s)
        z = rng.standard_normal(x.shape) if eta > 0 else np.zeros_like(x)
        x = ddim_step(cfg, x, tt, ss, eps_fn(x, tt), z, eta)
    return x


def guided_eps(eps_cond, eps_uncond, w):
    """eq-guidance: (1 + w) eps_theta(x, c) - w eps_theta(x); w = 0 is the conditional model."""
    return (1.0 + w) * eps_cond - w * eps_uncond


def gauss_kl(m1, v1, m2, v2):
    """KL(N(m1, v1 I) || N(m2, v2 I)) per row, for means (B, D) and variances broadcastable to (B, 1)."""
    D = m1.shape[-1]
    v1, v2 = (
        np.broadcast_to(v1, (len(m1), 1))[:, 0],
        np.broadcast_to(v2, (len(m1), 1))[:, 0],
    )
    return 0.5 * (D * (v1 / v2 - 1.0 - np.log(v1 / v2)) + np.sum((m1 - m2) ** 2, -1) / v2)


def log_normal(x, m, v):
    """log N(x; m, v I) per row."""
    D = x.shape[-1]
    v = np.broadcast_to(v, (len(x), 1))[:, 0]
    return -0.5 * (np.sum((x - m) ** 2, -1) / v + D * np.log(2 * math.pi * v))


def neg_elbo(eps_fn, cfg, x0, rng, var="beta"):
    """eq-elbo per row, one Monte Carlo draw of x_t ~ q(x_t | x_0) per term: L_T + sum_{t>1} L_{t-1} + L_0, in nats.
    L_0 is -log N(x_0; mu_theta(x_1, 1), beta_1 I), a Gaussian decoder for continuous data."""
    s = schedule(cfg)
    B, T = len(x0), cfg["n_steps"]
    abT = s["abar"][-1]
    total = gauss_kl(
        math.sqrt(abT) * x0,
        np.full((B, 1), 1.0 - abT),
        np.zeros_like(x0),
        np.ones((B, 1)),
    )
    for t in range(2, T + 1):
        tt = np.full(B, t)
        xt = q_sample(cfg, x0, tt, rng.standard_normal(x0.shape))
        mq, vq = posterior(cfg, xt, x0, tt)
        total = total + gauss_kl(mq, vq, reverse_mean(cfg, xt, tt, eps_fn(xt, tt)), sigma2(cfg, tt, var))
    one = np.ones(B, int)
    x1 = q_sample(cfg, x0, one, rng.standard_normal(x0.shape))
    return total - log_normal(x0, reverse_mean(cfg, x1, one, eps_fn(x1, one)), np.full((B, 1), s["beta"][0]))


def mixture_log_density(x, abar, means, s, weights):
    """log q_t(x) of the noisy marginal when x_0 is the mixture sum_k w_k N(m_k, s^2 I): each component becomes
    N(sqrt(abar) m_k, (abar s^2 + 1 - abar) I). abar is a scalar or a (B, 1) column."""
    abar = np.broadcast_to(np.asarray(abar, float), (len(x), 1))
    v = abar * s**2 + 1.0 - abar
    comp = np.stack([log_normal(x, np.sqrt(abar) * m, v) for m in np.atleast_2d(means)], 1)
    return np.logaddexp.reduce(comp + np.log(weights), 1)


def mixture_eps(x, abar, means, s, weights):
    """The exact noise prediction E[eps | x_t] = -sqrt(1 - abar) grad log q_t(x_t) for the mixture of
    mixture_log_density: a responsibility-weighted sum of each component's linear prediction."""
    abar = np.broadcast_to(np.asarray(abar, float), (len(x), 1))
    v = abar * s**2 + 1.0 - abar
    means = np.atleast_2d(means)
    comp = np.stack([log_normal(x, np.sqrt(abar) * m, v) for m in means], 1) + np.log(weights)
    r = np.exp(comp - np.logaddexp.reduce(comp, 1)[:, None])
    score = sum(r[:, [k]] * (np.sqrt(abar) * m - x) / v for k, m in enumerate(means))
    return -np.sqrt(1.0 - abar) * score


def adam(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] = st.get("t", 0) + 1
    for k in g:
        st[k + "m"] = b1 * st.get(k + "m", 0.0) + (1 - b1) * g[k]
        st[k + "v"] = b2 * st.get(k + "v", 0.0) + (1 - b2) * g[k] ** 2
        mh, vh = st[k + "m"] / (1 - b1 ** st["t"]), st[k + "v"] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)


def train(p, cfg, sample_data, steps, rng, batch=256, lr=3e-3):
    """Algorithm 1 of Ho et al. with Adam: x_0 from the data, t uniform on 1..T, eps ~ N(0, I); returns the losses."""
    st, trace = {}, []
    for _ in range(steps):
        x0 = sample_data(batch, rng)
        t = rng.integers(1, cfg["n_steps"] + 1, batch)
        loss, g = loss_and_grads(p, cfg, x0, t, rng.standard_normal(x0.shape))
        adam(p, g, st, lr)
        trace.append(loss)
    return trace
