# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/latent-diffusion-model/ (ai://NEW-latent-diffusion-model).
"""A latent diffusion model in NumPy, every gradient written by hand.

- The first stage: a linear autoencoder that cuts a signal of length N f into N patches of length f and maps each
  patch to c channels, so it downsamples by f as in eq-autoencoder; it is fitted in closed form by principal
  components of the patches. The KL penalty of a KL-regularized latent and the scale factor of eq-scale.
- Vector quantization: the nearest-codebook lookup of eq-vq, the codebook and commitment terms of eq-vq-loss with
  their straight-through gradients, the k-means step and the exponential moving average update of eq-ema.
- The second stage: the linear variance schedule, forward noising, the noise-prediction loss of eq-ldm, the exact
  noise predictor for latents drawn from a Gaussian mixture, deterministic DDIM sampling and classifier-free guidance.
- The denoiser: a one-hidden-layer network on the flattened latent with a sinusoidal time embedding, conditioned on a
  class through the cross-attention of eq-cross-attn over M learned context tokens per class, with a second tanh
  after the attention; class C is the null class of classifier-free guidance. It predicts the noise as the noisy
  latent plus a correction, because at large t the noise is almost all of z_t and DDIM multiplies any error there by
  up to 1 / sqrt(alpha_bar). Its loss gradient is written by hand and it is trained with Adam.

A batch of signals is a (B, N f) array, a batch of latents a (B, N, c) array, and the denoiser works on the flattened
latent of length D = N c.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# The first stage


def patches(x, f):
    return x.reshape(x.shape[0], -1, f)


def fit_autoencoder(x, f, c):
    """The linear autoencoder of the top c principal components of the patches of length f of the signals x."""
    P = patches(x, f).reshape(-1, f)
    mean = P.mean(0)
    _, vecs = np.linalg.eigh(np.cov(P - mean, rowvar=False, bias=True))
    We = vecs[:, ::-1][:, :c]
    return {"We": We, "be": -mean @ We, "Wd": We.T.copy(), "bd": mean}


def encode(ae, x, f):
    return patches(x, f) @ ae["We"] + ae["be"]


def decode(ae, z):
    return (z @ ae["Wd"] + ae["bd"]).reshape(z.shape[0], -1)


def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, exp(logvar)) || N(0, I)) summed over the latent, one value per row: the penalty of a KL-reg. latent."""
    return 0.5 * (mu**2 + np.exp(logvar) - 1.0 - logvar).reshape(mu.shape[0], -1).sum(1)


def latent_scale(z):
    """The standard deviation of all entries of a batch of latents, one number, as LDM estimates it (eq-scale)."""
    return float(np.sqrt(np.mean((z - z.mean()) ** 2)))


# ---------------------------------------------------------------------------------------------------------------------
# Vector quantization


def quantize(z, codebook):
    d = (z**2).sum(-1, keepdims=True) - 2.0 * z @ codebook.T + (codebook**2).sum(-1)
    idx = d.argmin(-1)
    return idx, codebook[idx]


def vq_terms(z_e, codebook, idx, beta):
    """The codebook term and beta times the commitment term of eq-vq-loss, each averaged over the latent vectors."""
    sq = ((z_e - codebook[idx]) ** 2).sum(-1).mean()
    return sq, beta * sq


def vq_grads(z_e, codebook, idx, g_zq, beta):
    """Gradients of a loss whose decoder sees z_q with gradient g_zq, plus the two terms of eq-vq-loss: the encoder gets
    g_zq copied straight through plus the commitment gradient, the codebook only the codebook-term gradient."""
    n = idx.size
    diff = (z_e - codebook[idx]).reshape(-1, codebook.shape[1])
    g_e = g_zq + 2.0 * beta * (z_e - codebook[idx]) / n
    g_cb = np.zeros_like(codebook)
    np.add.at(g_cb, idx.reshape(-1), -2.0 * diff / n)
    return g_e, g_cb


def kmeans_step(z, codebook):
    """One step of Lloyd's algorithm: assign each vector to its nearest code and move each used code to the mean."""
    z = z.reshape(-1, codebook.shape[1])
    idx, _ = quantize(z, codebook)
    out = codebook.copy()
    for k in np.unique(idx):
        out[k] = z[idx == k].mean(0)
    return out


def ema_init(codebook):
    return {"N": np.zeros(len(codebook)), "m": np.zeros_like(codebook), "e": codebook.copy()}


def ema_update(state, z_e, idx, gamma):
    """eq-ema: decayed counts N and sums m of the vectors assigned to each code; a code with N = 0 is left as it was."""
    z, idx = z_e.reshape(-1, state["e"].shape[1]), idx.reshape(-1)
    n = np.bincount(idx, minlength=len(state["e"])).astype(float)
    s = np.zeros_like(state["m"])
    np.add.at(s, idx, z)
    N = gamma * state["N"] + (1 - gamma) * n
    m = gamma * state["m"] + (1 - gamma) * s
    e = np.where(N[:, None] > 0, m / np.where(N > 0, N, 1.0)[:, None], state["e"])
    return {"N": N, "m": m, "e": e}


# ---------------------------------------------------------------------------------------------------------------------
# Diffusion in the latent space


def alpha_bars(T, b0=1e-4, b1=2e-2):
    return np.cumprod(1.0 - np.linspace(b0, b1, T))


def q_sample(z0, abar, eps):
    a = np.reshape(abar, (-1,) + (1,) * (z0.ndim - 1))
    return np.sqrt(a) * z0 + np.sqrt(1.0 - a) * eps


def snr(abar, var=1.0):
    return abar * var / (1.0 - abar)


def exact_eps(zt, abar, means, s2, weights):
    """E[eps | z_t] when z0 is a mixture of N(means[k], s2 I) with the given weights: the best noise predictor."""
    a = np.reshape(abar, (-1, 1))
    v = a * s2 + 1.0 - a
    r = zt[:, None, :] - np.sqrt(a)[:, None] * means[None]
    logw = np.log(weights)[None] - 0.5 * (r**2).sum(-1) / v
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    z0 = means[None] + (np.sqrt(a) * s2 / v)[:, None] * r
    return (zt - np.sqrt(a) * np.einsum("bk,bkd->bd", w, z0)) / np.sqrt(1.0 - a)


def ddim_sample(eps_fn, zT, abar, steps):
    """Deterministic DDIM (eta = 0) from z_T at the times steps[0] > steps[1] > ... down to a clean latent."""
    z = zT
    for i, t in enumerate(steps):
        a = abar[t]
        a_prev = abar[steps[i + 1]] if i + 1 < len(steps) else 1.0
        e = eps_fn(z, t)
        z0 = (z - math.sqrt(1.0 - a) * e) / math.sqrt(a)
        z = math.sqrt(a_prev) * z0 + math.sqrt(1.0 - a_prev) * e
    return z


def guided(e_cond, e_null, s):
    return e_null + s * (e_cond - e_null)


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(N, c, H, E, M, d, C):
    return {"positions": N, "channels": c, "hidden": H, "temb": E, "ctx_tokens": M, "attn": d, "classes": C}


def param_shapes(cfg):
    D, H, E, M, d, C = (cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"],
                        cfg["attn"], cfg["classes"])  # fmt: skip
    return {"W1": (D, H), "b1": (H,), "Wt": (E, H), "ctx": (C + 1, M, d), "Wq": (H, d), "Wk": (d, d),
            "Wv": (d, H), "W2": (H, D), "b2": (D,)}  # fmt: skip


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    D, H, E, M, d = cfg["positions"] * cfg["channels"], cfg["hidden"], cfg["temb"], cfg["ctx_tokens"], cfg["attn"]
    return 2 * B * (D * H + E * H + H * d + M * d * d + M * d * H + M * d + M * H + H * D)


def init(rng, cfg):
    out = {}
    for k, s in param_shapes(cfg).items():
        out[k] = np.zeros(s) if k.startswith("b") else rng.standard_normal(s) / math.sqrt(s[-2] if k != "ctx" else 1)
    return out


def time_embedding(t, E):
    w = np.exp(-math.log(10000.0) * np.arange(E // 2) / (E // 2))
    a = np.asarray(t, float)[:, None] * w
    return np.concatenate([np.sin(a), np.cos(a)], -1)


def _forward(p, cfg, z, t, y):
    B, d = z.shape[0], cfg["attn"]
    zf = z.reshape(B, -1)
    te = time_embedding(np.broadcast_to(t, (B,)), cfg["temb"])
    h = np.tanh(zf @ p["W1"] + te @ p["Wt"] + p["b1"])
    q, Cx = h @ p["Wq"], p["ctx"][y]
    K, V = Cx @ p["Wk"], Cx @ p["Wv"]
    s = np.einsum("bd,bmd->bm", q, K) / math.sqrt(d)
    w = np.exp(s - s.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    h2 = np.tanh(h + np.einsum("bm,bmh->bh", w, V))
    return zf + h2 @ p["W2"] + p["b2"], (zf, te, h, q, Cx, K, V, w, h2)


def forward(p, cfg, z, t, y):
    """The predicted noise, flattened to (B, D), for latents z at integer times t and classes y."""
    return _forward(p, cfg, z, t, y)[0]


def eps_loss(p, cfg, zt, t, y, eps):
    return float(((forward(p, cfg, zt, t, y) - eps.reshape(len(eps), -1)) ** 2).sum(1).mean())


def eps_loss_grad(p, cfg, zt, t, y, eps):
    out, (zf, te, h, q, Cx, K, V, w, h2) = _forward(p, cfg, zt, t, y)
    sd = math.sqrt(cfg["attn"])
    go = 2.0 * (out - eps.reshape(len(eps), -1)) / len(eps)
    g = {"W2": h2.T @ go, "b2": go.sum(0)}
    gh = (go @ p["W2"].T) * (1.0 - h2**2)
    dw = np.einsum("bh,bmh->bm", gh, V)
    dV = w[:, :, None] * gh[:, None, :]
    ds = w * (dw - (w * dw).sum(1, keepdims=True))
    dq, dK = np.einsum("bm,bmd->bd", ds, K) / sd, ds[:, :, None] * q[:, None, :] / sd
    g["Wq"], g["Wk"], g["Wv"] = h.T @ dq, np.einsum("bmd,bme->de", Cx, dK), np.einsum("bmd,bmh->dh", Cx, dV)
    g["ctx"] = np.zeros_like(p["ctx"])
    np.add.at(g["ctx"], y, dK @ p["Wk"].T + dV @ p["Wv"].T)
    da = (gh + dq @ p["Wq"].T) * (1.0 - h**2)
    g["W1"], g["Wt"], g["b1"] = zf.T @ da, te.T @ da, da.sum(0)
    return g


def adam_init(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()}, "v": {k: np.zeros_like(v) for k, v in p.items()}, "t": 0}


def adam_step(p, g, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    s["t"] += 1
    for k in p:
        s["m"][k] = b1 * s["m"][k] + (1 - b1) * g[k]
        s["v"][k] = b2 * s["v"][k] + (1 - b2) * g[k] ** 2
        mh, vh = s["m"][k] / (1 - b1 ** s["t"]), s["v"][k] / (1 - b2 ** s["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
    return p


def train(p, cfg, z0, y, abar, steps, rng, batch=256, lr=3e-3, p_null=0.1):
    """Adam on eq-ldm with random times and noise, dropping the class to the null class with probability p_null."""
    s, trace = adam_init(p), []
    for _ in range(steps):
        i = rng.integers(0, len(z0), batch)
        t = rng.integers(0, len(abar), batch)
        eps = rng.standard_normal(z0[i].shape)
        yi = np.where(rng.random(batch) < p_null, cfg["classes"], y[i])
        zt = q_sample(z0[i], abar[t], eps)
        trace.append(eps_loss(p, cfg, zt, t, yi, eps))
        p = adam_step(p, eps_loss_grad(p, cfg, zt, t, yi, eps), s, lr)
    return p, np.array(trace)
