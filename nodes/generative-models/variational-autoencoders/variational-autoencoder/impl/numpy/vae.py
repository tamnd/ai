# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""The variational autoencoder of the page in NumPy, with the backward pass written out.

The encoder is the Gaussian MLP of Kingma and Welling (2013), appendix C: h = tanh(x W + b), and the mean mu and the
log-variance log sigma^2 of a diagonal Gaussian q(z|x) are two affine maps of h. The prior is N(0, I). The decoder is
an MLP with one tanh layer, g = tanh(z V + c), and gives either the logits of independent Bernoulli pixels or the mean
and the log-variance of a diagonal Gaussian. A code is drawn as z = mu + sigma * eps with eps ~ N(0, I), the
reparameterization trick, so that the loss is a differentiable function of the parameters for a fixed eps.

Three objectives share one backward pass: "elbo", the negative ELBO with the KL term in closed form and the
reconstruction term averaged over the S draws (estimator B of the paper, eq. 7), "elbo_mc", the negative of the average
of the S log-weights log p(x, z) - log q(z|x) (estimator A, eq. 6), and "iwae", the negative importance weighted bound
of Burda et al. (2015) with k = S samples. Rows are examples: x is (B, D), eps is (S, B, J), and a layer computes
h @ W + b with W of shape (n_in, n_out).
"""

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))
DECODERS = ("bernoulli", "gaussian")
OBJECTIVES = ("elbo", "elbo_mc", "iwae")


def config(d_in=6, d_hidden=5, d_latent=2, decoder="bernoulli"):
    """The model of the shape contract: D inputs, one tanh layer of H units on each side, J latent dimensions."""
    assert decoder in DECODERS, decoder
    assert min(d_in, d_hidden, d_latent) >= 1
    return {"d_in": int(d_in), "d_hidden": int(d_hidden), "d_latent": int(d_latent), "decoder": decoder}


def param_shapes(cfg):
    D, H, J = cfg["d_in"], cfg["d_hidden"], cfg["d_latent"]
    s = {"enc_W": (D, H), "enc_b": (H,), "mu_W": (H, J), "mu_b": (J,), "lv_W": (H, J), "lv_b": (J,),
         "dec_W": (J, H), "dec_b": (H,), "out_W": (H, D), "out_b": (D,)}  # fmt: skip
    if cfg["decoder"] == "gaussian":
        s |= {"olv_W": (H, D), "olv_b": (D,)}  # the log-variance of each pixel
    return s


def param_count(cfg):
    """eq-params: (D + 1) H + 2 (H + 1) J + (J + 1) H + (H + 1) D, plus (H + 1) D for a Gaussian decoder."""
    return sum(int(np.prod(s)) for s in param_shapes(cfg).values())


def flops_forward(cfg, B, S=1):
    """eq-flops: the matrix products of one forward pass of a batch of B with S draws, a multiply-add as 2 FLOPs."""
    D, H, J = cfg["d_in"], cfg["d_hidden"], cfg["d_latent"]
    heads = 2 if cfg["decoder"] == "gaussian" else 1
    return 2 * B * D * H + 4 * B * H * J + S * (2 * B * J * H + heads * 2 * B * H * D)


def init(rng, cfg, scale=1.0):
    """Weights drawn from N(0, scale^2 / n_in), biases 0."""
    p = {}
    for k, s in param_shapes(cfg).items():
        p[k] = rng.normal(0.0, scale / np.sqrt(s[0]), s) if len(s) == 2 else np.zeros(s)
    return p


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # the logistic function, without overflow for large |a|


def encode(p, x):
    """The mean and the log-variance of q(z|x), each (B, J), and the hidden layer h."""
    h = np.tanh(x @ p["enc_W"] + p["enc_b"])
    return h @ p["mu_W"] + p["mu_b"], h @ p["lv_W"] + p["lv_b"], h


def decode(p, cfg, z):
    """The parameters of p(x|z) for codes z of shape (..., J): the logits, or the mean and the log-variance."""
    lead = z.shape[:-1]
    g = np.tanh(z.reshape(-1, z.shape[-1]) @ p["dec_W"] + p["dec_b"])
    out = {"g": g.reshape(*lead, -1), "a": (g @ p["out_W"] + p["out_b"]).reshape(*lead, -1)}
    if cfg["decoder"] == "gaussian":
        out["olv"] = (g @ p["olv_W"] + p["olv_b"]).reshape(*lead, -1)
    return out


def forward(p, cfg, x, eps):
    """The encoder, the S reparameterized codes z = mu + sigma * eps and the decoder; eps is (S, B, J)."""
    mu, lv, h = encode(p, x)
    std = np.exp(0.5 * lv)
    z = mu + std * eps
    out = {"mu": mu, "lv": lv, "z": z} | decode(p, cfg, z)
    return out, {"x": x, "h": h, "std": std, "eps": eps}


def nll(cfg, x, out):
    """-log p(x|z) of each draw, (S, B): the Bernoulli cross-entropy from the logits, or the Gaussian negative
    log-density."""
    a = out["a"]
    if cfg["decoder"] == "bernoulli":
        return np.sum(np.logaddexp(0.0, a) - x * a, -1)  # softplus(a) - x a
    return 0.5 * np.sum(LOG2PI + out["olv"] + (x - a) ** 2 * np.exp(-out["olv"]), -1)


def kl(mu, lv):
    """eq-kl: KL(N(mu, diag exp(lv)) || N(0, I)) of each row, in closed form."""
    return 0.5 * np.sum(mu**2 + np.exp(lv) - 1.0 - lv, -1)


def log_prior(z):
    return -0.5 * np.sum(LOG2PI + z**2, -1)


def log_q(lv, eps):
    """log q(z|x) at z = mu + sigma * eps, written in eps: the density of the draw under its own encoder."""
    return -0.5 * np.sum(LOG2PI + lv + eps**2, -1)


def log_weights(cfg, x, out, eps):
    """log w = log p(x|z) + log p(z) - log q(z|x) of each draw, (S, B)."""
    return -nll(cfg, x, out) + log_prior(out["z"]) - log_q(out["lv"], eps)


def logmeanexp(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def objective(p, cfg, x, eps, kind="elbo", beta=1.0):
    """The value of an objective, averaged over the batch, with the pieces needed by the backward pass."""
    assert kind in OBJECTIVES, kind
    out, cache = forward(p, cfg, x, eps)
    S, B = eps.shape[0], x.shape[0]
    if kind == "elbo":
        rate, dist = kl(out["mu"], out["lv"]), nll(cfg, x, out)
        value = float(np.mean(dist) + beta * np.mean(rate))
        coef = {"nll": np.full((S, B), 1.0 / (S * B)), "logp": np.zeros((S, B)), "logq": np.zeros((S, B)),
                "kl": np.full(B, beta / B)}  # fmt: skip
        return value, out, cache, coef
    lw = log_weights(cfg, x, out, eps)
    if kind == "elbo_mc":
        value, dlw = -float(np.mean(lw)), np.full((S, B), -1.0 / (S * B))
    else:
        value = -float(np.mean(logmeanexp(lw)))
        wn = np.exp(lw - np.max(lw, 0))
        dlw = -(wn / wn.sum(0)) / B  # the normalized importance weights, eq. 13 of Burda et al.
    # lw = -nll + log p(z) - log q, so the coefficient of each piece in the value follows from dlw
    return value, out, cache, {"nll": -dlw, "logp": dlw, "logq": -dlw, "kl": np.zeros(B)}


def backward(p, cfg, out, cache, coef):
    """The gradients of sum(coef.nll * nll) + sum(coef.logp * log p(z)) + sum(coef.logq * log q) + sum(coef.kl * kl)
    with respect to every parameter, by the chain rule through the decoder, the reparameterization and the encoder."""
    x, h, std, eps = cache["x"], cache["h"], cache["std"], cache["eps"]
    z, g, a = out["z"], out["g"], out["a"]
    S, B, J = eps.shape
    c = coef["nll"][..., None]
    grads = {}
    if cfg["decoder"] == "bernoulli":
        da = c * (sigmoid(a) - x)
    else:
        r2 = (x - a) ** 2 * np.exp(-out["olv"])
        da = c * (a - x) * np.exp(-out["olv"])
        dolv = c * 0.5 * (1.0 - r2)
        grads["olv_W"] = g.reshape(S * B, -1).T @ dolv.reshape(S * B, -1)
        grads["olv_b"] = dolv.sum((0, 1))
    g2, da2 = g.reshape(S * B, -1), da.reshape(S * B, -1)
    grads["out_W"], grads["out_b"] = g2.T @ da2, da2.sum(0)
    dg = da2 @ p["out_W"].T
    if cfg["decoder"] == "gaussian":
        dg = dg + dolv.reshape(S * B, -1) @ p["olv_W"].T
    dpre = dg * (1.0 - g2**2)
    grads["dec_W"], grads["dec_b"] = z.reshape(S * B, J).T @ dpre, dpre.sum(0)
    dz = (dpre @ p["dec_W"].T).reshape(S, B, J) - coef["logp"][..., None] * z  # d log p(z) / dz = -z
    # z = mu + std * eps; log q in eps depends on lv only, d log q / d lv = -1/2; kl has d/dmu = mu, d/dlv = (e^lv - 1)/2
    dmu = dz.sum(0) + coef["kl"][:, None] * out["mu"]
    dlv = (0.5 * std * (dz * eps).sum(0) - 0.5 * coef["logq"].sum(0)[:, None]
           + coef["kl"][:, None] * 0.5 * (np.exp(out["lv"]) - 1.0))  # fmt: skip
    grads["mu_W"], grads["mu_b"] = h.T @ dmu, dmu.sum(0)
    grads["lv_W"], grads["lv_b"] = h.T @ dlv, dlv.sum(0)
    dh = dmu @ p["mu_W"].T + dlv @ p["lv_W"].T
    dpre = dh * (1.0 - h**2)
    grads["enc_W"], grads["enc_b"] = x.T @ dpre, dpre.sum(0)
    return grads


def loss(p, cfg, x, eps, kind="elbo", beta=1.0):
    """The value of the objective and its gradients."""
    value, out, cache, coef = objective(p, cfg, x, eps, kind, beta)
    return value, backward(p, cfg, out, cache, coef)


def rate_distortion(p, cfg, x, eps):
    """The rate, the mean KL from the encoder to the prior, and the distortion, the mean reconstruction negative
    log-likelihood: the negative ELBO is their sum (Alemi et al., 2018)."""
    out, _ = forward(p, cfg, x, eps)
    return float(np.mean(kl(out["mu"], out["lv"]))), float(np.mean(nll(cfg, x, out)))


def iwae_bound(p, cfg, x, rng, k):
    """A draw of the k-sample bound L_k of each row of x; L_1 is a draw of the ELBO."""
    out, _ = forward(p, cfg, x, eps := rng.standard_normal((k, x.shape[0], cfg["d_latent"])))
    return logmeanexp(log_weights(cfg, x, out, eps))


def active_units(p, x, threshold=1e-2):
    """The activity A_u = Cov_x(E_q[u]) of each latent dimension u, and the dimensions with A_u above the threshold
    (Burda et al., 2015)."""
    mu = encode(p, x)[0]
    a = np.var(mu, 0)
    return a, np.flatnonzero(a > threshold)


def sample(p, cfg, rng, n):
    """n samples of the model: z from the prior, then the mean of p(x|z), the pixel probabilities for Bernoulli."""
    out = decode(p, cfg, rng.standard_normal((n, cfg["d_latent"])))
    return sigmoid(out["a"]) if cfg["decoder"] == "bernoulli" else out["a"]


def adam(p, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] = state.get("t", 0) + 1
    for k, g in grads.items():
        m = state.setdefault("m_" + k, np.zeros_like(g))
        v = state.setdefault("v_" + k, np.zeros_like(g))
        m[...] = b1 * m + (1 - b1) * g
        v[...] = b2 * v + (1 - b2) * g * g
        p[k] -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)


def train(p, cfg, data, rng, steps, lr=1e-2, batch=50, kind="elbo", S=1, beta=1.0, anneal=0):
    """Adam on minibatches; with anneal > 0 the weight of the KL term rises linearly from 0 to beta over that many
    steps (KL cost annealing, Bowman et al., 2016). Returns the objective of each step."""
    state, trace = {}, []
    for t in range(steps):
        xb = data[rng.integers(0, len(data), batch)]
        eps = rng.standard_normal((S, batch, cfg["d_latent"]))
        b = beta * min(1.0, t / anneal) if anneal else beta
        value, grads = loss(p, cfg, xb, eps, kind, b)
        adam(p, grads, state, lr)
        trace.append(value)
    return trace
