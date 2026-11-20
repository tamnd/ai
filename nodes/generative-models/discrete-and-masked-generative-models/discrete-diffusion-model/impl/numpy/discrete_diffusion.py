# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Discrete diffusion in NumPy, every gradient written by hand.

- The D3PM forward process of eq-d3pm-forward with the uniform and the absorbing transition matrices, their cumulative
  products, the posterior of eq-d3pm-posterior, the x0-parameterized reverse step of eq-x0-param, which for uniform
  noise is the multinomial diffusion posterior of Hoogeboom and colleagues, and the exact variational bound of eq-vlb
  for one token by enumeration.
- Masked (absorbing) diffusion in continuous time: the linear and cosine schedules, the masking of eq-mask-marginal,
  the posterior of eq-mask-posterior, the SUBS reverse step of eq-subs, the weighted cross-entropy bound of eq-mdlm, its
  Monte Carlo estimate, its exact value by enumerating masks, and the ancestral sampler with the exact distribution of
  its samples.
- The confidence-ordered parallel decoding of MaskGIT with the cosine mask schedule.
- Score entropy and denoising score entropy (eq-score-entropy), and the concrete score of the absorbing process.
- A denoiser: a one-hidden-layer network from the one-hot partly masked sequence to logits over the K data tokens at
  every position, trained by the exact masked-diffusion bound with Adam.

Tokens are integers. Data tokens are 0, ..., K - 1 and the mask is the integer K. A sequence of L tokens is a row, a
batch of B sequences is a (B, L) integer array, and probability vectors over the K + 1 states are rows.
"""

import itertools
import math

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# D3PM: transition matrices, marginals and posteriors of one token


def uniform_q(beta, K):
    """Q_t = (1 - beta) I + beta / K 1 1^T: with probability beta the token is redrawn uniformly from the K states."""
    return (1.0 - beta) * np.eye(K) + beta / K * np.ones((K, K))


def absorbing_q(beta, K):
    """Q_t = (1 - beta) I + beta 1 e_m^T over K data states and the mask m = K: with probability beta the token jumps to
    the mask, and the mask stays."""
    Q = (1.0 - beta) * np.eye(K + 1)
    Q[:, K] += beta
    return Q


def cumulative(Qs):
    """The products Qbar_t = Q_1 Q_2 ... Q_t for t = 0, ..., T, with Qbar_0 = I."""
    out = [np.eye(Qs[0].shape[0])]
    for Q in Qs:
        out.append(out[-1] @ Q)
    return out


def uniform_qbar(alpha_bar, K):
    """The closed form of the product of uniform matrices: alpha_bar I + (1 - alpha_bar) / K 1 1^T."""
    return alpha_bar * np.eye(K) + (1.0 - alpha_bar) / K * np.ones((K, K))


def absorbing_qbar(alpha_bar, K):
    """The closed form of the product of absorbing matrices: alpha_bar I + (1 - alpha_bar) 1 e_m^T."""
    return absorbing_q(1.0 - alpha_bar, K)


def one_hot(z, n):
    return np.eye(n)[np.asarray(z)]


def posterior(xt, x0, Qt, Qbar_prev):
    """q(x_{t-1} | x_t, x_0) of eq-d3pm-posterior, proportional to (x_t Q_t^T) * (x_0 Qbar_{t-1}).

    xt is one-hot; x0 is one-hot for the true posterior, or a probability vector p~(x_0 | x_t) for the reverse step
    of eq-x0-param, which sums the posterior against it. Rows of a batch are independent."""
    num = (xt @ Qt.T) * (x0 @ Qbar_prev)
    return num / np.sum(num, axis=-1, keepdims=True)


def multinomial_posterior(xt, x0, alpha_t, alpha_bar_prev, K):
    """theta_post of Hoogeboom and colleagues' eq. 13: [alpha_t x_t + (1 - alpha_t) / K] * [abar_{t-1} x_0 +
    (1 - abar_{t-1}) / K], normalized. With x0 a predicted probability vector it is their reverse step, eq. 14."""
    th = (alpha_t * xt + (1.0 - alpha_t) / K) * (alpha_bar_prev * x0 + (1.0 - alpha_bar_prev) / K)
    return th / np.sum(th, axis=-1, keepdims=True)


def kl(p, q):
    """KL(p || q) of probability rows, with 0 log 0 = 0."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    with np.errstate(divide="ignore"):
        lq = np.log(np.where(p > 0, q, 1.0))
    return np.sum(np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - lq), 0.0), axis=-1)


def d3pm_bound(x0, Qs, denoise, prior):
    """The negative variational bound of eq-vlb for one token x0 (an integer), every expectation over x_t summed
    exactly. denoise(t, xt) returns the predicted p~(x_0 | x_t) as a vector over the states; prior is p(x_T).

    Returns (L_T, [L_1, ..., L_{T-1}], L_0), where L_{t-1} is the KL term at step t."""
    n, T = Qs[0].shape[0], len(Qs)
    Qbar = cumulative(Qs)
    e0 = one_hot(x0, n)
    L_T = float(kl(e0 @ Qbar[T], prior))
    mids = []
    for t in range(2, T + 1):
        qt = e0 @ Qbar[t]
        term = 0.0
        for xt in range(n):
            if qt[xt] > 0:
                ext = one_hot(xt, n)
                q = posterior(ext, e0, Qs[t - 1], Qbar[t - 1])
                p = posterior(ext, denoise(t, xt), Qs[t - 1], Qbar[t - 1])
                term += qt[xt] * float(kl(q, p))
        mids.append(term)
    q1 = e0 @ Qbar[1]
    L_0 = 0.0
    for x1 in range(n):
        if q1[x1] > 0:
            p = posterior(one_hot(x1, n), denoise(1, x1), Qs[0], Qbar[0])
            L_0 -= q1[x1] * math.log(p[x0])
    return L_T, mids, L_0


def posterior_mean(p0, Qs):
    """The posterior q(x_0 | x_t) of a data distribution p0 over the states of one token. Under eq-x0-param it is the
    best p~ for absorbing noise but not for uniform noise, where the reverse step multiplies in q(x_t | x_0) a second
    time and the best p~ for a single token is p0 itself (derivations/x0-parameterization-optimum.md)."""
    Qbar = cumulative(Qs)

    def denoise(t, xt):
        w = p0 * Qbar[t][:, xt]
        return w / np.sum(w)

    return denoise


# ---------------------------------------------------------------------------------------------------------------------
# Masked diffusion in continuous time: schedules, masking, posterior and SUBS


def alpha(t, kind="linear"):
    """alpha_t, the probability that a token is still unmasked at time t; alpha_0 = 1 and alpha_1 = 0."""
    t = np.asarray(t, float)
    if kind == "linear":
        return 1.0 - t
    if kind == "cosine":
        return 1.0 - np.sin(0.5 * np.pi * t)  # = 1 - cos(pi (1 - t) / 2), exact at t = 0 and 1
    raise ValueError(kind)


def dalpha(t, kind="linear"):
    """The derivative alpha_t' of the schedule, which is negative."""
    t = np.asarray(t, float)
    if kind == "linear":
        return -np.ones_like(t)
    if kind == "cosine":
        return -0.5 * np.pi * np.cos(0.5 * np.pi * t)
    raise ValueError(kind)


def mask_tokens(x, t, u, K, kind="linear"):
    """Draw z_t from q(z_t | x) of eq-mask-marginal: token (b, i) is masked when u[b, i] < 1 - alpha_t[b]."""
    a = alpha(t, kind)
    return np.where(u < (1.0 - np.asarray(a))[..., None], K, x)


def mask_posterior(zt, x, a_s, a_t, K):
    """q(z_s | z_t, x) of eq-mask-posterior over the K + 1 states for one token: z_t itself if unmasked; otherwise the
    mask with probability (1 - a_s) / (1 - a_t) and x with probability (a_s - a_t) / (1 - a_t)."""
    if zt != K:
        return one_hot(zt, K + 1)
    v = np.zeros(K + 1)
    v[K] = (1.0 - a_s) / (1.0 - a_t)
    v[x] += (a_s - a_t) / (1.0 - a_t)
    return v


def subs_reverse(zt, probs, a_s, a_t, K):
    """p_theta(z_s | z_t) of eq-subs for one token: the posterior with x replaced by the denoiser's probabilities over
    the K data tokens, which give the mask no mass, and with an unmasked z_t carried over."""
    if zt != K:
        return one_hot(zt, K + 1)
    v = np.zeros(K + 1)
    v[K] = (1.0 - a_s) / (1.0 - a_t)
    v[:K] += (a_s - a_t) / (1.0 - a_t) * probs
    return v


def subs_probs(logits, z, K):
    """The SUBS denoiser of eq-subs: a softmax over the K data tokens at masked positions, so the mask gets no mass, and
    the one-hot of the token itself at unmasked positions. logits (B, L, K), z (B, L) -> (B, L, K)."""
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    sm = e / np.sum(e, axis=-1, keepdims=True)
    masked = (z == K)[..., None]
    return np.where(masked, sm, one_hot(np.where(z == K, 0, z), K))


def masked_kl_term(probs, x, a_s, a_t):
    """The step-s-to-t KL of eq-mask-kl for a masked token: (a_s - a_t) / (1 - a_t) times -log probs[x]."""
    return (a_s - a_t) / (1.0 - a_t) * -math.log(probs[x])


def weight(t, kind="linear"):
    """The weight -alpha_t' / (1 - alpha_t) of the cross-entropy at time t in eq-mdlm."""
    return -dalpha(t, kind) / (1.0 - alpha(t, kind))


def mdlm_estimate(logp, z, x, t, K, kind="linear"):
    """The one-sample estimate of eq-mdlm per sequence: weight(t) times the summed -log p_theta(x_i | z_t) over the
    masked positions. logp (B, L, K) are log-probabilities, z and x (B, L), t (B,) -> (B,)."""
    nll = -np.take_along_axis(logp, x[..., None], axis=-1)[..., 0]
    return weight(t, kind) * np.sum(np.where(z == K, nll, 0.0), axis=-1)


def mask_coefficient(m, L):
    """The integral over t of weight(t) P(a given set of m positions is exactly the masked set), for any schedule:
    the Beta function B(L - m + 1, m) = (L - m)! (m - 1)! / L!."""
    return math.factorial(L - m) * math.factorial(m - 1) / math.factorial(L)


def all_masks(L):
    """Every nonempty set of masked positions, as boolean rows (2^L - 1, L)."""
    return np.array([m for m in itertools.product([False, True], repeat=L) if any(m)])


def exact_nelbo(denoise, x, K):
    """eq-mdlm for one sequence x of length L, the integral over t and the expectation over masks done exactly by
    summing the masked sets with mask_coefficient. denoise(z) maps one (L,) sequence to (L, K) probabilities."""
    L = len(x)
    total = 0.0
    for m in all_masks(L):
        z = np.where(m, K, x)
        pr = denoise(z)
        total += mask_coefficient(int(m.sum()), L) * float(-np.sum(np.log(pr[m, x[m]])))
    return total


def nelbo_by_quadrature(denoise, x, K, kind, n=48):
    """eq-mdlm for one sequence by Gauss-Legendre quadrature in t, for a denoiser that ignores t: the expectation over
    masks at time t is a sum over sets, each weighted by (1 - alpha_t)^m alpha_t^(L - m)."""
    L = len(x)
    nodes, wts = np.polynomial.legendre.leggauss(n)
    ts, ws = 0.5 * (nodes + 1.0), 0.5 * wts
    total = 0.0
    for m in all_masks(L):
        z = np.where(m, K, x)
        ce = float(-np.sum(np.log(denoise(z)[m, x[m]])))
        k = int(m.sum())
        a, da = alpha(ts, kind), dalpha(ts, kind)
        total += ce * float(np.sum(ws * -da * (1.0 - a) ** (k - 1) * a ** (L - k)))
    return total


def conditional_denoiser(p, K):
    """The exact denoiser of a joint distribution p of shape (K,) * L: at a masked position i, the probability of each
    token given the unmasked positions, the other masked ones summed out; at an unmasked one, its one-hot."""
    L = p.ndim

    def denoise(z):
        idx = tuple(slice(None) if zi == K else int(zi) for zi in z)
        sub = p[idx]
        free = [i for i in range(L) if z[i] == K]
        out = one_hot(np.where(z == K, 0, z), K)
        for j, i in enumerate(free):
            marg = np.sum(sub, axis=tuple(a for a in range(len(free)) if a != j))
            out[i] = marg / np.sum(marg)
        return out

    return denoise


# ---------------------------------------------------------------------------------------------------------------------
# Sampling: ancestral steps of the masked reverse process, and MaskGIT


def categorical(probs, u):
    """Inverse-CDF draws: row b of probs with the uniform u[b]."""
    c = np.cumsum(probs, axis=-1)
    return np.minimum(np.sum(c < u[..., None], axis=-1), probs.shape[-1] - 1)


def ancestral_step(z, probs, a_s, a_t, u, g, K):
    """One step from t to s < t of the reverse process of eq-subs on a batch: a masked token is unmasked with
    probability (a_s - a_t) / (1 - a_t), when u < that, and takes a token drawn from probs with the uniform g."""
    tok = categorical(probs, g)
    unmask = (z == K) & (u < (a_s - a_t) / (1.0 - a_t))
    return np.where(unmask, tok, z)


def sample(denoise_batch, B, L, K, T, rng, kind="linear"):
    """Ancestral sampling with T equal steps from the all-mask sequence at t = 1 to t = 0. denoise_batch maps (B, L) to
    (B, L, K) probabilities. At t = 0 alpha = 1, so the last step unmasks every token left."""
    z = np.full((B, L), K)
    for i in range(T, 0, -1):
        t, s = i / T, (i - 1) / T
        z = ancestral_step(z, denoise_batch(z), float(alpha(s, kind)), float(alpha(t, kind)), rng.random((B, L)),
                           rng.random((B, L)), K)  # fmt: skip
    return z


def sampler_distribution(denoise, L, K, T, kind="linear"):
    """The exact distribution of the output of sample over K^L, as an array of shape (K,) * L, by carrying the
    probability of every partly masked state through the T steps."""
    states = {tuple([K] * L): 1.0}
    for i in range(T, 0, -1):
        a_s, a_t = float(alpha((i - 1) / T, kind)), float(alpha(i / T, kind))
        r = (a_s - a_t) / (1.0 - a_t)
        nxt = {}
        for z, pz in states.items():
            pr = denoise(np.array(z))
            opts = [[(z[j], 1.0)] if z[j] != K else [(K, 1.0 - r)] + [(y, r * pr[j, y]) for y in range(K)]
                    for j in range(L)]  # fmt: skip
            for combo in itertools.product(*opts):
                w = pz * math.prod(c[1] for c in combo)
                if w > 0:
                    key = tuple(c[0] for c in combo)
                    nxt[key] = nxt.get(key, 0.0) + w
        states = nxt
    out = np.zeros((K,) * L)
    for z, pz in states.items():
        out[z] += pz
    return out


def maskgit_gamma(r):
    """The cosine mask schedule of MaskGIT: the fraction of tokens still masked after a fraction r of the steps."""
    return np.cos(0.5 * np.pi * np.asarray(r, float))


def maskgit_counts(N, T):
    """How many tokens stay masked after each of the T iterations: ceil(gamma(t / T) N), capped so that every
    iteration unmasks at least one token; the last count is 0."""
    out, prev = [], N
    for t in range(1, T + 1):
        n = min(math.ceil(float(maskgit_gamma(t / T)) * N - 1e-9), prev - 1) if prev > 0 else 0
        n = max(n, 0)
        out.append(n)
        prev = n
    return out


def maskgit_decode(denoise_batch, B, L, K, T, u):
    """MaskGIT's iterative decoding. Each iteration predicts every masked token in parallel, draws one at each masked
    position with the uniforms u[t] (B, L), scores it by its own probability, keeps the tokens already fixed at
    confidence infinity, and masks again the n_t least confident positions. Returns the sequences and the states."""
    z = np.full((B, L), K)
    history = [z.copy()]
    for t, n in enumerate(maskgit_counts(L, T)):
        pr = denoise_batch(z)
        tok = categorical(pr, u[t])
        conf = np.where(z == K, np.take_along_axis(pr, tok[..., None], axis=-1)[..., 0], np.inf)
        filled = np.where(z == K, tok, z)
        order = np.argsort(conf, axis=-1, kind="stable")
        remask = np.zeros_like(z, dtype=bool)
        np.put_along_axis(remask, order[:, :n], True, axis=-1)
        z = np.where(remask, K, filled)
        history.append(z.copy())
    return z, history


# ---------------------------------------------------------------------------------------------------------------------
# Score entropy (SEDD)


def k_const(a):
    """K(a) = a (log a - 1), the constant that makes score entropy zero at the true ratios."""
    return a * (np.log(a) - 1.0)


def concrete_score(p):
    """The ratios p(y) / p(x) as an (N, N) array indexed [x, y]."""
    return p[None, :] / p[:, None]


def score_entropy(s, p, w):
    """eq-score-entropy: E_{x ~ p} sum_{y != x} w_xy (s_xy - r_xy log s_xy + K(r_xy)) with r = p(y) / p(x)."""
    r = concrete_score(p)
    off = ~np.eye(len(p), dtype=bool)
    terms = w * (s - r * np.log(s) + k_const(r))
    return float(np.sum(p[:, None] * np.where(off, terms, 0.0)))


def denoising_score_entropy(s, p0, kern, w):
    """Denoising score entropy for p(x) = sum_x0 p0(x0) kern[x0, x]: the ratios of p are replaced by those of the
    kernel, kern[x0, y] / kern[x0, x], and the constant K is dropped."""
    off = ~np.eye(kern.shape[1], dtype=bool)
    total = 0.0
    for a in range(len(p0)):
        k = kern[a]
        r = k[None, :] / k[:, None]
        total += p0[a] * float(np.sum(k[:, None] * np.where(off, w * (s - r * np.log(s)), 0.0)))
    return total


def masked_marginal(p, z, a, K):
    """p_t(z) for a partly masked sequence z under absorbing noise with alpha_t = a: the data probability of the
    unmasked tokens times a^(unmasked) (1 - a)^(masked)."""
    idx = tuple(slice(None) if zi == K else int(zi) for zi in z)
    m = int(np.sum(np.asarray(z) == K))
    return float(np.sum(p[idx])) * a ** (len(z) - m) * (1.0 - a) ** m


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser network: shapes, counts, forward, the weighted cross-entropy and its gradient


def config(L, K, H):
    return {"seq_len": L, "vocab": K, "hidden": H}


def param_shapes(cfg):
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return {"W1": (L * (K + 1), H), "b1": (H,), "W2": (H, L * K), "b2": (L * K,)}


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, B):
    """The two matrix products of the forward pass on B sequences, a multiply-add counted as 2 FLOPs."""
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return 2 * B * L * (K + 1) * H + 2 * B * H * L * K


def init(rng, cfg, scale=1.0):
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return {"W1": rng.normal(0.0, scale / math.sqrt(L), (L * (K + 1), H)), "b1": np.zeros(H),
            "W2": rng.normal(0.0, scale / math.sqrt(H), (H, L * K)), "b2": np.zeros(L * K)}  # fmt: skip


def forward(p, z, cfg):
    """Logits (B, L, K) over the data tokens at every position, from the one-hot of the partly masked z (B, L)."""
    L, K = cfg["seq_len"], cfg["vocab"]
    x = one_hot(z, K + 1).reshape(z.shape[0], L * (K + 1))
    h = np.tanh(x @ p["W1"] + p["b1"])
    return (h @ p["W2"] + p["b2"]).reshape(z.shape[0], L, K)


def log_softmax(a):
    m = np.max(a, axis=-1, keepdims=True)
    return a - m - np.log(np.sum(np.exp(a - m), axis=-1, keepdims=True))


def denoise_fn(p, cfg):
    """The SUBS denoiser of the network as a function of one (L,) sequence, for exact_nelbo and the samplers."""
    K = cfg["vocab"]

    def denoise(z):
        z = np.asarray(z)[None]
        return subs_probs(forward(p, z, cfg), z, K)[0]

    return denoise


def weighted_ce(p, cfg, Z, X, Wt):
    """sum_{n, i} Wt[n, i] (-log softmax(f(Z_n))[i, X[n, i]]). The Monte Carlo estimate of eq-mdlm and its exact value
    are both of this form; Wt is zero at unmasked positions."""
    lp = log_softmax(forward(p, Z, cfg))
    return float(np.sum(Wt * -np.take_along_axis(lp, X[..., None], axis=-1)[..., 0]))


def weighted_ce_grad(p, cfg, Z, X, Wt):
    """The gradient of weighted_ce in every parameter, by hand: d logits = Wt (softmax - one-hot), then through the
    output layer, the tanh and the input layer."""
    L, K = cfg["seq_len"], cfg["vocab"]
    B = Z.shape[0]
    x = one_hot(Z, K + 1).reshape(B, L * (K + 1))
    h = np.tanh(x @ p["W1"] + p["b1"])
    lp = log_softmax((h @ p["W2"] + p["b2"]).reshape(B, L, K))
    dlog = (Wt[..., None] * (np.exp(lp) - one_hot(X, K))).reshape(B, L * K)
    dh = (dlog @ p["W2"].T) * (1.0 - h**2)
    return {"W1": x.T @ dh, "b1": dh.sum(0), "W2": h.T @ dlog, "b2": dlog.sum(0)}


def exact_batch(data_p, K):
    """The rows (Z, X, Wt) that make weighted_ce equal the exact eq-mdlm averaged over the data distribution data_p of
    shape (K,) * L: one row per sequence and nonempty mask, weighted by p(x) times mask_coefficient at masked tokens."""
    L = data_p.ndim
    seqs = np.array(list(itertools.product(range(K), repeat=L)))
    masks = all_masks(L)
    Z, X, Wt = [], [], []
    for x in seqs:
        px = float(data_p[tuple(x)])
        for m in masks:
            Z.append(np.where(m, K, x))
            X.append(x)
            Wt.append(np.where(m, px * mask_coefficient(int(m.sum()), L), 0.0))
    return np.array(Z), np.array(X), np.array(Wt)


def mc_batch(x, t, u, K, kind="linear"):
    """The rows (Z, X, Wt) of the one-sample estimate of eq-mdlm averaged over a batch x (B, L) with times t (B,) and
    uniforms u (B, L) for the masks."""
    z = mask_tokens(x, t, u, K, kind)
    return z, x, np.where(z == K, weight(t, kind)[:, None], 0.0) / x.shape[0]


def adam_init(p):
    return {k: np.zeros_like(v) for k, v in p.items()}, {k: np.zeros_like(v) for k, v in p.items()}, 0


def adam_step(p, g, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, n = state
    n += 1
    out = {}
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        out[k] = p[k] - lr * (m[k] / (1 - b1**n)) / (np.sqrt(v[k] / (1 - b2**n)) + eps)
    return out, (m, v, n)


def train_exact(p, cfg, data_p, steps, lr=0.01):
    """Adam on the exact bound of eq-mdlm over the data distribution. Returns the parameters and the bound per step."""
    Z, X, Wt = exact_batch(data_p, cfg["vocab"])
    state, trace = adam_init(p), []
    for _ in range(steps):
        trace.append(weighted_ce(p, cfg, Z, X, Wt))
        p, state = adam_step(p, weighted_ce_grad(p, cfg, Z, X, Wt), state, lr)
    return p, np.array(trace)


def markov_data(L, K, stay=0.1, step=0.8):
    """A test distribution with dependencies: x_1 uniform, then x_{i+1} = x_i + 1 mod K with probability step, x_i with
    probability stay and the other tokens share the rest; K must be at least 3."""
    if K < 3:
        raise ValueError("markov_data needs K >= 3, so that a third token takes the remaining mass")
    T = np.full((K, K), (1.0 - stay - step) / max(K - 2, 1))
    for a in range(K):
        T[a, a] = stay
        T[a, (a + 1) % K] = step
    p = np.zeros((K,) * L)
    for x in itertools.product(range(K), repeat=L):
        p[x] = math.prod(T[x[i], x[i + 1]] for i in range(L - 1)) / K
    return p


def entropy(p):
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))
