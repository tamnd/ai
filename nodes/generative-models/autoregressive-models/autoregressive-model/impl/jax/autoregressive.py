# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Autoregressive models in JAX, as pure functions of a parameter dictionary: MADE with fixed masks (eq-made,
eq-made-masks), its negative log-likelihood (eq-nll) differentiated by jax.grad, a jitted Adam step, ancestral
sampling as a jax.lax.fori_loop, the mask ensemble of eq-ensemble, NADE's shared hidden layer (eq-nade) as one
jnp.cumsum, a WaveNet-style stack of dilated causal convolutions with gated units (eq-causal-conv, eq-gated) and the
mu-law codes of eq-mu-law.

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference. The masks are
arrays passed beside the parameters, not parameters, so jax.grad gives them no gradient.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np


def config(d_data=4, hidden=(6,), direct=True):
    hidden = tuple(int(h) for h in hidden)
    if d_data < 2 or min(hidden, default=1) < 1:
        raise ValueError("MADE needs d_data >= 2 and positive hidden widths")
    if not hidden and not direct:
        raise ValueError("with no hidden layer the direct connection is the only path")
    return {"d_data": int(d_data), "hidden": hidden, "direct": bool(direct)}


def param_shapes(cfg):
    D, widths = cfg["d_data"], (cfg["d_data"],) + cfg["hidden"]
    s = {}
    for i in range(len(cfg["hidden"])):
        s[f"W{i + 1}"], s[f"b{i + 1}"] = (widths[i], widths[i + 1]), (widths[i + 1],)
    if cfg["hidden"]:
        s["V"] = (widths[-1], D)
    s["c"] = (D,)
    if cfg["direct"]:
        s["A"] = (D, D)
    return s


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg, batch):
    """eq-flops: the matrix products of one pass, a multiply-add as 2."""
    w = (cfg["d_data"],) + cfg["hidden"] + (cfg["d_data"],)
    n = sum(a * b for a, b in itertools.pairwise(w)) + (cfg["d_data"] ** 2 if cfg["direct"] else 0)
    return 2 * batch * n


def init(key, cfg):
    """Hidden weights random, the output and direct weights and every bias 0: the model starts uniform."""
    p = {k: jnp.zeros(s) for k, s in param_shapes(cfg).items()}
    for i in range(len(cfg["hidden"])):
        key, sub = jax.random.split(key)
        shape = p[f"W{i + 1}"].shape
        p[f"W{i + 1}"] = jax.random.normal(sub, shape) * math.sqrt(2.0 / shape[0])
    return p


def degrees(cfg, order=None, rng=None):
    """m0 = order (a permutation of 1..D), then each hidden layer's degrees in [min m_{l-1}, D - 1]: drawn by the
    NumPy generator rng if given, else cycled through that range. Degrees are host-side integers."""
    D = cfg["d_data"]
    m = [np.arange(1, D + 1) if order is None else np.asarray(order, int)]
    for n in cfg["hidden"]:
        lo = int(m[-1].min())
        m.append(rng.integers(lo, D, n) if rng is not None else lo + np.arange(n) % (D - lo))
    return m


def masks(cfg, m):
    """eq-made-masks, each mask of the shape of its weight."""
    out = {f"W{i}": jnp.asarray(m[i][None, :] >= m[i - 1][:, None], float) for i in range(1, len(m))}
    if cfg["hidden"]:
        out["V"] = jnp.asarray(m[0][None, :] > m[-1][:, None], float)
    if cfg["direct"]:
        out["A"] = jnp.asarray(m[0][None, :] > m[0][:, None], float)
    return out


def logits(p, mk, x):
    """eq-made: the logits of p(x_d = 1 | inputs before d) for a batch x of shape (B, D)."""
    h, i = x, 1
    while f"W{i}" in p:
        h = jax.nn.relu(h @ (p[f"W{i}"] * mk[f"W{i}"]) + p[f"b{i}"])
        i += 1
    a = jnp.broadcast_to(p["c"], x.shape)
    if "V" in p:
        a = a + h @ (p["V"] * mk["V"])
    if "A" in p:
        a = a + x @ (p["A"] * mk["A"])
    return a


def log_prob(p, mk, x):
    a = logits(p, mk, x)
    return jnp.sum(x * a - jax.nn.softplus(a), 1)


def nll(p, mk, x):
    """eq-nll: the mean negative log-likelihood over the batch."""
    return -jnp.mean(log_prob(p, mk, x))


def sample(p, mk, m0, key, n):
    """Ancestral sampling as a fori_loop over the D positions of the ordering m0."""
    D = len(m0)
    pos = jnp.asarray(np.argsort(m0))
    u = jax.random.uniform(key, (n, D))

    def body(s, x):
        d = pos[s]
        prob = jax.nn.sigmoid(logits(p, mk, x)[:, d])
        return x.at[:, d].set((u[:, d] < prob).astype(x.dtype))

    return jax.lax.fori_loop(0, D, body, jnp.zeros((n, D)))


def all_binary(D):
    return jnp.array(list(itertools.product([0.0, 1.0], repeat=D)))


def ensemble_log_prob(p, mask_list, x):
    """eq-ensemble: the log of the mean probability over a list of mask sets sharing the weights."""
    lp = jnp.stack([log_prob(p, mk, x) for mk in mask_list])
    return jax.scipy.special.logsumexp(lp, 0) - math.log(len(mask_list))


def adam_init(p):
    return {"m": jax.tree.map(jnp.zeros_like, p), "v": jax.tree.map(jnp.zeros_like, p), "t": 0}


def make_step(lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    """One jitted step of maximum likelihood with Adam on a batch x under the masks mk."""

    @jax.jit
    def step(p, st, mk, x):
        loss, g = jax.value_and_grad(nll)(p, mk, x)
        k = st["t"] + 1
        m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, st["m"], g)
        v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b**2, st["v"], g)
        p = jax.tree.map(lambda w, mi, vi: w - lr * (mi / (1 - b1**k)) / (jnp.sqrt(vi / (1 - b2**k)) + eps), p, m, v)
        return p, {"m": m, "v": v, "t": k}, loss

    return step


# NADE: h_i = sigm(c + sum_{j<i} x_j W[j]), p(x_i = 1 | x_<i) = sigm(b_i + h_i . V[:, i]).


def nade_logits(q, x):
    """eq-nade in O(BDH): an exclusive cumulative sum over the rows of W weighted by x."""
    terms = x[:, :, None] * q["W"][None]
    pre = q["c"] + jnp.cumsum(terms, 1) - terms
    return q["b"] + jnp.einsum("bdh,hd->bd", jax.nn.sigmoid(pre), q["V"])


# WaveNet-style stack over K-valued sequences: logits[t] predict x_t from x_<t.


def wavenet_shapes(K, C, kernel, dilations):
    s: dict[str, tuple[int, ...]] = {"E": (K, C)}
    for i in range(len(dilations)):
        s.update({f"F{i}": (kernel, C, C), f"G{i}": (kernel, C, C), f"R{i}": (C, C), f"S{i}": (C, C)})
    s.update({"O": (C, K), "o": (K,)})
    return s


def receptive_field(kernel, dilations):
    """eq-receptive-field."""
    return 1 + (kernel - 1) * sum(dilations)


def shift(h, s):
    """h delayed by s steps along axis 1, zeros in front."""
    return jnp.pad(h, ((0, 0), (s, 0), (0, 0)))[:, : h.shape[1]] if s else h


def causal_conv(h, W, dilation):
    """eq-causal-conv: out[t] = sum_j h[t - (k - 1 - j) dilation] @ W[j], for h (B, T, C_in) and W (k, C_in, C_out)."""
    k = W.shape[0]
    out = shift(h, (k - 1) * dilation) @ W[0]
    for j in range(1, k):
        out = out + shift(h, (k - 1 - j) * dilation) @ W[j]
    return out


def wavenet_logits(w, x, dilations):
    """Logits (B, T, K) for integer sequences x (B, T): embed x shifted by one step, gated residual layers (eq-gated)
    with skip connections, a ReLU and an output layer."""
    h = shift(w["E"][x], 1)
    skip = jnp.zeros_like(h)
    for i, d in enumerate(dilations):
        z = jnp.tanh(causal_conv(h, w[f"F{i}"], d)) * jax.nn.sigmoid(causal_conv(h, w[f"G{i}"], d))
        skip = skip + z @ w[f"S{i}"]
        h = h + z @ w[f"R{i}"]
    return jax.nn.relu(skip) @ w["O"] + w["o"]


def wavenet_log_prob(w, x, dilations):
    lp = jax.nn.log_softmax(wavenet_logits(w, x, dilations), -1)
    return jnp.take_along_axis(lp, x[..., None], -1)[..., 0].sum(1)


def wavenet_sample(w, dilations, key, n, T):
    """T passes in a fori_loop, each drawing x_t from the softmax at t given the values drawn so far."""
    keys = jax.random.split(key, T)

    def body(t, x):
        a = wavenet_logits(w, x, dilations)[:, t]
        return x.at[:, t].set(jax.random.categorical(keys[t], a))

    return jax.lax.fori_loop(0, T, body, jnp.zeros((n, T), int))


def mu_law_encode(x, K=256, mu=255.0):
    """eq-mu-law, then K uniform bins of [-1, 1]."""
    y = jnp.sign(x) * jnp.log1p(mu * jnp.abs(x)) / math.log1p(mu)
    return jnp.clip(jnp.floor((y + 1.0) / 2.0 * K), 0, K - 1).astype(int)
