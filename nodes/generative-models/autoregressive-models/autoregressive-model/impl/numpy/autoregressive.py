# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoregressive-model/ (ai://NEW-autoregressive-model).
"""Autoregressive models in NumPy: MADE, a masked autoencoder whose outputs are the conditionals of eq-chain
(eq-made, eq-made-masks), with the negative log-likelihood of eq-nll and every backward pass written by hand; exact
sequential sampling; orderings, mask sets and the ensemble of eq-ensemble; NADE with the shared hidden layer of
eq-nade, computed in O(DH) and naively in O(D^2 H); a WaveNet-style stack of dilated causal convolutions with gated
units (eq-causal-conv, eq-gated) over mu-law codes (eq-mu-law); and the masked convolutions of PixelCNN with the
dependency fields of a masked stack and of the vertical and horizontal stacks of Gated PixelCNN.

Vectors are rows and a layer computes h @ W + b with W of shape (n_in, n_out). The degrees m follow Germain et al.:
m0[d] is the position of input d in the ordering, from 1 to D, and a hidden unit of degree m sees the inputs of
position at most m.
"""

import itertools
import math

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
    """eq-flops: the matrix products of one pass, a multiply-add as 2; the masks are fixed and not counted."""
    w = (cfg["d_data"],) + cfg["hidden"] + (cfg["d_data"],)
    n = sum(a * b for a, b in itertools.pairwise(w)) + (cfg["d_data"] ** 2 if cfg["direct"] else 0)
    return 2 * batch * n


def init(rng, cfg):
    """Hidden weights random, the output and direct weights and every bias 0: the model starts uniform."""
    p = {k: np.zeros(s) for k, s in param_shapes(cfg).items()}
    for i in range(len(cfg["hidden"])):
        W = p[f"W{i + 1}"]
        p[f"W{i + 1}"] = rng.standard_normal(W.shape) * math.sqrt(2.0 / W.shape[0])
    return p


def degrees(cfg, order=None, rng=None):
    """m0 = order (a permutation of 1..D, default 1..D), then each hidden layer's degrees in [min m_{l-1}, D - 1]:
    drawn uniformly if rng is given, as in Germain et al., else cycled through that range."""
    D = cfg["d_data"]
    m = [np.arange(1, D + 1) if order is None else np.asarray(order, int)]
    for n in cfg["hidden"]:
        lo = int(m[-1].min())
        m.append(rng.integers(lo, D, n) if rng is not None else lo + np.arange(n) % (D - lo))
    return m


def masks(cfg, m):
    """eq-made-masks: hidden masks 1[m_l(k') >= m_{l-1}(k)], output mask 1[m0(d) > m_L(k)], direct mask
    1[m0(d) > m0(d')], each of the shape of its weight."""
    out = {f"W{i}": (m[i][None, :] >= m[i - 1][:, None]).astype(float) for i in range(1, len(m))}
    if cfg["hidden"]:
        out["V"] = (m[0][None, :] > m[-1][:, None]).astype(float)
    if cfg["direct"]:
        out["A"] = (m[0][None, :] > m[0][:, None]).astype(float)
    return out


def logits(p, cfg, mk, x):
    """eq-made: a_d, the logit of p(x_d = 1 | inputs before d), for a batch x of shape (B, D); also the cache."""
    hs, zs = [x], []
    for i in range(1, len(cfg["hidden"]) + 1):
        zs.append(hs[-1] @ (p[f"W{i}"] * mk[f"W{i}"]) + p[f"b{i}"])
        hs.append(np.maximum(zs[-1], 0.0))
    a = p["c"] + np.zeros((len(x), cfg["d_data"]))
    if cfg["hidden"]:
        a = a + hs[-1] @ (p["V"] * mk["V"])
    if cfg["direct"]:
        a = a + x @ (p["A"] * mk["A"])
    return a, (hs, zs)


def softplus(a):
    return np.logaddexp(0.0, a)


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def log_prob(p, cfg, mk, x):
    """eq-nll with the sign flipped: log p(x) = sum_d x_d a_d - softplus(a_d), per row."""
    a, _ = logits(p, cfg, mk, x)
    return np.sum(x * a - softplus(a), 1)


def loss_and_grads(p, cfg, mk, x):
    """The mean negative log-likelihood over the batch and its gradients, backpropagated by hand through the masks."""
    B = len(x)
    a, (hs, zs) = logits(p, cfg, mk, x)
    ga = (sigmoid(a) - x) / B
    g = {"c": ga.sum(0)}
    if cfg["direct"]:
        g["A"] = (x.T @ ga) * mk["A"]
    if cfg["hidden"]:
        g["V"] = (hs[-1].T @ ga) * mk["V"]
        gh = ga @ (p["V"] * mk["V"]).T
        for i in range(len(cfg["hidden"]), 0, -1):
            gz = gh * (zs[i - 1] > 0)
            g[f"W{i}"], g[f"b{i}"] = (hs[i - 1].T @ gz) * mk[f"W{i}"], gz.sum(0)
            gh = gz @ (p[f"W{i}"] * mk[f"W{i}"]).T
    return float(-np.mean(np.sum(x * a - softplus(a), 1))), g


def sample(p, cfg, mk, m0, rng, n):
    """Ancestral sampling: D passes, each drawing the next input in the ordering m0 from its conditional."""
    x = np.zeros((n, cfg["d_data"]))
    for d in np.argsort(m0):
        x[:, d] = rng.random(n) < sigmoid(logits(p, cfg, mk, x)[0][:, d])
    return x


def all_binary(D):
    return np.array(list(itertools.product([0.0, 1.0], repeat=D)))


def ensemble_log_prob(p, cfg, mask_list, x):
    """eq-ensemble: log of the mean probability over a list of masks, each mask set one model sharing the weights."""
    lp = np.stack([log_prob(p, cfg, mk, x) for mk in mask_list])
    return np.logaddexp.reduce(lp, 0) - math.log(len(mask_list))


def mask_list(cfg, rng, n, shuffle_order=True):
    """n mask sets with random degrees and, if shuffle_order, a random ordering each."""
    D = cfg["d_data"]
    return [masks(cfg, degrees(cfg, rng.permutation(D) + 1 if shuffle_order else None, rng)) for _ in range(n)]


def adam(p, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    st["t"] = st.get("t", 0) + 1
    for k in g:
        st[k + "m"] = b1 * st.get(k + "m", 0.0) + (1 - b1) * g[k]
        st[k + "v"] = b2 * st.get(k + "v", 0.0) + (1 - b2) * g[k] ** 2
        mh, vh = st[k + "m"] / (1 - b1 ** st["t"]), st[k + "v"] / (1 - b2 ** st["t"])
        p[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)


def train(p, cfg, data, steps, rng, mks, batch=100, lr=1e-2):
    """Maximum likelihood with Adam on minibatches of the rows of data, cycling through the mask sets mks, one per
    update, as in the finite mask lists of Germain et al.; returns the losses."""
    st, trace = {}, []
    for s in range(steps):
        loss, g = loss_and_grads(p, cfg, mks[s % len(mks)], data[rng.integers(0, len(data), batch)])
        adam(p, g, st, lr)
        trace.append(loss)
    return trace


# NADE (Larochelle and Murray): h_i = sigm(c + sum_{j<i} x_j W[j]), p(x_i = 1 | x_<i) = sigm(b_i + h_i . V[:, i]).


def nade_shapes(D, H):
    return {"W": (D, H), "c": (H,), "V": (H, D), "b": (D,)}


def nade_logits(q, x):
    """eq-nade in O(BDH): the pre-activations of all D hidden layers are one exclusive cumulative sum."""
    terms = x[:, :, None] * q["W"][None]
    pre = q["c"] + np.cumsum(terms, 1) - terms
    return q["b"] + np.einsum("bdh,hd->bd", sigmoid(pre), q["V"])


def nade_logits_naive(q, x):
    """The same logits with one separate pass per conditional, O(BD^2 H)."""
    cols = [q["b"][i] + sigmoid(q["c"] + x[:, :i] @ q["W"][:i]) @ q["V"][:, i] for i in range(x.shape[1])]
    return np.stack(cols, 1)


# WaveNet-style stack over K-valued sequences: logits[t] predict x_t from x_<t.


def wavenet_shapes(K, C, kernel, dilations):
    s: dict[str, tuple[int, ...]] = {"E": (K, C)}
    for i in range(len(dilations)):
        s.update({f"F{i}": (kernel, C, C), f"G{i}": (kernel, C, C), f"R{i}": (C, C), f"S{i}": (C, C)})
    s.update({"O": (C, K), "o": (K,)})
    return s


def receptive_field(kernel, dilations):
    """eq-receptive-field: logits[t] depend on the last 1 + (kernel - 1) sum(dilations) values before t."""
    return 1 + (kernel - 1) * sum(dilations)


def shift(h, s):
    """h delayed by s steps along axis 1, zeros in front: out[:, t] = h[:, t - s]."""
    if s == 0:
        return h
    out = np.zeros_like(h)
    out[:, s:] = h[:, : h.shape[1] - s] if s < h.shape[1] else 0.0
    return out


def causal_conv(h, W, dilation):
    """eq-causal-conv: out[t] = sum_j h[t - (k - 1 - j) dilation] @ W[j], for h (B, T, C_in) and W (k, C_in, C_out)."""
    k = W.shape[0]
    return np.sum([shift(h, (k - 1 - j) * dilation) @ W[j] for j in range(k)], 0)


def wavenet_logits(w, x, dilations):
    """Logits (B, T, K) for integer sequences x (B, T): embed x_{t-1}, then gated residual layers (eq-gated) with skip
    connections, a ReLU and an output layer."""
    E = w["E"]
    h = shift(E[x], 1)
    skip = np.zeros_like(h)
    for i, d in enumerate(dilations):
        z = np.tanh(causal_conv(h, w[f"F{i}"], d)) * sigmoid(causal_conv(h, w[f"G{i}"], d))
        skip = skip + z @ w[f"S{i}"]
        h = h + z @ w[f"R{i}"]
    return np.maximum(skip, 0.0) @ w["O"] + w["o"]


def log_softmax(a):
    return a - np.logaddexp.reduce(a, -1)[..., None]


def wavenet_log_prob(w, x, dilations):
    lp = log_softmax(wavenet_logits(w, x, dilations))
    return np.take_along_axis(lp, x[..., None], -1)[..., 0].sum(1)


def wavenet_sample(w, dilations, rng, n, T, K):
    """T passes, each drawing x_t from the softmax at t given the values drawn so far."""
    x = np.zeros((n, T), int)
    for t in range(T):
        pr = np.exp(log_softmax(wavenet_logits(w, x, dilations)[:, t]))
        x[:, t] = (rng.random((n, 1)) > np.cumsum(pr, 1)).sum(1).clip(0, K - 1)
    return x


def mu_law_encode(x, K=256, mu=255.0):
    """eq-mu-law, then K uniform bins of [-1, 1]: integer codes of samples x in [-1, 1]."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / math.log1p(mu)
    return np.clip(np.floor((y + 1.0) / 2.0 * K), 0, K - 1).astype(int)


def mu_law_decode(q, K=256, mu=255.0):
    """The centre of bin q mapped back through the inverse of eq-mu-law."""
    y = (np.asarray(q) + 0.5) / K * 2.0 - 1.0
    return np.sign(y) * np.expm1(np.abs(y) * math.log1p(mu)) / mu


# PixelCNN: masked k x k convolutions in raster order, and the fields they see.


def conv_mask(k, kind):
    """The k x k mask of PixelCNN: rows above the centre, the centre row left of the centre, and the centre itself
    only for mask B. Colour channels are not split here."""
    m, c = np.zeros((k, k)), k // 2
    m[:c], m[c, :c] = 1.0, 1.0
    m[c, c] = 1.0 if kind == "B" else 0.0
    return m


def masked_conv2d(x, W, mask):
    """'Same' convolution (cross-correlation) of x (B, H, W, C_in) with W (k, k, C_in, C_out) times the mask."""
    k, c = W.shape[0], W.shape[0] // 2
    xp = np.pad(x, ((0, 0), (c, c), (c, c), (0, 0)))
    n, m = x.shape[1], x.shape[2]
    return np.sum([xp[:, r : r + n, s : s + m] @ (W[r, s] * mask[r, s]) for r in range(k) for s in range(k)], 0)


def _grow(field, offsets):
    return {(a + u, b + v) for a, b in field for u, v in offsets}


def masked_stack_field(n_layers, k=3):
    """Offsets (di, dj) of the inputs that one output of a mask A layer and n_layers - 1 mask B layers depends on."""
    c = k // 2
    off = {kind: {(r - c, s - c) for r in range(k) for s in range(k) if conv_mask(k, kind)[r, s]} for kind in "AB"}
    field = {(0, 0)}
    for layer in range(n_layers, 0, -1):
        field = _grow(field, off["A" if layer == 1 else "B"])
    return field


def gated_stack_field(n_layers, k=3):
    """The same for the vertical and horizontal stacks of Gated PixelCNN: the vertical stack sees rows up to its own
    (a ceil(k/2) x k kernel) and feeds the horizontal stack one row down; the horizontal stack sees its own row to the
    left, itself excluded in the first layer."""
    r = k // 2
    ov = {(di, dj) for di in range(-r, 1) for dj in range(-r, r + 1)}
    v, h = [{(0, 0)}], {(0, 0)}
    for layer in range(1, n_layers + 1):
        v.append(_grow(v[-1], ov))
        oh = {(0, dj) for dj in range(-r, 0 if layer == 1 else 1)}
        h = _grow(h, oh) | _grow(v[-1], {(-1, 0)})
    return h
