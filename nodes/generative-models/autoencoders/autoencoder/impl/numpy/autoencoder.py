# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/autoencoder/ (ai://NEW-autoencoder).
"""The autoencoder of the page in NumPy: an encoder of affine layers and activations maps an input x to a code y, a
decoder maps y back to a reconstruction z, and training lowers a reconstruction loss L(x, z) by backpropagation.

The same code runs the classical bottleneck autoencoder, tied weights (each decoder matrix the transpose of an encoder
matrix), the denoising autoencoder of Vincent et al. (2010) with its three corruptions and its emphasis on the
corrupted components, the contractive penalty of Rifai et al. (2011), and the linear autoencoder that finds the
principal subspace. Rows are examples: x is (B, D) and a layer computes h @ W + b with W of shape (n_in, n_out).
"""

import numpy as np

ACTIVATIONS = ("sigmoid", "relu", "tanh", "linear")


def config(d_in=6, d_hidden=5, n_hidden=1, d_code=2, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    """The autoencoder of the shape contract: n_hidden layers of d_hidden units on each side of a code of d_code."""
    return from_widths([d_in] + [d_hidden] * n_hidden + [d_code], act, code_act, out, tied)


def from_widths(widths, act="sigmoid", code_act="linear", out="sigmoid", tied=False):
    """An autoencoder with the encoder widths [D, n_1, ..., C]; the decoder runs through them in reverse.

    act is the activation of the hidden layers, code_act that of the code layer, and out is "sigmoid" (a Bernoulli
    reconstruction with the cross-entropy loss) or "linear" (a Gaussian reconstruction with the squared error)."""
    widths = [int(w) for w in widths]
    assert len(widths) >= 2 and min(widths) >= 1, widths
    assert act in ACTIVATIONS and code_act in ACTIVATIONS and out in ("sigmoid", "linear"), (act, code_act, out)
    return {"widths": widths, "act": act, "code_act": code_act, "out": out, "tied": bool(tied)}


def n_layers(cfg):
    return len(cfg["widths"]) - 1


def enc_act(cfg, i):
    return cfg["code_act"] if i == n_layers(cfg) - 1 else cfg["act"]


def dec_act(cfg, j):
    return cfg["out"] if j == n_layers(cfg) - 1 else cfg["act"]


def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # the logistic function, without overflow for large |a|


def act(name, a):
    if name == "sigmoid":
        return sigmoid(a)
    if name == "relu":
        return np.maximum(a, 0.0)
    if name == "tanh":
        return np.tanh(a)
    return a


def act_grad(name, a, h):
    """dh/da of the activation, from its input a and its output h."""
    if name == "sigmoid":
        return h * (1.0 - h)
    if name == "relu":
        return (a > 0.0).astype(a.dtype)
    if name == "tanh":
        return 1.0 - h * h
    return np.ones_like(a)


def init(rng, cfg, gain=1.0):
    """Glorot-uniform weights times gain and zero biases. With tied weights the decoder has biases only."""
    w, K = cfg["widths"], n_layers(cfg)

    def glorot(n_in, n_out):
        r = gain * np.sqrt(6.0 / (n_in + n_out))
        return rng.uniform(-r, r, (n_in, n_out))

    p = {"enc_W": [glorot(w[i], w[i + 1]) for i in range(K)], "enc_b": [np.zeros(w[i + 1]) for i in range(K)],
         "dec_b": [np.zeros(w[K - j - 1]) for j in range(K)]}  # fmt: skip
    if not cfg["tied"]:
        p["dec_W"] = [glorot(w[K - j], w[K - j - 1]) for j in range(K)]
    return p


def dec_weights(p, cfg):
    """The decoder matrices; with tied weights decoder layer j uses the transpose of encoder layer K - 1 - j."""
    if cfg["tied"]:
        return [W.T for W in reversed(p["enc_W"])]
    return p["dec_W"]


def forward(p, cfg, x):
    """Encode and decode a (B, D) batch. Returns {y: the code, a: the output before its activation (the logits when
    out is sigmoid), z: the reconstruction} and the cache of the backward pass."""
    K = n_layers(cfg)
    hs, pre = [x], []
    for i in range(K):
        a = hs[-1] @ p["enc_W"][i] + p["enc_b"][i]
        pre.append(a)
        hs.append(act(enc_act(cfg, i), a))
    y = hs[-1]
    for j, W in enumerate(dec_weights(p, cfg)):
        a = hs[-1] @ W + p["dec_b"][j]
        pre.append(a)
        hs.append(act(dec_act(cfg, j), a))
    return {"y": y, "a": pre[-1], "z": hs[-1]}, {"hs": hs, "pre": pre}


def encode(p, cfg, x):
    return forward(p, cfg, x)[0]["y"]


def reconstruct(p, cfg, x):
    return forward(p, cfg, x)[0]["z"]


def loss(cfg, target, a, weight=None):
    """The reconstruction loss, summed over the D components and averaged over the batch, and its gradient in a.

    With out = "sigmoid" it is the cross-entropy -sum_j [x_j log z_j + (1 - x_j) log(1 - z_j)] with z = sigmoid(a),
    written as softplus(a) - x a so that it is finite for any a; with out = "linear" it is ||x - z||^2. weight, a
    (B, D) array, scales each component's term: the alpha and beta of the emphasized denoising autoencoder."""
    if cfg["out"] == "sigmoid":
        e = np.logaddexp(0.0, a) - target * a
        de = sigmoid(a) - target
    else:
        e = (target - a) ** 2
        de = 2.0 * (a - target)
    w = 1.0 if weight is None else weight
    B = target.shape[0]
    return float(np.sum(w * e) / B), w * de / B


def backward(p, cfg, cache, da):
    """The gradients of the loss given its gradient da in the output pre-activation, and the gradient in the input."""
    K, hs, pre = n_layers(cfg), cache["hs"], cache["pre"]
    Ws = dec_weights(p, cfg)
    g = {"enc_W": [None] * K, "enc_b": [None] * K, "dec_b": [None] * K}
    dec_W = [None] * K
    d = da
    for j in reversed(range(K)):
        if j < K - 1:
            d = d * act_grad(dec_act(cfg, j), pre[K + j], hs[K + j + 1])
        dec_W[j] = hs[K + j].T @ d
        g["dec_b"][j] = d.sum(0)
        d = d @ Ws[j].T
    for i in reversed(range(K)):
        d = d * act_grad(enc_act(cfg, i), pre[i], hs[i + 1])
        g["enc_W"][i] = hs[i].T @ d
        g["enc_b"][i] = d.sum(0)
        d = d @ p["enc_W"][i].T
    if cfg["tied"]:
        for i in range(K):
            g["enc_W"][i] = g["enc_W"][i] + dec_W[K - 1 - i].T
    else:
        g["dec_W"] = dec_W
    return g, d


def param_count(cfg):
    w, K = cfg["widths"], n_layers(cfg)
    weights = sum(w[i] * w[i + 1] for i in range(K))
    return (1 if cfg["tied"] else 2) * weights + sum(w[1:]) + sum(w[:-1])


def flops(cfg, B):
    """FLOPs of the matrix products over a batch of B, a multiply-add counted as 2: (forward, backward)."""
    w = cfg["widths"]
    fwd = 2 * 2 * B * sum(w[i] * w[i + 1] for i in range(n_layers(cfg)))
    return fwd, 2 * fwd


def corrupt(rng, x, kind, level, low=0.0, high=1.0):
    """The corruptions of Vincent et al. (2010, Section 3.3). Returns x~ and the (B, D) mask of corrupted components.

    "gaussian": x + N(0, level^2), every component corrupted. "mask": a fraction level of the components of each
    example, chosen at random, set to 0. "salt_pepper": the same fraction set to low or high by a fair coin."""
    if kind == "gaussian":
        return x + level * rng.normal(size=x.shape), np.ones(x.shape, dtype=bool)
    B, D = x.shape
    k = int(round(level * D))
    chosen = np.argsort(rng.random((B, D)), axis=1)[:, :k]
    mask = np.zeros((B, D), dtype=bool)
    np.put_along_axis(mask, chosen, True, axis=1)
    if kind == "mask":
        return np.where(mask, 0.0, x), mask
    if kind == "salt_pepper":
        return np.where(mask, np.where(rng.random((B, D)) < 0.5, high, low), x), mask
    raise ValueError(kind)


def emphasis(mask, alpha, beta):
    """The weights of the emphasized loss: alpha on the corrupted components and beta on the others."""
    return np.where(mask, alpha, beta)


def jacobian(p, cfg, x1, upto="code"):
    """The Jacobian of the code (upto="code") or of the reconstruction z (upto="recon") in one input x1 of size D,
    as a (n_out, D) matrix, by the chain rule through the layers."""
    K = n_layers(cfg)
    layers = [(p["enc_W"][i], p["enc_b"][i], enc_act(cfg, i)) for i in range(K)]
    if upto == "recon":
        layers += [(W, p["dec_b"][j], dec_act(cfg, j)) for j, W in enumerate(dec_weights(p, cfg))]
    h, J = x1, np.eye(x1.shape[0])
    for W, b, name in layers:
        a = h @ W + b
        h = act(name, a)
        J = act_grad(name, a, h)[:, None] * (W.T @ J)
    return J


def contractive_penalty(p, cfg, x):
    """||J_f(x)||_F^2 averaged over the batch for an encoder of one sigmoid layer, and its gradients in W and b.

    For h = sigmoid(x W + b) the Jacobian has entries h_j (1 - h_j) W_ij, so ||J||_F^2 = sum_j (h_j (1 - h_j))^2
    sum_i W_ij^2 (Rifai et al., 2011, p. 4)."""
    assert n_layers(cfg) == 1 and cfg["code_act"] == "sigmoid", "the closed form is that of one sigmoid layer"
    W, b = p["enc_W"][0], p["enc_b"][0]
    h = sigmoid(x @ W + b)
    g2, s = (h * (1.0 - h)) ** 2, (W**2).sum(0)  # (B, C) and (C,)
    B = x.shape[0]
    r = 2.0 * s * g2 * (1.0 - 2.0 * h) / B  # d penalty / d pre-activation
    return float(np.sum(g2 * s) / B), {"W": 2.0 * W * g2.sum(0) / B + x.T @ r, "b": r.sum(0)}


def tree_map(f, *trees):
    return {k: [f(*vs) for vs in zip(*(t[k] for t in trees))] for k in trees[0]}


def objective(p, cfg, x, rng=None, noise=None, alpha_beta=None, contraction=0.0, weight_decay=0.0):
    """One evaluation of the training objective on a batch and its gradients.

    noise = (kind, level) corrupts the input and keeps the clean x as the target (the denoising autoencoder);
    alpha_beta = (alpha, beta) weights the corrupted and the other components; contraction is the lambda of the
    contractive penalty; weight_decay is the lambda of lambda * sum ||W||_F^2 over the weight matrices."""
    xin, weight = x, None
    if noise is not None:
        xin, mask = corrupt(rng, x, *noise)
        if alpha_beta is not None:
            weight = emphasis(mask, *alpha_beta)
    out, cache = forward(p, cfg, xin)
    value, da = loss(cfg, x, out["a"], weight)
    grads, _ = backward(p, cfg, cache, da)
    if contraction:
        pen, gp = contractive_penalty(p, cfg, xin)
        value += contraction * pen
        grads["enc_W"][0] = grads["enc_W"][0] + contraction * gp["W"]
        grads["enc_b"][0] = grads["enc_b"][0] + contraction * gp["b"]
    if weight_decay:
        for k in ("enc_W", "dec_W"):
            if k in p:
                value += weight_decay * sum(float(np.sum(W**2)) for W in p[k])
                grads[k] = [G + 2.0 * weight_decay * W for G, W in zip(grads[k], p[k])]
    return value, grads


def train(rng, cfg, X, steps, lr=1e-2, batch=32, p=None, **kw):
    """Adam on minibatches of the rows of X; kw goes to objective. Returns the parameters and the losses."""
    p = init(rng, cfg) if p is None else p
    m, v = tree_map(np.zeros_like, p), tree_map(np.zeros_like, p)
    b1, b2, eps, history = 0.9, 0.999, 1e-8, []
    for t in range(1, steps + 1):
        idx = rng.choice(X.shape[0], size=min(batch, X.shape[0]), replace=False)
        value, g = objective(p, cfg, X[idx], rng=rng, **kw)
        m = tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
        v = tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
        c1, c2 = 1 - b1**t, 1 - b2**t
        p = tree_map(lambda w, a, b: w - lr * (a / c1) / (np.sqrt(b / c2) + eps), p, m, v)
        history.append(value)
    return p, history


def pca(X, m):
    """The mean, the first m principal directions as a (D, m) matrix, and all the squared singular values of the
    centred data, which is what a linear autoencoder with a code of m finds the span of."""
    mean = X.mean(0)
    U, s, _ = np.linalg.svd((X - mean).T, full_matrices=False)
    return mean, U[:, :m], s**2


def pca_reconstruct(X, m):
    mean, P, _ = pca(X, m)
    return mean + (X - mean) @ P @ P.T
