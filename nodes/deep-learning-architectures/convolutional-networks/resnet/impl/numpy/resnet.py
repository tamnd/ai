# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""The residual network of He et al. (2016), the NumPy reference.

It computes eq-residual, eq-projection and eq-bottleneck of the page for the ImageNet networks with bottleneck
blocks (ResNet-50, -101 and -152 and their smaller kin): a 7x7 stem, four stages of bottleneck blocks, global
average pooling and one fully connected layer. Batch normalization follows every convolution. With state=None
it normalizes with the statistics of the batch, as in training; with a state of running means and variances, it
normalizes with those, as at inference. It is the forward pass only.

The same file holds the fully connected residual network of the experiments of the page, with its backward pass:
dense_init, dense_forward and dense_backward. Its blocks are the two-layer basic blocks of eq-residual, written
with matrix products instead of convolutions.
"""

import numpy as np

EPS = 1e-5  # the eps of batch normalization; the paper does not give one, 1e-5 is the one of Ioffe and Szegedy's code
STAGES = 4
EXPANSION = 4  # a block of stage s maps 4 m channels to m and back to 4 m, with m = width * 2^s


def config(width=64, blocks=(3, 4, 6, 3), classes=1000, stride_on="1x1"):
    """Return the config. stride_on is "1x1", as in the paper, or "3x3", as in the ResNet-50 v1.5 of later code."""
    if len(blocks) != STAGES or min(width, classes, *blocks) < 1 or stride_on not in ("1x1", "3x3"):
        raise ValueError("width, classes and the 4 block counts must be at least 1; stride_on is 1x1 or 3x3")
    cfg = {"width": width, "classes": classes, "stride_on": stride_on, "blocks": tuple(blocks)}
    cfg.update({f"n{s + 1}": n for s, n in enumerate(blocks)})
    return cfg


def depth(cfg):
    """The depth as the paper counts it: the stem, three convolutions per block and the fully connected layer."""
    return 3 * sum(cfg["blocks"]) + 2


def plan(cfg):
    """Yield (stage, index, c_in, mid, c_out, stride, projection) for every block, in order."""
    c_in = cfg["width"]
    for s, n in enumerate(cfg["blocks"]):
        mid = cfg["width"] * 2**s
        for i in range(n):
            stride = 2 if (s > 0 and i == 0) else 1
            yield s, i, c_in, mid, EXPANSION * mid, stride, i == 0
            c_in = EXPANSION * mid


def convs(cfg, R):
    """Yield (c_in, c_out, k, h_out) for every convolution on an R by R image, the projections included."""
    w = cfg["width"]
    yield 3, w, 7, R // 2
    h = R // 4  # after the stride 2 of the stem and of the max pooling
    for _, _, c_in, mid, c_out, stride, proj in plan(cfg):
        h_out = h // stride
        h_1 = h_out if cfg["stride_on"] == "1x1" else h  # the 1x1 runs at the input size when the 3x3 strides
        yield c_in, mid, 1, h_1
        yield mid, mid, 3, h_out
        yield mid, c_out, 1, h_out
        if proj:
            yield c_in, c_out, 1, h_out
        h = h_out


def param_count(cfg):
    """Convolution weights (no biases), the scale and shift of each batch normalization, and the classifier."""
    n = sum(c_in * c_out * k * k + 2 * c_out for c_in, c_out, k, _ in convs(cfg, 32))
    return n + EXPANSION * 8 * cfg["width"] * cfg["classes"] + cfg["classes"]


def flops(cfg, B, R):
    """FLOPs of one forward pass of B images of R by R pixels, a multiply-add counted as 2, as in shapes.yaml.

    Batch normalization, ReLU, the poolings and the additions of the shortcuts are left out.
    """
    if R % 32:
        raise ValueError("R must be a multiple of 32")
    macs = sum(c_in * c_out * k * k * h * h for c_in, c_out, k, h in convs(cfg, R))
    return 2 * B * (macs + 32 * cfg["width"] * cfg["classes"])


def _conv_w(rng, c_out, c_in, k):
    """He et al. (2015) initialization: a Gaussian with variance 2 / (k^2 c_in)."""
    return rng.normal(0.0, np.sqrt(2.0 / (k * k * c_in)), (c_out, c_in, k, k))  # (c_out, c_in, k, k)


def _bn(c):
    return {"gamma": np.ones(c), "beta": np.zeros(c)}  # (c,)


def init(rng, cfg, zero_gamma=False):
    """Return the parameter tree. zero_gamma sets the scale of the last norm of each block to 0 (Goyal et al. 2017)."""
    w, K = cfg["width"], cfg["classes"]
    params = {"stem": {"conv": _conv_w(rng, w, 3, 7), "bn": _bn(w)}, "stages": [[] for _ in range(STAGES)]}
    for s, _, c_in, mid, c_out, _, proj in plan(cfg):
        p = {
            "conv1": _conv_w(rng, mid, c_in, 1),
            "bn1": _bn(mid),
            "conv2": _conv_w(rng, mid, mid, 3),
            "bn2": _bn(mid),
            "conv3": _conv_w(rng, c_out, mid, 1),
            "bn3": _bn(c_out),
        }
        if zero_gamma:
            p["bn3"]["gamma"][:] = 0.0
        if proj:
            p["proj"], p["bn_proj"] = _conv_w(rng, c_out, c_in, 1), _bn(c_out)
        params["stages"][s].append(p)
    params["fc"] = {"W": rng.normal(0.0, (32 * w) ** -0.5, (K, 32 * w)), "b": np.zeros(K)}  # (K, 32C), (K,)
    return params


def init_state(params):
    """Running statistics for every batch normalization of the tree: mean 0 and variance 1."""
    if isinstance(params, dict) and "gamma" in params:
        return {"mean": np.zeros_like(params["gamma"]), "var": np.ones_like(params["gamma"])}
    if isinstance(params, dict):
        return {k: init_state(v) for k, v in params.items() if isinstance(v, (dict, list))}
    return [init_state(v) for v in params]


def conv2d(x, W, stride=1, pad=0):
    """x (B, c_in, H, H) and W (c_out, c_in, k, k) give (B, c_out, H', H'), with H' = (H + 2 pad - k) // stride + 1."""
    k = W.shape[-1]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    return np.tensordot(win, W, axes=((1, 4, 5), (1, 2, 3))).transpose(0, 3, 1, 2)  # (B, c_out, H', H')


def max_pool(x):
    """3x3 max pooling with stride 2 and padding 1: (B, C, H, H) to (B, C, H / 2, H / 2)."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    return np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))[:, :, ::2, ::2].max(axis=(4, 5))


def batch_norm(x, p, st=None):
    """Normalize each channel of x (B, C, H, H), by the batch statistics or by the running ones in st."""
    if st is None:
        mean, var = x.mean(axis=(0, 2, 3)), x.var(axis=(0, 2, 3))
    else:
        mean, var = st["mean"], st["var"]
    r = (slice(None), None, None)
    return p["gamma"][r] * (x - mean[r]) / np.sqrt(var[r] + EPS) + p["beta"][r]


def relu(x):
    return np.maximum(x, 0.0)


def bottleneck(p, x, stride, stride_on="1x1", st=None):
    """One bottleneck block. Return y and the cache of its internals a, b, c and shortcut."""
    st = st or {}
    s1, s2 = (stride, 1) if stride_on == "1x1" else (1, stride)
    a = relu(batch_norm(conv2d(x, p["conv1"], s1), p["bn1"], st.get("bn1")))  # (B, m, H', H')
    b = relu(batch_norm(conv2d(a, p["conv2"], s2, pad=1), p["bn2"], st.get("bn2")))  # (B, m, H', H')
    c = batch_norm(conv2d(b, p["conv3"]), p["bn3"], st.get("bn3"))  # eq: eq-bottleneck  (B, 4m, H', H')
    if "proj" in p:
        shortcut = batch_norm(conv2d(x, p["proj"], stride), p["bn_proj"], st.get("bn_proj"))  # eq: eq-projection
    else:
        shortcut = x  # the identity shortcut adds no parameter
    y = relu(c + shortcut)  # eq: eq-residual  (B, 4m, H', H')
    return y, {"a": a, "b": b, "c": c, "shortcut": shortcut}


def forward(params, x, cfg, state=None):
    """Return (logits (B, K), cache) for images x (B, 3, R, R), R a multiple of 32."""
    st = state or {"stem": {}, "stages": [[{} for _ in stage] for stage in params["stages"]]}
    conv = conv2d(x, params["stem"]["conv"], 2, pad=3)  # (B, C, R/2, R/2)
    h = max_pool(relu(batch_norm(conv, params["stem"]["bn"], st["stem"].get("bn"))))  # (B, C, R/4, R/4)
    cache = {"stem": {"conv": conv, "y": h}, "blocks": {}}
    for s, i, _, _, _, stride, _ in plan(cfg):
        h, cache["blocks"][(s, i)] = bottleneck(params["stages"][s][i], h, stride, cfg["stride_on"], st["stages"][s][i])
        cache["blocks"][(s, i)]["y"] = h
    pooled = h.mean(axis=(2, 3))  # (B, 32C)
    cache["pooled"] = pooled
    return pooled @ params["fc"]["W"].T + params["fc"]["b"], cache  # (B, K)


# The fully connected residual network of the experiments.


def dense_init(rng, d_in, d, n_blocks, classes, dtype=np.float64):
    """An input layer d_in -> d, n_blocks blocks of two d -> d layers, and a classifier d -> classes.

    Return (params, state): the weights, scales and shifts, and the running statistics of the batch norms.
    """

    def lin(n_in, n_out, gain=2.0):
        return rng.normal(0.0, np.sqrt(gain / n_in), (n_in, n_out)).astype(dtype)

    p = {"W0": lin(d_in, d), "g0": np.ones(d, dtype), "b0": np.zeros(d, dtype)}
    for i in range(n_blocks):
        for j in (1, 2):
            p[f"W{i}_{j}"], p[f"g{i}_{j}"], p[f"b{i}_{j}"] = lin(d, d), np.ones(d, dtype), np.zeros(d, dtype)
    p["Wo"], p["bo"] = lin(d, classes, 1.0), np.zeros(classes, dtype)
    norms = [k[1:] for k in p if k.startswith("g")]
    state = {f"m{k}": np.zeros(d, dtype) for k in norms} | {f"v{k}": np.ones(d, dtype) for k in norms}
    return p, state


def _bn_dense(z, g, b, state, key, train, momentum):
    if train:
        mean, var = z.mean(axis=0), z.var(axis=0)
        n = len(z)
        state["m" + key] = (1 - momentum) * state["m" + key] + momentum * mean
        state["v" + key] = (1 - momentum) * state["v" + key] + momentum * var * n / (n - 1)
    else:
        mean, var = state["m" + key], state["v" + key]
    inv = 1.0 / np.sqrt(var + EPS)
    zh = (z - mean) * inv
    return g * zh + b, (zh, inv, g, train)


def _bn_dense_back(dy, cache):
    zh, inv, g, train = cache
    dg, db, dzh = (dy * zh).sum(axis=0), dy.sum(axis=0), dy * g
    if not train:  # the statistics are constants
        return dzh * inv, dg, db
    n = len(dy)
    return inv / n * (n * dzh - dzh.sum(axis=0) - zh * (dzh * zh).sum(axis=0)), dg, db


def dense_forward(p, state, x, residual, train=False, momentum=0.1):
    """Return (logits, cache). In train mode the norms use the batch statistics and update the running ones."""
    n = sum(1 for k in p if k.endswith("_1") and k.startswith("W"))
    y, c0 = _bn_dense(x @ p["W0"], p["g0"], p["b0"], state, "0", train, momentum)
    h = relu(y)
    cache = {"x": x, "in": (c0, h), "blocks": []}
    for i in range(n):
        y1, c1 = _bn_dense(h @ p[f"W{i}_1"], p[f"g{i}_1"], p[f"b{i}_1"], state, f"{i}_1", train, momentum)
        a = relu(y1)
        f, c2 = _bn_dense(a @ p[f"W{i}_2"], p[f"g{i}_2"], p[f"b{i}_2"], state, f"{i}_2", train, momentum)
        h_next = relu(f + h if residual else f)  # eq: eq-residual, with F two layers and the identity shortcut
        cache["blocks"].append((h, c1, a, c2, h_next))
        h = h_next
    cache["h"] = h
    return h @ p["Wo"] + p["bo"], cache


def dense_backward(p, cache, dlogits, residual):
    """The gradient of the loss with respect to every parameter, given its gradient dlogits at the logits."""
    g = {"Wo": cache["h"].T @ dlogits, "bo": dlogits.sum(axis=0)}
    dh = dlogits @ p["Wo"].T
    for i in reversed(range(len(cache["blocks"]))):
        h, c1, a, c2, h_next = cache["blocks"][i]
        ds = dh * (h_next > 0)
        dz2, g[f"g{i}_2"], g[f"b{i}_2"] = _bn_dense_back(ds, c2)
        g[f"W{i}_2"] = a.T @ dz2
        dz1, g[f"g{i}_1"], g[f"b{i}_1"] = _bn_dense_back((dz2 @ p[f"W{i}_2"].T) * (a > 0), c1)
        g[f"W{i}_1"] = h.T @ dz1
        dh = dz1 @ p[f"W{i}_1"].T + (ds if residual else 0.0)  # eq: eq-residual-grad, the shortcut passes ds as it is
    c0, h0 = cache["in"]
    dz, g["g0"], g["b0"] = _bn_dense_back(dh * (h0 > 0), c0)
    g["W0"] = cache["x"].T @ dz
    return g


def cross_entropy(logits, y):
    """The mean cross-entropy over the batch and its gradient with respect to the logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(prob[np.arange(n), y] + 1e-12).mean()
    prob[np.arange(n), y] -= 1.0
    return loss, prob / n
