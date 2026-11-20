# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""The residual network of He et al. (2016) as pure functions, the JAX reference.

It computes eq-residual, eq-projection and eq-bottleneck of the page with jax.lax, as the NumPy reference does,
and leaves the gradient to jax.grad. init and apply use the parameter tree of the NumPy reference, so the shared
fixture loads as it is. With state=None the norms use the statistics of the batch; with a state, the running ones.
"""

import jax
import jax.numpy as jnp

EPS = 1e-5
STAGES = 4
EXPANSION = 4


def config(width=64, blocks=(3, 4, 6, 3), classes=1000, stride_on="1x1"):
    """Return the config. stride_on is "1x1", as in the paper, or "3x3", as in the ResNet-50 v1.5 of later code."""
    if len(blocks) != STAGES or min(width, classes, *blocks) < 1 or stride_on not in ("1x1", "3x3"):
        raise ValueError("width, classes and the 4 block counts must be at least 1; stride_on is 1x1 or 3x3")
    cfg = {"width": width, "classes": classes, "stride_on": stride_on, "blocks": tuple(blocks)}
    cfg.update({f"n{s + 1}": n for s, n in enumerate(blocks)})
    return cfg


def param_count(cfg):
    """The closed form of shapes.yaml, summed over the stages."""
    w, K = cfg["width"], cfg["classes"]
    n, c_in = 149 * w + 32 * w * K + K, w
    for s, blocks in enumerate(cfg["blocks"]):
        m = w * 2**s
        n += 5 * m * c_in + 13 * m * m + 20 * m + (blocks - 1) * (17 * m * m + 12 * m)
        c_in = EXPANSION * m
    return n


def init(key, cfg, zero_gamma=False):
    """Return the parameter tree of the NumPy reference, with He et al. (2015) initialization of the convolutions."""
    w, K = cfg["width"], cfg["classes"]
    keys = iter(jax.random.split(key, 4 * sum(cfg["blocks"]) + 2))

    def conv(c_out, c_in, k):
        return jax.random.normal(next(keys), (c_out, c_in, k, k)) * (2.0 / (k * k * c_in)) ** 0.5

    def bn(c, gamma=1.0):
        return {"gamma": jnp.full((c,), gamma), "beta": jnp.zeros(c)}

    params = {"stem": {"conv": conv(w, 3, 7), "bn": bn(w)}, "stages": []}
    c_in = w
    for s, n in enumerate(cfg["blocks"]):
        m, stage = w * 2**s, []
        for i in range(n):
            p = {
                "conv1": conv(m, c_in, 1),
                "bn1": bn(m),
                "conv2": conv(m, m, 3),
                "bn2": bn(m),
                "conv3": conv(EXPANSION * m, m, 1),
                "bn3": bn(EXPANSION * m, 0.0 if zero_gamma else 1.0),
            }
            if i == 0:
                p["proj"], p["bn_proj"] = conv(EXPANSION * m, c_in, 1), bn(EXPANSION * m)
            stage.append(p)
            c_in = EXPANSION * m
        params["stages"].append(stage)
    params["fc"] = {"W": jax.random.normal(next(keys), (K, c_in)) * c_in**-0.5, "b": jnp.zeros(K)}
    return params


def conv2d(x, W, stride=1, pad=0):
    return jax.lax.conv_general_dilated(
        x, W, (stride, stride), [(pad, pad), (pad, pad)], dimension_numbers=("NCHW", "OIHW", "NCHW")
    )


def max_pool(x):
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, 3, 3), (1, 1, 2, 2), [(0, 0), (0, 0), (1, 1), (1, 1)])


def batch_norm(x, p, st=None):
    if st is None:
        mean, var = x.mean(axis=(0, 2, 3)), x.var(axis=(0, 2, 3))
    else:
        mean, var = st["mean"], st["var"]
    r = (slice(None), None, None)
    return p["gamma"][r] * (x - mean[r]) / jnp.sqrt(var[r] + EPS) + p["beta"][r]


def bottleneck(p, x, stride, stride_on="1x1", st=None):
    st = st or {}
    s1, s2 = (stride, 1) if stride_on == "1x1" else (1, stride)
    a = jax.nn.relu(batch_norm(conv2d(x, p["conv1"], s1), p["bn1"], st.get("bn1")))
    b = jax.nn.relu(batch_norm(conv2d(a, p["conv2"], s2, 1), p["bn2"], st.get("bn2")))
    c = batch_norm(conv2d(b, p["conv3"]), p["bn3"], st.get("bn3"))  # eq: eq-bottleneck
    if "proj" in p:
        shortcut = batch_norm(conv2d(x, p["proj"], stride), p["bn_proj"], st.get("bn_proj"))  # eq: eq-projection
    else:
        shortcut = x
    y = jax.nn.relu(c + shortcut)  # eq: eq-residual
    return y, {"a": a, "b": b, "c": c, "shortcut": shortcut, "y": y}


def apply(params, x, cfg, state=None):
    """Return (logits (B, K), cache) for images x (B, 3, R, R)."""
    st = state or {"stem": {}, "stages": [[{} for _ in stage] for stage in params["stages"]]}
    conv = conv2d(x, params["stem"]["conv"], 2, 3)
    h = max_pool(jax.nn.relu(batch_norm(conv, params["stem"]["bn"], st["stem"].get("bn"))))
    cache = {"stem": {"conv": conv, "y": h}, "blocks": {}}
    for s, stage in enumerate(params["stages"]):
        for i, p in enumerate(stage):
            stride = 2 if (s > 0 and i == 0) else 1
            h, cache["blocks"][(s, i)] = bottleneck(p, h, stride, cfg["stride_on"], st["stages"][s][i])
    pooled = h.mean(axis=(2, 3))
    cache["pooled"] = pooled
    return pooled @ params["fc"]["W"].T + params["fc"]["b"], cache
