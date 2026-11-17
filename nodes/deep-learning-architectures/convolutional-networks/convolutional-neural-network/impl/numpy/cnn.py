# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""The convolutional layer and a small convolutional network, the NumPy reference.

conv2d computes eq-conv of the page, the cross-correlation that deep learning libraries call convolution, with a
stride, zero or circular padding and several input and output channels. It forms the windows of the input with a
strided view (im2col) and contracts them with the kernel; conv2d_loop computes the same sum with explicit loops
and is the oracle of the tests. conv2d_backward gives the gradients of a loss with respect to the input, the
kernel and the bias. out_size is eq-out-size, receptive_field is eq-rf, and conv_params is eq-conv-params.

The network of shapes.yaml is LeNet-5 with the substitutions of later practice: two 5x5 convolutions without
padding, each followed by a ReLU and a 2x2 max pooling with stride 2, then three fully connected layers. Every
feature map of the second convolution sees every map of the first, and the output layer is linear. It is the
forward pass only.
"""

import numpy as np

KERNEL = 5  # the kernel size of both convolutions of the network, as in LeNet-5
POOL = 2  # the window and the stride of both max poolings


def out_size(n, k, s=1, p=0):
    """The number of positions of a k-wide window with stride s on n inputs padded by p on each side."""
    if min(n, k, s) < 1 or p < 0 or n + 2 * p < k:
        raise ValueError("need n, k, s >= 1, p >= 0 and n + 2p >= k")
    return (n + 2 * p - k) // s + 1  # eq: eq-out-size


def receptive_field(layers):
    """Return (r, j) for a stack of (k, s) layers: the receptive field and the jump of the last layer, in inputs."""
    r, j = 1, 1
    for k, s in layers:
        r, j = r + (k - 1) * j, j * s  # eq: eq-rf
    return r, j


def conv_params(k, c_in, c_out, bias=True):
    """The parameters of a k by k convolution from c_in to c_out channels."""
    return k * k * c_in * c_out + (c_out if bias else 0)  # eq: eq-conv-params


def _pad(x, p, mode):
    if mode not in ("zeros", "circular"):
        raise ValueError("mode is zeros or circular")
    width = ((0, 0), (0, 0), (p, p), (p, p))
    return np.pad(x, width, mode="wrap") if mode == "circular" else np.pad(x, width)


def im2col(x, k, stride=1, pad=0, mode="zeros"):
    """The windows of x (B, C, H, W): an array (B, C, H', W', k, k), a view of the padded input."""
    xp = _pad(x, pad, mode)
    out_size(x.shape[2], k, stride, pad)  # checks that a window fits
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))  # (B, C, H + 2p - k + 1, W + 2p - k + 1, k, k)
    return win[:, :, ::stride, ::stride]  # (B, C, H', W', k, k)


def conv2d(x, w, b=None, stride=1, pad=0, mode="zeros"):
    """x (B, C, H, W) and w (D, C, k, k) give (B, D, H', W') with H' = out_size(H, k, stride, pad)."""
    if x.shape[1] != w.shape[1] or w.shape[2] != w.shape[3]:
        raise ValueError("w must be (D, C, k, k) with the C of x")
    cols = im2col(x, w.shape[2], stride, pad, mode)  # (B, C, H', W', k, k)
    y = np.einsum("bchwuv,dcuv->bdhw", cols, w)  # eq: eq-conv  (B, D, H', W')
    return y if b is None else y + b[None, :, None, None]


def conv2d_loop(x, w, b=None, stride=1, pad=0, mode="zeros"):
    """The same sum as conv2d, one output entry at a time: the oracle of the tests."""
    xp = _pad(x, pad, mode)
    B, _, H, W = x.shape
    D, C, k, _ = w.shape
    Ho, Wo = out_size(H, k, stride, pad), out_size(W, k, stride, pad)
    y = np.zeros((B, D, Ho, Wo), dtype=np.result_type(x, w))  # (B, D, H', W')
    for n in range(B):
        for d in range(D):
            for i in range(Ho):
                for j in range(Wo):
                    acc = 0.0 if b is None else b[d]
                    for c in range(C):
                        for u in range(k):
                            for v in range(k):
                                acc += w[d, c, u, v] * xp[n, c, stride * i + u, stride * j + v]
                    y[n, d, i, j] = acc
    return y


def conv2d_backward(x, w, dy, stride=1, pad=0):
    """Gradients of a loss L for zero padding, given dy = dL/dy (B, D, H', W'): return (dx, dw, db)."""
    k = w.shape[2]
    cols = im2col(x, k, stride, pad)  # (B, C, H', W', k, k)
    dw = np.einsum("bchwuv,bdhw->dcuv", cols, dy)  # (D, C, k, k)
    db = dy.sum(axis=(0, 2, 3))  # (D,)
    dcols = np.einsum("bdhw,dcuv->bchwuv", dy, w)  # (B, C, H', W', k, k)
    B, C, H, W = x.shape
    dxp = np.zeros((B, C, H + 2 * pad, W + 2 * pad), dtype=dcols.dtype)  # (B, C, H + 2p, W + 2p)
    Ho, Wo = dy.shape[2], dy.shape[3]
    for u in range(k):
        for v in range(k):
            dxp[:, :, u : u + stride * Ho : stride, v : v + stride * Wo : stride] += dcols[:, :, :, :, u, v]
    return dxp[:, :, pad : pad + H, pad : pad + W], dw, db


def max_pool(x, z=POOL, s=POOL):
    """Max over z by z windows with stride s, channel by channel: (B, C, H, W) to (B, C, H', W')."""
    win = np.lib.stride_tricks.sliding_window_view(x, (z, z), axis=(2, 3))[:, :, ::s, ::s]  # (B, C, H', W', z, z)
    return win.max(axis=(4, 5))  # eq: eq-maxpool


def relu(x):
    return np.maximum(x, 0.0)


def config(channels=1, c1=6, c2=16, f1=120, f2=84, classes=10, size=32):
    """Return the config. The defaults are the sizes of LeNet-5; size - 12 must be a positive multiple of 4."""
    if min(channels, c1, c2, f1, f2, classes) < 1 or size <= 12 or (size - 12) % 4:
        raise ValueError("all widths must be at least 1, and size - 12 a positive multiple of 4")
    return {"channels": channels, "c1": c1, "c2": c2, "f1": f1, "f2": f2, "classes": classes, "size": size}


def spatial(cfg):
    """The height of the maps after conv1, pool1, conv2 and pool2."""
    a = out_size(cfg["size"], KERNEL)
    b = out_size(a, POOL, POOL)
    c = out_size(b, KERNEL)
    return a, b, c, out_size(c, POOL, POOL)


def layers(cfg):
    """Yield (name, n_in, n_out, kind) for the five layers with weights."""
    flat = cfg["c2"] * spatial(cfg)[3] ** 2
    yield "conv1", cfg["channels"], cfg["c1"], "conv"
    yield "conv2", cfg["c1"], cfg["c2"], "conv"
    yield "fc1", flat, cfg["f1"], "dense"
    yield "fc2", cfg["f1"], cfg["f2"], "dense"
    yield "fc3", cfg["f2"], cfg["classes"], "dense"


def param_count(cfg):
    """The kernels and biases of the two convolutions and the weights and biases of the three dense layers."""
    return sum(conv_params(KERNEL, a, b) if kind == "conv" else a * b + b for _, a, b, kind in layers(cfg))


def flops(cfg, B):
    """FLOPs of one forward pass over B images, a multiply-add counted as 2; biases, ReLU and pooling left out."""
    a, _, c, _ = spatial(cfg)
    macs = KERNEL**2 * cfg["channels"] * cfg["c1"] * a * a + KERNEL**2 * cfg["c1"] * cfg["c2"] * c * c
    macs += sum(n_in * n_out for _, n_in, n_out, kind in layers(cfg) if kind == "dense")
    return 2 * B * macs


def init(rng, cfg):
    """He initialization for every weight, zero biases."""
    params = {}
    for name, n_in, n_out, kind in layers(cfg):
        if kind == "conv":
            fan = KERNEL * KERNEL * n_in
            w = rng.normal(0.0, np.sqrt(2.0 / fan), (n_out, n_in, KERNEL, KERNEL))  # (D, C, k, k)
        else:
            w = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))  # (n_in, n_out)
        params[name] = {"w": w, "b": np.zeros(n_out)}
    return params


def forward(params, x, cfg):
    """Return (logits (B, K), cache) for images x (B, C, R, R)."""
    p = params
    h1 = relu(conv2d(x, p["conv1"]["w"], p["conv1"]["b"]))  # (B, C1, R - 4, R - 4)
    q1 = max_pool(h1)  # (B, C1, (R - 4) / 2, (R - 4) / 2)
    h2 = relu(conv2d(q1, p["conv2"]["w"], p["conv2"]["b"]))  # (B, C2, (R - 12) / 2, (R - 12) / 2)
    q2 = max_pool(h2)  # (B, C2, (R - 12) / 4, (R - 12) / 4)
    flat = q2.reshape(q2.shape[0], -1)  # (B, C2 * ((R - 12) / 4) ^ 2)
    z1 = relu(flat @ p["fc1"]["w"] + p["fc1"]["b"])  # (B, F1)
    z2 = relu(z1 @ p["fc2"]["w"] + p["fc2"]["b"])  # (B, F2)
    logits = z2 @ p["fc3"]["w"] + p["fc3"]["b"]  # (B, K)
    return logits, {"conv1": h1, "pool1": q1, "conv2": h2, "pool2": q2, "fc1": z1, "fc2": z2}
