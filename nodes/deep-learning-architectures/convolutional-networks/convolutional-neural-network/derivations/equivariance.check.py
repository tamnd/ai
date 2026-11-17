# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/convolutional-neural-network/ (ai://GNXZCP).
"""Numeric check of the derivation equivariance on two-dimensional, multichannel inputs."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import cnn  # noqa: E402

rng = np.random.default_rng(1)


def shift(x, t, u=0):
    return np.roll(x, (t, u), axis=(2, 3))  # (T_t x)[i] = x[i - t]


x = rng.normal(size=(2, 3, 12, 12))
w = rng.normal(size=(4, 3, 3, 3))
b = rng.normal(size=4)

# 1. stride 1, circular padding: equivariant to every shift
for t in range(-5, 6):
    for u in (0, 2, 7):
        lhs = cnn.conv2d(shift(x, t, u), w, b, pad=1, mode="circular")
        assert np.allclose(lhs, shift(cnn.conv2d(x, w, b, pad=1, mode="circular"), t, u))

# 2. no padding: the outputs of the windows inside the moved part move with it
big = rng.normal(size=(1, 3, 16, 16))
y = cnn.conv2d(big[:, :, :12, :12], w)  # (1, 4, 10, 10)
y2 = cnn.conv2d(big[:, :, 2:14, 1:13], w)  # the input moved by (-2, -1)
assert np.allclose(y[:, :, 2:, 1:], y2[:, :, :-2, :-1])
# zero padding: the border outputs do not move with the signal
z = cnn.conv2d(shift(x, 1), w, pad=1)
assert not np.allclose(z, shift(cnn.conv2d(x, w, pad=1), 1))
assert np.allclose(z[:, :, 2:-1], shift(cnn.conv2d(x, w, pad=1), 1)[:, :, 2:-1])

# 3. a stack with ReLU stays equivariant
w2 = rng.normal(size=(2, 4, 3, 3))


def net(v):
    return cnn.conv2d(cnn.relu(cnn.conv2d(v, w, b, pad=1, mode="circular")), w2, pad=1, mode="circular")


assert np.allclose(net(shift(x, 3, -2)), shift(net(x), 3, -2))

# 4. stride s: periodic-s equivariance, for the convolution and for max pooling
for s in (2, 3):
    for t in (1, 2):
        lhs = cnn.conv2d(shift(x, s * t), w, stride=s, pad=1, mode="circular")
        assert np.allclose(lhs, shift(cnn.conv2d(x, w, stride=s, pad=1, mode="circular"), t))
assert not np.allclose(cnn.conv2d(shift(x, 1), w, stride=2, pad=1, mode="circular"),
                       shift(cnn.conv2d(x, w, stride=2, pad=1, mode="circular"), 1))
assert np.allclose(cnn.max_pool(shift(x, 2)), shift(cnn.max_pool(x), 1))

# 5. global pooling after equivariant layers: invariant
for g in (lambda v: v.max(axis=(2, 3)), lambda v: v.mean(axis=(2, 3))):
    assert np.allclose(g(net(shift(x, 5, 4))), g(net(x)))

# 6. convolution = cross-correlation with the flipped kernel, offset by k - 1
for _ in range(20):
    sig, v = rng.normal(size=11), rng.normal(size=4)
    full = np.convolve(sig, v, mode="full")
    img, ker = np.zeros((1, 1, 4, 11)), np.zeros((1, 1, 4, 4))  # one nonzero row in each: a 1-D correlation
    img[0, 0, 0], ker[0, 0, 0] = sig, v[::-1]
    xc = cnn.conv2d(img, ker)  # (1, 1, 1, 8)
    assert np.allclose(xc[0, 0, 0], full[3:11])
print("equivariance: ok")
