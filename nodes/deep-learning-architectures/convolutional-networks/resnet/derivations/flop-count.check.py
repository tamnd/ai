# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Numeric check of the derivation flop-count: sum the multiply-adds of every convolution and compare with the closed form."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import resnet  # noqa: E402


def closed_form(w, N, R, K):
    return w * w * R * R * (17 * N + 19) // 16 + 147 * w * R * R // 4 + 32 * w * K


def layer_by_layer(cfg, R):
    """Multiply-adds of the convolutions listed by the reference, plus the head."""
    convs = sum(k * k * c_in * c_out * h * h for c_in, c_out, k, h in resnet.convs(cfg, R))
    return convs + 32 * cfg["width"] * cfg["classes"]


for w, blocks, R, K in [(2, (1, 1, 1, 1), 32, 5), (8, (2, 3, 1, 2), 64, 7), (64, (3, 4, 6, 3), 224, 1000),
                        (64, (3, 4, 23, 3), 224, 1000), (64, (3, 8, 36, 3), 224, 1000)]:
    cfg = resnet.config(width=w, blocks=blocks, classes=K)
    m = closed_form(w, sum(blocks), R, K)
    assert layer_by_layer(cfg, R) == m == resnet.flops(cfg, 1, R) // 2, (w, blocks, R)

macs = [closed_form(64, N, 224, 1000) for N in (16, 33, 50)]
assert macs == [3_857_973_248, 7_570_194_432, 11_282_415_616]
table1 = [3.8, 7.6, 11.3]
assert [round(m / 1e9, 1) for m in macs[1:]] == table1[1:]
assert all(abs(m / 1e9 - t) < 0.07 for m, t in zip(macs, table1))

# A block that is not the first of its stage costs the same in every stage.
w, R = 64, 224
rest = [17 * (w * 2**s) ** 2 * (R // (4 * 2**s)) ** 2 for s in range(4)]
assert len(set(rest)) == 1 and rest[0] == 17 * w * w * R * R // 16

# The stride in the 3x3 convolution: same parameters, 18 w^2 R^2 / 16 more multiply-adds.
a = resnet.config(stride_on="1x1")
b = resnet.config(stride_on="3x3")
assert resnet.param_count(a) == resnet.param_count(b)
extra = 18 * w * w * R * R // 16
assert extra == 231_211_008
assert layer_by_layer(b, R) == macs[0] + extra == 4_089_184_256
print(f"flop-count: closed form matches layer sums; ResNet-50/101/152 {[round(m / 1e9, 2) for m in macs]} G; stride in 3x3: {layer_by_layer(b, R) / 1e9:.2f} G")
