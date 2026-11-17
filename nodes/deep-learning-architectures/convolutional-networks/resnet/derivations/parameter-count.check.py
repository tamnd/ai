# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Numeric check of the derivation parameter-count: count the entries of built networks, then evaluate the closed forms."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import resnet  # noqa: E402


def closed_form(w, blocks, K):
    m = [w * 2**s for s in range(4)]
    p = 149 * w + 18 * w * w + 20 * w
    p += sum(23 * m[s] ** 2 + 20 * m[s] for s in (1, 2, 3))
    p += sum((n - 1) * (17 * m[s] ** 2 + 12 * m[s]) for s, n in enumerate(blocks))
    return p + 32 * w * K + K


def leaves(tree):
    if isinstance(tree, dict):
        return sum(leaves(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(leaves(v) for v in tree)
    return np.asarray(tree).size


rng = np.random.default_rng(0)
for w, blocks, K in [(2, (1, 1, 1, 1), 5), (4, (2, 1, 3, 2), 3), (8, (3, 4, 6, 3), 10)]:
    cfg = resnet.config(width=w, blocks=blocks, classes=K)
    built = leaves(resnet.init(rng, cfg))
    assert built == closed_form(w, blocks, K) == resnet.param_count(cfg), (w, blocks, built)

counts = {b: closed_form(64, b, 1000) for b in [(3, 4, 6, 3), (3, 4, 23, 3), (3, 8, 36, 3)]}
assert list(counts.values()) == [25_557_032, 44_549_160, 60_192_808], counts

block = 256 * 64 + 9 * 64 * 64 + 64 * 256
assert block == 69_632 and 256 * 256 == 65_536
assert 0.9 < 65_536 / block < 1.0


def cifar(n):
    return 464 + 4672 * n + 13952 + 18560 * (n - 1) + 55552 + 73984 * (n - 1) + 650


def cifar_built(n):
    convs = [(3, 16)] + [(16, 16)] * 2 * n
    convs += [(16, 32)] + [(32, 32)] * (2 * n - 1)
    convs += [(32, 64)] + [(64, 64)] * (2 * n - 1)
    return sum(9 * a * b + 2 * b for a, b in convs) + 64 * 10 + 10


for n in (3, 5, 7, 9, 18, 200):
    assert cifar(n) == cifar_built(n), n
assert cifar(3) == 269_722 and cifar(200) == 19_421_274
table6 = {3: 0.27, 5: 0.46, 7: 0.66, 9: 0.85, 18: 1.7}  # Table 6, in millions
assert all(round(cifar(n) / 1e6, 2 if n < 18 else 1) == v for n, v in table6.items())
assert round(cifar(3) / 1e6, 2) == 0.27 and round(cifar(200) / 1e6, 1) == 19.4
print(f"parameter-count: built trees match; ResNet-50/101/152 {list(counts.values())}; block {block} + projection 65536; CIFAR n=3 {cifar(3)}, n=200 {cifar(200)}")
