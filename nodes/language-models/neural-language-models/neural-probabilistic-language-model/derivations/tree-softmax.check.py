# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""Check of tree-softmax: the leaf probabilities of a binary tree of sigmoid decisions sum to one; Huffman trees of
equal counts have depths floor(log2 V) and ceil(log2 V); and the mean Huffman depth lies in [H, H + 1)."""

import heapq
import math
import random


def huffman_depths(counts):
    heap = [(c, i, (i,)) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    depth = [0] * len(counts)
    tie = len(counts)
    while len(heap) > 1:
        c0, _, a = heapq.heappop(heap)
        c1, _, b = heapq.heappop(heap)
        for w in a + b:
            depth[w] += 1
        heapq.heappush(heap, (c0 + c1, tie, a + b))
        tie += 1
    return depth


def leaf_probs(counts, rng):
    """Probabilities of the leaves of the Huffman tree with a random branch probability at each internal node."""
    heap = [(c, i, ((i, 1.0),)) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    tie = len(counts)
    while len(heap) > 1:
        c0, _, a = heapq.heappop(heap)
        c1, _, b = heapq.heappop(heap)
        s = 1 / (1 + math.exp(-rng.gauss(0, 3)))
        merged = tuple((w, s * p) for w, p in a) + tuple((w, (1 - s) * p) for w, p in b)
        heapq.heappush(heap, (c0 + c1, tie, merged))
        tie += 1
    return dict(heap[0][2])


def main():
    rng = random.Random(5)
    for V in range(2, 300):
        d = huffman_depths([1] * V)
        assert min(d) == math.floor(math.log2(V)) and max(d) == math.ceil(math.log2(V))
        assert sum(2.0**-k for k in d) == 1.0
    for _ in range(100):
        V = rng.randint(2, 400)
        counts = [rng.paretovariate(1.2) for _ in range(V)]
        d = huffman_depths(counts)
        assert abs(sum(2.0**-k for k in d) - 1.0) < 1e-12
        z = sum(counts)
        H = -sum(c / z * math.log2(c / z) for c in counts)
        mean = sum(c / z * k for c, k in zip(counts, d, strict=True))
        assert H - 1e-12 <= mean < H + 1
        probs = leaf_probs(counts, rng)
        assert len(probs) == V and abs(sum(probs.values()) - 1.0) < 1e-12
    assert round(math.log2(10_000), 2) == 13.29 and round(10_000 / math.log2(10_000)) == 753
    print(f"tree-softmax: log2 10000 = {math.log2(10_000):.2f}, ratio {10_000 / math.log2(10_000):.0f}")


if __name__ == "__main__":
    main()
