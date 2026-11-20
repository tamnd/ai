# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/perplexity/ (ai://NEW-perplexity).
# Check of derivations/units-and-context.md: perplexity across units, bits per byte, the canonical-segmentation
# bound, the strided windows, and the nats removed by a relative cut of perplexity.
import itertools
import math

import numpy as np

rng = np.random.default_rng(1121)
for _ in range(2000):
    nt, nw, nb = (int(x) for x in rng.integers(1, 5000, size=3))
    s = float(rng.uniform(0, 20)) * min(nt, nb, nw)
    ell = s / nt
    assert math.isclose(
        math.exp(s / nw), math.exp(ell) ** (nt / nw), rel_tol=1e-9
    )  # item 1
    bpb = s / (nb * math.log(2))
    assert math.isclose(bpb, nt / nb * ell / math.log(2), rel_tol=1e-12)  # item 2
    assert math.isclose(2**bpb, math.exp(s / nb), rel_tol=1e-9)

# item 3: a unigram model over subword units; p(string) sums over all segmentations into known units
units = ["a", "b", "c", "ab", "bc", "abc", "ca"]
for _ in range(200):
    w = rng.dirichlet(np.ones(len(units)))
    prob = dict(zip(units, w, strict=True))

    def segs(x, prob=prob):
        if not x:
            yield []
            return
        for k in range(1, len(x) + 1):
            if x[:k] in prob:
                for rest in segs(x[k:], prob):
                    yield [x[:k], *rest]

    for n in range(1, 7):
        for t in itertools.product("abc", repeat=n):
            x = "".join(t)
            all_segs = list(segs(x))
            total = sum(math.prod(prob[u] for u in sg) for sg in all_segs)
            greedy = []  # a canonical segmentation: longest match first
            i = 0
            while i < len(x):
                k = max(k for k in range(1, len(x) - i + 1) if x[i : i + k] in prob)
                greedy.append(x[i : i + k])
                i += k
            assert greedy in all_segs
            canon = math.prod(prob[u] for u in greedy)
            assert canon <= total * (1 + 1e-12)
            assert math.exp(-math.log(canon) / n) >= math.exp(-math.log(total) / n) * (
                1 - 1e-12
            )


def windows(n, window, stride):
    out, prev_end, begin = [], 0, 0
    while True:
        end = min(begin + window, n)
        out.append((begin, end, prev_end))
        prev_end = end
        if end == n:
            return out
        begin += stride


for n in range(1, 201):  # item 4
    for w in range(1, 41):
        for s in range(1, w + 1):
            ws = windows(n, w, s)
            assert [i for _, e, f in ws for i in range(f, e)] == list(range(n))
            assert len(ws) == (1 if n <= w else 1 + math.ceil((n - w) / s))
            assert all(f - b >= w - s for b, _, f in ws[1:])

for _ in range(2000):  # item 5
    ppl = float(rng.uniform(1.01, 1e5))
    r = float(rng.uniform(0, 0.99))
    removed = math.log(ppl) - math.log((1 - r) * ppl)
    assert math.isclose(removed, -math.log(1 - r), rel_tol=1e-9, abs_tol=1e-12)
assert math.isclose(math.log2(100) - math.log2(90), -math.log2(0.9), rel_tol=1e-12)
assert round(math.log2(100), 2) == 6.64 and round(math.log2(90), 2) == 6.49

print("units-and-context: ok")
