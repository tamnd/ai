"""Check every number in contrastive-loss-arithmetic.md."""

import math

import numpy as np


def close(x, y, tol=5e-5):
    assert abs(x - y) <= tol * max(1.0, abs(y)), (x, y)


# 1. The loss with uninformative embeddings.
close(math.log(32768), 10.3972)
close(math.log(16384), 9.7041)
rng = np.random.default_rng(0)
n, d = 512, 256
u = rng.standard_normal((n, d))
v = rng.standard_normal((n, d))
u /= np.linalg.norm(u, axis=1, keepdims=True)
v /= np.linalg.norm(v, axis=1, keepdims=True)
logits = 1.0 * u @ v.T
lse = np.log(np.exp(logits).sum(1))
loss = float(np.mean(lse - np.diag(logits)))
assert abs(loss - math.log(n)) < 0.01, loss


# 2. The floor set by the logit scale.
def floor(s, nn, neg_cos):
    return math.log1p((nn - 1) * math.exp(-s * (1 - neg_cos)))


close(floor(1.0, 32768, 0.0), 9.3972)
close(floor(1 / 0.07, 32768, 0.0), 0.020268, 1e-4)
close(floor(1 / 0.07, 32768, 0.2), 0.3049, 1e-3)
assert floor(100.0, 32768, 0.0) < 1e-38
close(floor(1.0, 32768, -1.0), 8.3972)
close(1 / 0.07, 14.2857)


# 3. The sigmoid loss at initialization.
def softplus(x):
    return math.log1p(math.exp(x)) if x < 30 else x


pos = softplus(10.0)  # -log sigmoid(-10): the positive pair at logit -10
neg = softplus(-10.0)  # -log sigmoid(10): a negative pair at logit -10
close(pos, 10.0000454, 1e-7)
close(neg, 4.53989e-5, 1e-4)
close((32768 - 1) * neg, 1.4876, 1e-4)
close(pos + (32768 - 1) * neg, 11.4876, 1e-4)
close(32767 * math.log(2), 22712.3, 1e-5)
close(math.log(2), 0.6931, 1e-4)
close(1 / (1 + math.exp(10)), 4.54e-5, 1e-3)
close(math.exp(10), 22026.5, 1e-5)
close(math.log(1 / 32767), -10.3972, 1e-4)

# 4. Memory of the similarity matrix.
assert 32768**2 * 4 == 2**32
assert (2**20) ** 2 * 4 == 4 * 2**40
assert (2**20 // 2**10) ** 2 * 4 == 4 * 2**20
assert (2**20 // 2**10) * 2**20 * 4 == 2**32

# 5. Pairs and steps.
assert 32768**2 - 32768 == 1_073_709_056
assert 32 * 400_000_000 == 12_800_000_000
assert 12_800_000_000 / 32768 == 390_625

# 6. The zero-shot classifier.
assert 1000 * 80 == 80_000
k, p = 5, 7
t = rng.standard_normal((k, p, d))
t /= np.linalg.norm(t, axis=2, keepdims=True)
w = t.mean(1)
w /= np.linalg.norm(w, axis=1, keepdims=True)
img = rng.standard_normal(d)
img /= np.linalg.norm(img)
assert np.argmax(w @ img) == np.argmax(t.mean(1) @ img / np.linalg.norm(t.mean(1), axis=1))
print("ok")
