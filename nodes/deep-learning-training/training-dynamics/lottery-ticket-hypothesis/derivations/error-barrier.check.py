# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
# Check of derivations/error-barrier.md: the barrier sup_alpha E(alpha W1 + (1 - alpha) W2) - (E(W1) + E(W2)) / 2 is at
# least |E(W1) - E(W2)| / 2 >= 0; for a convex error it equals |E(W1) - E(W2)| / 2; the grid of 30 alphas gives a lower
# bound within L / 58 of the sup when E_alpha is L-Lipschitz in alpha; and two ReLU networks that differ only by the order
# of their hidden units both have zero error while the midpoint of the line between them does not.
import numpy as np


def barrier(err, w1, w2, alphas):
    path = np.array([err([a * x + (1 - a) * y for x, y in zip(w1, w2)]) for a in alphas])
    return path.max() - 0.5 * (err(w1) + err(w2)), path


grid = np.linspace(0.0, 1.0, 30)
fine = np.linspace(0.0, 1.0, 2901)  # holds the 30-point grid, since 2900 = 29 * 100
assert np.allclose(fine[::100], grid)
rng = np.random.default_rng(0)

# 1. A convex error: the mean squared error of a linear model. The sup is at an end, so the barrier is |E1 - E2| / 2.
x = rng.standard_normal((200, 5))
y = x @ rng.standard_normal(5) + 0.1 * rng.standard_normal(200)


def mse(ws):
    return float(np.mean((x @ ws[0] - y) ** 2))


for _ in range(20):
    a, b = [rng.standard_normal(5)], [rng.standard_normal(5)]
    h, path = barrier(mse, a, b, fine)
    assert abs(h - abs(mse(a) - mse(b)) / 2) < 1e-9
    assert path.max() <= max(mse(a), mse(b)) + 1e-9

# 2. Any error, with the ends in the grid: the barrier is at least |E1 - E2| / 2, so it is never negative.
c = rng.standard_normal(5)


def wavy(ws):
    return float(np.sin(3 * ws[0] @ c) + 0.1 * ws[0] @ ws[0])


for _ in range(50):
    a, b = [rng.standard_normal(5)], [rng.standard_normal(5)]
    h, _ = barrier(wavy, a, b, grid)
    assert h >= abs(wavy(a) - wavy(b)) / 2 - 1e-12 >= -1e-12
    # 3. The 30-point grid is a lower bound on the sup, within L / (2 * 29) for an L-Lipschitz path.
    hf, path = barrier(wavy, a, b, fine)
    lip = np.max(np.abs(np.diff(path))) / (fine[1] - fine[0])
    assert h <= hf + 1e-12 and hf - h <= lip / 58 + 1e-9

# 4. A ReLU network and the same network with its hidden units swapped: the same function, so both have zero error on
# labels from the first; the midpoint averages the units and makes errors.
d, k = 10, 8
w1, w2 = rng.standard_normal((d, k)), rng.standard_normal((k, 3))
perm = rng.permutation(k)
assert not np.array_equal(perm, np.arange(k))
net_a, net_b = [w1, w2], [w1[:, perm], w2[perm]]
xs = rng.standard_normal((2000, d))


def predict(ws, z):
    return np.argmax(np.maximum(z @ ws[0], 0) @ ws[1], axis=1)


labels = predict(net_a, xs)


def error01(ws):
    return float(np.mean(predict(ws, xs) != labels))


assert error01(net_a) == 0.0 and error01(net_b) == 0.0
h, path = barrier(error01, net_a, net_b, grid)
assert path[0] == 0.0 and path[-1] == 0.0
assert h > 0.1, h  # a large barrier: the two networks are not linearly connected
print(f"permuted ReLU net: barrier {h:.3f} over 30 alphas")

# 5. The same network twice has barrier 0.
assert barrier(error01, net_a, net_a, grid)[0] == 0.0
print("error-barrier: ok")
