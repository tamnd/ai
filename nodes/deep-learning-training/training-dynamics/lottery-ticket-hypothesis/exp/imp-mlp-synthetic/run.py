# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
"""Iterative magnitude pruning with reset on a small MLP: one arm and one seed per run.

aik exp run gives the arm, the seed and the configuration in job.json (spec 9.1), and this writes metrics.json.
The task is fixed by data_seed: a random teacher network and its training and test data. A seed draws the initial
weights theta_0 of a student MLP and the order of its minibatches.
The dense arm trains the student once. The ticket arm trains it, prunes 30% of the hidden weights of each layer by
magnitude (15% of the output layer), resets the survivors to theta_0, and repeats; it reports the test accuracy of the
last pruned network. The reinit arm takes the same final mask and trains it from a fresh random initialization.
All arms of a seed share the data, theta_0 and the order of the minibatches.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "impl", "numpy"))
from lottery_ticket import imp_round, sparsity  # noqa: E402  The reference implementation of the node.


def make_data(rng, cfg):
    """A random teacher tanh network labels Gaussian inputs with the argmax of its outputs."""
    d, h, k = cfg["dim"], cfg["teacher_hidden"], cfg["classes"]
    a = rng.standard_normal((d, h)) / np.sqrt(d)
    b = rng.standard_normal((h, k)) / np.sqrt(h)
    x = rng.standard_normal((cfg["train"] + cfg["test"], d))
    y = np.argmax(np.tanh(2.0 * x @ a) @ b, axis=1)
    n = cfg["train"]
    return x[:n], y[:n], x[n:], y[n:]


def init(rng, sizes):
    """Gaussian weights with variance 2 / (fan_in + fan_out); the layers have no biases."""
    return [rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b)) for a, b in zip(sizes[:-1], sizes[1:])]


def forward(ws, x):
    hs = [x]
    for w in ws[:-1]:
        hs.append(np.maximum(hs[-1] @ w, 0.0))
    return hs, hs[-1] @ ws[-1]


def train(ws, masks, x, y, cfg, seed):
    """Adam on the softmax cross-entropy, with the masked weights held at 0. The minibatches depend only on seed."""
    rng = np.random.default_rng(seed)
    ws = [w * m for w, m in zip(ws, masks)]
    m1 = [np.zeros_like(w) for w in ws]
    v1 = [np.zeros_like(w) for w in ws]
    lr, bs = cfg["lr"], cfg["batch_size"]
    for t in range(1, cfg["steps"] + 1):
        idx = rng.integers(0, len(x), bs)
        hs, z = forward(ws, x[idx])
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(bs), y[idx]] -= 1.0
        g = p / bs
        for i in range(len(ws) - 1, -1, -1):
            grad = (hs[i].T @ g) * masks[i]
            if i:
                g = (g @ ws[i].T) * (hs[i] > 0)
            m1[i] = 0.9 * m1[i] + 0.1 * grad
            v1[i] = 0.999 * v1[i] + 0.001 * grad * grad
            step = lr * (m1[i] / (1 - 0.9**t)) / (np.sqrt(v1[i] / (1 - 0.999**t)) + 1e-8)
            ws[i] = (ws[i] - step) * masks[i]
    return ws


def accuracy(ws, x, y):
    return float(np.mean(np.argmax(forward(ws, x)[1], axis=1) == y))


def run(cfg, seed):
    """Run one arm and return its metrics."""
    x, y, xt, yt = make_data(np.random.default_rng(cfg["data_seed"]), cfg)
    sizes = [cfg["dim"], *cfg["hidden"], cfg["classes"]]
    theta0 = init(np.random.default_rng(seed), sizes)
    batches = seed + 1
    masks = [np.ones_like(w) for w in theta0]
    trained = train(theta0, masks, x, y, cfg, batches)
    rates = [cfg["rate"]] * (len(sizes) - 2) + [cfg["output_rate"]]
    arm = cfg["procedure"]
    if arm != "dense":
        for _ in range(cfg["rounds"]):
            masks, start = imp_round(trained, theta0, masks, rates)
            trained = train(start, masks, x, y, cfg, batches)
        if arm == "reinit":
            fresh = init(np.random.default_rng(seed + 2), sizes)
            trained = train(fresh, masks, x, y, cfg, batches)
    return {"test_acc": accuracy(trained, xt, yt), "train_acc": accuracy(trained, x, y), "remaining": sparsity(masks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.job) as fh:
        job = json.load(fh)
    metrics = run(job["config"], int(job["seed"]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"final": metrics, "device": "cpu"}, fh, indent=2)


if __name__ == "__main__":
    main()
