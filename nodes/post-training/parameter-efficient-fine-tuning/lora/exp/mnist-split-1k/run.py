# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Learning and forgetting of LoRA and full fine-tuning on a split of MNIST: one arm and one seed per run.

aik exp run gives the arm, the seed, the configuration and the paths of the pinned data in job.json (spec 9.1), and
this writes metrics.json. Each seed pretrains a 2-hidden-layer MLP on images of the digits 0 to 4, then fine-tunes it
on images of the digits 5 to 9 in the way the arm names: the output layer only, LoRA of rank r on both hidden layers
(with the output layer trained), or every weight. The same seed gives the same pretrained model and the same order
of the fine-tuning batches in every arm, so the arms differ only in what is trained and in the learning rate.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "impl", "numpy"))
import lora  # noqa: E402  The reference implementation of the node.


def read_idx(path):
    """Read an IDX file of MNIST: a magic number, the sizes, and then unsigned bytes."""
    with gzip.open(path, "rb") as f:
        b = f.read()
    ndim = b[3]
    shape = tuple(int.from_bytes(b[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(data):
    def part(images, labels):
        x = read_idx(os.path.join(data, images)).reshape(-1, 784).astype(np.float64) / 255.0
        return x, read_idx(os.path.join(data, labels)).astype(np.int64)

    return (part("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
            part("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"))


def cross_entropy(logits, y):
    """The mean cross-entropy and its gradient with respect to the logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    p[np.arange(n), y] -= 1.0
    return loss, p / n


class Adam:
    """Adam with the defaults of Kingma and Ba (2015): beta1 0.9, beta2 0.999, eps 1e-8."""

    def __init__(self, lr):
        self.lr, self.m, self.v, self.t = lr, {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m.get(k, 0.0) + 0.1 * g
            self.v[k] = 0.999 * self.v.get(k, 0.0) + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class Net:
    """784 -> H -> H -> 10 with ReLU. mode is full, head or lora; in lora mode W1 and W2 carry adapters."""

    def __init__(self, params, mode, cfgs=None):
        self.p, self.mode, self.cfgs = params, mode, cfgs

    def layer(self, i, x):
        if self.mode == "lora":
            ad = {"A": self.p[f"A{i}"], "B": self.p[f"B{i}"]}
            return lora.forward(ad, self.p[f"W{i}"], x, self.cfgs[i - 1], self.p[f"b{i}"])
        return x @ self.p[f"W{i}"].T + self.p[f"b{i}"], None

    def logits(self, x):
        h1 = np.maximum(self.layer(1, x)[0], 0.0)
        h2 = np.maximum(self.layer(2, h1)[0], 0.0)
        return h2 @ self.p["W3"].T + self.p["b3"]

    def grads(self, x, y):
        a1, c1 = self.layer(1, x)
        h1 = np.maximum(a1, 0.0)
        a2, c2 = self.layer(2, h1)
        h2 = np.maximum(a2, 0.0)
        loss, g = cross_entropy(h2 @ self.p["W3"].T + self.p["b3"], y)
        out = {"W3": g.T @ h2, "b3": g.sum(axis=0)}
        if self.mode == "head":
            return loss, out
        d2 = (g @ self.p["W3"]) * (a2 > 0)
        if self.mode == "full":
            out["W2"], out["b2"] = d2.T @ h1, d2.sum(axis=0)
            d1 = (d2 @ self.p["W2"]) * (a1 > 0)
            out["W1"], out["b1"] = d1.T @ x, d1.sum(axis=0)
            return loss, out
        ad2 = {"A": self.p["A2"], "B": self.p["B2"]}
        dh1, g2 = lora.backward(ad2, self.p["W2"], c2, d2, self.cfgs[1])
        d1 = dh1 * (a1 > 0)
        ad1 = {"A": self.p["A1"], "B": self.p["B1"]}
        _, g1 = lora.backward(ad1, self.p["W1"], c1, d1, self.cfgs[0])
        out.update(A1=g1["A"], B1=g1["B"], A2=g2["A"], B2=g2["B"])
        return loss, out


def train(net, x, y, lr, epochs, batch, rng):
    opt = Adam(lr)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            j = order[i : i + batch]
            _, g = net.grads(x[j], y[j])
            opt.step(net.p, g)


def error(net, x, y, classes=None):
    """The share of wrong predictions; with classes, the argmax is over those logits only."""
    z = net.logits(x)
    if classes is not None:
        z = z[:, classes]
    return float((z.argmax(axis=1) != y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.job) as f:
        job = json.load(f)
    cfg, seed = job["config"], job["seed"]
    (xtr, ytr), (xte, yte) = load(job["data"]["mnist-ai-data"])
    h, pre, ft = cfg["model"]["hidden"], cfg["pretrain"], cfg["finetune"]

    rng = np.random.default_rng(seed)
    old_pool, new_pool = np.where(ytr < 5)[0], np.where(ytr >= 5)[0]
    pre_idx = rng.choice(old_pool, pre["images"], replace=False)
    ft_idx = rng.permutation(new_pool)[: ft["images"]]
    p = {
        "W1": rng.normal(0.0, np.sqrt(2 / 784), (h, 784)), "b1": np.zeros(h),
        "W2": rng.normal(0.0, np.sqrt(2 / h), (h, h)), "b2": np.zeros(h),
        "W3": rng.normal(0.0, 0.5 * np.sqrt(2 / h), (10, h)), "b3": np.zeros(10),
    }
    start = time.monotonic()
    train(Net(p, "full"), xtr[pre_idx], ytr[pre_idx], pre["lr"], pre["epochs"], pre["batch_size"], rng)
    new_te, old_te = np.where(yte >= 5)[0], np.where(yte < 5)[0]
    base_old = error(Net(p, "full"), xte[old_te], yte[old_te], classes=slice(0, 5))

    frng = np.random.default_rng([seed, 1])  # the same batches and adapter start in every arm
    mode, r = cfg["mode"], int(cfg.get("rank", 0))
    cfgs = None
    if mode == "lora":
        cfgs = [lora.config(784, h, r), lora.config(h, h, r)]
        for i, c in enumerate(cfgs, start=1):
            ad = lora.init(frng, c)
            p[f"A{i}"], p[f"B{i}"] = ad["A"], ad["B"]
    net = Net(p, mode, cfgs)
    train(net, xtr[ft_idx], ytr[ft_idx], cfg["lr"], ft["epochs"], ft["batch_size"], frng)

    trainable = {"head": 10 * h + 10, "full": 784 * h + h + h * h + h + 10 * h + 10}.get(mode)
    if mode == "lora":
        trainable = sum(lora.num_trainable(c) for c in cfgs) + 10 * h + 10
    final = {
        "new_error": error(net, xte[new_te], yte[new_te]),
        "old_error": error(net, xte[old_te], yte[old_te], classes=slice(0, 5)),
        "pretrained_old_error": base_old,
        "train_error": error(net, xtr[ft_idx], ytr[ft_idx]),
        "trainable": trainable,
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "device": "cpu", "train_seconds": time.monotonic() - start}, f)


if __name__ == "__main__":
    main()
