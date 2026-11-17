# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Plain and residual fully connected networks of two depths on MNIST: one arm and one seed per run.

aik exp run gives the arm, the seed, the configuration and the paths of the pinned data in job.json (spec 9.1), and
this writes metrics.json. Each run trains a network of an input layer, `blocks` blocks of two layers with batch
normalization, and a classifier, with SGD and momentum, and reports its error on the training images (with the
running statistics of the norms, as at inference) and on the test set. In the residual arms each block adds its
input to its output (eq-residual); in the plain arms it does not, and nothing else differs. The same seed draws the
same training images and the same order of batches in every arm, and the same weights for the plain and the
residual network of one depth.
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
import resnet  # noqa: E402  The reference implementation of the node.


def read_idx(path):
    """Read an IDX file of MNIST: a magic number, the sizes, and then unsigned bytes."""
    with gzip.open(path, "rb") as f:
        b = f.read()
    ndim = b[3]
    shape = tuple(int.from_bytes(b[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(data):
    def part(images, labels):
        x = read_idx(os.path.join(data, images)).reshape(-1, 784).astype(np.float32) / 255.0
        return x, read_idx(os.path.join(data, labels)).astype(np.int64)

    (xtr, ytr), (xte, yte) = (part("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
                              part("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"))
    mean, sd = xtr.mean(), xtr.std()  # one mean and one scale for all pixels, from the training set
    return ((xtr - mean) / sd, ytr), ((xte - mean) / sd, yte)


def error(p, state, x, y, residual):
    wrong = 0
    for i in range(0, len(x), 2000):
        logits, _ = resnet.dense_forward(p, state, x[i : i + 2000], residual, train=False)
        wrong += int((logits.argmax(axis=1) != y[i : i + 2000]).sum())
    return wrong / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.job) as f:
        job = json.load(f)
    cfg, seed = job["config"], job["seed"]
    (xall, yall), (xte, yte) = load(job["data"]["mnist-ai-data"])
    tr, opt = cfg["train"], cfg["optimizer"]
    n, residual, width = int(cfg["blocks"]), bool(cfg["residual"]), int(cfg["model"]["width"])

    data_rng = np.random.default_rng([seed, 0])  # the images and the batches: the same in every arm
    idx = data_rng.permutation(len(xall))[: tr["images"]]
    x, y = xall[idx], yall[idx]
    p, state = resnet.dense_init(np.random.default_rng([seed, 1, n]), 784, width, n, 10, dtype=np.float32)
    velocity = {k: np.zeros_like(v) for k, v in p.items()}
    batch, per_epoch = tr["batch_size"], tr["images"] // tr["batch_size"]
    losses, start = [], time.monotonic()
    for step in range(tr["steps"]):
        if step % per_epoch == 0:
            order = data_rng.permutation(tr["images"])
        j = order[(step % per_epoch) * batch : (step % per_epoch + 1) * batch]
        logits, cache = resnet.dense_forward(p, state, x[j], residual, train=True, momentum=cfg["bn_momentum"])
        loss, dlogits = resnet.cross_entropy(logits, y[j])
        losses.append(float(loss))
        grads = resnet.dense_backward(p, cache, dlogits, residual)
        for k in p:
            g = grads[k] + (opt["weight_decay"] * p[k] if k.startswith("W") else 0.0)  # no decay on norms
            velocity[k] = opt["momentum"] * velocity[k] + g
            p[k] -= cfg["lr"] * velocity[k]
    final = {
        "train_error": error(p, state, x, y, residual),
        "test_error": error(p, state, xte, yte, residual),
        "final_loss": float(np.mean(losses[-50:])),
        "layers": 2 * n + 2,
        "params": int(sum(v.size for v in p.values())),
    }
    if not np.isfinite(final["final_loss"]):
        raise SystemExit("the loss diverged")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "device": "cpu", "train_seconds": time.monotonic() - start}, f)


if __name__ == "__main__":
    main()
