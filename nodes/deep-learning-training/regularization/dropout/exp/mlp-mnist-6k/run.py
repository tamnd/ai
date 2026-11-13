# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/dropout/ (ai://Y8VNMB).
"""Train a small MLP on a subset of MNIST, with and without dropout, and write the final metrics.

aik exp run gives the arm, the seed, the configuration and the paths of the pinned data in job.json (spec 9.1).
"""

import argparse
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "impl", "pytorch"))
from dropout import Dropout  # noqa: E402  The reference implementation of the node.


def read_idx(path):
    """Read an IDX file of MNIST: a magic number, the sizes, and then unsigned bytes."""
    with gzip.open(path, "rb") as f:
        b = f.read()
    ndim = b[3]
    shape = tuple(int.from_bytes(b[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(data, n_train, n_test):
    """Give the first n_train training images and the first n_test test images, scaled to [0, 1]."""
    def part(images, labels, n):
        x = read_idx(os.path.join(data, images))[:n].reshape(n, -1).astype(np.float32) / 255.0
        y = read_idx(os.path.join(data, labels))[:n].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    return (part("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", n_train),
            part("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", n_test))


class MLP(torch.nn.Module):
    """Fully connected layers with ReLU. Dropout with p_input on the input and p on each hidden layer."""

    def __init__(self, hidden, p, p_input):
        super().__init__()
        sizes = [784, *hidden]
        self.inp = Dropout(p_input)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.drops = torch.nn.ModuleList(Dropout(p) for _ in hidden)
        self.out = torch.nn.Linear(sizes[-1], 10)

    def forward(self, x):
        x = self.inp(x)
        for layer, drop in zip(self.layers, self.drops):
            x = drop(torch.relu(layer(x)))
        return self.out(x)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        error = (logits.argmax(dim=1) != y).float().mean().item()
    return loss, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.job) as f:
        job = json.load(f)
    cfg, seed = job["config"], job["seed"]

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

    (xtr, ytr), (xte, yte) = load(job["data"]["mnist-ai-data"], cfg["train_images"], cfg["test_images"])
    model = MLP(cfg["model"]["hidden"], cfg["p"], cfg["p_input"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["optimizer"]["lr"], momentum=cfg["optimizer"]["momentum"])
    g = torch.Generator().manual_seed(seed)
    start = time.monotonic()
    for _ in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), cfg["batch_size"]):
            idx = order[i : i + cfg["batch_size"]]
            loss = torch.nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_loss, train_error = evaluate(model, xtr, ytr)
    test_loss, test_error = evaluate(model, xte, yte)
    final = {"test_error": test_error, "test_loss": test_loss, "train_loss": train_loss, "train_error": train_error}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"final": final, "device": "cpu", "train_seconds": time.monotonic() - start}, f)


if __name__ == "__main__":
    main()
