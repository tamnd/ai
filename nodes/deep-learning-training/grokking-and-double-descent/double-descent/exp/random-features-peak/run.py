# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/double-descent/ (provisional id NEW-double-descent).
"""Double descent in minimum-norm least squares on random ReLU features: one arm and one seed per run.

aik exp run gives the arm, the seed and the configuration in job.json (spec 9.1), and this writes metrics.json.
Each seed draws the target, the training, validation and test data, and the random features.
"""

import argparse
import json
import os

import numpy as np


def target(x, t):
    """The fixed nonlinear target of a seed: a single-index tanh and a sine of the first coordinate."""
    return np.tanh(2.0 * x @ t) + 0.5 * np.sin(x[:, 0])


def features(x, w):
    """Random ReLU features, scaled so that the feature matrix has entries of order 1/sqrt(N)."""
    return np.maximum(x @ w, 0.0) / np.sqrt(w.shape[1])


def fit(f, y, lam):
    """Least squares on the features: the minimum-norm solution when lam is 0, ridge regression otherwise."""
    if lam == 0:
        return np.linalg.pinv(f) @ y
    n, p = f.shape
    if p <= n:
        return np.linalg.solve(f.T @ f + lam * np.eye(p), f.T @ y)
    return f.T @ np.linalg.solve(f @ f.T + lam * np.eye(n), y)


def run(cfg, seed):
    """Fit one model and return its metrics."""
    rng = np.random.default_rng(seed)
    d, n, nv, nt, p = cfg["dim"], cfg["train"], cfg["validation"], cfg["test"], cfg["features"]
    t = rng.standard_normal(d) / np.sqrt(d)
    x, xv, xt = (rng.standard_normal((m, d)) for m in (n, nv, nt))
    y = target(x, t) + cfg["noise_sd"] * rng.standard_normal(n)
    yv = target(xv, t) + cfg["noise_sd"] * rng.standard_normal(nv)
    yt = target(xt, t)  # the test error is measured against the noiseless target
    w = rng.standard_normal((d, p)) / np.sqrt(d)
    f, fv, ft = features(x, w), features(xv, w), features(xt, w)
    lam = 0.0
    if cfg["ridge"] == "tuned":
        lam = min(cfg["ridge_grid"], key=lambda l: float(np.mean((fv @ fit(f, y, l) - yv) ** 2)))
    a = fit(f, y, lam)
    return {
        "test_mse": float(np.mean((ft @ a - yt) ** 2)),
        "train_mse": float(np.mean((f @ a - y) ** 2)),
        "ridge_lambda": float(lam),
    }


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
