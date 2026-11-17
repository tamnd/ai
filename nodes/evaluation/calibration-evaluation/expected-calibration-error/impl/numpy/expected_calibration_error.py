# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
"""Expected calibration error with equal-width bins, the bin statistics of a reliability diagram, and temperature
scaling fitted by negative log-likelihood (Guo et al., 2017), in NumPy.

The confidence of a prediction is the largest probability of its row and the predicted class is the first index that
reaches it. Bin m of M holds the confidences in (m/M - 1/M, m/M], with the edges computed as m / M in float64, and
bin 1 also holds a confidence of exactly 0. The ECE is the sum over the bins of |sum of (correct - confidence)| / n
(eq-ece-hat), the MCE the largest gap |accuracy - confidence| over the non-empty bins. The arithmetic is in float64.
"""

from collections.abc import Sequence

import numpy as np

T_MIN, T_MAX = 0.05, 100.0


def _n_bins(n_bins: object) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError(f"n_bins must be an integer, got {n_bins!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    return int(n_bins)


def _labels(references: Sequence[int] | np.ndarray, n: int, k: int) -> np.ndarray:
    y = np.asarray(references)
    if y.ndim != 1 or len(y) != n:
        raise ValueError(f"need one label for each of the {n} predictions, got shape {y.shape}")
    if y.dtype == bool or not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"labels must be integers, got dtype {y.dtype}")
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError(f"labels must be in 0..{k - 1}")
    return y.astype(np.int64)


def _probabilities(predictions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    p = np.asarray(predictions, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError(f"predictions must be a non-empty n x K matrix of probabilities, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.any(np.abs(p.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("each row of probabilities must sum to 1 within 1e-6")
    return p


def bin_index(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    """The 0-based bin of each confidence: bin m holds (edges[m], edges[m + 1]], and bin 0 also holds 0."""
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    return np.searchsorted(edges[1:-1], confidence, side="left")


def reliability_bins(
    predictions: Sequence[Sequence[float]] | np.ndarray,
    references: Sequence[int] | np.ndarray,
    n_bins: int | np.integer = 15,
) -> dict:
    """The count, accuracy and mean confidence of each of n_bins equal-width bins; 0.0 for an empty bin."""
    m = _n_bins(n_bins)
    p = _probabilities(predictions)
    y = _labels(references, p.shape[0], p.shape[1])
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    idx = bin_index(conf, m)
    count = np.bincount(idx, minlength=m)
    hits = np.bincount(idx, weights=correct, minlength=m)
    mass = np.bincount(idx, weights=conf, minlength=m)
    gap_sum = np.bincount(idx, weights=correct - conf, minlength=m)
    safe = np.maximum(count, 1)
    return {
        "count": count.astype(np.int64),
        "accuracy": hits / safe,
        "confidence": mass / safe,
        "gap_sum": gap_sum,
        "n": p.shape[0],
        "correct": correct,
        "conf": conf,
    }


def score(
    predictions: Sequence[Sequence[float]] | np.ndarray,
    references: Sequence[int] | np.ndarray,
    n_bins: int | np.integer = 15,
) -> dict:
    """ECE and MCE with n_bins equal-width bins, with the bin statistics of the reliability diagram."""
    b = reliability_bins(predictions, references, n_bins)
    n, count = b["n"], b["count"]
    ece = float(np.sum(np.abs(b["gap_sum"])) / n)
    full = count > 0
    mce = float(np.max(np.abs(b["accuracy"][full] - b["confidence"][full])))
    return {
        "ece": ece,
        "mce": mce,
        "accuracy": float(b["correct"].mean()),
        "confidence": float(b["conf"].mean()),
        "bin_count": count.tolist(),
        "bin_accuracy": b["accuracy"].tolist(),
        "bin_confidence": b["confidence"].tolist(),
    }


def _logits(logits: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"logits must be a non-empty n x K matrix, got shape {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite")
    return z


def _temperature(t: object) -> float:
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"the temperature must be positive and finite, got {t}")
    return t


def softmax(logits: Sequence[Sequence[float]] | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """softmax(z / T) of each row (eq-temperature), computed after subtracting the row maximum."""
    z = _logits(logits) / _temperature(temperature)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll(
    logits: Sequence[Sequence[float]] | np.ndarray, references: Sequence[int] | np.ndarray, temperature: float = 1.0
) -> float:
    """The mean negative log-likelihood of the labels under softmax(z / T)."""
    z = _logits(logits) / _temperature(temperature)
    y = _labels(references, z.shape[0], z.shape[1])
    zmax = z.max(axis=1)
    lse = zmax + np.log(np.exp(z - zmax[:, None]).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def _slope(z: np.ndarray, y: np.ndarray, beta: float) -> float:
    """d/d(beta) of the mean NLL at inverse temperature beta: mean of E_softmax(beta z)[z] - z_y (eq-stationary)."""
    s = beta * z
    e = np.exp(s - s.max(axis=1, keepdims=True))
    q = e / e.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(q * z, axis=1) - z[np.arange(len(y)), y]))


def fit_temperature(
    logits: Sequence[Sequence[float]] | np.ndarray,
    references: Sequence[int] | np.ndarray,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    iters: int = 200,
) -> float:
    """The T in [t_min, t_max] that minimizes the mean NLL of softmax(z / T).

    The mean NLL is convex in beta = 1/T, so its slope does not decrease in beta; bisection on the slope finds the
    minimizer. When the slope keeps one sign on the whole interval, the minimizer is the nearer end.
    """
    z = _logits(logits)
    y = _labels(references, z.shape[0], z.shape[1])
    t_min, t_max = _temperature(t_min), _temperature(t_max)
    if not t_min < t_max:
        raise ValueError(f"need t_min < t_max, got {t_min} and {t_max}")
    lo, hi = 1.0 / t_max, 1.0 / t_min
    if _slope(z, y, lo) >= 0:
        return t_max
    if _slope(z, y, hi) <= 0:
        return t_min
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if _slope(z, y, mid) > 0:
            hi = mid
        else:
            lo = mid
    return 1.0 / (0.5 * (lo + hi))
