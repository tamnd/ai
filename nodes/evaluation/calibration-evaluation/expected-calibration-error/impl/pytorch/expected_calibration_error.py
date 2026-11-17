# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/expected-calibration-error/ (ai://01R1CF).
"""Expected calibration error with equal-width bins, the bin statistics of a reliability diagram, and temperature
scaling fitted by negative log-likelihood (Guo et al., 2017), in PyTorch.

The confidence of a prediction is the largest probability of its row and the predicted class is the first index that
reaches it. Bin m of M holds the confidences in (m/M - 1/M, m/M], with the edges computed as m / M in float64, and
bin 1 also holds a confidence of exactly 0. The ECE is the sum over the bins of |sum of (correct - confidence)| / n
(eq-ece-hat), the MCE the largest gap |accuracy - confidence| over the non-empty bins. The arithmetic is in
torch.float64 on the CPU.
"""

import math
from collections.abc import Sequence

import torch

T_MIN, T_MAX = 0.05, 100.0


def _n_bins(n_bins: object) -> int:
    if isinstance(n_bins, torch.Tensor) and n_bins.dim() == 0 and not n_bins.is_floating_point():
        n_bins = int(n_bins.item())
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise TypeError(f"n_bins must be an integer, got {n_bins!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    return n_bins


def _labels(references: Sequence[int] | torch.Tensor, n: int, k: int) -> torch.Tensor:
    y = references if isinstance(references, torch.Tensor) else torch.as_tensor(references)
    if y.dim() != 1 or y.shape[0] != n:
        raise ValueError(f"need one label for each of the {n} predictions, got shape {tuple(y.shape)}")
    if y.dtype == torch.bool or y.is_floating_point() or y.is_complex():
        raise TypeError(f"labels must be integers, got dtype {y.dtype}")
    if bool(torch.any(y < 0)) or bool(torch.any(y >= k)):
        raise ValueError(f"labels must be in 0..{k - 1}")
    return y.to(torch.int64)


def _matrix(x: Sequence[Sequence[float]] | torch.Tensor, what: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float64).cpu()
    if t.dim() != 2 or t.shape[0] == 0 or t.shape[1] == 0:
        raise ValueError(f"{what} must be a non-empty n x K matrix, got shape {tuple(t.shape)}")
    if not bool(torch.all(torch.isfinite(t))):
        raise ValueError(f"{what} must be finite")
    return t


def _probabilities(predictions: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    p = _matrix(predictions, "predictions")
    if bool(torch.any(p < 0)) or bool(torch.any(p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if bool(torch.any(torch.abs(p.sum(dim=1) - 1.0) > 1e-6)):
        raise ValueError("each row of probabilities must sum to 1 within 1e-6")
    return p


def bin_index(confidence: torch.Tensor, n_bins: int) -> torch.Tensor:
    """The 0-based bin of each confidence: bin m holds (edges[m], edges[m + 1]], and bin 0 also holds 0."""
    edges = torch.arange(n_bins + 1, dtype=torch.float64) / n_bins
    return torch.searchsorted(edges[1:-1].contiguous(), confidence.contiguous(), right=False)


def reliability_bins(
    predictions: Sequence[Sequence[float]] | torch.Tensor,
    references: Sequence[int] | torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """The count, accuracy and mean confidence of each of n_bins equal-width bins; 0.0 for an empty bin."""
    m = _n_bins(n_bins)
    p = _probabilities(predictions)
    y = _labels(references, p.shape[0], p.shape[1])
    conf, pred = p.max(dim=1).values, torch.argmax(p, dim=1)
    correct = (pred == y).to(torch.float64)
    idx = bin_index(conf, m)
    count = torch.bincount(idx, minlength=m)
    hits = torch.zeros(m, dtype=torch.float64).index_add_(0, idx, correct)
    mass = torch.zeros(m, dtype=torch.float64).index_add_(0, idx, conf)
    gap_sum = torch.zeros(m, dtype=torch.float64).index_add_(0, idx, correct - conf)
    safe = torch.clamp(count, min=1).to(torch.float64)
    return {
        "count": count,
        "accuracy": hits / safe,
        "confidence": mass / safe,
        "gap_sum": gap_sum,
        "n": p.shape[0],
        "correct": correct,
        "conf": conf,
    }


def score(
    predictions: Sequence[Sequence[float]] | torch.Tensor,
    references: Sequence[int] | torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """ECE and MCE with n_bins equal-width bins, with the bin statistics of the reliability diagram."""
    b = reliability_bins(predictions, references, n_bins)
    n, count = b["n"], b["count"]
    ece = float(torch.sum(torch.abs(b["gap_sum"])) / n)
    full = count > 0
    mce = float(torch.max(torch.abs(b["accuracy"][full] - b["confidence"][full])))
    return {
        "ece": ece,
        "mce": mce,
        "accuracy": float(b["correct"].mean()),
        "confidence": float(b["conf"].mean()),
        "bin_count": [int(c) for c in count.tolist()],
        "bin_accuracy": b["accuracy"].tolist(),
        "bin_confidence": b["confidence"].tolist(),
    }


def _temperature(t: object) -> float:
    t = float(t)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"the temperature must be positive and finite, got {t}")
    return t


def softmax(logits: Sequence[Sequence[float]] | torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """softmax(z / T) of each row (eq-temperature)."""
    return torch.softmax(_matrix(logits, "logits") / _temperature(temperature), dim=1)


def nll(
    logits: Sequence[Sequence[float]] | torch.Tensor, references: Sequence[int] | torch.Tensor, temperature: float = 1.0
) -> float:
    """The mean negative log-likelihood of the labels under softmax(z / T)."""
    z = _matrix(logits, "logits") / _temperature(temperature)
    y = _labels(references, z.shape[0], z.shape[1])
    return float(torch.nn.functional.cross_entropy(z, y, reduction="mean"))


def _slope(z: torch.Tensor, y: torch.Tensor, beta: float) -> float:
    """d/d(beta) of the mean NLL at inverse temperature beta: mean of E_softmax(beta z)[z] - z_y (eq-stationary)."""
    q = torch.softmax(beta * z, dim=1)
    return float(torch.mean(torch.sum(q * z, dim=1) - z[torch.arange(len(y)), y]))


def fit_temperature(
    logits: Sequence[Sequence[float]] | torch.Tensor,
    references: Sequence[int] | torch.Tensor,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    iters: int = 200,
) -> float:
    """The T in [t_min, t_max] that minimizes the mean NLL of softmax(z / T).

    The mean NLL is convex in beta = 1/T, so its slope does not decrease in beta; bisection on the slope finds the
    minimizer. When the slope keeps one sign on the whole interval, the minimizer is the nearer end.
    """
    z = _matrix(logits, "logits")
    y = _labels(references, z.shape[0], z.shape[1])
    t_min, t_max = _temperature(t_min), _temperature(t_max)
    if not t_min < t_max:
        raise ValueError(f"need t_min < t_max, got {t_min} and {t_max}")
    lo, hi = 1.0 / t_max, 1.0 / t_min
    with torch.no_grad():
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
