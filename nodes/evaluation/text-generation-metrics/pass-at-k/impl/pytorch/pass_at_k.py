# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/pass-at-k/ (ai://AH634X).
"""pass@k (Kulal et al., 2019; Chen et al., 2021) estimated from n samples per problem, in PyTorch.

The inputs are counts: for each problem, c samples out of n pass all of its unit tests. The unbiased estimate of one
problem is 1 - C(n - c, k) / C(n, k) (eq-estimator), computed as the product of eq-product-form, as in Figure 3 of
Chen et al. (2021), so that no binomial coefficient is formed. The arithmetic is in torch.float64 on the CPU.
"""

from collections.abc import Sequence

import torch

ESTIMATORS = ("unbiased", "plug-in")


def _int(x: object, what: str) -> int:
    if isinstance(x, torch.Tensor) and x.dim() == 0 and not x.is_floating_point() and x.dtype != torch.bool:
        return int(x.item())
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{what} must be an integer, got {x!r}")
    return x


def estimate(n: int, c: int, k: int) -> float:
    """1 - C(n - c, k) / C(n, k) for one problem, as 1 - prod_{i = n - c + 1}^{n} (1 - k / i) (eq-product-form)."""
    if n - c < k:
        return 1.0  # every subset of k samples holds a correct one
    i = torch.arange(n - c + 1, n + 1, dtype=torch.float64)
    return float(1.0 - torch.prod(1.0 - k / i))


def plug_in(n: int, c: int, k: int) -> float:
    """1 - (1 - c/n)^k for one problem: biased low for k >= 2 (Chen et al., 2021, Appendix A)."""
    return float(1.0 - (1.0 - torch.tensor(c, dtype=torch.float64) / n) ** k)


def score(
    predictions: Sequence[int] | torch.Tensor,
    references: Sequence[int] | torch.Tensor,
    k: int | torch.Tensor = 1,
    estimator: str = "unbiased",
) -> dict:
    """pass@k over problems: predictions[j] = c_j samples pass, references[j] = n_j samples were drawn."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} counts of correct samples but {len(references)} sample counts")
    if len(predictions) == 0:
        raise ValueError("pass@k needs at least one problem")
    k = _int(k, "k")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    f = estimate if estimator == "unbiased" else plug_in
    per = torch.empty(len(predictions), dtype=torch.float64)
    for j, (c, n) in enumerate(zip(predictions, references, strict=True)):
        c, n = _int(c, "a count of correct samples"), _int(n, "a sample count")
        if not 0 <= c <= n:
            raise ValueError(f"problem {j}: need 0 <= c <= n, got c = {c} and n = {n}")
        if n < k:
            raise ValueError(f"problem {j}: n = {n} samples cannot estimate pass@{k}; draw at least k samples")
        per[j] = f(n, c, k)
    return {"pass_at_k": float(per.mean()), "per_problem": per.tolist()}
