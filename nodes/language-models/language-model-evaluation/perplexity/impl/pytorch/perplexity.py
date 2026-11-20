# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/perplexity/ (ai://NEW-perplexity).
"""Perplexity of a language model on a corpus, from the log-probabilities it gave the observed tokens, in PyTorch.

The inputs are, for each sequence of the corpus, the natural-log probabilities log p(x_i | x_<i) of its scored
tokens, and the number of units (tokens, words, characters or UTF-8 bytes) over which the sequence is normalised.
The corpus value is exp(total negative log-likelihood / total units) (eq-perplexity), computed in the log domain.
The arithmetic is in torch.float64 on the CPU. token_logprobs turns logits and targets into the inputs.
"""

import math
from collections.abc import Sequence

import torch

AGGREGATES = ("corpus", "mean-of-sequences")
BASES = ("e", "2")


def _units(n: object, j: int) -> int:
    if (
        isinstance(n, torch.Tensor)
        and n.dim() == 0
        and not n.is_floating_point()
        and n.dtype != torch.bool
    ):
        n = int(n.item())
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"sequence {j}: the unit count must be an integer, got {n!r}")
    if n < 1:
        raise ValueError(f"sequence {j}: the unit count must be at least 1, got {n}")
    return n


def _nll(logprobs: Sequence[float] | torch.Tensor, j: int) -> torch.Tensor:
    """The negative log-likelihood of one sequence, -sum log p, in nats; inf when a token had probability 0."""
    lp = torch.as_tensor(logprobs, dtype=torch.float64).reshape(-1)
    if torch.isnan(lp).any():
        raise ValueError(f"sequence {j}: a log-probability is NaN")
    if (lp > 0).any():
        raise ValueError(
            f"sequence {j}: a log-probability is above 0, so it is not the log of a probability"
        )
    return -lp.sum()


def score(
    predictions: Sequence[Sequence[float] | torch.Tensor],
    references: Sequence[int] | torch.Tensor,
    aggregate: str = "corpus",
    base: str = "e",
) -> dict:
    """Perplexity: predictions[j] holds the log-probabilities of sequence j, references[j] its number of units."""
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
    if base not in BASES:
        raise ValueError(f"base must be one of {BASES}, got {base!r}")
    if len(predictions) != len(references):
        raise ValueError(
            f"{len(predictions)} sequences of log-probabilities but {len(references)} unit counts"
        )
    if len(predictions) == 0:
        raise ValueError("perplexity needs at least one sequence")
    nll = torch.stack([_nll(lp, j) for j, lp in enumerate(predictions)])
    units = torch.tensor(
        [_units(n, j) for j, n in enumerate(references)], dtype=torch.float64
    )
    per = torch.exp(nll / units)
    total = nll.sum() / units.sum()  # nats per unit
    value = torch.exp(total) if aggregate == "corpus" else per.mean()
    cross_entropy = total if base == "e" else total / math.log(2.0)
    return {
        "perplexity": float(value),
        "cross_entropy": float(cross_entropy),
        "per_sequence": per.tolist(),
    }


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(logits)[target] for each position: logits (..., T, V) and targets (..., T) give (..., T).

    The caller shifts the sequence so that logits[t] is the prediction for targets[t], the token after position t.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} do not match targets {tuple(targets.shape)}"
        )
    lp = torch.log_softmax(logits.to(torch.float64), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)


def bits_per_byte(nll_per_token: float, tokens: int, nbytes: int) -> float:
    """(L_T / L_B) * loss / ln 2 of Gao et al. (2020): a per-token loss in nats, as bits per UTF-8 byte."""
    if tokens < 1 or nbytes < 1:
        raise ValueError("the token and byte counts must be at least 1")
    return float(tokens / nbytes * nll_per_token / math.log(2.0))


def windows(n_tokens: int, window: int, stride: int) -> list[tuple[int, int, int]]:
    """Strided windows for a sequence longer than the context: (begin, end, first_scored) per window.

    The model reads tokens [begin, end) and only tokens [first_scored, end) are scored, so every token is scored
    exactly once, and a scored token outside the first window sees at least window - stride tokens of its window
    before it (derivations/units-and-context.md).
    """
    if window < 1 or not 1 <= stride <= window:
        raise ValueError(
            f"need window >= 1 and 1 <= stride <= window, got window = {window}, stride = {stride}"
        )
    if n_tokens < 1:
        raise ValueError(f"need at least one token, got {n_tokens}")
    out, prev_end, begin = [], 0, 0
    while True:
        end = min(begin + window, n_tokens)
        out.append((begin, end, prev_end))
        prev_end = end
        if end == n_tokens:
            return out
        begin += stride
