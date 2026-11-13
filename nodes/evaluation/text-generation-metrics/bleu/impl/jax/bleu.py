# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bleu/ (ai://V3JG9E).
"""Corpus BLEU (Papineni et al., 2002) as sacreBLEU 2.6.0 computes it, with the precisions and the penalty in JAX.

The tokenizers and the n-gram counts are plain Python; the modified precisions (eq-modified-precision), the smoothing
of the orders with no match (eq-smooth-exp), the brevity penalty (eq-brevity-penalty) and the geometric mean
(eq-bleu) are JAX arrays in float64. The score is on the 0 to 100 scale of sacreBLEU.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # float64, as sacreBLEU computes in Python floats

# The regular expressions of mteval-v13a, as sacreBLEU's tokenizer_13a and tokenizer_re apply them.
_RE_13A = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),  # split off punctuation and symbols
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),  # a period or a comma not after a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),  # a period or a comma not before a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),  # a dash after a digit
]


def _tok_13a(line: str) -> str:
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    line = f" {line} "
    for pattern, repl in _RE_13A:
        line = pattern.sub(repl, line)
    return " ".join(line.split())


def _cat(prefix: str) -> Callable[[str], bool]:
    return lambda c: unicodedata.category(c).startswith(prefix)


_P, _N, _S = _cat("P"), _cat("N"), _cat("S")


def _pairs(line: str, first: Callable[[str], bool], second: Callable[[str], bool], fmt: str) -> str:
    """Rewrite each pair (a, b) with first(a) and second(b), left to right and without overlap, as re.sub does."""
    out, i = [], 0
    while i < len(line):
        if i + 1 < len(line) and first(line[i]) and second(line[i + 1]):
            out.append(fmt.format(line[i], line[i + 1]))
            i += 2
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def _tok_intl(line: str) -> str:
    """mteval-v14 international tokenization: Unicode punctuation unless between digits, and every symbol."""
    line = _pairs(line, lambda c: not _N(c), _P, "{0} {1} ")  # (\P{N})(\p{P}) -> \1 \2
    line = _pairs(line, _P, lambda c: not _N(c), " {0} {1}")  # (\p{P})(\P{N}) -> \1 \2
    line = "".join(f" {c} " if _S(c) else c for c in line)  # (\p{S}) -> \1
    return " ".join(line.split())


TOKENIZERS: dict[str, Callable[[str], str]] = {
    "13a": _tok_13a,
    "intl": _tok_intl,
    "char": lambda line: " ".join(line),
    "none": lambda line: line,
}
SMOOTH_DEFAULTS = {"none": None, "floor": 0.1, "add-k": 1.0, "exp": None}


def _ngrams(tokens: list[str], max_order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for n in range(1, max_order + 1) for i in range(len(tokens) - n + 1))


def statistics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenize: str = "13a",
    lowercase: bool = False,
    max_ngram_order: int = 4,
) -> tuple[int, int, list[int], list[int]]:
    """The sufficient statistics of the corpus: sys_len, ref_len, and the clipped and total counts of each order."""
    if tokenize not in TOKENIZERS:
        raise ValueError(f"tokenize must be one of {sorted(TOKENIZERS)}, not {tokenize!r}")
    if max_ngram_order < 1:
        raise ValueError("max_ngram_order must be at least 1")
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions but references for {len(references)} segments")
    if any(len(refs) == 0 for refs in references) or len({len(refs) for refs in references}) > 1:
        raise ValueError("every segment needs the same number of references, at least one")
    tok = TOKENIZERS[tokenize]

    def prep(s: str) -> list[str]:
        return tok((s.lower() if lowercase else s).rstrip()).split()

    sys_len = ref_len = 0
    correct, total = [0] * max_ngram_order, [0] * max_ngram_order
    for hyp, refs in zip(predictions, references, strict=True):
        h = prep(hyp)
        ref_max: Counter[tuple[str, ...]] = Counter()
        ref_lens = []
        for ref in refs:
            r = prep(ref)
            ref_lens.append(len(r))
            ref_max |= _ngrams(r, max_ngram_order)  # the largest count of each n-gram in any one reference
        sys_len += len(h)
        ref_len += min(ref_lens, key=lambda x: (abs(len(h) - x), x))  # the closest length, the shorter on a tie
        for gram, count in _ngrams(h, max_ngram_order).items():
            total[len(gram) - 1] += count
            correct[len(gram) - 1] += min(count, ref_max[gram])  # clipping, eq: eq-modified-precision
    return sys_len, ref_len, correct, total


def score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    tokenize: str = "13a",
    lowercase: bool = False,
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    max_ngram_order: int = 4,
    effective_order: bool = False,
) -> dict:
    """Corpus BLEU of the predictions against the references; references[i] holds the references of segment i."""
    if smooth_method not in SMOOTH_DEFAULTS:
        raise ValueError(f"smooth_method must be one of {sorted(SMOOTH_DEFAULTS)}, not {smooth_method!r}")
    if smooth_value is None:
        smooth_value = SMOOTH_DEFAULTS[smooth_method]
    sys_len, ref_len, correct, total = statistics(predictions, references, tokenize, lowercase, max_ngram_order)
    big_n = max_ngram_order

    # eq: eq-brevity-penalty
    bp = 1.0
    if sys_len < ref_len:
        bp = float(jnp.exp(jnp.float64(1.0 - ref_len / sys_len))) if sys_len > 0 else 0.0
    out = {"score": 0.0, "precisions": [0.0] * big_n, "bp": bp, "sys_len": sys_len, "ref_len": ref_len}
    if not any(correct):
        return out

    c = jnp.asarray(correct, dtype=jnp.float64)
    t = jnp.asarray(total, dtype=jnp.float64)
    if smooth_method == "add-k":
        assert smooth_value is not None
        c = c.at[1:].add(smooth_value)
        t = t.at[1:].add(smooth_value)
    # The orders that have at least one hypothesis n-gram; the totals do not grow with n, so they are a prefix.
    reached = jnp.cumprod(t > 0).astype(bool)
    safe_t = jnp.where(reached, t, 1.0)
    zero = reached & (c == 0)  # an order that is not reached has no match either, so its precision stays 0
    p = 100.0 * c / safe_t
    if smooth_method == "exp":  # eq: eq-smooth-exp, 1/2, 1/4, ... of a count for each order with no match
        p = jnp.where(zero, 100.0 / (2.0 ** jnp.cumsum(zero) * safe_t), p)
    elif smooth_method == "floor":
        assert smooth_value is not None
        p = jnp.where(zero, 100.0 * smooth_value / safe_t, p)

    order = int(reached.sum()) if effective_order else big_n
    used = p[:order]
    out["precisions"] = [float(x) for x in p.tolist()]
    if order == 0 or bool((used == 0).any()):
        return out
    out["score"] = float(bp * jnp.exp(jnp.sum(jnp.log(used)) / order))  # eq: eq-bleu
    return out
