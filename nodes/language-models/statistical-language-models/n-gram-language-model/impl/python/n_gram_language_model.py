# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""N-gram language models in plain Python: counting, maximum likelihood, add-k, interpolated and modified Kneser-Ney,
Stupid Backoff, the infinity-gram over a suffix array, perplexity and sampling.

A sentence is a list of word strings. pad() puts order - 1 copies of BOS in front and EOS at the end, so every predicted
word, EOS included, ends exactly one N-gram, and BOS is never predicted. The vocabulary of a smoothed model is the set
of words that are predicted in training plus UNK, which stands for every word not seen; UNK has count zero.

KneserNey follows Section 3 of Heafield et al. (2013), the estimation of KenLM:

* adjusted counts: a(w) = c(w) for an N-gram or for an n-gram that starts with BOS, and otherwise the number of distinct
  words v such that v w was seen (the continuation count);
* discounts: for each order n, t[n][k] counts the n-grams with adjusted count k, Y = t1 / (t1 + 2 t2), and
  D(k) = k - (k + 1) Y t[k + 1] / t[k] for k = 1, 2, 3, with D(k) = D(3) above 3 (modified Kneser-Ney); with
  discount="interpolated" every k gets the single discount Y, which is interpolated Kneser-Ney;
* normalisation: u(w | h) = (a(h w) - D(a(h w))) / sum_x a(h x) and b(h) = sum_x D(a(h x)) / sum_x a(h x);
* interpolation: p(w | h) = u(w | h) + b(h) p(w | h'), where h' drops the first word of h, ending in the uniform
  distribution over the vocabulary. A context that was never seen gives u = 0 and b = 1.

Every probability is a sum and product of exact fractions; the model keeps them as floats."""

import math
import random
from collections import Counter, defaultdict
from typing import Any

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def pad(sentence, order):
    """order - 1 copies of BOS, the words, then EOS."""
    if order < 1:
        raise ValueError("the order must be at least 1")
    return [BOS] * (order - 1) + list(sentence) + [EOS]


def ngrams(tokens, n):
    """The n-grams of a token list as tuples, left to right."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(sentences, order):
    """Counts of the order-grams of the padded sentences: one per predicted word."""
    counts = Counter()
    for s in sentences:
        counts.update(ngrams(pad(s, order), order))
    return counts


def all_counts(sentences, order):
    """Raw counts of every n-gram with 1 <= n <= order in the padded sentences, keyed by tuple."""
    counts = Counter()
    for s in sentences:
        toks = pad(s, order)
        for n in range(1, order + 1):
            counts.update(ngrams(toks, n))
    return counts


def mle_prob(counts, word, context):
    """The maximum likelihood estimate c(h w) / sum_x c(h x) from counts of one order; 0/0 raises."""
    context = tuple(context)
    total = sum(c for g, c in counts.items() if g[:-1] == context)
    if total == 0:
        raise ZeroDivisionError("the context was never seen")
    return counts.get(context + (word,), 0) / total


def additive_prob(counts, vocab_size, word, context, k=1.0):
    """Add-k smoothing: (c(h w) + k) / (sum_x c(h x) + k |V|); k = 1 is add-one (Laplace)."""
    context = tuple(context)
    total = sum(c for g, c in counts.items() if g[:-1] == context)
    return (counts.get(context + (word,), 0) + k) / (total + k * vocab_size)


def good_turing_leftover(counts):
    """Turing's estimate n1 / N of the probability that the next n-gram is one never seen: the share of tokens
    whose n-gram occurred once."""
    total = sum(counts.values())
    return sum(1 for c in counts.values() if c == 1) / total


def closed_form_discounts(t, modified=True):
    """Discounts [D(0), D(1), D(2), D(3+)] of one order from t = {k: number of n-grams with adjusted count k}."""
    t1, t2 = t.get(1, 0), t.get(2, 0)
    if t1 + 2 * t2 == 0:
        raise ValueError("no n-gram has adjusted count 1 or 2")
    y = t1 / (t1 + 2 * t2)
    if not modified:
        return [0.0, y, y, y]
    out = [0.0]
    for k in (1, 2, 3):
        if t.get(k, 0) == 0:
            raise ValueError(
                f"no n-gram has adjusted count {k}; use discount='interpolated'"
            )
        d = k - (k + 1) * y * t.get(k + 1, 0) / t[k]
        if not 0.0 <= d <= k:
            raise ValueError(f"discount D({k}) = {d} is outside [0, {k}]")
        out.append(d)
    return out


class KneserNey:
    """Interpolated modified Kneser-Ney (discount="modified"), interpolated Kneser-Ney (discount="interpolated"), or
    fixed discounts given as {n: [0, D1, D2, D3+]}."""

    def __init__(
        self, sentences, order: int, discount: str | dict[int, Any] = "modified"
    ):
        if order < 1:
            raise ValueError("the order must be at least 1")
        self.order = order
        top = count_ngrams(sentences, order)
        # entries[n] maps each n-gram that is a suffix of a seen N-gram to its adjusted count; raw[n] to its count.
        self.entries: dict[int, dict[tuple[str, ...], int]] = {order: dict(top)}
        raw: dict[int, dict[tuple[str, ...], int]] = {order: dict(top)}
        for n in range(order - 1, 0, -1):
            cont, r = Counter(), Counter()
            for g in self.entries[n + 1]:
                cont[g[1:]] += 1
                r[g[1:]] += raw[n + 1][g]
            raw[n] = dict(r)
            self.entries[n] = {g: (r[g] if g[0] == BOS else cont[g]) for g in cont}
        self.raw = raw
        self.vocab = sorted({g[0] for g in self.entries[1]} | {UNK})
        self._vocab = set(self.vocab)
        self.t = {
            n: Counter(a for a in e.values() if a <= 4) for n, e in self.entries.items()
        }
        if isinstance(discount, dict):
            self.discounts = {
                n: list(map(float, discount[n])) for n in range(1, order + 1)
            }
        elif discount in ("modified", "interpolated"):
            self.discounts = {
                n: closed_form_discounts(self.t[n], discount == "modified")
                for n in self.t
            }
        else:
            raise ValueError(
                "discount must be 'modified', 'interpolated' or a dict of lists"
            )
        # denominator, backoff numerator and u for every context.
        self.total, self.gamma = defaultdict(float), defaultdict(float)
        for n, e in self.entries.items():
            for g, a in e.items():
                self.total[g[:-1]] += a
                self.gamma[g[:-1]] += self.D(n, a)
        self.u = {
            g: (a - self.D(n, a)) / self.total[g[:-1]]
            for n, e in self.entries.items()
            for g, a in e.items()
        }

    def D(self, n, a):
        """The discount of an n-gram of order n with adjusted count a."""
        return self.discounts[n][min(a, 3)]

    def backoff(self, context):
        """b(h); 1 for a context that was never seen."""
        context = tuple(context)
        return (
            self.gamma[context] / self.total[context] if context in self.total else 1.0
        )

    def map_word(self, w):
        return w if w in self._vocab or w == BOS else UNK

    def prob(self, word, context=()):
        """p(word | context) with the context truncated to its last order - 1 words."""
        word = self.map_word(word)
        context = tuple(self.map_word(w) for w in context)[
            max(0, len(context) - self.order + 1) :
        ]
        p = 1.0 / len(self.vocab)
        for i in range(len(context), -1, -1):
            h = context[i:]
            if h in self.total:
                p = self.u.get(h + (word,), 0.0) + self.backoff(h) * p
        return p

    def distribution(self, context=()):
        """p(. | context) over the vocabulary, EOS included."""
        return {w: self.prob(w, context) for w in self.vocab}

    def sentence_logprob(self, sentence):
        """Natural log of the probability of a sentence, EOS included, and the number of predicted words."""
        toks = pad(sentence, self.order)
        lp = sum(
            math.log(self.prob(toks[i], toks[i - self.order + 1 : i]))
            for i in range(self.order - 1, len(toks))
        )
        return lp, len(toks) - self.order + 1

    def perplexity(self, sentences):
        return perplexity(self, sentences)

    def sample(self, rng, max_len=50):
        """Draw a sentence word by word from p(. | last order - 1 words) until EOS or max_len words."""
        hist, out = [BOS] * (self.order - 1), []
        while len(out) < max_len:
            dist = self.distribution(tuple(hist[len(hist) - self.order + 1 :]))
            words = [w for w in dist if w != UNK]
            w = rng.choices(words, weights=[dist[v] for v in words])[0]
            if w == EOS:
                break
            out.append(w)
            hist.append(w)
        return out


def perplexity(model, sentences):
    """exp(-(1/T) sum log p) over the T predicted words of the sentences, EOS included."""
    lp, n = 0.0, 0
    for s in sentences:
        a, b = model.sentence_logprob(s)
        lp, n = lp + a, n + b
    return math.exp(-lp / n)


class StupidBackoff:
    """The score S(w | h) of Brants et al. (2007): f(h w) / f(h) when h w was seen, else alpha S(w | h'), ending in
    f(w) / N with N the number of predicted words. The scores do not sum to one."""

    def __init__(self, sentences, order, alpha=0.4):
        self.order, self.alpha = order, alpha
        self.f = all_counts(sentences, order)
        self.N = sum(len(s) + 1 for s in sentences)

    def score(self, word, context=()):
        context = tuple(context)[max(0, len(context) - self.order + 1) :]
        mult = 1.0
        for i in range(len(context) + 1):
            h = context[i:]
            if h and self.f.get(h + (word,), 0) > 0:
                return mult * self.f[h + (word,)] / self.f[h]
            if not h:
                return mult * self.f.get((word,), 0) / self.N
            mult *= self.alpha
        raise AssertionError("unreachable")


class InfiniGram:
    """The infinity-gram of Liu et al. (2024) over a suffix array of the training documents.

    P(w | h) = cnt(s w) / cnt(s .), where s is the longest suffix of h that occurs followed by a token (so that the
    estimate is a distribution) and cnt(s .) counts those occurrences; the effective n is len(s) + 1. Documents are
    joined with a separator that never matches a word."""

    SEP = None

    def __init__(self, documents):
        self.tokens = []
        for d in documents:
            self.tokens.extend(d)
            self.tokens.append(self.SEP)
        # The separator sorts below every word.
        order = sorted({w for w in self.tokens if w is not None})
        rank = {w: i + 1 for i, w in enumerate(order)}
        self.ids = [rank[w] if w is not None else 0 for w in self.tokens]
        self.rank = rank
        self.sa = suffix_array(self.ids)
        self.vocab = order

    def _range(self, pattern):
        """The block [lo, hi) of the suffix array whose suffixes start with pattern."""
        if any(w not in self.rank for w in pattern):
            return 0, 0
        pat, m, ids = [self.rank[w] for w in pattern], len(pattern), self.ids
        lo, hi = 0, len(self.sa)
        while lo < hi:  # the first suffix whose first m ids are not below the pattern
            mid = (lo + hi) // 2
            if ids[self.sa[mid] : self.sa[mid] + m] < pat:
                lo = mid + 1
            else:
                hi = mid
        first, hi = lo, len(self.sa)
        while lo < hi:  # the first suffix whose first m ids are above the pattern
            mid = (lo + hi) // 2
            if ids[self.sa[mid] : self.sa[mid] + m] <= pat:
                lo = mid + 1
            else:
                hi = mid
        return first, lo

    def count(self, pattern):
        lo, hi = self._range(pattern)
        return hi - lo

    def next_counts(self, pattern):
        """Counter of the tokens that follow each occurrence of pattern (the separator excluded)."""
        lo, hi = self._range(pattern)
        m, out = len(pattern), Counter()
        for i in self.sa[lo:hi]:
            if i + m < len(self.tokens) and self.tokens[i + m] is not None:
                out[self.tokens[i + m]] += 1
        return out

    def distribution(self, context):
        """(P(. | context) as a dict over the tokens that can follow, effective n)."""
        context = tuple(context)
        for start in range(len(context) + 1):
            nxt = self.next_counts(context[start:])
            if nxt:
                total = sum(nxt.values())
                return {w: c / total for w, c in nxt.items()}, len(context) - start + 1
        raise ValueError("the training data is empty")

    def prob(self, word, context):
        dist, _ = self.distribution(context)
        return dist.get(word, 0.0)


def suffix_array(ids):
    """The suffix array of a list of integers by prefix doubling: sort by (rank of the first k, rank of the next k)."""
    n = len(ids)
    rank, sa, k = list(ids), list(range(n)), 1
    while True:

        def key(i, rank=rank, k=k):
            return rank[i], rank[i + k] if i + k < n else -1

        sa.sort(key=key)
        new = [0] * n
        for j in range(1, n):
            new[sa[j]] = new[sa[j - 1]] + (key(sa[j]) != key(sa[j - 1]))
        rank = new
        if n == 0 or rank[sa[-1]] == n - 1:
            return sa
        k *= 2


def sample_sentences(model, n, seed=0, max_len=50):
    rng = random.Random(seed)
    return [model.sample(rng, max_len) for _ in range(n)]
