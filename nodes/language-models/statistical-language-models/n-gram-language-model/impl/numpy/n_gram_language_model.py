# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/n-gram-language-model/ (ai://NEW-n-gram-language-model).
"""N-gram language models in NumPy: counting with np.unique over sliding windows, modified and interpolated Kneser-Ney
as sorted key tables queried with np.searchsorted, Stupid Backoff, perplexity, sampling, and the infinity-gram over a
suffix array built by prefix doubling.

Words are integer ids from a Vocab: 0 is UNK, 1 is BOS, 2 is EOS and the training words follow in sorted order. A
sentence is padded with order - 1 BOS ids in front and EOS at the end, so each predicted id ends exactly one N-gram.
An n-gram is packed into one int64 key, the digits of its ids in base len(vocab), so tables are sorted 1-D arrays.

KneserNey follows Section 3 of Heafield et al. (2013): adjusted counts (the raw count for N-grams and for n-grams that
start with BOS, the number of distinct left extensions otherwise), closed-form discounts D(k) = k - (k + 1) Y t[k + 1] /
t[k] with Y = t1 / (t1 + 2 t2) for k = 1, 2, 3 (or the single discount Y with discount="interpolated"), pseudo
probabilities u = (a - D(a)) / sum a, backoffs b = sum D(a) / sum a, and p(w | h) = u(w | h) + b(h) p(w | h') down to the
uniform distribution over the vocabulary. An unseen context has u = 0 and b = 1."""

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

UNK, BOS, EOS = 0, 1, 2
SPECIALS = ("<unk>", "<s>", "</s>")


class Vocab:
    def __init__(self, sentences):
        words = sorted({w for s in sentences for w in s} - set(SPECIALS))
        self.itos = list(SPECIALS) + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, order):
        """The padded id array of a sentence; unknown words become UNK."""
        ids = [self.stoi.get(w, UNK) for w in sentence]
        return np.array([BOS] * (order - 1) + ids + [EOS], dtype=np.int64)


def windows(padded, n):
    """All n-grams of a list of padded id arrays as one (count, n) array."""
    parts = [sliding_window_view(p, n) for p in padded if len(p) >= n]
    return np.concatenate(parts) if parts else np.zeros((0, n), dtype=np.int64)


def pack(grams, base):
    """One int64 key per row: the ids as digits in the given base, most significant first."""
    key = np.zeros(len(grams), dtype=np.int64)
    for j in range(grams.shape[1]):
        key = key * base + grams[:, j]
    return key


def count(grams):
    """The distinct rows of grams in lexicographic order and their counts."""
    if len(grams) == 0:
        return grams, np.zeros(0, dtype=np.int64)
    return np.unique(grams, axis=0, return_counts=True)


def lookup(keys, sorted_keys):
    """(index, found) of each key in a sorted key array."""
    idx = np.searchsorted(sorted_keys, keys)
    idx = np.minimum(idx, max(len(sorted_keys) - 1, 0))
    found = sorted_keys[idx] == keys if len(sorted_keys) else np.zeros(len(keys), dtype=bool)
    return idx, found


def closed_form_discounts(t, modified=True):
    """[D(0), D(1), D(2), D(3+)] from t[k] = number of n-grams with adjusted count k (t has length at least 5)."""
    t = np.asarray(t, dtype=np.float64)
    if t[1] + 2 * t[2] == 0:
        raise ValueError("no n-gram has adjusted count 1 or 2")
    y = t[1] / (t[1] + 2 * t[2])
    if not modified:
        return np.array([0.0, y, y, y])
    if np.any(t[1:4] == 0):
        raise ValueError("some adjusted count in 1..3 does not occur; use discount='interpolated'")
    k = np.arange(1, 4)
    d = k - (k + 1) * y * t[2:5] / t[1:4]
    if np.any(d < 0) or np.any(d > k):
        raise ValueError(f"discounts {d} are outside [0, k]")
    return np.concatenate([[0.0], d])


class KneserNey:
    def __init__(self, sentences, order: int, discount: str | dict[int, Any] = "modified", vocab=None):
        if order < 1:
            raise ValueError("the order must be at least 1")
        self.order = order
        self.vocab = vocab if vocab is not None else Vocab(sentences)
        self.V = len(self.vocab)
        if float(self.V) ** order >= 2.0**62:
            raise ValueError("the vocabulary is too large to pack an n-gram in an int64 key")
        padded = [self.vocab.encode(s, order) for s in sentences]
        grams, c = count(windows(padded, order))
        self.grams: dict[int, np.ndarray] = {order: grams}
        self.adj: dict[int, np.ndarray] = {order: c}
        raw: dict[int, np.ndarray] = {order: c}
        for n in range(order - 1, 0, -1):
            suf, inv = np.unique(self.grams[n + 1][:, 1:], axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            cont = np.bincount(inv, minlength=len(suf))
            raw[n] = np.bincount(inv, weights=raw[n + 1], minlength=len(suf)).astype(np.int64)
            self.grams[n], self.adj[n] = suf, np.where(suf[:, 0] == BOS, raw[n], cont)
        self.raw = raw
        # The unigram entries plus UNK make the model's support; a word id not among them gets UNK's probability.
        self.support = np.union1d(self.grams[1][:, 0], [UNK])
        self.t = {n: np.bincount(np.minimum(a, 5), minlength=6)[:5] for n, a in self.adj.items()}
        if isinstance(discount, dict):
            self.discounts = {n: np.asarray(discount[n], dtype=np.float64) for n in range(1, order + 1)}
        elif discount in ("modified", "interpolated"):
            self.discounts = {n: closed_form_discounts(self.t[n], discount == "modified") for n in self.t}
        else:
            raise ValueError("discount must be 'modified', 'interpolated' or a dict of arrays")
        self.keys, self.u, self.ctx, self.b = {}, {}, {}, {}
        for n in range(1, order + 1):
            g, a = self.grams[n], self.adj[n]
            d = self.discounts[n][np.minimum(a, 3)]
            ck = pack(g[:, :-1], self.V)
            ctx, cinv = np.unique(ck, return_inverse=True)
            total = np.bincount(cinv, weights=a.astype(np.float64))
            gamma = np.bincount(cinv, weights=d)
            self.keys[n], self.u[n] = pack(g, self.V), (a - d) / total[cinv]
            self.ctx[n], self.b[n] = ctx, gamma / total

    def prob(self, words, contexts):
        """p(words[i] | contexts[i]) for an id array words of shape (M,) and contexts of shape (M, order - 1)."""
        words = np.asarray(words, dtype=np.int64)
        words = np.where(np.isin(words, self.support), words, UNK)
        contexts = np.asarray(contexts, dtype=np.int64).reshape(len(words), self.order - 1)
        p = np.full(len(words), 1.0 / len(self.support))
        for n in range(1, self.order + 1):
            h = contexts[:, self.order - n :]
            ci, cf = lookup(pack(h, self.V), self.ctx[n])
            gi, gf = lookup(pack(np.column_stack([h, words]), self.V), self.keys[n])
            u = np.where(gf, self.u[n][gi], 0.0)
            p = np.where(cf, u + np.where(cf, self.b[n][ci], 1.0) * p, p)
        return p

    def distribution(self, context):
        """p(. | context) over the support ids, returned as (ids, probabilities)."""
        context = np.asarray(context, dtype=np.int64)
        context = context[len(context) - self.order + 1 :] if self.order > 1 else context[:0]
        return self.support, self.prob(self.support, np.tile(context, (len(self.support), 1)))

    def token_logprobs(self, sentences):
        """Natural log of the probability of every predicted word, EOS included, of the sentences."""
        win = windows([self.vocab.encode(s, self.order) for s in sentences], self.order)
        return np.log(self.prob(win[:, -1], win[:, :-1]))

    def perplexity(self, sentences):
        return float(np.exp(-self.token_logprobs(sentences).mean()))

    def sample(self, rng, max_len=50):
        """A sentence drawn word by word until EOS or max_len words; UNK is never drawn."""
        hist, out = [BOS] * (self.order - 1), []
        while len(out) < max_len:
            ids, p = self.distribution(np.array(hist[len(hist) - self.order + 1 :], dtype=np.int64))
            p = np.where(ids == UNK, 0.0, p)
            w = int(rng.choice(ids, p=p / p.sum()))
            if w == EOS:
                break
            out.append(self.vocab.itos[w])
            hist.append(w)
        return out


class StupidBackoff:
    """S(w | h) = f(h w) / f(h) if h w was seen, else alpha S(w | h'), ending in f(w) / N (Brants et al., 2007)."""

    def __init__(self, sentences, order, alpha=0.4, vocab=None):
        self.order, self.alpha = order, alpha
        self.vocab = vocab if vocab is not None else Vocab(sentences)
        self.V = len(self.vocab)
        padded = [self.vocab.encode(s, order) for s in sentences]
        self.keys, self.f = {}, {}
        for n in range(1, order + 1):
            g, c = count(windows(padded, n))
            self.keys[n], self.f[n] = pack(g, self.V), c
        self.N = sum(len(s) + 1 for s in sentences)

    def _f(self, grams):
        n = grams.shape[1]
        i, found = lookup(pack(grams, self.V), self.keys[n])
        return np.where(found, self.f[n][i], 0)

    def score(self, words, contexts):
        words = np.asarray(words, dtype=np.int64)
        contexts = np.asarray(contexts, dtype=np.int64).reshape(len(words), self.order - 1)
        s = self._f(words[:, None]) / self.N
        for n in range(2, self.order + 1):
            h = contexts[:, self.order - n :]
            num = self._f(np.column_stack([h, words]))
            s = np.where(num > 0, num / np.maximum(self._f(h), 1), self.alpha * s)
        return s


def suffix_array(tokens):
    """The suffix array of an int array by prefix doubling: ranks of 2^k-prefixes sorted with np.lexsort."""
    n = len(tokens)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rank = np.unique(tokens, return_inverse=True)[1].reshape(-1).astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if k < n:
            second[: n - k] = rank[k:]
        sa = np.lexsort((second, rank))
        pair = np.stack([rank[sa], second[sa]], axis=1)
        new = np.empty(n, dtype=np.int64)
        new[sa] = np.concatenate([[0], np.cumsum(np.any(pair[1:] != pair[:-1], axis=1))])
        rank = new
        if rank.max() == n - 1 or k >= n:
            return sa
        k *= 2


class InfiniGram:
    """P(w | h) = cnt(s w) / cnt(s .) for the longest suffix s of h that occurs followed by a token (Liu et al., 2024).
    Documents are joined with the separator id -1, which matches no word and sorts first."""

    SEP = -1

    def __init__(self, documents, vocab=None):
        self.vocab = vocab if vocab is not None else Vocab(documents)
        parts = []
        for d in documents:
            parts += [self.vocab.stoi.get(w, UNK) for w in d] + [self.SEP]
        self.tokens = np.array(parts, dtype=np.int64)
        self.sa = suffix_array(self.tokens)

    def _cmp(self, start, pattern):
        """-1, 0 or 1 as the suffix at start is below, begins with, or is above pattern."""
        seg = self.tokens[start : start + len(pattern)]
        m = min(len(seg), len(pattern))
        diff = np.flatnonzero(seg[:m] != pattern[:m])
        if len(diff):
            return -1 if seg[diff[0]] < pattern[diff[0]] else 1
        return 0 if len(seg) == len(pattern) else -1

    def _range(self, pattern):
        pattern = np.asarray(pattern, dtype=np.int64)
        lo, hi = 0, len(self.sa)
        while lo < hi:  # first suffix not below the pattern
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if self._cmp(self.sa[mid], pattern) < 0 else (lo, mid)
        first, hi = lo, len(self.sa)
        while lo < hi:  # first suffix above the pattern
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if self._cmp(self.sa[mid], pattern) <= 0 else (lo, mid)
        return first, lo

    def count(self, pattern):
        lo, hi = self._range(pattern)
        return hi - lo

    def next_counts(self, pattern):
        """(ids, counts) of the tokens that follow the occurrences of pattern, the separator excluded."""
        lo, hi = self._range(pattern)
        pos = self.sa[lo:hi] + len(pattern)
        nxt = self.tokens[pos[pos < len(self.tokens)]]
        return np.unique(nxt[nxt != self.SEP], return_counts=True)

    def distribution(self, context):
        """(ids, probabilities, effective n) for a context given as words."""
        ids = np.array([self.vocab.stoi.get(w, UNK) for w in context], dtype=np.int64)
        for start in range(len(ids) + 1):
            w, c = self.next_counts(ids[start:])
            if len(w):
                return w, c / c.sum(), len(ids) - start + 1
        raise ValueError("the training data is empty")
