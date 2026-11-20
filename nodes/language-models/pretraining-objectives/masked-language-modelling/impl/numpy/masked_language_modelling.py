# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Masked language modelling in NumPy: the corruption schemes and a small bidirectional model with the backward pass
written out.

Token ids 0 and 1 are the special tokens [PAD] and [MASK]; ordinary tokens are 2 to V - 1.

Corruption schemes:

* BERT (Devlin et al., 2019): choose 15% of the ordinary positions; of those, replace 80% by [MASK], 10% by a random
  token and leave 10% unchanged; predict the original token at every chosen position. bert_corrupt is the deterministic
  part, given the chosen positions, one uniform number per position and one random token per position.
* SpanBERT (Joshi et al., 2019): choose whole spans whose lengths follow a geometric distribution with p = 0.2
  truncated at 10, at uniform starting points, until the budget of 15% of the tokens is spent.
* T5 (Raffel et al., 2020): choose the noise positions as random spans with a given mean length, replace each run of
  noise positions in the input by one sentinel token and make the target the runs, each after its sentinel, followed
  by a final sentinel.
* ELECTRA (Clark et al., 2020): a small masked language model, the generator, fills the masked positions with samples;
  a discriminator labels every position as original or replaced.

The model is a window model, the smallest network that is bidirectional: position t sees the embeddings of the
corrupted tokens at the offsets in `offsets` (by default -w to w, its own position included), concatenated, then

    z = tanh(d + H x),   logits = b + U z,   D = sigmoid(c + g . z)

where D is the discriminator's probability that the token at t is original. With offsets -w to -1 the same network is
a left-to-right (causal) model, which predicts x_t from the tokens before it. A Transformer encoder replaces the window
by attention over the whole sequence; the losses and the corruption are the same."""

from typing import Any

import numpy as np

PAD, MASK = 0, 1
FIRST = 2  # the first ordinary token id
Params = dict[str, np.ndarray]


# Corruption schemes


def select_positions(ids: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """BERT's selection: max(1, round(rate * c)) of the c ordinary positions, uniformly without replacement."""
    ok = np.flatnonzero(ids >= FIRST)
    k = max(1, int(round(rate * len(ok))))
    sel = np.zeros(ids.shape, dtype=bool)
    sel[rng.choice(ok, size=k, replace=False)] = True
    return sel


def bert_corrupt(
    ids: np.ndarray,
    selected: np.ndarray,
    u: np.ndarray,
    random_ids: np.ndarray,
    p_mask: float = 0.8,
    p_random: float = 0.1,
) -> np.ndarray:
    """At each selected position: [MASK] if u < p_mask, the random token if u < p_mask + p_random, else unchanged."""
    out = ids.copy()
    to_mask = selected & (u < p_mask)
    to_random = selected & (u >= p_mask) & (u < p_mask + p_random)
    out[to_mask] = MASK
    out[to_random] = random_ids[to_random]
    return out


def bert_mask(
    ids: np.ndarray, V: int, rng: np.random.Generator, rate: float = 0.15, p_mask: float = 0.8, p_random: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """A fresh BERT corruption of each row of ids: (corrupted ids, selected positions)."""
    ids = np.atleast_2d(ids)
    sel = np.stack([select_positions(row, rate, rng) for row in ids])
    u = rng.random(ids.shape)
    rnd = rng.integers(FIRST, V, size=ids.shape)
    return bert_corrupt(ids, sel, u, rnd, p_mask, p_random), sel


def truncated_geometric(p: float, lmax: int) -> np.ndarray:
    """P(l) for l = 1 .. lmax of a geometric distribution with success probability p, truncated and renormalised."""
    lengths = np.arange(1, lmax + 1)
    w = p * (1 - p) ** (lengths - 1)
    return w / w.sum()


def span_mask(
    n: int, rng: np.random.Generator, rate: float = 0.15, p: float = 0.2, lmax: int = 10
) -> tuple[np.ndarray, list[int]]:
    """SpanBERT's selection over n tokens (each token one word here): spans of truncated-geometric length at uniform
    starts, skipping any span that touches one already chosen, until at least round(rate * n) tokens are chosen.
    Returns the selected positions and the lengths of the chosen spans. A long span is more likely to touch an
    earlier one, so the kept spans average a little less than the mean of the distribution they are drawn from."""
    budget = max(1, int(round(rate * n)))
    pmf = truncated_geometric(p, lmax)
    sel = np.zeros(n, dtype=bool)
    lengths = []
    while sel.sum() < budget:
        length = 1 + int(rng.choice(lmax, p=pmf))
        if length > n:
            continue
        start = int(rng.integers(n - length + 1))
        lo, hi = max(0, start - 1), min(n, start + length + 1)
        if sel[lo:hi].any():
            continue
        sel[start : start + length] = True
        lengths.append(length)
    return sel, lengths


def random_spans_noise_mask(n: int, rate: float, mean_span: float, rng: np.random.Generator) -> np.ndarray:
    """T5's span selection: round(rate * n) noise tokens in round(noise / mean_span) spans; the lengths of the noise
    spans and of the kept spans between them are uniform random compositions; the sequence starts with a kept span."""
    n_noise = min(max(1, int(round(rate * n))), n - 1)
    n_spans = max(1, int(round(n_noise / mean_span)))
    n_spans = min(n_spans, n_noise, n - n_noise)

    def composition(total: int, k: int) -> np.ndarray:
        cuts = np.sort(rng.choice(np.arange(1, total), size=k - 1, replace=False))
        return np.diff(np.concatenate([[0], cuts, [total]]))

    keep, noise = composition(n - n_noise, n_spans), composition(n_noise, n_spans)
    mask = np.zeros(n, dtype=bool)
    at = 0
    for k, m in zip(keep, noise, strict=True):
        at += int(k)
        mask[at : at + int(m)] = True
        at += int(m)
    return mask


def span_corrupt(ids: np.ndarray, noise: np.ndarray, sentinels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T5's input and target: each run of noise positions becomes one sentinel in the input; the target is every run
    after its sentinel, then one more sentinel."""
    inp, tgt = [], []
    s = 0
    for t in range(len(ids)):
        if noise[t]:
            if t == 0 or not noise[t - 1]:
                inp.append(int(sentinels[s]))
                tgt.append(int(sentinels[s]))
                s += 1
            tgt.append(int(ids[t]))
        else:
            inp.append(int(ids[t]))
    tgt.append(int(sentinels[s]))
    return np.array(inp, dtype=np.int64), np.array(tgt, dtype=np.int64)


def run_count(mask: np.ndarray) -> int:
    """The number of maximal runs of True in a 1-D boolean array."""
    m = mask.astype(np.int8)
    return int(m[0] + np.sum(np.diff(m) == 1))


# The window model


def softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * a))


class WindowMLM:
    """A bidirectional window model with an MLM head (U, b) and a discriminator head (g, c)."""

    def __init__(self, V: int, m: int, h: int, offsets: tuple[int, ...], seed: int = 0, scale: float = 0.3):
        rng = np.random.default_rng(seed)
        self.V, self.m, self.h, self.offsets = V, m, h, tuple(offsets)
        k = len(self.offsets) * m
        self.p: Params = {
            "E": rng.normal(0, scale, (V, m)),
            "H": rng.normal(0, 1 / np.sqrt(k), (h, k)),
            "d": np.zeros(h),
            "U": rng.normal(0, 1 / np.sqrt(h), (V, h)),
            "b": np.zeros(V),
            "g": rng.normal(0, 1 / np.sqrt(h), h),
            "c": np.zeros(1),
        }

    def windows(self, inp: np.ndarray) -> np.ndarray:
        """(B, L, len(offsets)) token ids around each position, [PAD] beyond the ends."""
        w = max(abs(o) for o in self.offsets)
        B, L = inp.shape
        padded = np.full((B, L + 2 * w), PAD, dtype=np.int64)
        padded[:, w : w + L] = inp
        idx = np.arange(L)[:, None] + w + np.array(self.offsets)[None, :]
        return padded[:, idx]

    def hidden(self, win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.p["E"][win].reshape(*win.shape[:-1], -1)
        return x, np.tanh(self.p["d"] + x @ self.p["H"].T)

    def logits(self, inp: np.ndarray) -> np.ndarray:
        _, z = self.hidden(self.windows(inp))
        return self.p["b"] + z @ self.p["U"].T

    def _trunk_grads(self, win: np.ndarray, x: np.ndarray, z: np.ndarray, dz: np.ndarray, g: Params) -> None:
        da = dz * (1 - z**2)
        da2 = da.reshape(-1, self.h)
        g["H"] = da2.T @ x.reshape(len(da2), -1)
        g["d"] = da2.sum(0)
        dx = (da @ self.p["H"]).reshape(-1, self.m)
        g["E"] = np.zeros_like(self.p["E"])
        np.add.at(g["E"], win.reshape(-1), dx)

    def mlm_loss_and_grads(
        self, inp: np.ndarray, targets: np.ndarray, predict: np.ndarray, decay: float = 0.0
    ) -> tuple[float, Params]:
        """Mean cross-entropy of the original tokens at the positions in `predict`, plus decay / 2 times the squared
        norm of E, H and U; and its gradient. Positions outside `predict` give no loss."""
        win = self.windows(inp)[predict]  # (K, n_offsets)
        x, z = self.hidden(win)
        lg = self.p["b"] + z @ self.p["U"].T
        P = softmax(lg)
        tgt = targets[predict]
        K = len(tgt)
        loss = -np.mean(np.log(P[np.arange(K), tgt]))
        dlg = P
        dlg[np.arange(K), tgt] -= 1
        dlg /= K
        g: Params = {"U": dlg.T @ z, "b": dlg.sum(0), "g": np.zeros(self.h), "c": np.zeros(1)}
        self._trunk_grads(win, x, z, dlg @ self.p["U"], g)
        if decay:
            loss += 0.5 * decay * sum(float(np.sum(self.p[k] ** 2)) for k in ("E", "H", "U"))
            for k in ("E", "H", "U"):
                g[k] = g[k] + decay * self.p[k]
        return float(loss), g

    def disc_loss_and_grads(self, inp: np.ndarray, original: np.ndarray, valid: np.ndarray) -> tuple[float, Params]:
        """ELECTRA's discriminator loss, the mean binary cross-entropy of D = P(original) over the valid positions."""
        win = self.windows(inp)[valid]
        x, z = self.hidden(win)
        D = sigmoid(self.p["c"][0] + z @ self.p["g"])
        y = original[valid].astype(float)
        n = len(y)
        eps = 1e-300
        loss = -np.mean(y * np.log(D + eps) + (1 - y) * np.log(1 - D + eps))
        ds = (D - y) / n
        g: Params = {"U": np.zeros_like(self.p["U"]), "b": np.zeros(self.V), "g": ds @ z, "c": np.array([ds.sum()])}
        self._trunk_grads(win, x, z, ds[:, None] * self.p["g"][None, :], g)
        return float(loss), g

    def cloze_logprobs(self, ids: np.ndarray) -> np.ndarray:
        """log p(x_t | the other tokens) with only position t replaced by [MASK], for every position of every row.
        A window model sees position t only through the window at t, so one pass computes all of them. For a model
        whose offsets exclude 0 this is the left-to-right log-probability."""
        win = self.windows(ids).copy()
        if 0 in self.offsets:
            win[..., self.offsets.index(0)] = MASK
        _, z = self.hidden(win)
        lg = self.p["b"] + z @ self.p["U"].T
        lse = lg.max(-1) + np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1))
        return np.take_along_axis(lg, ids[..., None], -1)[..., 0] - lse

    def sgd(self, g: Params, lr: float) -> None:
        for k, v in g.items():
            self.p[k] -= lr * v


def sample_tokens(logits: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF samples from softmax(logits) over the ordinary tokens, one uniform number per row."""
    P = softmax(logits[:, FIRST:])
    cdf = np.cumsum(P, axis=-1)
    idx = np.minimum((cdf < u[:, None]).sum(-1), P.shape[1] - 1)
    return FIRST + idx


def electra_losses(
    gen: WindowMLM, disc: WindowMLM, ids: np.ndarray, selected: np.ndarray, u: np.ndarray, lam: float
) -> dict[str, Any]:
    """One ELECTRA example batch: mask the selected positions, let the generator sample replacements, and score the
    discriminator on every ordinary position. The total is the generator loss plus lam times the discriminator loss;
    no gradient flows through the sampling, so each network gets the gradient of its own term."""
    masked = np.where(selected, MASK, ids)
    gen_loss, gen_grads = gen.mlm_loss_and_grads(masked, ids, selected)
    corrupt = ids.copy()
    corrupt[selected] = sample_tokens(gen.logits(masked)[selected], u[selected])
    original = corrupt == ids
    disc_loss, disc_grads = disc.disc_loss_and_grads(corrupt, original, ids >= FIRST)
    disc_grads = {k: lam * v for k, v in disc_grads.items()}
    return {
        "corrupt": corrupt,
        "gen_loss": gen_loss,
        "disc_loss": disc_loss,
        "total": gen_loss + lam * disc_loss,
        "gen_grads": gen_grads,
        "disc_grads": disc_grads,
    }


def train_mlm(
    model: WindowMLM,
    seqs: np.ndarray,
    epochs: int,
    lr: float,
    batch: int,
    rng: np.random.Generator,
    rate: float = 0.15,
    p_mask: float = 0.8,
    p_random: float = 0.1,
    dynamic: bool = True,
    decay: float = 0.0,
) -> list[float]:
    """SGD on the masked LM loss. With dynamic=False each sequence keeps the corruption drawn before the first epoch,
    as in the original BERT preprocessing; with dynamic=True it gets a fresh one every epoch (RoBERTa). A model whose
    offsets exclude 0 is trained as a left-to-right model on every position instead. Returns the mean loss per
    epoch."""
    causal = 0 not in model.offsets
    fixed = None if dynamic or causal else bert_mask(seqs, model.V, rng, rate, p_mask, p_random)
    hist = []
    for _ in range(epochs):
        if causal:
            inp, pred = seqs, seqs >= FIRST
        elif fixed is not None:
            inp, pred = fixed
        else:
            inp, pred = bert_mask(seqs, model.V, rng, rate, p_mask, p_random)
        order = rng.permutation(len(seqs))
        losses = []
        for i in range(0, len(seqs), batch):
            rows = order[i : i + batch]
            loss, g = model.mlm_loss_and_grads(inp[rows], seqs[rows], pred[rows], decay)
            model.sgd(g, lr)
            losses.append(loss)
        hist.append(float(np.mean(losses)))
    return hist


# A source with known conditionals


def class_chain(rng: np.random.Generator, K: int, per: int, concentration: float) -> np.ndarray:
    """A first-order Markov chain over K word classes, (K + 1) x K with row K the start distribution."""
    return rng.dirichlet(np.full(K, concentration), size=K + 1)


def sample_sequences(rng: np.random.Generator, trans: np.ndarray, per: int, n: int, L: int) -> np.ndarray:
    """n sequences of L words; the class follows the chain and the word is uniform among the per words of its class.
    Word ids are FIRST + class * per + j."""
    K = trans.shape[1]
    out = np.empty((n, L), dtype=np.int64)
    for i in range(n):
        c = K
        for t in range(L):
            c = int(rng.choice(K, p=trans[c]))
            out[i, t] = FIRST + c * per + int(rng.integers(per))
    return out


def source_cloze_logprobs(ids: np.ndarray, trans: np.ndarray, per: int, causal: bool = False) -> np.ndarray:
    """The source's own log p(x_t | all other tokens) (or | the tokens before t, if causal). In a first-order chain
    over classes only the classes at t - 1 and t + 1 matter: p(c_t | rest) is proportional to
    T[c_(t-1), c_t] T[c_t, c_(t+1)], and the word adds -log(per)."""
    K = trans.shape[1]
    cls = (ids - FIRST) // per
    n, L = ids.shape
    out = np.empty((n, L))
    for t in range(L):
        prev = trans[cls[:, t - 1]] if t > 0 else np.broadcast_to(trans[K], (n, K))
        score = prev.copy()
        if t + 1 < L and not causal:
            score = score * trans[:K, cls[:, t + 1]].T
        p = score[np.arange(n), cls[:, t]] / score.sum(1)
        out[:, t] = np.log(p) - np.log(per)
    return out


def independence_gap(joint: np.ndarray) -> tuple[float, float, float]:
    """For two masked tokens with joint distribution `joint` (given the visible context), the best possible masked LM
    loss H(A) + H(B), the joint entropy H(A, B), and their difference, the mutual information I(A; B), in nats."""
    pa, pb = joint.sum(1), joint.sum(0)

    def ent(q: np.ndarray) -> float:
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    mlm, h = ent(pa) + ent(pb), ent(joint.ravel())
    return mlm, h, mlm - h
