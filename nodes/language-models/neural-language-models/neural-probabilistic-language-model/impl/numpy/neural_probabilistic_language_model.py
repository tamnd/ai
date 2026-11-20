# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""The neural probabilistic language model of Bengio et al. (2003) in NumPy, with the backward pass written out.

The model maps the n - 1 previous words to their feature vectors, rows of a |V| x m table C, concatenates them newest
first into x = (C(w[t-1]), ..., C(w[t-n+1])), and computes the unnormalised log-probabilities

    y = b + W x + U tanh(d + H x)

of every word, followed by a softmax. W (the direct connections) is optional. With tie=True the output matrix U is the
input table C itself (Press and Wolf, 2017), which needs h = m.

Training minimises the mean negative log-likelihood plus decay / 2 times the squared norm of C, H, U and W (not of the
biases b and d), by stochastic gradient descent with the learning rate eps_t = eps_0 / (1 + r t) after t updates.

Two cheaper output layers are included: noise-contrastive estimation (Mnih and Teh, 2012), which scores only the target
and k noise words with the normalising constant fixed to 1, and a hierarchical softmax over a Huffman tree (Morin and
Bengio, 2005; Mikolov et al., 2013), which needs one sigmoid per node on the path of a word."""

import heapq
from typing import Any

import numpy as np

Params = dict[str, np.ndarray]
WEIGHTS = ("C", "H", "U", "W")  # the parameters under weight decay; the biases b and d are not


def n_params(V: int, m: int, h: int, n: int, direct: bool = True, tie: bool = False) -> int:
    """|V|(1 + nm + h) + h(1 + (n - 1)m) with direct connections, as counted by Bengio et al.; W adds |V|(n - 1)m and
    a tied U removes |V|h."""
    total = V * (1 + n * m + h) + h * (1 + (n - 1) * m)
    if not direct:
        total -= V * (n - 1) * m
    if tie:
        total -= V * h
    return total


def output_fraction(V: int, m: int, h: int, n: int) -> float:
    """Bengio et al.'s estimate of the share of the multiply-adds of one example spent on the output units."""
    out = V * (1 + (n - 1) * m + h)
    return out / (out + h * (1 + (n - 1) * m) + (n - 1) * m)


def windows(ids: np.ndarray, n: int, bos: int) -> tuple[np.ndarray, np.ndarray]:
    """The (T, n - 1) contexts, oldest word first, and the T targets of a token stream padded in front with bos."""
    padded = np.concatenate([np.full(n - 1, bos, dtype=np.int64), np.asarray(ids, dtype=np.int64)])
    idx = np.arange(len(ids))[:, None] + np.arange(n - 1)[None, :]
    return padded[idx], padded[n - 1 :]


def log_softmax(y: np.ndarray) -> np.ndarray:
    z = y - y.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(log_sigmoid(z))


def perplexity(logp: np.ndarray) -> float:
    """exp of the mean negative log-likelihood of the predicted tokens."""
    return float(np.exp(-np.mean(logp)))


def mix(logp_a: np.ndarray, logp_b: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """log(weight p_a + (1 - weight) p_b), the linear mixture used with the interpolated trigram."""
    return np.logaddexp(np.log(weight) + logp_a, np.log1p(-weight) + logp_b)


class NPLM:
    def __init__(
        self, V: int, m: int, h: int, n: int, direct: bool = True, tie: bool = False, seed: int = 0, scale: float = 0.1
    ):
        if tie and h != m:
            raise ValueError("tying U to C needs h = m")
        self.V, self.m, self.h, self.n, self.direct, self.tie = V, m, h, n, direct, tie
        rng = np.random.default_rng(seed)
        k = (n - 1) * m
        self.p: Params = {
            "C": rng.normal(0.0, scale, (V, m)),
            "H": rng.uniform(-1, 1, (h, k)) / np.sqrt(k),
            "d": np.zeros(h),
            "b": np.zeros(V),
        }
        if not tie:
            self.p["U"] = rng.uniform(-1, 1, (V, h)) / np.sqrt(h)
        if direct:
            self.p["W"] = rng.uniform(-1, 1, (V, k)) / np.sqrt(k)
        self.updates = 0

    @property
    def U(self) -> np.ndarray:
        return self.p["C"] if self.tie else self.p["U"]

    def size(self) -> int:
        return sum(v.size for v in self.p.values())

    def hidden(self, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The input words newest first, the features x and the hidden activations a = tanh(d + H x)."""
        rev = np.asarray(ctx, dtype=np.int64)[:, ::-1]
        x = self.p["C"][rev].reshape(len(rev), -1)
        a = np.tanh(self.p["d"] + x @ self.p["H"].T)
        return rev, x, a

    def logits(self, ctx: np.ndarray) -> np.ndarray:
        _, x, a = self.hidden(ctx)
        y = self.p["b"] + a @ self.U.T
        if self.direct:
            y = y + x @ self.p["W"].T
        return y

    def log_probs(self, ctx: np.ndarray) -> np.ndarray:
        return log_softmax(self.logits(ctx))

    def token_logprobs(self, ctx: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        lp = self.log_probs(ctx)
        return lp[np.arange(len(tgt)), tgt]

    def penalty(self, decay: float) -> float:
        return 0.5 * decay * sum(float((self.p[k] ** 2).sum()) for k in WEIGHTS if k in self.p)

    def _backward(self, rev: np.ndarray, x: np.ndarray, a: np.ndarray, g_a: np.ndarray, g_x: np.ndarray, g: Params):
        """Back-propagate the gradients at a and x through the tanh layer and the table look-up into g."""
        g_o = g_a * (1.0 - a**2)
        g["d"] = g_o.sum(axis=0)
        g["H"] = g_o.T @ x
        g_x = g_x + g_o @ self.p["H"]
        np.add.at(g["C"], rev.reshape(-1), g_x.reshape(-1, self.m))

    def _decay(self, g: Params, decay: float):
        for k in WEIGHTS:
            if k in self.p:
                g[k] += decay * self.p[k]

    def loss_and_grads(self, ctx: np.ndarray, tgt: np.ndarray, decay: float = 0.0) -> tuple[float, Params]:
        """The mean negative log-likelihood plus the weight-decay penalty, and its gradient for every parameter."""
        B = len(tgt)
        rev, x, a = self.hidden(ctx)
        y = self.p["b"] + a @ self.U.T
        if self.direct:
            y = y + x @ self.p["W"].T
        lp = log_softmax(y)
        loss = -float(lp[np.arange(B), tgt].mean()) + self.penalty(decay)
        g_y = np.exp(lp)
        g_y[np.arange(B), tgt] -= 1.0  # the derivative of -log p_w is p_j - 1{j = w}
        g_y /= B
        g: Params = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["b"] = g_y.sum(axis=0)
        g_out = g_y.T @ a
        if self.tie:
            g["C"] += g_out
        else:
            g["U"] = g_out
        g_a = g_y @ self.U
        g_x = np.zeros_like(x)
        if self.direct:
            g["W"] = g_y.T @ x
            g_x = g_y @ self.p["W"]
        self._backward(rev, x, a, g_a, g_x, g)
        self._decay(g, decay)
        return loss, g

    def nce_loss_and_grads(
        self, ctx: np.ndarray, tgt: np.ndarray, noise: np.ndarray, log_pn: np.ndarray, decay: float = 0.0
    ) -> tuple[float, Params]:
        """Noise-contrastive estimation with k = noise.shape[1] noise words per example drawn from the distribution
        exp(log_pn), and the normalising constant fixed to 1: the model's log-probability of word j is the logit y_j.
        Only the rows of U, b and W of the target and noise words enter the loss and get a gradient."""
        B, k = noise.shape
        rev, x, a = self.hidden(ctx)
        rows = np.concatenate([np.asarray(tgt)[:, None], noise], axis=1)  # (B, k + 1)
        U = self.U
        y = self.p["b"][rows] + np.einsum("bh,bjh->bj", a, U[rows])
        if self.direct:
            y = y + np.einsum("bx,bjx->bj", x, self.p["W"][rows])
        delta = y - np.log(k) - log_pn[rows]  # log of P_model / (k P_noise)
        label = np.zeros((B, k + 1))
        label[:, 0] = 1.0
        loss = -float((label * log_sigmoid(delta) + (1 - label) * log_sigmoid(-delta)).sum(axis=1).mean())
        loss += self.penalty(decay)
        g_y = (sigmoid(delta) - label) / B
        g: Params = {key: np.zeros_like(v) for key, v in self.p.items()}
        np.add.at(g["b"], rows, g_y)
        g_out = np.zeros_like(U)
        np.add.at(g_out, rows, g_y[:, :, None] * a[:, None, :])
        if self.tie:
            g["C"] += g_out
        else:
            g["U"] = g_out
        g_a = np.einsum("bj,bjh->bh", g_y, U[rows])
        g_x = np.zeros_like(x)
        if self.direct:
            np.add.at(g["W"], rows, g_y[:, :, None] * x[:, None, :])
            g_x = np.einsum("bj,bjx->bx", g_y, self.p["W"][rows])
        self._backward(rev, x, a, g_a, g_x, g)
        self._decay(g, decay)
        return loss, g

    def learning_rate(self, lr0: float, r: float) -> float:
        return lr0 / (1.0 + r * self.updates)

    def step(self, g: Params, lr0: float, r: float = 0.0):
        lr = self.learning_rate(lr0, r)
        for key, v in g.items():
            self.p[key] -= lr * v
        self.updates += 1

    def train(
        self,
        ctx: np.ndarray,
        tgt: np.ndarray,
        epochs: int,
        lr0: float,
        r: float = 0.0,
        decay: float = 0.0,
        batch: int = 1,
        rng: Any = None,
        orders: Any = None,
    ) -> list[float]:
        """Minibatch SGD over the examples in a fresh random order each epoch, or in the given orders (one
        permutation per epoch); returns the mean training loss of each epoch."""
        rng = np.random.default_rng(0) if rng is None else rng
        history = []
        for e in range(epochs):
            order = rng.permutation(len(tgt)) if orders is None else np.asarray(orders[e])
            total = 0.0
            for s in range(0, len(order), batch):
                i = order[s : s + batch]
                loss, g = self.loss_and_grads(ctx[i], tgt[i], decay)
                self.step(g, lr0, r)
                total += loss * len(i)
            history.append(total / len(order))
        return history


def huffman(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Huffman tree of the counts as padded (V, L) arrays: the internal nodes on the path from the root to each
    word, the branch taken at each (0 or 1) and a mask of the real steps. Ties break by creation order, so equal
    counts give a tree whose depths differ by at most one."""
    V = len(counts)
    if V < 2:
        raise ValueError("a tree needs at least two words")
    heap = [(float(c), i, i) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    parent = np.zeros(2 * V - 1, dtype=np.int64)
    branch = np.zeros(2 * V - 1, dtype=np.int64)
    nxt = V
    while len(heap) > 1:
        c0, _, n0 = heapq.heappop(heap)
        c1, _, n1 = heapq.heappop(heap)
        parent[n0], parent[n1], branch[n0], branch[n1] = nxt, nxt, 0, 1
        heapq.heappush(heap, (c0 + c1, nxt, nxt))
        nxt += 1
    root = 2 * V - 2
    paths: list[list[int]] = []
    codes: list[list[int]] = []
    for w in range(V):
        node, p, c = w, [], []
        while node != root:
            p.append(int(parent[node] - V))
            c.append(int(branch[node]))
            node = parent[node]
        paths.append(p[::-1])
        codes.append(c[::-1])
    L = max(len(p) for p in paths)
    path = np.zeros((V, L), dtype=np.int64)
    code = np.zeros((V, L), dtype=np.int64)
    mask = np.zeros((V, L))
    for w in range(V):
        k = len(paths[w])
        path[w, :k], code[w, :k], mask[w, :k] = paths[w], codes[w], 1.0
    return path, code, mask


class HierarchicalSoftmax:
    """p(w | a) = prod over the nodes j on the path of w of sigmoid(s (v_j . a + c_j)), with s = +1 on branch 0 and
    -1 on branch 1, so the two children of a node share its probability and every distribution sums to one."""

    def __init__(self, counts: np.ndarray, h: int, seed: int = 0, scale: float = 0.1):
        self.path, self.code, self.mask = huffman(np.asarray(counts, dtype=np.float64))
        rng = np.random.default_rng(seed)
        nodes = len(counts) - 1
        self.v = rng.normal(0.0, scale, (nodes, h))
        self.c = rng.normal(0.0, scale, nodes)

    def depth(self) -> np.ndarray:
        return self.mask.sum(axis=1).astype(np.int64)

    def _scores(self, a: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes, sign, mask = self.path[tgt], 1.0 - 2.0 * self.code[tgt], self.mask[tgt]
        z = np.einsum("bh,blh->bl", a, self.v[nodes]) + self.c[nodes]
        return nodes, sign, mask, z

    def log_prob(self, a: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        _, sign, mask, z = self._scores(a, tgt)
        return (mask * log_sigmoid(sign * z)).sum(axis=1)

    def all_log_probs(self, a: np.ndarray) -> np.ndarray:
        """The log-probability of every word, (B, V): one score per internal node, shared by all words below it."""
        z = a @ self.v.T + self.c
        zs = z[:, self.path] * (1.0 - 2.0 * self.code)[None]
        return (self.mask[None] * log_sigmoid(zs)).sum(axis=2)

    def loss_and_grads(self, a: np.ndarray, tgt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """The mean negative log-likelihood and its gradient at a, at the node vectors v and at the node biases c."""
        B = len(tgt)
        nodes, sign, mask, z = self._scores(a, tgt)
        loss = -float((mask * log_sigmoid(sign * z)).sum(axis=1).mean())
        g_z = -mask * sign * sigmoid(-sign * z) / B
        g_a = np.einsum("bl,blh->bh", g_z, self.v[nodes])
        g_v = np.zeros_like(self.v)
        np.add.at(g_v, nodes, g_z[:, :, None] * a[:, None, :])
        g_c = np.zeros_like(self.c)
        np.add.at(g_c, nodes, g_z)
        return loss, g_a, g_v, g_c
