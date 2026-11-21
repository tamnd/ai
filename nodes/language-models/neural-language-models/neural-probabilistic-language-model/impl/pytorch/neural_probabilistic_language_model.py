# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-probabilistic-language-model/ (ai://NEW-neural-probabilistic-language-model).
"""The neural probabilistic language model of Bengio et al. (2003) in PyTorch, with autograd for the gradients.

y = b + W x + U tanh(d + H x), where x = (C(w[t-1]), ..., C(w[t-n+1])) concatenates the feature vectors of the
context words newest first, followed by a softmax over the vocabulary. W is optional, and tie=True uses C as U
(Press and Wolf, 2017). The loss is the mean negative log-likelihood plus decay / 2 times the squared norm of C, H, U
and W, and SGD uses eps_t = eps_0 / (1 + r t). nce_loss is noise-contrastive estimation with the normalising constant
fixed to 1 (Mnih and Teh, 2012), and HierarchicalSoftmax is a Huffman-tree softmax (Morin and Bengio, 2005)."""

import heapq
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

WEIGHTS = ("C", "H", "U", "W")


class NPLM(nn.Module):
    def __init__(self, V: int, m: int, h: int, n: int, direct: bool = True, tie: bool = False, seed: int = 0):
        super().__init__()
        if tie and h != m:
            raise ValueError("tying U to C needs h = m")
        self.V, self.m, self.h, self.n, self.direct, self.tie = V, m, h, n, direct, tie
        g = torch.Generator().manual_seed(seed)
        k = (n - 1) * m
        dt = torch.get_default_dtype()
        self.C = nn.Parameter(0.1 * torch.randn(V, m, generator=g, dtype=dt))
        self.H = nn.Parameter((2 * torch.rand(h, k, generator=g, dtype=dt) - 1) / k**0.5)
        self.d = nn.Parameter(torch.zeros(h, dtype=dt))
        self.b = nn.Parameter(torch.zeros(V, dtype=dt))
        self.U = None if tie else nn.Parameter((2 * torch.rand(V, h, generator=g, dtype=dt) - 1) / h**0.5)
        self.W = nn.Parameter((2 * torch.rand(V, k, generator=g, dtype=dt) - 1) / k**0.5) if direct else None
        self.updates = 0

    def load(self, arrays: dict) -> "NPLM":
        """Copy the named arrays (C, H, d, b and, when present, U and W) into the parameters."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.copy_(torch.as_tensor(arrays[name], dtype=p.dtype))
        return self

    def out(self) -> torch.Tensor:
        return self.C if self.U is None else self.U

    def hidden(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.C[ctx.flip(1)].reshape(len(ctx), -1)
        return x, torch.tanh(self.d + x @ self.H.T)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        x, a = self.hidden(ctx)
        y = self.b + a @ self.out().T
        if self.W is not None:
            y = y + x @ self.W.T
        return y

    def penalty(self, decay: float) -> torch.Tensor:
        weights = [p for name, p in self.named_parameters() if name in WEIGHTS]
        return 0.5 * decay * torch.stack([(p**2).sum() for p in weights]).sum()

    def loss(self, ctx: torch.Tensor, tgt: torch.Tensor, decay: float = 0.0) -> torch.Tensor:
        return F.cross_entropy(self(ctx), tgt) + self.penalty(decay)

    def token_logprobs(self, ctx: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self(ctx), dim=-1).gather(1, tgt[:, None])[:, 0]

    def nce_loss(
        self, ctx: torch.Tensor, tgt: torch.Tensor, noise: torch.Tensor, log_pn: torch.Tensor, decay: float = 0.0
    ) -> torch.Tensor:
        """Noise-contrastive estimation with the normalising constant fixed to 1, scoring only the target and the k
        noise words of each example."""
        k = noise.shape[1]
        x, a = self.hidden(ctx)
        rows = torch.cat([tgt[:, None], noise], dim=1)
        y = self.b[rows] + torch.einsum("bh,bjh->bj", a, self.out()[rows])
        if self.W is not None:
            y = y + torch.einsum("bx,bjx->bj", x, self.W[rows])
        delta = y - torch.log(torch.tensor(float(k), dtype=y.dtype)) - log_pn[rows]
        loss = -(F.logsigmoid(delta[:, 0]) + F.logsigmoid(-delta[:, 1:]).sum(dim=1)).mean()
        return loss + self.penalty(decay)

    def step(self, loss: torch.Tensor, lr0: float, r: float = 0.0):
        lr = lr0 / (1.0 + r * self.updates)
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                assert p.grad is not None
                p -= lr * p.grad
        self.updates += 1

    def fit(
        self,
        ctx: torch.Tensor,
        tgt: torch.Tensor,
        orders: Sequence[torch.Tensor],
        lr0: float,
        r: float = 0.0,
        decay: float = 0.0,
        batch: int = 1,
    ) -> list[float]:
        """Minibatch SGD, one epoch per given permutation; returns the mean training loss of each epoch."""
        history = []
        for order in orders:
            total = 0.0
            for s in range(0, len(order), batch):
                i = order[s : s + batch]
                loss = self.loss(ctx[i], tgt[i], decay)
                total += loss.item() * len(i)
                self.step(loss, lr0, r)
            history.append(total / len(order))
        return history


def huffman(counts: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The padded (V, L) path, branch and mask tensors of the Huffman tree of the counts; ties break by creation
    order."""
    V = len(counts)
    if V < 2:
        raise ValueError("a tree needs at least two words")
    heap = [(float(c), i, i) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    parent, branch = [0] * (2 * V - 1), [0] * (2 * V - 1)
    nxt = V
    while len(heap) > 1:
        c0, _, n0 = heapq.heappop(heap)
        c1, _, n1 = heapq.heappop(heap)
        parent[n0], parent[n1], branch[n0], branch[n1] = nxt, nxt, 0, 1
        heapq.heappush(heap, (c0 + c1, nxt, nxt))
        nxt += 1
    root = 2 * V - 2
    paths, codes = [], []
    for w in range(V):
        node, p, c = w, [], []
        while node != root:
            p.append(parent[node] - V)
            c.append(branch[node])
            node = parent[node]
        paths.append(p[::-1])
        codes.append(c[::-1])
    L = max(len(p) for p in paths)
    path = torch.zeros(V, L, dtype=torch.long)
    code = torch.zeros(V, L, dtype=torch.long)
    mask = torch.zeros(V, L)
    for w in range(V):
        k = len(paths[w])
        path[w, :k] = torch.tensor(paths[w], dtype=torch.long)
        code[w, :k] = torch.tensor(codes[w], dtype=torch.long)
        mask[w, :k] = 1.0
    return path, code, mask


class HierarchicalSoftmax(nn.Module):
    """p(w | a) is the product over the nodes j on the path of w of sigmoid(s (v_j . a + c_j)), s = +1 on branch 0
    and -1 on branch 1."""

    def __init__(self, counts: Sequence[float], h: int, seed: int = 0):
        super().__init__()
        path, code, mask = huffman(counts)
        self.path: torch.Tensor
        self.sign: torch.Tensor
        self.mask: torch.Tensor
        self.register_buffer("path", path)
        self.register_buffer("sign", 1.0 - 2.0 * code.to(torch.get_default_dtype()))
        self.register_buffer("mask", mask.to(torch.get_default_dtype()))
        g = torch.Generator().manual_seed(seed)
        self.v = nn.Parameter(0.1 * torch.randn(len(counts) - 1, h, generator=g))
        self.c = nn.Parameter(0.1 * torch.randn(len(counts) - 1, generator=g))

    def log_prob(self, a: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        nodes = self.path[tgt]
        z = torch.einsum("bh,blh->bl", a, self.v[nodes]) + self.c[nodes]
        return (self.mask[tgt] * F.logsigmoid(self.sign[tgt] * z)).sum(dim=1)

    def all_log_probs(self, a: torch.Tensor) -> torch.Tensor:
        z = a @ self.v.T + self.c
        return (self.mask[None] * F.logsigmoid(z[:, self.path] * self.sign[None])).sum(dim=2)
