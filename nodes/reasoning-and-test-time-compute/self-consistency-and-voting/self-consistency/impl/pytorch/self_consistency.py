# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/self-consistency/ (ai://NEW-self-consistency).
"""Self-consistency over a small autoregressive model, in PyTorch.

A language model samples m reasoning paths in parallel with temperature and top-k truncation; a parser reads an
answer off each path; the vote returns the most common answer. The file has:

* TinyLM, a small causal model (embedding, GRU, output layer) whose outputs are a stand-in for a real model's;
* sample_paths, batched ancestral sampling with temperature and top-k, returning the tokens and their log-probabilities;
* greedy_path, the single path that chain-of-thought prompting with greedy decoding would use;
* self_consistency, with plain, length-normalized and unnormalized weighting;
* exact_answer_marginal, which enumerates every path to get P(a) = sum_r P(r, a) for tests;
* adaptive_self_consistency, which samples in rounds and stops with the Beta rule.
"""

import itertools
import math

import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab=5, d=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.emb = nn.Embedding(vocab + 1, d)  # the extra id is the prompt token
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.8)

    @property
    def bos(self):
        return self.vocab

    def logits(self, tokens):
        """Next-token logits after every prefix of `tokens` (batch, length)."""
        h, _ = self.rnn(self.emb(tokens))
        return self.out(h)


def _filter(logits, temperature, top_k):
    if temperature <= 0:
        raise ValueError("temperature must be positive; use greedy_path for T = 0")
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def sample_paths(model, length, m, temperature=1.0, top_k=None, generator=None):
    """m paths of `length` tokens. Returns tokens (m, length) and the model's token log-probs at T = 1 (m, length)."""
    seq = torch.full((m, 1), model.bos, dtype=torch.long)
    lps = []
    for _ in range(length):
        raw = model.logits(seq)[:, -1]
        logp = _filter(raw, temperature, top_k)
        nxt = torch.multinomial(logp.exp(), 1, generator=generator)
        lps.append(torch.log_softmax(raw, -1).gather(1, nxt))
        seq = torch.cat([seq, nxt], 1)
    return seq[:, 1:], torch.cat(lps, 1)


@torch.no_grad()
def greedy_path(model, length):
    seq = torch.full((1, 1), model.bos, dtype=torch.long)
    for _ in range(length):
        nxt = model.logits(seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
    return seq[0, 1:]


@torch.no_grad()
def sequence_logprob(model, tokens):
    """log P(tokens) under the model at T = 1, the sum of the token log-probabilities, for a batch of paths."""
    seq = torch.cat([torch.full((tokens.shape[0], 1), model.bos, dtype=torch.long), tokens], 1)
    logp = torch.log_softmax(model.logits(seq[:, :-1]), -1)
    return logp.gather(2, tokens.unsqueeze(-1)).squeeze(-1).sum(1)


def vote(answers, weights=None):
    """The answer with the largest total weight (count when weights is None) and the totals as a dict."""
    answers = torch.as_tensor(answers)
    w = (
        torch.ones(len(answers), dtype=torch.float64)
        if weights is None
        else torch.as_tensor(weights, dtype=torch.float64)
    )
    keys, inv = torch.unique(answers, return_inverse=True)
    totals = torch.zeros(len(keys), dtype=torch.float64).index_add_(0, inv, w)
    best = int(keys[totals.argmax()])
    return best, {int(k): float(t) for k, t in zip(keys, totals)}


def self_consistency(model, parse, length, m, temperature=0.7, top_k=None, weighting="none", generator=None):
    """Sample m paths, parse an answer from each, and vote. weighting is "none", "normalized" or "unnormalized"."""
    tokens, lps = sample_paths(model, length, m, temperature, top_k, generator)
    answers = [parse(t) for t in tokens]
    if weighting == "none":
        w = None
    elif weighting == "normalized":
        w = lps.mean(1).exp()
    elif weighting == "unnormalized":
        w = lps.sum(1).exp()
    else:
        raise ValueError(weighting)
    best, totals = vote(answers, w)
    return best, totals, tokens


@torch.no_grad()
def exact_answer_marginal(model, parse, length, temperature=1.0, top_k=None):
    """P(a) under the sampling distribution, summing over every one of vocab^length paths."""
    paths = torch.tensor(list(itertools.product(range(model.vocab), repeat=length)), dtype=torch.long)
    seq = torch.cat([torch.full((len(paths), 1), model.bos, dtype=torch.long), paths], 1)
    logp = _filter(model.logits(seq[:, :-1]), temperature, top_k)
    lp = logp.gather(2, paths.unsqueeze(-1)).squeeze(-1).sum(1)
    marg = {}
    for t, l in zip(paths, lp.exp().tolist()):
        a = parse(t)
        marg[a] = marg.get(a, 0.0) + l
    return marg


def beta_confidence(v1, v2):
    """P(q > 1/2) for q ~ Beta(v1 + 1, v2 + 1)."""
    n = v1 + v2 + 1
    return 1.0 - sum(math.comb(n, j) for j in range(v1 + 1, n + 1)) / 2**n


def adaptive_self_consistency(model, parse, length, max_samples=40, threshold=0.95, temperature=0.7, generator=None):
    """Sample one path at a time and stop when the Beta rule is confident in the leader."""
    answers = []
    while len(answers) < max_samples:
        tokens, _ = sample_paths(model, length, 1, temperature, generator=generator)
        answers.append(parse(tokens[0]))
        _, totals = vote(answers)
        top = sorted(totals.values(), reverse=True) + [0.0]
        if beta_confidence(int(top[0]), int(top[1])) > threshold:
            break
    return vote(answers)[0], len(answers)
