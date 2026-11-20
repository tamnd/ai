# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-relative-policy-optimization/ (ai://NEW-group-relative-policy-optimization).
"""Group relative policy optimization (GRPO) with a verifiable reward, in PyTorch.

A small recurrent policy reads a question token and writes up to `max_len` tokens, ending with EOS. The verifier
gives reward 1 if the output ends with EOS and the token before EOS is the question's answer; an unfinished
(overlong) output gets 0, or is dropped from the loss with `overlong_filter`. `grpo_loss` implements the clipped
objective with separate lower and upper ranges, the k3 KL penalty against a frozen reference, and the three loss
reductions: GRPO's per-sequence mean, DAPO's per-token mean and Dr. GRPO's constant normalizer.
"""

import copy

import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, n_questions, vocab, hidden=64):
        super().__init__()
        self.vocab, self.bos = vocab, vocab  # tokens 0..vocab-1, with vocab-1 as EOS; index vocab is BOS
        self.q_emb = nn.Embedding(n_questions, hidden)
        self.tok_emb = nn.Embedding(vocab + 1, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, q, inputs, h=None):
        h = self.q_emb(q)[None] if h is None else h
        out, h = self.gru(self.tok_emb(inputs), h)
        return self.head(out), h


@torch.no_grad()
def generate(policy, q, G, max_len, generator=None):
    """G samples per question. Returns repeated questions, tokens (N, max_len), mask (N, max_len), finished (N,)."""
    eos = policy.vocab - 1
    qq = q.repeat_interleave(G)
    n = len(qq)
    tok = torch.full((n, 1), policy.bos)
    h, out = None, []
    done = torch.zeros(n, dtype=torch.bool)
    mask = torch.zeros(n, max_len)
    for t in range(max_len):
        logits, h = policy(qq, tok, h)
        nxt = torch.multinomial(logits[:, -1].softmax(-1), 1, generator=generator)
        mask[:, t] = (~done).float()
        out.append(nxt)
        done = done | (nxt[:, 0] == eos)
        tok = nxt
    return qq, torch.cat(out, 1), mask, done


def verify(tokens, mask, finished, answers, qq):
    """1 if the output ended with EOS and the token before EOS equals the answer."""
    length = mask.sum(1).long()
    before = tokens.gather(1, (length - 2).clamp(min=0)[:, None])[:, 0]
    return (finished & (length >= 2) & (before == answers[qq])).float()


def token_logprobs(policy, qq, tokens):
    inputs = torch.cat([torch.full((len(qq), 1), policy.bos), tokens[:, :-1]], 1)
    logits, _ = policy(qq, inputs)
    return logits.log_softmax(-1).gather(2, tokens[..., None])[..., 0]


def group_advantages(rewards, G, normalize=True, eps=1e-6):
    r = rewards.view(-1, G)
    a = r - r.mean(1, keepdim=True)
    if normalize:
        a = a / (r.std(1, unbiased=False, keepdim=True) + eps)
    return a.reshape(-1)


def grpo_loss(logp, old_logp, ref_logp, adv, mask, eps_low=0.2, eps_high=0.2, beta=0.0, mode="sequence", max_len=None):
    """Negative GRPO objective. logp etc. are (N, T) per-token log-probabilities; adv is (N,)."""
    ratio = (logp - old_logp).exp()
    A = adv[:, None]
    surr = torch.minimum(ratio * A, ratio.clamp(1 - eps_low, 1 + eps_high) * A)
    d = ref_logp - logp
    per_token = -(surr - beta * (d.exp() - d - 1)) * mask
    if mode == "sequence":
        return (per_token.sum(1) / mask.sum(1).clamp(min=1)).mean()
    if mode == "token":
        return per_token.sum() / mask.sum().clamp(min=1)
    if mode == "constant":
        return per_token.sum() / (len(adv) * max_len)
    raise ValueError(mode)


def train(
    policy,
    answers,
    steps=120,
    G=8,
    batch=8,
    max_len=4,
    mu=2,
    lr=3e-3,
    beta=0.01,
    mode="sequence",
    eps_low=0.2,
    eps_high=0.28,
    overlong_filter=False,
    seed=0,
):
    """GRPO with a frozen reference. Returns the mean reward of every step."""
    g = torch.Generator().manual_seed(seed)
    ref = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        q = torch.randperm(len(answers), generator=g)[:batch]
        qq, tokens, mask, finished = generate(policy, q, G, max_len, g)
        r = verify(tokens, mask, finished, answers, qq)
        adv = group_advantages(r, G)
        if overlong_filter:
            mask = mask * finished[:, None]
        with torch.no_grad():
            old_logp = token_logprobs(policy, qq, tokens)
            ref_logp = token_logprobs(ref, qq, tokens)
        for _ in range(mu):
            loss = grpo_loss(
                token_logprobs(policy, qq, tokens),
                old_logp,
                ref_logp,
                adv,
                mask,
                eps_low,
                eps_high,
                beta,
                mode,
                max_len,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append(r.mean().item())
    return history


@torch.no_grad()
def success_rate(policy, answers, G=64, max_len=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.arange(len(answers))
    qq, tokens, mask, finished = generate(policy, q, G, max_len, g)
    return verify(tokens, mask, finished, answers, qq).mean().item()
