# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Masked language modelling in PyTorch: BERT's 80/10/10 corruption, T5's span corruption, the bidirectional window
model of the NumPy reference with its masked LM and discriminator heads, and ELECTRA's two losses, with autograd.

Token ids 0 and 1 are [PAD] and [MASK]; ordinary tokens are 2 to V - 1. The random draws (chosen positions, uniform
numbers, random tokens, noise masks) are inputs, so the same draws give the same result as the NumPy reference."""

import torch
from torch import nn

PAD, MASK = 0, 1
FIRST = 2


def bert_corrupt(
    ids: torch.Tensor,
    selected: torch.Tensor,
    u: torch.Tensor,
    random_ids: torch.Tensor,
    p_mask: float = 0.8,
    p_random: float = 0.1,
) -> torch.Tensor:
    """At each selected position: [MASK] if u < p_mask, the random token if u < p_mask + p_random, else unchanged."""
    out = torch.where(selected & (u < p_mask), torch.full_like(ids, MASK), ids)
    to_random = selected & (u >= p_mask) & (u < p_mask + p_random)
    return torch.where(to_random, random_ids, out)


def span_corrupt(ids: torch.Tensor, noise: torch.Tensor, sentinels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """T5's input and target for one sequence: each run of noise positions becomes one sentinel in the input; the
    target is every run after its sentinel, then one more sentinel."""
    starts = noise & ~torch.cat([torch.zeros(1, dtype=torch.bool), noise[:-1]])
    run = torch.cumsum(starts.long(), 0) - 1  # the index of the run each position belongs to
    sent = sentinels[run.clamp(min=0)]
    keep_inp = ~noise | starts
    inp = torch.where(noise, sent, ids)[keep_inp]
    # each noise position contributes (its run's sentinel, if it starts the run) and then its own token
    pairs = torch.stack([sent, ids], 1)[noise]
    keep = torch.stack([starts, torch.ones_like(noise)], 1)[noise]
    tgt = pairs.reshape(-1)[keep.reshape(-1)]
    n_runs = int(starts.sum())
    return inp, torch.cat([tgt, sentinels[n_runs : n_runs + 1]])


class WindowMLM(nn.Module):
    """z = tanh(d + H x) over the concatenated embeddings of the tokens at `offsets`; logits = b + U z for the masked
    LM; D = sigmoid(c + g . z) for the discriminator."""

    def __init__(self, V: int, m: int, h: int, offsets: tuple[int, ...]):
        super().__init__()
        self.V, self.m, self.h, self.offsets = V, m, h, tuple(offsets)
        k = len(self.offsets) * m
        self.E = nn.Parameter(torch.zeros(V, m))
        self.H = nn.Parameter(torch.zeros(h, k))
        self.d = nn.Parameter(torch.zeros(h))
        self.U = nn.Parameter(torch.zeros(V, h))
        self.b = nn.Parameter(torch.zeros(V))
        self.g = nn.Parameter(torch.zeros(h))
        self.c = nn.Parameter(torch.zeros(1))

    def load(self, arrays: dict[str, torch.Tensor]) -> "WindowMLM":
        with torch.no_grad():
            for name, p in self.named_parameters():
                p.copy_(arrays[name])
        return self

    def windows(self, inp: torch.Tensor) -> torch.Tensor:
        w = max(abs(o) for o in self.offsets)
        B, L = inp.shape
        padded = torch.full((B, L + 2 * w), PAD, dtype=torch.long)
        padded[:, w : w + L] = inp
        idx = torch.arange(L)[:, None] + w + torch.tensor(self.offsets)[None, :]
        return padded[:, idx]

    def hidden(self, win: torch.Tensor) -> torch.Tensor:
        x = self.E[win].reshape(*win.shape[:-1], -1)
        return torch.tanh(self.d + x @ self.H.T)

    def logits(self, inp: torch.Tensor) -> torch.Tensor:
        return self.b + self.hidden(self.windows(inp)) @ self.U.T

    def mlm_loss(self, inp: torch.Tensor, targets: torch.Tensor, predict: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy of the original tokens at the positions in `predict`."""
        z = self.hidden(self.windows(inp)[predict])
        return nn.functional.cross_entropy(self.b + z @ self.U.T, targets[predict])

    def disc_loss(self, inp: torch.Tensor, original: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean binary cross-entropy of D = P(original) over the valid positions."""
        z = self.hidden(self.windows(inp)[valid])
        return nn.functional.binary_cross_entropy_with_logits(self.c + z @ self.g, original[valid].double())

    def cloze_logprobs(self, ids: torch.Tensor) -> torch.Tensor:
        """log p(x_t | the other tokens) with only position t masked, for every position."""
        win = self.windows(ids).clone()
        if 0 in self.offsets:
            win[..., self.offsets.index(0)] = MASK
        lg = self.b + self.hidden(win) @ self.U.T
        return torch.log_softmax(lg, -1).gather(-1, ids[..., None])[..., 0]


def sample_tokens(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF samples from softmax(logits) over the ordinary tokens, one uniform number per row."""
    cdf = torch.cumsum(torch.softmax(logits[:, FIRST:], -1), -1)
    idx = (cdf < u[:, None]).sum(-1).clamp(max=cdf.shape[1] - 1)
    return FIRST + idx


def electra_losses(
    gen: WindowMLM, disc: WindowMLM, ids: torch.Tensor, selected: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The generator's masked LM loss, the discriminator loss and the corrupted input. The samples are drawn without
    gradient, so the discriminator loss does not reach the generator."""
    masked = torch.where(selected, torch.full_like(ids, MASK), ids)
    gen_loss = gen.mlm_loss(masked, ids, selected)
    with torch.no_grad():
        fake = sample_tokens(gen.logits(masked)[selected], u[selected])
    corrupt = ids.clone()
    corrupt[selected] = fake
    disc_loss = disc.disc_loss(corrupt, corrupt == ids, ids >= FIRST)
    return gen_loss, disc_loss, corrupt
