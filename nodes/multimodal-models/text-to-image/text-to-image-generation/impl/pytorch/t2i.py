# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/text-to-image-generation/ (ai://NEW-text-to-image-generation).
"""A small DALL-E-style text-to-image model: one causal Transformer over [text tokens, image tokens], trained with
next-token cross-entropy, text dropout for classifier-free guidance, guided sampling on logits, and reranking.

The "images" are already tokenized, as a dVAE or ViT-VQGAN would do: a scene is a grid of SLOTS cells and each cell
holds one colour token. A caption names every cell's colour with a position word, in a random order, for example
"blue at 2, red at 0, green at 1, ...". The model must bind each colour to its position.

Two ways to condition are compared. "sequence" puts the caption tokens in the Transformer's context, so image tokens
can attend to each word in order (DALL-E, Parti, and the cross-attention of Imagen). "pooled" replaces the caption by
the mean of its word embeddings, one vector, which keeps which words occur but not which colour went with which
position.
"""

import torch
import torch.nn.functional as F
from torch import nn


class World:
    """Captions and token images. Colour tokens are 0..colors-1; text and image vocabularies are separate."""

    def __init__(self, colors=6, slots=4, noise=0.0):
        self.colors, self.slots, self.noise = colors, slots, noise
        self.POS = colors  # position words are colors .. colors+slots-1
        self.PAD = colors + slots  # the null caption token used for guidance
        self.text_vocab = colors + slots + 1
        self.text_len = 2 * slots

    def sample(self, n, gen):
        img = torch.randint(self.colors, (n, self.slots), generator=gen)
        order = torch.argsort(torch.rand(n, self.slots, generator=gen), dim=1)
        text = torch.empty(n, self.text_len, dtype=torch.long)
        text[:, 0::2] = torch.gather(img, 1, order)
        text[:, 1::2] = self.POS + order
        if self.noise > 0:  # web alt-text: some captions do not describe their image
            bad = torch.rand(n, generator=gen) < self.noise
            img = torch.where(bad[:, None], torch.randint(self.colors, (n, self.slots), generator=gen), img)
        return text, img

    def target(self, text):
        """The image a caption describes."""
        img = torch.empty(text.shape[0], self.slots, dtype=torch.long)
        img.scatter_(1, text[:, 1::2] - self.POS, text[:, 0::2])
        return img

    def swap(self, text, i=0, j=1):
        """The caption with the colours of two positions exchanged, words in the same order."""
        t = text.clone()
        pi, pj = (t[:, 1::2] == self.POS + i), (t[:, 1::2] == self.POS + j)
        ci, cj = t[:, 0::2][pi].clone(), t[:, 0::2][pj].clone()
        colors = t[:, 0::2]
        colors[pi], colors[pj] = cj, ci
        t[:, 0::2] = colors
        return t


class TextToImage(nn.Module):
    """A decoder-only Transformer over the caption (or its pooled embedding) followed by the image tokens."""

    def __init__(self, world, cond="sequence", d=64, heads=4, layers=2):
        super().__init__()
        self.w, self.cond = world, cond
        self.txt = nn.Embedding(world.text_vocab, d)
        self.img = nn.Embedding(world.colors + 1, d)  # + a start-of-image token
        self.BOI = world.colors
        n_ctx = (world.text_len if cond == "sequence" else 1) + 1 + world.slots
        self.pos = nn.Parameter(0.02 * torch.randn(n_ctx, d))
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, world.colors)

    def text_context(self, text):
        e = self.txt(text)
        return e if self.cond == "sequence" else e.mean(dim=1, keepdim=True)

    def forward(self, text, img_prefix):
        """Logits for the next image token after each of [BOI, img_prefix...]."""
        n = text.shape[0]
        boi = torch.full((n, 1), self.BOI, dtype=torch.long)
        h = torch.cat([self.text_context(text), self.img(torch.cat([boi, img_prefix], 1))], 1)
        t = h.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
        z = self.norm(self.blocks(h + self.pos[:t], mask=mask, is_causal=True))
        k = img_prefix.shape[1] + 1
        return self.out(z[:, -k:])


def drop_text(text, p, pad, gen):
    """Replace whole captions by the null caption with probability p, so one model learns both p(x|y) and p(x)."""
    drop = torch.rand(text.shape[0], generator=gen) < p
    return torch.where(drop[:, None], torch.full_like(text, pad), text)


def train(model, steps=600, batch=128, lr=3e-3, p_drop=0.1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        text, img = model.w.sample(batch, gen)
        text = drop_text(text, p_drop, model.w.PAD, gen)
        logits = model(text, img[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), img.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def guided_logits(model, text, prefix, scale):
    """Classifier-free guidance on logits: l_u + scale * (l_c - l_u). scale = 1 is the conditional model."""
    lc = model(text, prefix)[:, -1]
    if scale == 1.0:
        return lc
    lu = model(torch.full_like(text, model.w.PAD), prefix)[:, -1]
    return lu + scale * (lc - lu)


@torch.no_grad()
def sample(model, text, scale=1.0, temperature=1.0, gen=None):
    prefix = torch.empty(text.shape[0], 0, dtype=torch.long)
    for _ in range(model.w.slots):
        logits = guided_logits(model, text, prefix, scale) / temperature
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        prefix = torch.cat([prefix, nxt], 1)
    return prefix


@torch.no_grad()
def log_likelihood(model, text, img, scale=1.0):
    """log p(img | text) under the (guided, renormalized) next-token distributions."""
    total = torch.zeros(text.shape[0])
    for k in range(img.shape[1]):
        lp = F.log_softmax(guided_logits(model, text, img[:, :k], scale), -1)
        total += lp.gather(1, img[:, k : k + 1]).squeeze(1)
    return total


@torch.no_grad()
def rerank_sample(model, text, n, scorer, gen=None, scale=1.0):
    """Draw n images per caption and keep the one the scorer likes best (DALL-E used 512 and CLIP; Parti 16 and CoCa)."""
    b = text.shape[0]
    cands = sample(model, text.repeat_interleave(n, 0), scale=scale, gen=gen).view(b, n, -1)
    scores = scorer(text.repeat_interleave(n, 0), cands.view(b * n, -1)).view(b, n)
    return cands[torch.arange(b), scores.argmax(1)]


def accuracy(world, text, img):
    """Fraction of images whose every cell matches the caption."""
    return (img == world.target(text)).all(1).float().mean().item()


def entropy_per_image(model, text, scale, n=64, gen=None):
    """Monte Carlo estimate of the entropy of the sampled image distribution, in nats."""
    imgs = sample(model, text.repeat_interleave(n, 0), scale=scale, gen=gen)
    return -log_likelihood(model, text.repeat_interleave(n, 0), imgs, scale).mean().item()
