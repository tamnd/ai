# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/prompt-sensitivity/ (ai://NEW-prompt-sensitivity).
"""A small transformer trained to classify in context, used to locate where order sensitivity comes from.

Each sequence draws a hidden direction w, k demonstrations (x, sign(w . x)) and a query x. Every demonstration is
one token and the query is the last token; the model reads the label of the query from its output. Attention without
positional information treats the demonstrations as a set, so the prediction cannot depend on their order. With
learned positional embeddings the same architecture can, and does, change some predictions when the demonstrations
are reordered. Contextual calibration subtracts the logit the model gives a content-free query, x = 0.
"""


import torch
from torch import nn


def make_batch(n, d=4, k=8, gen=None):
    w = torch.randn(n, d, generator=gen)
    x = torch.randn(n, k + 1, d, generator=gen)
    y = torch.sign((x @ w.unsqueeze(-1)).squeeze(-1))
    return x, y


def tokens(x, y):
    """Demonstration tokens carry (x, y, 0); the last token is the query (x, 0, 1)."""
    n, t, _ = x.shape
    lab = y.clone()
    lab[:, -1] = 0
    flag = torch.zeros(n, t)
    flag[:, -1] = 1
    return torch.cat([x, lab.unsqueeze(-1), flag.unsqueeze(-1)], -1)


class InContextClassifier(nn.Module):
    def __init__(self, d=4, k=8, width=64, layers=2, positional=True):
        super().__init__()
        self.embed = nn.Linear(d + 2, width)
        self.pos = nn.Parameter(torch.randn(k + 1, width) * 0.5) if positional else None
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, batch_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)

    def forward(self, tok):
        h = self.embed(tok)
        if self.pos is not None:
            h = h + self.pos[: tok.shape[1]]
        return self.head(self.body(h)[:, -1]).squeeze(-1)


def train(positional=True, steps=1500, batch=256, lr=2e-3, seed=0, d=4, k=8):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = InContextClassifier(d, k, positional=positional)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(batch, d, k, gen)
        loss = nn.functional.softplus(-y[:, -1] * model(tokens(x, y))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model, n=4000, seed=1, d=4, k=8):
    x, y = make_batch(n, d, k, torch.Generator().manual_seed(seed))
    return float((torch.sign(model(tokens(x, y))) == y[:, -1]).float().mean())


@torch.no_grad()
def order_flips(model, n=500, orders=24, seed=2, d=4, k=8):
    """Fraction of queries whose predicted label changes under some reordering of the demonstrations."""
    gen = torch.Generator().manual_seed(seed)
    x, y = make_batch(n, d, k, gen)
    perms = [torch.randperm(k, generator=gen) for _ in range(orders)]
    preds = []
    for p in perms:
        idx = torch.cat([p, torch.tensor([k])])
        preds.append(torch.sign(model(tokens(x[:, idx], y[:, idx]))))
    preds = torch.stack(preds)
    return float((preds != preds[0]).any(0).float().mean())


@torch.no_grad()
def content_free_logit(model, x, y):
    x = x.clone()
    x[:, -1] = 0
    return model(tokens(x, y))


@torch.no_grad()
def label_balance_shift(model, n=2000, seed=3, d=4, k=8):
    """Mean content-free logit with all-positive minus all-negative demonstration labels."""
    x, _ = make_batch(n, d, k, torch.Generator().manual_seed(seed))
    pos = torch.ones(n, k + 1)
    return float((content_free_logit(model, x, pos) - content_free_logit(model, x, -pos)).mean())
