# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/contrastive-language-image-pretraining/ (ai://NEW-contrastive-language-image-pretraining).
"""A dual encoder trained with the CLIP softmax loss or the SigLIP sigmoid loss on paired synthetic views, with
zero-shot classification and a measurement of the modality gap.

The encoders are small ReLU networks, so their outputs at initialization already lie in a narrow cone, one cone per
encoder; this is the starting point Liang et al. give for the modality gap. Training aligns paired views while the
two cones stay apart, and the gap shift experiment measures the loss when one modality is moved toward the other.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def make_pairs(n, k=8, dx=32, dy=24, noise=0.3, gen=None, mix=None):
    """n paired views x = z A + noise, y = z B + noise of a k-dimensional latent z."""
    gen = gen or torch.Generator().manual_seed(0)
    a, b = mix if mix is not None else (torch.randn(k, dx, generator=gen), torch.randn(k, dy, generator=gen))
    z = torch.randn(n, k, generator=gen)
    x = z @ a + noise * torch.randn(n, dx, generator=gen)
    y = z @ b + noise * torch.randn(n, dy, generator=gen)
    return x, y, z, (a, b)


def encoder(d_in, d_out, width=128, depth=3):
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def clip_loss(u, v, scale):
    logits = scale * u @ v.T
    labels = torch.arange(len(u))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def sigmoid_loss(u, v, t, b):
    z = t * u @ v.T + b
    y = 2 * torch.eye(len(u)) - 1
    return -F.logsigmoid(y * z).sum() / len(u)


class DualEncoder(nn.Module):
    def __init__(self, dx, dy, d=16, loss="clip", learn_scale=True, scale=1 / 0.07):
        super().__init__()
        self.image, self.text, self.loss_kind = encoder(dx, d), encoder(dy, d), loss
        init = math.log(10.0) if loss == "sigmoid" else math.log(scale)
        self.log_scale = nn.Parameter(torch.tensor(init), requires_grad=learn_scale)
        self.bias = nn.Parameter(torch.tensor(-10.0), requires_grad=loss == "sigmoid")

    def embed_images(self, x):
        return F.normalize(self.image(x), dim=-1)

    def embed_texts(self, y):
        return F.normalize(self.text(y), dim=-1)

    def scale(self):
        return self.log_scale.clamp(max=math.log(100.0)).exp()  # CLIP caps the logit scale at 100

    def loss(self, u, v):
        if self.loss_kind == "clip":
            return clip_loss(u, v, self.scale())
        return sigmoid_loss(u, v, self.scale(), self.bias)

    def forward(self, x, y):
        return self.loss(self.embed_images(x), self.embed_texts(y))


def train(model, x, y, steps=600, batch=256, lr=1e-3, seed=1):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        idx = torch.randperm(len(x), generator=gen)[:batch]
        loss = model(x[idx], y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def retrieval_accuracy(model, x, y):
    sim = model.embed_images(x) @ model.embed_texts(y).T
    return (sim.argmax(1) == torch.arange(len(x))).float().mean().item()


@torch.no_grad()
def zero_shot_accuracy(model, images, labels, captions):
    """captions: (K, P, dy); the P prompt embeddings of each class are averaged, then normalized."""
    k, p, dy = captions.shape
    w = F.normalize(model.embed_texts(captions.reshape(k * p, dy)).reshape(k, p, -1).mean(1), dim=-1)
    pred = (model.embed_images(images) @ w.T).argmax(1)
    return (pred == labels).float().mean().item()


@torch.no_grad()
def modality_gap(model, x, y):
    """Distance between the centres of the image and text embeddings (Liang et al.)."""
    return (model.embed_images(x).mean(0) - model.embed_texts(y).mean(0)).norm().item()


@torch.no_grad()
def mean_cosine_within(emb):
    """Average cosine between different embeddings of one modality; near 1 means a narrow cone."""
    n = len(emb)
    return ((emb @ emb.T).sum() - n).item() / (n * (n - 1))


@torch.no_grad()
def shifted_loss(model, x, y, lam):
    """Loss after moving image embeddings by -lam/2 * gap and text by +lam/2 * gap, then renormalizing."""
    u, v = model.embed_images(x), model.embed_texts(y)
    delta = u.mean(0) - v.mean(0)
    return model.loss(F.normalize(u - lam / 2 * delta, dim=-1), F.normalize(v + lam / 2 * delta, dim=-1)).item()
