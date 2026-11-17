# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
"""Sparse autoencoders in PyTorch. Tokens are rows: x has shape (B, d), W_enc (n, d), W_dec (d, n), n latents.

SparseAutoencoder is the ReLU encoder of eq-relu-encoder or the TopK encoder of eq-topk-encoder with the decoder of
eq-decoder; its encoder starts as the transpose of a decoder with unit-norm columns. normalize_decoder_ puts the columns
back at unit norm after a step, and project_decoder_grad_ removes the part of their gradient along them. l1_loss is
eq-l1-loss, fvu the normalized MSE, relative_reconstruction_bias eq-gamma, DeadLatentTracker counts the tokens since each
latent last fired, and auxk_loss is the auxiliary loss of Gao et al. on the dead latents."""

import torch
from torch import nn


def topk_relu(pre, k):
    """Keep the k largest entries of each row and apply a ReLU; the rest are zero."""
    if not 1 <= k <= pre.shape[-1]:
        raise ValueError(f"k must be in [1, {pre.shape[-1]}], got {k}")
    values, idx = torch.topk(pre, k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, idx, torch.relu(values))


class SparseAutoencoder(nn.Module):
    """A one-hidden-layer autoencoder with an overcomplete, sparse hidden layer.

    kind "relu": z = ReLU(W_enc (x - b_pre) + b_enc), trained with l1_loss.
    kind "topk": z = ReLU(TopK(W_enc (x - b_pre))), trained with the squared error alone; b_enc is unused.
    """

    def __init__(self, d, n, kind="topk", k=None, generator=None):
        super().__init__()
        if kind not in ("relu", "topk"):
            raise ValueError(f"kind must be relu or topk, got {kind}")
        if kind == "topk" and (k is None or not 1 <= k <= n):
            raise ValueError(f"a topk autoencoder needs 1 <= k <= {n}, got {k}")
        self.kind, self.k = kind, k
        W_dec = torch.randn(d, n, generator=generator)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_pre = nn.Parameter(torch.zeros(d))

    def pre_activations(self, x):
        pre = (x - self.b_pre) @ self.W_enc.T
        return pre + self.b_enc if self.kind == "relu" else pre

    def encode(self, x):
        pre = self.pre_activations(x)
        return torch.relu(pre) if self.kind == "relu" else topk_relu(pre, self.k)

    def decode(self, z):
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x):
        """Return (x_hat, z, pre)."""
        pre = self.pre_activations(x)
        z = torch.relu(pre) if self.kind == "relu" else topk_relu(pre, self.k)
        return self.decode(z), z, pre

    @torch.no_grad()
    def normalize_decoder_(self, eps=1e-12):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(eps))

    @torch.no_grad()
    def project_decoder_grad_(self):
        g = self.W_dec.grad
        if g is not None:
            g.sub_(self.W_dec * (self.W_dec * g).sum(dim=0, keepdim=True))


def mse_loss(x, x_hat):
    """Squared error summed over coordinates, averaged over the batch."""
    return ((x - x_hat) ** 2).sum(dim=-1).mean()


def l1_loss(x, x_hat, z, lam):
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    return mse_loss(x, x_hat) + lam * z.abs().sum(dim=-1).mean()


def fvu(x, x_hat):
    """Fraction of variance unexplained: squared error over the squared error of the batch mean."""
    base = ((x - x.mean(dim=0, keepdim=True)) ** 2).sum()
    if base == 0:
        raise ValueError("x has no variance")
    return ((x - x_hat) ** 2).sum() / base


def relative_reconstruction_bias(x, x_hat):
    """gamma = E||x_hat||^2 / E[x_hat . x]."""
    return (x_hat * x_hat).sum() / (x_hat * x).sum()


class DeadLatentTracker:
    """Counts, for each latent, the tokens seen since it last fired; a latent is dead past a threshold."""

    def __init__(self, n, threshold):
        self.since = torch.zeros(n, dtype=torch.long)
        self.threshold = threshold

    def update(self, z):
        fired = (z != 0).any(dim=0)
        self.since += z.shape[0]
        self.since[fired] = 0

    def dead(self):
        return self.since >= self.threshold

    def fraction(self):
        return self.dead().double().mean().item()


def auxk_loss(x, x_hat, pre, dead, W_dec, k_aux):
    """||e - W_dec z_aux||^2 averaged over the batch, e = x - x_hat with no gradient, z_aux the k_aux largest
    pre-activations among the dead latents, with a ReLU. Zero if no latent is dead."""
    n_dead = int(dead.sum())
    if n_dead == 0:
        return x.new_zeros(())
    masked = pre.masked_fill(~dead, float("-inf"))
    z_aux = topk_relu(masked, min(k_aux, n_dead))
    e = (x - x_hat).detach()
    return mse_loss(e, z_aux @ W_dec.T)


def train(sae, data, steps, lr=1e-3, batch=256, lam=0.0, generator=None):
    """Adam on mse_loss (topk) or l1_loss (relu), projecting the decoder gradient and renormalizing each step."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        x = data[torch.randint(0, data.shape[0], (batch,), generator=generator)]
        x_hat, z, _ = sae(x)
        loss = mse_loss(x, x_hat) if sae.kind == "topk" else l1_loss(x, x_hat, z, lam)
        opt.zero_grad()
        loss.backward()
        sae.project_decoder_grad_()
        opt.step()
        sae.normalize_decoder_()
    return sae
