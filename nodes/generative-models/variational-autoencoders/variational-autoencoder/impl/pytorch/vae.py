# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""The variational autoencoder of the page in PyTorch, with autograd for the backward pass: the encoder and decoder of
eq-encoder and eq-decoder, the reparameterization of eq-reparam, the closed-form KL of eq-kl, and the three objectives
of the NumPy reference, "elbo" (eq-elbo with the KL in closed form), "elbo_mc" (every term sampled) and "iwae"
(eq-iwae).

The weights have the shape (n_in, n_out) and a layer computes h @ W + b, as in the NumPy reference, so from_params
loads the same arrays. The noise eps of shape (S, B, J) is an argument, so a fixture can fix it; sample_eps draws it.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

DECODERS = ("bernoulli", "gaussian")
OBJECTIVES = ("elbo", "elbo_mc", "iwae")
LOG2PI = math.log(2.0 * math.pi)


def config(d_in=6, d_hidden=5, d_latent=2, decoder="bernoulli"):
    if decoder not in DECODERS or min(d_in, d_hidden, d_latent) < 1:
        raise ValueError("positive sizes, and a decoder among DECODERS")
    return {"d_in": int(d_in), "d_hidden": int(d_hidden), "d_latent": int(d_latent), "decoder": decoder}


def param_shapes(cfg):
    D, H, J = cfg["d_in"], cfg["d_hidden"], cfg["d_latent"]
    s = {"enc_W": (D, H), "enc_b": (H,), "mu_W": (H, J), "mu_b": (J,), "lv_W": (H, J), "lv_b": (J,),
         "dec_W": (J, H), "dec_b": (H,), "out_W": (H, D), "out_b": (D,)}  # fmt: skip
    if cfg["decoder"] == "gaussian":
        s |= {"olv_W": (H, D), "olv_b": (D,)}
    return s


def param_count(cfg):
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def kl(mu, lv):
    """eq-kl: KL(N(mu, diag exp(lv)) || N(0, I)) of each row."""
    return 0.5 * (mu**2 + lv.exp() - 1.0 - lv).sum(-1)


def log_prior(z):
    return -0.5 * (LOG2PI + z**2).sum(-1)


def log_q(lv, eps):
    """log q(z|x) at z = mu + sigma * eps, written in eps."""
    return -0.5 * (LOG2PI + lv + eps**2).sum(-1)


class VAE(nn.Module):
    def __init__(self, cfg, scale=1.0, generator=None):
        super().__init__()
        self.cfg = cfg
        self.p = nn.ParameterDict()
        for k, s in param_shapes(cfg).items():
            w = torch.randn(s, generator=generator) * scale / math.sqrt(s[0]) if len(s) == 2 else torch.zeros(s)
            self.p[k] = nn.Parameter(w)

    @classmethod
    def from_params(cls, params, cfg):
        """A model holding copies of the arrays of a NumPy parameter dict, the keys of param_shapes."""
        m = cls(cfg)
        with torch.no_grad():
            for k in m.p:
                m.p[k].copy_(torch.as_tensor(params[k]))
        return m

    def encode(self, x):
        """The mean and the log-variance of q(z|x), each (B, J)."""
        h = torch.tanh(x @ self.p["enc_W"] + self.p["enc_b"])
        return h @ self.p["mu_W"] + self.p["mu_b"], h @ self.p["lv_W"] + self.p["lv_b"]

    def decode(self, z):
        """The logits of p(x|z), and for the Gaussian decoder the log-variance too; z is (..., J)."""
        g = torch.tanh(z @ self.p["dec_W"] + self.p["dec_b"])
        a = g @ self.p["out_W"] + self.p["out_b"]
        return a, (g @ self.p["olv_W"] + self.p["olv_b"] if self.cfg["decoder"] == "gaussian" else None)

    def forward(self, x, eps):
        mu, lv = self.encode(x)
        z = mu + (0.5 * lv).exp() * eps  # eq-reparam, broadcast over the S draws
        a, olv = self.decode(z)
        return {"mu": mu, "lv": lv, "z": z, "a": a, "olv": olv}

    def nll(self, x, out):
        """-log p(x|z) of each draw, (S, B)."""
        a = out["a"]
        if self.cfg["decoder"] == "bernoulli":
            return F.binary_cross_entropy_with_logits(a, x.expand_as(a), reduction="none").sum(-1)
        return 0.5 * (LOG2PI + out["olv"] + (x - a) ** 2 * (-out["olv"]).exp()).sum(-1)

    def log_weights(self, x, out, eps):
        return -self.nll(x, out) + log_prior(out["z"]) - log_q(out["lv"], eps)

    def objective(self, x, eps, kind="elbo", beta=1.0):
        """The objective to minimize, averaged over the batch."""
        out = self(x, eps)
        if kind == "elbo":
            return self.nll(x, out).mean() + beta * kl(out["mu"], out["lv"]).mean()
        lw = self.log_weights(x, out, eps)
        if kind == "elbo_mc":
            return -lw.mean()
        if kind == "iwae":
            return -(torch.logsumexp(lw, 0) - math.log(lw.shape[0])).mean()
        raise ValueError(kind)

    def sample_eps(self, S, B, generator=None):
        return torch.randn(S, B, self.cfg["d_latent"], generator=generator)

    def sample(self, n, generator=None):
        """n samples of the mean of p(x|z), z from the prior."""
        a, _ = self.decode(torch.randn(n, self.cfg["d_latent"], generator=generator))
        return torch.sigmoid(a) if self.cfg["decoder"] == "bernoulli" else a
