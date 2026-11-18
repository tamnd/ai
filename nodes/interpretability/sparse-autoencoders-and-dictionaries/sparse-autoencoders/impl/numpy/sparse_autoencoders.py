# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://ZTPNXN).
"""Sparse autoencoders in NumPy. Tokens are rows: x has shape (B, d), W_enc (n, d), W_dec (d, n), n latents.

encode_relu is eq-relu-encoder and encode_topk eq-topk-encoder, decode eq-decoder; l1_loss is eq-l1-loss and mse_loss
the TopK loss; normalize_decoder puts the decoder columns at unit norm and project_decoder_grad removes the part of a
gradient that would change their norms; soft_threshold is the one-latent optimum of the derivation l1-shrinkage;
fvu is the normalized MSE, relative_reconstruction_bias eq-gamma, fire_counts and dead_fraction the dead-latent count,
and auxk_loss the auxiliary loss of Gao et al. on the dead latents."""

import numpy as np


def _check_k(k, n):
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")


def relu(a):
    return np.maximum(np.asarray(a, dtype=np.float64), 0.0)


def topk(a, k):
    """Keep the k largest entries of each row, ties to the lower index, zero the rest."""
    a = np.asarray(a, dtype=np.float64)
    _check_k(k, a.shape[-1])
    idx = np.argsort(-a, axis=-1, kind="stable")[..., :k]
    out = np.zeros_like(a)
    np.put_along_axis(out, idx, np.take_along_axis(a, idx, axis=-1), axis=-1)
    return out


def pre_activations(x, W_enc, b_pre, b_enc=None):
    """W_enc (x - b_pre) + b_enc for each row of x."""
    a = (np.asarray(x, dtype=np.float64) - b_pre) @ np.asarray(
        W_enc, dtype=np.float64
    ).T
    return a if b_enc is None else a + b_enc


def encode_relu(x, W_enc, b_enc, b_pre):
    """The baseline encoder: z = ReLU(W_enc (x - b_pre) + b_enc)."""
    return relu(pre_activations(x, W_enc, b_pre, b_enc))


def encode_topk(x, W_enc, b_pre, k):
    """The TopK encoder: the k largest pre-activations of each row, then a ReLU so no kept value is negative."""
    return relu(topk(pre_activations(x, W_enc, b_pre), k))


def decode(z, W_dec, b_pre):
    """x_hat = W_dec z + b_pre for each row of z."""
    return (
        np.asarray(z, dtype=np.float64) @ np.asarray(W_dec, dtype=np.float64).T + b_pre
    )


def mse_loss(x, x_hat):
    """The squared reconstruction error, summed over the d coordinates and averaged over the batch."""
    return float(np.mean(np.sum((np.asarray(x) - x_hat) ** 2, axis=-1)))


def l1_loss(x, x_hat, z, lam):
    """The ReLU SAE loss: ||x - x_hat||^2 + lam * ||z||_1, averaged over the batch."""
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    return mse_loss(x, x_hat) + lam * float(np.mean(np.sum(np.abs(z), axis=-1)))


def normalize_decoder(W_dec, eps=1e-12):
    """Scale each column of W_dec (one per latent) to unit L2 norm."""
    W_dec = np.asarray(W_dec, dtype=np.float64)
    return W_dec / np.maximum(np.linalg.norm(W_dec, axis=0, keepdims=True), eps)


def project_decoder_grad(W_dec, grad):
    """Remove from each column of grad its component along the matching unit-norm column of W_dec."""
    W_dec = np.asarray(W_dec, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    return grad - W_dec * np.sum(W_dec * grad, axis=0, keepdims=True)


def soft_threshold(a, lam):
    """The minimizer over f >= 0 of ||x - f d||^2 + lam * f for a unit-norm d, where a = <d, x>: max(0, a - lam/2)."""
    return np.maximum(np.asarray(a, dtype=np.float64) - lam / 2.0, 0.0)


def fvu(x, x_hat):
    """Fraction of variance unexplained: the MSE over the MSE of always predicting the batch mean of x."""
    x = np.asarray(x, dtype=np.float64)
    base = np.sum((x - x.mean(axis=0, keepdims=True)) ** 2)
    if base == 0:
        raise ValueError("x has no variance")
    return float(np.sum((x - x_hat) ** 2) / base)


def relative_reconstruction_bias(x, x_hat):
    """gamma = E||x_hat||^2 / E[x_hat . x]; 1 means no shrinkage, below 1 means the reconstructions are too short."""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    return float(np.sum(x_hat * x_hat) / np.sum(x_hat * x))


def fire_counts(z):
    """How many rows each latent is active (nonzero) on."""
    return np.count_nonzero(np.asarray(z), axis=0)


def dead_fraction(counts):
    """The fraction of latents that never fired in the counted tokens."""
    counts = np.asarray(counts)
    return float(np.mean(counts == 0))


def auxk_loss(x, x_hat, pre, dead, W_dec, k_aux):
    """||e - e_hat||^2 averaged over the batch, where e = x - x_hat and e_hat = W_dec z_aux decodes the k_aux largest
    pre-activations among the dead latents only, with a ReLU, and no bias. Zero if no latent is dead."""
    dead = np.asarray(dead, dtype=bool)
    if not dead.any():
        return 0.0
    k_aux = min(k_aux, int(dead.sum()))
    masked = np.where(dead, np.asarray(pre, dtype=np.float64), -np.inf)
    z_aux = relu(topk(masked, k_aux))
    e = np.asarray(x, dtype=np.float64) - x_hat
    e_hat = z_aux @ np.asarray(W_dec, dtype=np.float64).T
    return mse_loss(e, e_hat)
