# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/text-to-image-generation/ (ai://NEW-text-to-image-generation).
"""Pieces of a text-to-image system that can be checked without a trained network.

- Classifier-free guidance in the x-prediction form Imagen uses, and Imagen's static and dynamic thresholding,
  run inside a deterministic DDIM sampler whose denoisers are exact posterior means of Gaussian "images".
- Contrastive reranking, as in DALL-E and Parti: draw N candidates, score each against the caption with a
  contrastive model, keep the top k.
- The PCA-and-buckets tokenization unCLIP uses for its autoregressive prior over CLIP image embeddings.
- A pooled bag-of-words text embedding, which cannot tell "red left, blue right" from "blue left, red right".
"""

import numpy as np

# ----- guidance and thresholding -----


def guided_x0(x0_cond, x0_uncond, w):
    """Classifier-free guidance on x-predictions: w * x0_c + (1 - w) * x0_u. w = 1 is the conditional model."""
    return w * x0_cond + (1.0 - w) * x0_uncond


def static_threshold(x0):
    """Clip every pixel of the x-prediction to the training range [-1, 1]."""
    return np.clip(x0, -1.0, 1.0)


def dynamic_threshold(x0, p=99.5):
    """Imagen: s = the p-th percentile of |x0| per sample; if s > 1, clip to [-s, s] and divide by s."""
    flat = np.abs(x0).reshape(x0.shape[0], -1)
    s = np.maximum(np.percentile(flat, p, axis=1), 1.0).reshape((-1,) + (1,) * (x0.ndim - 1))
    return np.clip(x0, -s, s) / s


def gaussian_posterior_mean(z, alpha, sigma, mu, s):
    """E[x | z] for x ~ N(mu, s^2) per pixel and z = alpha x + sigma eps: the exact denoiser of that data."""
    gain = alpha * s**2 / (alpha**2 * s**2 + sigma**2)
    return mu + gain * (z - alpha * mu)


def cosine_schedule(t):
    """Variance-preserving schedule: alpha = cos(pi t / 2), sigma = sin(pi t / 2)."""
    return np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)


def ddim_guided(z, mu_c, s_c, mu_u, s_u, w, steps=50, threshold=None, t_max=0.999):
    """Deterministic DDIM from t_max to 0 with classifier-free guidance and an optional threshold on x0.

    The conditional "model" is the exact denoiser of N(mu_c, s_c^2), the unconditional one that of N(mu_u, s_u^2).
    Returns the final x0 and the largest |x0| seen before thresholding, over the whole run.
    """
    ts = np.linspace(t_max, 0.0, steps + 1)
    peak = 0.0
    x0 = z
    for t, t_next in zip(ts[:-1], ts[1:]):
        a, sg = cosine_schedule(t)
        x0 = guided_x0(
            gaussian_posterior_mean(z, a, sg, mu_c, s_c), gaussian_posterior_mean(z, a, sg, mu_u, s_u), w
        )
        peak = max(peak, float(np.abs(x0).max()))
        if threshold is not None:
            x0 = threshold(x0)
        eps = (z - a * x0) / sg
        a2, s2 = cosine_schedule(t_next)
        z = a2 * x0 + s2 * eps
    return x0, peak


def saturation(x, tol=1e-6):
    """The fraction of pixels sitting on the boundary of [-1, 1]."""
    return float(np.mean(np.abs(x) >= 1.0 - tol))


# ----- contrastive reranking -----


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rerank(image_emb, text_emb, k):
    """Indices of the k candidates whose embeddings have the highest cosine similarity with the caption."""
    scores = normalize(image_emb) @ normalize(text_emb)
    return np.argsort(-scores, kind="stable")[:k], scores


def best_of_n(p, n):
    """Probability that at least one of n independent samples is right when each is right with probability p.

    With a perfect reranker this is the top-1 accuracy of best-of-n sampling.
    """
    return 1.0 - (1.0 - p) ** n


# ----- unCLIP's autoregressive-prior tokens -----


def fit_pca(Z, k):
    """Mean and the top-k principal directions of the rows of Z, ordered by decreasing eigenvalue."""
    mean = Z.mean(axis=0)
    _, sv, vt = np.linalg.svd(Z - mean, full_matrices=False)
    return mean, vt[:k], sv


def pca_tokens(Z, mean, comps, lo, hi, buckets):
    """Project onto the components, then quantize each coordinate into one of `buckets` equal bins."""
    c = (Z - mean) @ comps.T
    q = np.floor((np.clip(c, lo, hi) - lo) / (hi - lo) * buckets).astype(int)
    return np.minimum(q, buckets - 1)


def pca_detokenize(tokens, mean, comps, lo, hi, buckets):
    """Bin centres back to coordinates, then back to the embedding space."""
    c = lo + (tokens + 0.5) * (hi - lo) / buckets
    return mean + c @ comps


# ----- a pooled text embedding -----


def pooled_text(tokens, E):
    """Mean of the word embeddings: a bag of words, so it is the same for any order of the same words."""
    return E[np.asarray(tokens)].mean(axis=0)
