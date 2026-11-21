# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/visual-instruction-tuning/ (ai://NEW-visual-instruction-tuning).
"""The pieces of a vision-language model built from frozen parts: a linear projection that turns image features
into soft tokens for a frozen language model, a tanh-gated cross-attention block, a Perceiver-style resampler, and
the POPE metrics for object hallucination.

The frozen language model is a toy: it averages the embeddings of its input tokens and scores every word of the
vocabulary against the average with the tied embedding matrix. That is enough to show the LLaVA stage-1 idea. The
model never changes, only the projection W learns, and after training the image features land where the right word
embedding is.
"""

import math

import numpy as np


# ----- a frozen toy language model and a linear visual connector -----


class FrozenBagLM:
    """Next-token logits = E h, with h the mean of the input embeddings. E is fixed."""

    def __init__(self, vocab, d, seed=0):
        self.E = np.random.default_rng(seed).standard_normal((vocab, d)) / math.sqrt(d)

    def logits_from_embeddings(self, H):
        """H: (batch, tokens, d) of input embeddings -> (batch, vocab) logits for the next token."""
        return H.mean(axis=1) @ self.E.T * math.sqrt(self.E.shape[1])

    def embed(self, ids):
        return self.E[np.asarray(ids)]


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def make_images(n, words, d_img=16, n_patches=4, noise=0.3, seed=0, protos=None):
    """n images of the given word ids. Each image is n_patches noisy copies of its class prototype."""
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((max(words) + 1, d_img)) if protos is None else protos
    y = rng.choice(np.asarray(words), size=n)
    Z = protos[y][:, None, :] + noise * rng.standard_normal((n, n_patches, d_img))
    return Z, y, protos


def soft_prompt_loss(W, lm, Z, prompt_ids, y):
    """Cross-entropy of the caption word y given [W z_1, ..., W z_P, prompt tokens], and its gradient in W.

    Z: (n, P, d_img) patch features. W: (d_img, d). Only W is trained; lm is frozen.
    """
    n, P, _ = Z.shape
    Hv = Z @ W  # (n, P, d): one soft token per patch, as in H_v = W . Z_v
    Ht = np.broadcast_to(lm.embed(prompt_ids), (n, len(prompt_ids), W.shape[1]))
    H = np.concatenate([Hv, Ht], axis=1)
    logits = lm.logits_from_embeddings(H)
    lp = log_softmax(logits)
    loss = -lp[np.arange(n), y].mean()
    dlogits = np.exp(lp)
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    dh = dlogits @ lm.E * math.sqrt(lm.E.shape[1])  # (n, d): gradient at the mean
    dHv = np.broadcast_to(dh[:, None, :] / H.shape[1], Hv.shape)  # the mean spreads it evenly
    dW = np.einsum("npi,npd->id", Z, dHv)
    return float(loss), dW, logits


def train_projection(lm, Z, y, prompt_ids, d_img, steps=300, lr=0.5, seed=0):
    W = 0.01 * np.random.default_rng(seed).standard_normal((d_img, lm.E.shape[1]))
    losses = []
    for _ in range(steps):
        loss, dW, _ = soft_prompt_loss(W, lm, Z, prompt_ids, y)
        W -= lr * dW
        losses.append(loss)
    return W, losses


# ----- Flamingo: tanh-gated cross-attention and the Perceiver Resampler -----


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_attention(y, x, Wq, Wk, Wv):
    """Queries from the text states y (T, d), keys and values from the visual tokens x (M, d)."""
    q, k, v = y @ Wq, x @ Wk, x @ Wv
    P = softmax(q @ k.T / math.sqrt(q.shape[1]))
    return P @ v, P


def gated_xattn(y, x, alpha, Wq, Wk, Wv, mask=None):
    """y' = y + tanh(alpha) * xattn(y, x). With alpha = 0 this returns y unchanged.

    mask (T, M) of booleans limits which visual tokens each text token sees (Flamingo's per-image mask).
    """
    q, k, v = y @ Wq, x @ Wk, x @ Wv
    s = q @ k.T / math.sqrt(q.shape[1])
    if mask is not None:
        s = np.where(mask, s, -1e30)
    P = softmax(s)
    if mask is not None:
        P = np.where(mask.any(axis=1, keepdims=True), P, 0.0)  # text before any image sees nothing
    f = P @ v
    return y + math.tanh(alpha) * f, f, P


def gated_xattn_grads(y, x, alpha, Wq, Wk, Wv, g):
    """Gradients of <g, y'> with respect to alpha and Wv (the analytic forms of the derivation)."""
    _, f, P = gated_xattn(y, x, alpha, Wq, Wk, Wv)
    d_alpha = (1 - math.tanh(alpha) ** 2) * float(np.sum(g * f))
    d_Wv = math.tanh(alpha) * x.T @ (P.T @ g)
    return d_alpha, d_Wv


def image_causal_mask(text_image_index, image_of_token):
    """mask[t, j] is True when visual token j belongs to the image that most recently preceded text token t.

    text_image_index[t]: index of the last image before text token t, or -1. image_of_token[j]: image of visual token j.
    """
    t = np.asarray(text_image_index)[:, None]
    return (t >= 0) & (t == np.asarray(image_of_token)[None, :])


def perceiver_resampler(X, latents, Wq, Wk, Wv, layers=1):
    """Map a variable number of visual features X (n, d) to len(latents) tokens.

    Each layer lets the latents attend to the concatenation of X and themselves, as in Flamingo, and adds the result
    to the latents. Without positional terms the output does not depend on the order of the rows of X.
    """
    L = latents.copy()
    for _ in range(layers):
        kv = np.concatenate([X, L], axis=0)
        out, _ = cross_attention(L, kv, Wq, Wk, Wv)
        L = L + out
    return L


def visual_tokens(resolution, patch, cls_token=False):
    side = resolution // patch
    return side * side + (1 if cls_token else 0)


# ----- POPE -----


def pope_questions(image_objects, freq, cooc, l=6, strategy="random", rng=None):
    """Build l yes-or-no probes for one image: l/2 objects it contains and l/2 it does not.

    freq: dict object -> count over the dataset. cooc: dict (a, b) -> co-occurrence count.
    random samples absent objects uniformly; popular takes the most frequent absent objects; adversarial takes the
    absent objects that co-occur most with the image's objects.
    """
    rng = rng or np.random.default_rng(0)
    present = list(image_objects)
    absent = [o for o in freq if o not in image_objects]
    k = l // 2
    yes = list(rng.choice(present, size=min(k, len(present)), replace=False))
    if strategy == "random":
        no = list(rng.choice(absent, size=k, replace=False))
    elif strategy == "popular":
        no = sorted(absent, key=lambda o: (-freq[o], o))[:k]
    elif strategy == "adversarial":
        score = {o: sum(cooc.get((p, o), 0) + cooc.get((o, p), 0) for p in present) for o in absent}
        no = sorted(absent, key=lambda o: (-score[o], o))[:k]
    else:
        raise ValueError(strategy)
    qs = [(f"Is there a {o} in the image?", "yes", o) for o in yes] + [(f"Is there a {o} in the image?", "no", o) for o in no]
    return qs


def pope_metrics(pred_yes, label_yes):
    """Accuracy, precision, recall and F1 with yes as the positive class, and the share of yes answers."""
    p, t = np.asarray(pred_yes, bool), np.asarray(label_yes, bool)
    tp, fp, fn = int(np.sum(p & t)), int(np.sum(p & ~t)), int(np.sum(~p & t))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": float(np.mean(p == t)), "precision": prec, "recall": rec, "f1": f1, "yes": float(np.mean(p))}
