# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/continued-pretraining/ (ai://NEW-continued-pretraining).
"""The pieces of continued pretraining, in NumPy.

Continued pretraining takes a model that has finished pretraining and trains it further with the same objective on
new data: a domain, a language, longer documents, or a fresher crawl. The file has:

* learning rate schedules: linear warmup into a cosine decay (re-warming and re-decaying), Parmar et al.'s schedule
  with its switch point, and the infinite schedule of Ibrahim et al. (warmup, cooldown, constant, annealing);
* replay: the number of old and new tokens under compute-equivalent replay, and a batch sampler that draws a fixed
  fraction of every batch from the old data;
* elastic weight consolidation: the diagonal Fisher information from per-example gradients and the penalty;
* vocabulary extension: new embedding and output rows, initialized at the mean of the old ones or at random;
* RoPE frequencies and wavelengths for a given base;
* measurements: forgetting and adaptation, tokens per pass over a corpus, and the bound on how much mean-initialized
  rows change the next-token distribution;
* a softmax bigram language model with SGD, small enough to show forgetting and replay in a test.
"""

import math

import numpy as np

# Learning rate schedules.


def cosine_decay(s, eta_max, eta_min):
    """eta_min + (eta_max - eta_min)(1 + cos(pi s))/2 for s in [0, 1], the fraction of the decay done."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return eta_min + (eta_max - eta_min) * (1.0 + np.cos(np.pi * s)) / 2.0


def rewarm_cosine(step, warmup, total, eta_max, eta_min):
    """The schedule of a continued-pretraining run: linear warmup from 0 to eta_max over `warmup` steps, then cosine
    decay to eta_min at step `total`. Re-warming and re-decaying in Ibrahim et al.'s sense is running this again from
    the final checkpoint of a run that already decayed."""
    step = np.asarray(step, dtype=float)
    warm = eta_max * step / max(warmup, 1)
    decay = cosine_decay((step - warmup) / max(total - warmup, 1), eta_max, eta_min)
    return np.where(step < warmup, warm, decay)


def cosine_crossing(eta, eta_max, eta_min):
    """The fraction s of a cosine decay at which the learning rate equals eta."""
    if not min(eta_min, eta_max) <= eta <= max(eta_min, eta_max):
        raise ValueError("eta is outside the schedule")
    frac = (eta - eta_min) / (eta_max - eta_min)
    return math.acos(2.0 * frac - 1.0) / math.pi


def parmar_schedule(pretrain_eta_min, switch_fraction_of_start=0.2, end_fraction=0.01):
    """Parmar et al.'s recipe: start at the pretrained model's eta_min, cosine-decay to a hundredth of it, and switch
    from the general blend to the second blend when the rate reaches a fifth of its start. Returns the start, the end
    and the fraction of the decay at which to switch."""
    start = pretrain_eta_min
    end = end_fraction * start
    return start, end, cosine_crossing(switch_fraction_of_start * start, start, end)


def infinite_schedule(step, warmup, cooldown, t_ann, t_end, eta_max, eta_const, eta_min):
    """Ibrahim et al.'s infinite schedule: linear warmup to eta_max, cosine cooldown to eta_const by step
    warmup + cooldown, constant until t_ann, then exponential annealing eta_const (eta_min/eta_const)^((t - t_ann) /
    (t_end - t_ann)). A new dataset can start in the constant phase without re-warming."""
    t = float(step)
    if t < warmup:
        return eta_max * t / max(warmup, 1)
    if t < warmup + cooldown:
        return float(cosine_decay((t - warmup) / cooldown, eta_max, eta_const))
    if t < t_ann:
        return eta_const
    s = min((t - t_ann) / (t_end - t_ann), 1.0)
    return eta_const * (eta_min / eta_const) ** s


# Replay.


def compute_equivalent_replay(new_tokens, replay_fraction):
    """Ibrahim et al.'s compute-equivalent replay: the run keeps new_tokens of compute, of which a replay_fraction
    comes from the old data. Returns (unique new tokens seen, replayed old tokens)."""
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("replay_fraction must be in [0, 1)")
    old = replay_fraction * new_tokens
    return new_tokens - old, old


def tokens_for_new(new_tokens, replay_fraction):
    """Total tokens needed to see new_tokens of new data when a replay_fraction of every batch is old data."""
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("replay_fraction must be in [0, 1)")
    return new_tokens / (1.0 - replay_fraction)


def replay_batches(n_old, n_new, batch, replay_fraction, steps, rng):
    """Index batches for continued pretraining with replay: each batch has round(replay_fraction * batch) indices
    into the old data and the rest into the new data. New data is read in order without repetition, wrapping around
    when it runs out; old data is sampled uniformly. Yields (old_idx, new_idx)."""
    n_rep = round(replay_fraction * batch)
    pos = 0
    for _ in range(steps):
        old_idx = rng.integers(n_old, size=n_rep)
        new_idx = (pos + np.arange(batch - n_rep)) % n_new
        pos = (pos + batch - n_rep) % n_new
        yield old_idx, new_idx


def passes_over(total_tokens, weight, corpus_tokens):
    """How many times a mixture with this weight on a corpus goes over it in total_tokens."""
    return total_tokens * weight / corpus_tokens


# Elastic weight consolidation.


def fisher_diagonal(per_example_grads):
    """Empirical diagonal Fisher information: the mean of the squared per-example gradients of the log-likelihood,
    shape (n_examples, n_params) -> (n_params,). EWC takes the gradients at the old optimum."""
    g = np.asarray(per_example_grads, dtype=float)
    return np.mean(g * g, axis=0)


def ewc_penalty(theta, theta_old, fisher, lam):
    """(lam/2) sum_i F_i (theta_i - theta_old_i)^2 and its gradient lam F (theta - theta_old)."""
    d = np.asarray(theta, dtype=float) - np.asarray(theta_old, dtype=float)
    return 0.5 * lam * float(np.sum(fisher * d * d)), lam * fisher * d


# Vocabulary extension.


def extend_vocab(embed, head, k, rng=None, init="mean"):
    """Append k rows to the input embedding (V, H) and the output head (V, H). init="mean" sets every new row to
    the mean of the old rows of its matrix, which bounds how much the next-token distribution changes before
    training; init="normal" draws rows with the old rows' mean and standard deviation per column. The old rows are
    unchanged."""
    embed, head = np.asarray(embed, dtype=float), np.asarray(head, dtype=float)
    if init == "mean":
        new_e = np.tile(embed.mean(0), (k, 1))
        new_h = np.tile(head.mean(0), (k, 1))
    elif init == "normal":
        new_e = embed.mean(0) + embed.std(0) * rng.standard_normal((k, embed.shape[1]))
        new_h = head.mean(0) + head.std(0) * rng.standard_normal((k, head.shape[1]))
    else:
        raise ValueError("init must be 'mean' or 'normal'")
    return np.vstack([embed, new_e]), np.vstack([head, new_h])


def new_parameters(old_vocab, new_vocab, hidden, tied=False):
    """Parameters added by growing the vocabulary: one matrix if embedding and head are tied, two otherwise."""
    return (1 if tied else 2) * (new_vocab - old_vocab) * hidden


def mean_init_kl_bound(old_vocab, k):
    """Upper bound log(1 + k/V) on KL(old next-token distribution || new one) after k mean-initialized rows."""
    return math.log1p(k / old_vocab)


# RoPE.


def rope_frequencies(d, base):
    """theta_i = base^(-2i/d) for the d/2 rotated pairs."""
    return base ** (-2.0 * np.arange(d // 2) / d)


def rope_wavelengths(d, base):
    """Positions after which each pair's rotation repeats, 2 pi base^(2i/d)."""
    return 2.0 * np.pi / rope_frequencies(d, base)


def rope_rotate(x, positions, base):
    """Apply RoPE to x of shape (T, d) at the given positions, rotating consecutive pairs (x_2i, x_2i+1)."""
    x = np.asarray(x, dtype=float)
    ang = np.outer(positions, rope_frequencies(x.shape[-1], base))
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out


# Measurements.


def forgetting(old_loss_before, old_loss_after):
    """Increase of the loss on the old data over continued pretraining; negative means it improved."""
    return old_loss_after - old_loss_before


def adaptation(new_loss_before, new_loss_after):
    """Decrease of the loss on the new data over continued pretraining."""
    return new_loss_before - new_loss_after


# A bigram language model to show forgetting.


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class BigramLM:
    """p(next | prev) = softmax(W[prev]); W has one row of logits per previous token."""

    def __init__(self, vocab, rng, scale=0.01):
        self.W = scale * rng.standard_normal((vocab, vocab))

    def loss(self, pairs):
        """Mean cross-entropy of the next token over an array of (prev, next) pairs."""
        p = softmax(self.W[pairs[:, 0]])
        return float(-np.mean(np.log(p[np.arange(len(pairs)), pairs[:, 1]])))

    def grad(self, pairs):
        p = softmax(self.W[pairs[:, 0]])
        p[np.arange(len(pairs)), pairs[:, 1]] -= 1.0
        g = np.zeros_like(self.W)
        np.add.at(g, pairs[:, 0], p)
        return g / len(pairs)

    def per_example_grads(self, pairs):
        """Gradients of each pair's log-likelihood, flattened, for the Fisher information."""
        out = np.zeros((len(pairs), self.W.size))
        for j, (a, b) in enumerate(pairs):
            p = softmax(self.W[a])
            p[b] -= 1.0
            g = np.zeros_like(self.W)
            g[a] = -p
            out[j] = g.ravel()
        return out


def markov_pairs(transition, n, rng):
    """n (prev, next) pairs from a Markov chain with this row-stochastic transition matrix, started uniformly."""
    v = transition.shape[0]
    prev = rng.integers(v, size=n)
    nxt = np.array([rng.choice(v, p=transition[a]) for a in prev])
    return np.stack([prev, nxt], axis=1)


def train(model, data, steps, batch, lr, rng, old=None, replay_fraction=0.0, ewc=None):
    """SGD on `data` with an optional replay of `old` and an optional EWC term (theta_old, fisher, lam). lr is a
    function of the step. Returns the model."""
    batches = replay_batches(len(old) if old is not None else 1, len(data), batch, replay_fraction, steps, rng)
    for t, (oi, ni) in enumerate(batches):
        parts = [data[ni]] + ([old[oi]] if old is not None and len(oi) else [])
        g = model.grad(np.concatenate(parts))
        if ewc is not None:
            theta_old, fisher, lam = ewc
            g = g + ewc_penalty(model.W.ravel(), theta_old, fisher, lam)[1].reshape(model.W.shape)
        model.W -= lr(t) * g
    return model
