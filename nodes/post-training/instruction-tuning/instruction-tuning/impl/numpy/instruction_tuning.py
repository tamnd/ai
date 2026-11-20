# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/instruction-tuning/ (ai://NEW-instruction-tuning).
"""The data side and the loss of instruction tuning in NumPy.

Instruction tuning is ordinary maximum-likelihood fine-tuning; what makes it instruction tuning is the data. This file
has the pieces that turn labelled datasets into training sequences and the loss trained on them:

* templates that phrase an example as an instruction and a target, and input inversion, which swaps the two;
* mixing rates across datasets: examples-proportional with a cap (FLAN, T5), with a temperature, and T0's size rule;
* encoding a prompt and a response into next-token inputs, targets and a loss mask that is 1 only on the response;
* packing several examples into fixed-length rows with segment ids;
* the masked negative log-likelihood, averaged over tokens or over sequences, and its gradient with respect to logits.

Token ids are plain integers; any tokenizer works. The model that produces the logits is not part of this file.
"""

import numpy as np

# Templates.


def render(template, example):
    """Fill a template, a pair (prompt_format, target_format) of str.format strings, with the fields of a dict."""
    prompt_format, target_format = template
    return prompt_format.format(**example), target_format.format(**example)


def invert(template_inverse, example):
    """Input inversion: phrase the example so that the model is given the original output and asked for the input.

    `template_inverse` is an ordinary template whose prompt mentions the original output field and whose target is the
    original input field, for example ("Write a question whose answer is: {answer}", "{question}")."""
    return render(template_inverse, example)


# Mixing.


def mixing_rates(sizes, cap=None, temperature=1.0):
    """Sampling rate of each dataset: proportional to min(size, cap) ** (1 / temperature).

    cap=None means no cap. temperature=1 is examples-proportional mixing; larger temperatures move toward uniform."""
    e = np.asarray(sizes, dtype=float)
    if np.any(e <= 0):
        raise ValueError("dataset sizes must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    w = e if cap is None else np.minimum(e, float(cap))
    w = w ** (1.0 / temperature)
    return w / w.sum()


def t0_sizes(sizes, templates, threshold=500_000):
    """The effective sizes T0 uses for proportional sampling: a dataset with more than `threshold` examples counts as
    threshold / (its number of templates)."""
    e = np.asarray(sizes, dtype=float)
    t = np.broadcast_to(np.asarray(templates, dtype=float), e.shape)
    return np.where(e > threshold, threshold / t, e)


def expected_passes(sizes, rates, num_samples):
    """Expected number of times each example of each dataset is drawn in `num_samples` draws."""
    return num_samples * np.asarray(rates, dtype=float) / np.asarray(sizes, dtype=float)


def sample_mixture(rng, sizes, rates, num_samples):
    """Draw (dataset, index) pairs: a dataset with probability rate, then an example uniformly within it."""
    sizes = np.asarray(sizes)
    which = rng.choice(len(sizes), size=num_samples, p=rates)
    index = np.floor(rng.random(num_samples) * sizes[which]).astype(int)
    return which, index


# Encoding and packing.


def encode(prompt_ids, response_ids, eos_id):
    """Inputs, next-token targets and loss mask for one example.

    The sequence is prompt + response + [eos]. Position t predicts token t + 1, and the mask is 1 exactly where that
    token belongs to the response or is the closing eos, so the summed loss is -log p(response, eos | prompt)."""
    prompt_ids, response_ids = list(prompt_ids), list(response_ids)
    if not prompt_ids:
        raise ValueError("the prompt must have at least one token")
    seq = prompt_ids + response_ids + [eos_id]
    inputs = np.array(seq[:-1], dtype=int)
    targets = np.array(seq[1:], dtype=int)
    mask = np.zeros(len(targets), dtype=float)
    mask[len(prompt_ids) - 1 :] = 1.0
    return inputs, targets, mask


def pack(encoded, length, pad_id=0):
    """Greedily pack encoded examples, in order, into rows of `length` positions.

    Returns inputs, targets, mask and segment ids, each of shape (rows, length). Segment ids start at 1 within each
    row and are 0 on padding, where the mask is also 0. An example longer than `length` raises ValueError. Attention
    must not cross segments; see `segment_attention_mask`."""
    rows, current, used = [], [], 0
    for inputs, targets, mask in encoded:
        n = len(inputs)
        if n > length:
            raise ValueError(f"example of {n} tokens does not fit in {length}")
        if used + n > length:
            rows.append(current)
            current, used = [], 0
        current.append((inputs, targets, mask))
        used += n
    if current:
        rows.append(current)
    out_inputs = np.full((len(rows), length), pad_id, dtype=int)
    out_targets = np.full((len(rows), length), pad_id, dtype=int)
    out_mask = np.zeros((len(rows), length), dtype=float)
    out_segments = np.zeros((len(rows), length), dtype=int)
    for r, row in enumerate(rows):
        pos = 0
        for s, (inputs, targets, mask) in enumerate(row, start=1):
            n = len(inputs)
            out_inputs[r, pos : pos + n] = inputs
            out_targets[r, pos : pos + n] = targets
            out_mask[r, pos : pos + n] = mask
            out_segments[r, pos : pos + n] = s
            pos += n
    return out_inputs, out_targets, out_mask, out_segments


def segment_attention_mask(segments):
    """Boolean (..., T, T) mask: position i may attend to j when j <= i and both are in the same nonzero segment."""
    seg = np.asarray(segments)
    t = seg.shape[-1]
    causal = np.tril(np.ones((t, t), dtype=bool))
    same = (seg[..., :, None] == seg[..., None, :]) & (seg[..., :, None] > 0)
    return causal & same


def segment_positions(segments):
    """Position ids that restart at 0 at the start of every segment (0 on padding)."""
    seg = np.asarray(segments)
    pos = np.zeros_like(seg)
    for idx in np.ndindex(seg.shape[:-1]):
        row = seg[idx]
        p = 0
        for t in range(len(row)):
            if row[t] == 0:
                pos[idx + (t,)] = 0
                continue
            p = 0 if t == 0 or row[t] != row[t - 1] else p + 1
            pos[idx + (t,)] = p
    return pos


# Loss.


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    zs = z - z.max(axis=-1, keepdims=True)
    return zs - np.log(np.exp(zs).sum(axis=-1, keepdims=True))


def token_nll(logits, targets):
    """Per-position cross-entropy -log softmax(logits)[target]; logits (..., T, V), targets (..., T)."""
    logp = log_softmax(logits)
    return -np.take_along_axis(logp, np.asarray(targets)[..., None], axis=-1)[..., 0]


def _weights(mask, reduction):
    m = np.asarray(mask, dtype=float)
    if reduction == "token":
        total = m.sum()
        if total == 0:
            raise ValueError("the mask selects no tokens")
        return m / total
    if reduction == "sequence":
        m2 = m.reshape(-1, m.shape[-1])
        per = m2.sum(axis=-1, keepdims=True)
        keep = per[:, 0] > 0
        if not keep.any():
            raise ValueError("the mask selects no tokens")
        w = np.where(per > 0, m2 / np.where(per > 0, per, 1.0), 0.0) / keep.sum()
        return w.reshape(m.shape)
    raise ValueError(f"unknown reduction {reduction!r}")


def masked_nll(logits, targets, mask, reduction="token"):
    """Mean cross-entropy over the positions where mask is 1.

    reduction="token" averages over all unmasked tokens of the batch, so long responses weigh more; "sequence"
    averages within each sequence first, then over sequences that have at least one unmasked token."""
    return float((_weights(mask, reduction) * token_nll(logits, targets)).sum())


def masked_nll_grad(logits, targets, mask, reduction="token"):
    """Gradient of `masked_nll` with respect to the logits: w_t (softmax(z_t) - onehot(y_t))."""
    z = np.asarray(logits, dtype=float)
    p = np.exp(log_softmax(z))
    onehot = np.zeros_like(p)
    np.put_along_axis(onehot, np.asarray(targets)[..., None], 1.0, axis=-1)
    return _weights(mask, reduction)[..., None] * (p - onehot)
