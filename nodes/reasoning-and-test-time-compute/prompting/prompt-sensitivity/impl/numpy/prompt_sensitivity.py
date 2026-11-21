# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/prompt-sensitivity/ (ai://NEW-prompt-sensitivity).
"""A toy few-shot classifier with the biases of Zhao et al., contextual calibration, entropy-based order selection
and the statistics used to measure prompt sensitivity.

The toy model scores a test input by a fixed semantic term plus three prompt-dependent shifts: a majority-label term
that grows with the share of one label among the demonstrations, a recency term that weighs later demonstrations
more, and a format term. The shifts do not depend on the input, so a content-free input reads them off exactly and
contextual calibration removes them. The statistics functions give the expected spread across sampled formats, the
probability that two models swap order under another format, the optimism of choosing a prompt on a small
development set, and the rank agreement statistics Kendall's W and tau-b.
"""

import itertools
import math

import numpy as np

SEPARATORS = [": ", " - ", ":: ", ":\n", " "]
SPACES = [" ", "\n", " || ", "\n\n"]
CASINGS = [str.title, str.upper, str.lower]


def render_format(fmt, fields):
    """Render (descriptor, value) pairs with a format (separator, space, casing), as in Sclar et al.'s grammar."""
    sep, space, case = fmt
    return space.join(case(name) + sep + value for name, value in fields)


def all_formats():
    return list(itertools.product(SEPARATORS, SPACES, CASINGS))


def prompt_bias(labels, maj=1.5, rec=1.0, decay=0.5, fmt=0.0):
    """Log-odds shift toward the positive class caused by the demonstration labels (+1/-1) and the format."""
    labels = np.asarray(labels, dtype=float)
    k = len(labels)
    majority = maj * labels.mean() if k else 0.0
    weights = decay ** np.arange(k - 1, -1, -1)
    recency = rec * float(weights @ labels) / weights.sum() if k else 0.0
    return majority + recency + fmt


def semantic_logodds(x, w, scale=3.0):
    return scale * (np.asarray(x) @ w)


def icl_prob(x, w, labels, **bias):
    """Positive-class probability of the toy model for inputs x given the demonstration labels in order."""
    return 1 / (1 + np.exp(-(semantic_logodds(x, w) + prompt_bias(labels, **bias))))


def content_free_prob(w, labels, **bias):
    return float(icl_prob(np.zeros(len(w)), w, labels, **bias))


def calibrate(p, p_cf):
    """Contextual calibration: argmax of p / p_cf over classes. p has shape (n, c), p_cf shape (c,)."""
    return np.argmax(np.asarray(p) / np.asarray(p_cf), axis=1)


def binary_calibrated(p, a):
    """Calibrated binary prediction, equal to thresholding p at the content-free probability a."""
    return calibrate(np.stack([1 - p, p], 1), np.array([1 - a, a])) == 1


def order_accuracies(demo_labels, x, y, w, calibrated=False, **bias):
    """Accuracy for every distinct order of the demonstration labels."""
    out = {}
    for order in sorted(set(itertools.permutations(demo_labels))):
        p = icl_prob(x, w, order, **bias)
        pred = binary_calibrated(p, content_free_prob(w, order, **bias)) if calibrated else p > 0.5
        out[order] = float((pred == (y > 0)).mean())
    return out


def global_entropy(pred, n_classes=2):
    """GlobalE of Lu et al.: entropy of the predicted-label distribution over a probing set."""
    q = np.bincount(np.asarray(pred), minlength=n_classes) / len(pred)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def local_entropy(p):
    """LocalE of Lu et al.: mean entropy of the per-input predicted distributions. p has shape (n, c)."""
    p = np.clip(np.asarray(p), 1e-12, 1)
    return float(-(p * np.log(p)).sum(1).mean())


def expected_max_normal(m):
    x = np.linspace(-12, 12, 240_001)
    pdf = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return float(np.trapezoid(x * m * pdf * cdf ** (m - 1), x))


def expected_range(m, sigma=1.0):
    return 2 * sigma * expected_max_normal(m)


def flip_probability(delta, sigma, rho=0.0):
    z = -delta / (sigma * math.sqrt(2 * (1 - rho)))
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def selection_optimism(p, n, m):
    """Expected excess of the best development score over the true accuracy, m equal prompts, n items."""
    return math.sqrt(p * (1 - p) / n) * expected_max_normal(m)


def ranks(scores):
    """Ranks 1..n (1 = highest score), ties broken by position."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    r = np.empty(len(order), dtype=float)
    r[order] = np.arange(1, len(order) + 1)
    return r


def kendall_w(rank_matrix):
    """Kendall's coefficient of concordance for an (m judges, n objects) matrix of ranks without ties."""
    r = np.asarray(rank_matrix, dtype=float)
    m, n = r.shape
    totals = r.sum(0)
    return float(12 * ((totals - totals.mean()) ** 2).sum() / (m * m * (n**3 - n)))


def kendall_tau_b(a, b):
    a, b = np.asarray(a), np.asarray(b)
    con = dis = ta = tb = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s, t = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        if s == 0 and t == 0:
            continue
        if s == 0:
            ta += 1
        elif t == 0:
            tb += 1
        elif s == t:
            con += 1
        else:
            dis += 1
    return (con - dis) / math.sqrt((con + dis + ta) * (con + dis + tb))
