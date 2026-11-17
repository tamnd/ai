# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/speculative-decoding/ (ai://NEW-speculative-decoding).
"""Speculative decoding in NumPy, over a vocabulary of V tokens. A distribution is a 1-D array of V probabilities;
p is the target model's next-token distribution and q the draft model's, in the notation of Leviathan et al.

acceptance_probability is the rule of eq-accept and residual the adjusted distribution of eq-residual; verify_token
is one step of speculative sampling, which keeps a draft token x ~ q with probability min(1, p(x)/q(x)) and otherwise
resamples from the residual; output_distribution is the exact law of the token it returns, which equals p;
acceptance_rate is eq-alpha and tv_distance the total variation distance it complements; verify is the accept loop of
Algorithm 1 over gamma draft tokens, with the bonus token from the last target distribution; speculative_step drafts
gamma tokens from a draft model and verifies them against a target model; expected_tokens is eq-expected-tokens,
token_count_pmf the capped geometric law behind it, walltime_improvement is eq-walltime and operations_factor
eq-ops."""

import numpy as np

TOL = 1e-9


def _dist(p, name="p"):
    """p as a float64 vector, checked to be a probability distribution."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.any(p < 0) or abs(p.sum() - 1.0) > TOL:
        raise ValueError(f"{name} must be non-negative and sum to 1")
    return p


def _pair(p, q):
    p, q = _dist(p, "p"), _dist(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must be over the same vocabulary")
    return p, q


def sample(p, rng):
    """Draw one token index from the distribution p with the numpy Generator rng."""
    p = _dist(p)
    return int(rng.choice(p.size, p=p))


def acceptance_rate(p, q):
    """alpha = sum_x min(p(x), q(x)), the probability that a draft token x ~ q is accepted (eq-alpha)."""
    p, q = _pair(p, q)
    return float(np.minimum(p, q).sum())


def tv_distance(p, q):
    """The total variation distance (1/2) sum_x |p(x) - q(x)|, equal to 1 - acceptance_rate(p, q)."""
    p, q = _pair(p, q)
    return float(0.5 * np.abs(p - q).sum())


def acceptance_probability(p, q, x):
    """min(1, p(x) / q(x)) for a draft token x, which must have q(x) > 0 since it was drawn from q (eq-accept)."""
    p, q = _pair(p, q)
    if q[x] <= 0:
        raise ValueError("the draft token has probability 0 under q, so it cannot have been drawn from q")
    return min(1.0, p[x] / q[x])


def residual(p, q):
    """norm(max(0, p - q)), the distribution to resample from after a rejection (eq-residual).

    It is undefined when p == q, where no rejection can happen."""
    p, q = _pair(p, q)
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    if s <= 0:
        raise ValueError("p equals q: a rejection cannot happen and the residual is undefined")
    return r / s


def verify_token(p, q, x, rng):
    """One step of speculative sampling for a draft token x ~ q. Returns (accepted, token)."""
    a = acceptance_probability(p, q, x)
    if rng.random() < a:
        return True, int(x)
    return False, sample(residual(p, q), rng)


def output_distribution(p, q):
    """The exact law of the token returned by verify_token when x ~ q.

    It is q(x) min(1, p(x)/q(x)) for acceptance plus P(reject) times the residual; the derivation shows it is p."""
    p, q = _pair(p, q)
    ratio = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
    accepted = q * np.minimum(1.0, ratio)
    rejected = 1.0 - accepted.sum()
    if np.all(p == q):
        return accepted
    return accepted + rejected * residual(p, q)


def verify(target_probs, draft_probs, draft_tokens, rng):
    """The accept loop of Algorithm 1 (Leviathan et al.).

    target_probs has gamma + 1 rows, p_1 .. p_{gamma+1}, where p_i is the target distribution after the prefix and the
    first i - 1 draft tokens; draft_probs has gamma rows q_1 .. q_gamma; draft_tokens holds x_i ~ q_i. Returns the new
    tokens: the accepted draft tokens, then one token from the residual of the first rejected position or, when all
    gamma are accepted, a bonus token from p_{gamma+1}. The length is between 1 and gamma + 1."""
    P = np.asarray(target_probs, dtype=np.float64)
    Q = np.asarray(draft_probs, dtype=np.float64)
    xs = [int(x) for x in draft_tokens]
    gamma = len(xs)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[0] != gamma + 1 or Q.shape != (gamma, P.shape[1]):
        raise ValueError("need gamma + 1 target rows and gamma draft rows over one vocabulary")
    out = []
    for i in range(gamma):
        accepted, token = verify_token(P[i], Q[i], xs[i], rng)
        out.append(token)
        if not accepted:
            return out
    out.append(sample(P[gamma], rng))
    return out


def speculative_step(prefix, target, draft, gamma, rng):
    """One run of Algorithm 1. target(seq) and draft(seq) return the next-token distribution after a list of token ids.

    Samples gamma guesses from the draft autoregressively, evaluates the target after each of the gamma + 1 prefixes
    (one parallel call on real hardware, a loop here) and verifies. Returns prefix + the new tokens."""
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    seq = [int(t) for t in prefix]
    qs, xs = [], []
    for _ in range(gamma):
        q = _dist(draft(seq + xs), "q")
        xs.append(sample(q, rng))
        qs.append(q)
    ps = [_dist(target(seq + xs[:i]), "p") for i in range(gamma + 1)]
    return seq + verify(np.stack(ps), np.stack(qs), xs, rng)


def _check(alpha, gamma):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")


def token_count_pmf(alpha, gamma):
    """P(N = n) for n = 1 .. gamma + 1, N the tokens from one run under i.i.d. acceptance with probability alpha.

    alpha^(n-1) (1 - alpha) for n <= gamma, and alpha^gamma at the cap n = gamma + 1."""
    _check(alpha, gamma)
    n = np.arange(1, gamma + 2)
    pmf = alpha ** (n - 1) * (1.0 - alpha)
    pmf[-1] = alpha**gamma
    return pmf


def expected_tokens(alpha, gamma):
    """(1 - alpha^(gamma+1)) / (1 - alpha), the expected tokens per run (eq-expected-tokens); gamma + 1 at alpha = 1."""
    _check(alpha, gamma)
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def walltime_improvement(alpha, gamma, c):
    """The expected walltime improvement (eq-walltime): expected_tokens / (gamma c + 1), c the cost coefficient."""
    if c < 0:
        raise ValueError("the cost coefficient c must be non-negative")
    return expected_tokens(alpha, gamma) / (gamma * c + 1.0)


def operations_factor(alpha, gamma, c_hat):
    """The expected factor by which total arithmetic grows (eq-ops): (gamma c_hat + gamma + 1) / expected_tokens."""
    if c_hat < 0:
        raise ValueError("the arithmetic cost coefficient c_hat must be non-negative")
    return (gamma * c_hat + gamma + 1.0) / expected_tokens(alpha, gamma)
