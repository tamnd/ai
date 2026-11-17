# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/speculative-decoding/ (ai://NEW-speculative-decoding).
"""Speculative decoding in PyTorch, over a vocabulary of V tokens. A distribution is a 1-D float64 tensor of V
probabilities; p is the target model's next-token distribution and q the draft model's, as in Leviathan et al.
Randomness comes from a torch.Generator.

acceptance_probability is the rule of eq-accept and residual the adjusted distribution of eq-residual; verify_token
keeps a draft token x ~ q with probability min(1, p(x)/q(x)) and otherwise resamples from the residual;
output_distribution is the exact law of its output, which equals p; acceptance_rate is eq-alpha and tv_distance the
total variation distance it complements; verify is the accept loop of Algorithm 1 with the bonus token;
speculative_step drafts gamma tokens and verifies them; expected_tokens is eq-expected-tokens, token_count_pmf the
capped geometric law behind it, walltime_improvement eq-walltime and operations_factor eq-ops."""

import torch

TOL = 1e-9


def _dist(p, name="p"):
    p = torch.as_tensor(p, dtype=torch.float64)
    if p.dim() != 1 or p.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1-D tensor")
    if bool((p < 0).any()) or abs(float(p.sum()) - 1.0) > TOL:
        raise ValueError(f"{name} must be non-negative and sum to 1")
    return p


def _pair(p, q):
    p, q = _dist(p, "p"), _dist(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must be over the same vocabulary")
    return p, q


def sample(p, gen):
    """Draw one token index from the distribution p with the torch.Generator gen."""
    p = _dist(p)
    return int(torch.multinomial(p, 1, generator=gen))


def _uniform(gen):
    return float(torch.rand((), generator=gen, dtype=torch.float64))


def acceptance_rate(p, q):
    """alpha = sum_x min(p(x), q(x)) (eq-alpha)."""
    p, q = _pair(p, q)
    return float(torch.minimum(p, q).sum())


def tv_distance(p, q):
    """(1/2) sum_x |p(x) - q(x)|, equal to 1 - acceptance_rate(p, q)."""
    p, q = _pair(p, q)
    return float(0.5 * (p - q).abs().sum())


def acceptance_probability(p, q, x):
    """min(1, p(x) / q(x)) for a draft token x with q(x) > 0 (eq-accept)."""
    p, q = _pair(p, q)
    if float(q[x]) <= 0:
        raise ValueError("the draft token has probability 0 under q, so it cannot have been drawn from q")
    return min(1.0, float(p[x] / q[x]))


def residual(p, q):
    """norm(max(0, p - q)) (eq-residual); undefined when p == q."""
    p, q = _pair(p, q)
    r = torch.clamp(p - q, min=0.0)
    s = float(r.sum())
    if s <= 0:
        raise ValueError("p equals q: a rejection cannot happen and the residual is undefined")
    return r / s


def verify_token(p, q, x, gen):
    """One step of speculative sampling for a draft token x ~ q. Returns (accepted, token)."""
    a = acceptance_probability(p, q, x)
    if _uniform(gen) < a:
        return True, int(x)
    return False, sample(residual(p, q), gen)


def output_distribution(p, q):
    """The exact law of the token returned by verify_token when x ~ q; the derivation shows it is p."""
    p, q = _pair(p, q)
    ratio = torch.where(q > 0, p / torch.where(q > 0, q, torch.ones_like(q)), torch.zeros_like(p))
    accepted = q * torch.clamp(ratio, max=1.0)
    rejected = 1.0 - float(accepted.sum())
    if bool(torch.equal(p, q)):
        return accepted
    return accepted + rejected * residual(p, q)


def verify(target_probs, draft_probs, draft_tokens, gen):
    """The accept loop of Algorithm 1: target_probs has gamma + 1 rows, draft_probs gamma rows, draft_tokens x_i ~ q_i.

    Returns the accepted draft tokens, then a token from the residual of the first rejection or, when all gamma are
    accepted, a bonus token from the last target row."""
    P = torch.as_tensor(target_probs, dtype=torch.float64)
    Q = torch.as_tensor(draft_probs, dtype=torch.float64)
    xs = [int(x) for x in draft_tokens]
    gamma = len(xs)
    if P.dim() != 2 or Q.dim() != 2 or P.shape[0] != gamma + 1 or tuple(Q.shape) != (gamma, P.shape[1]):
        raise ValueError("need gamma + 1 target rows and gamma draft rows over one vocabulary")
    out = []
    for i in range(gamma):
        accepted, token = verify_token(P[i], Q[i], xs[i], gen)
        out.append(token)
        if not accepted:
            return out
    out.append(sample(P[gamma], gen))
    return out


def speculative_step(prefix, target, draft, gamma, gen):
    """One run of Algorithm 1 with callables target(seq) and draft(seq) giving next-token distributions."""
    if gamma < 1:
        raise ValueError("gamma must be at least 1")
    seq = [int(t) for t in prefix]
    qs, xs = [], []
    for _ in range(gamma):
        q = _dist(draft(seq + xs), "q")
        xs.append(sample(q, gen))
        qs.append(q)
    ps = [_dist(target(seq + xs[:i]), "p") for i in range(gamma + 1)]
    return seq + verify(torch.stack(ps), torch.stack(qs), xs, gen)


def _check(alpha, gamma):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")


def token_count_pmf(alpha, gamma):
    """P(N = n) for n = 1 .. gamma + 1: alpha^(n-1) (1 - alpha) below the cap and alpha^gamma at it."""
    _check(alpha, gamma)
    n = torch.arange(1, gamma + 2, dtype=torch.float64)
    pmf = alpha ** (n - 1) * (1.0 - alpha)
    pmf[-1] = alpha**gamma
    return pmf


def expected_tokens(alpha, gamma):
    """(1 - alpha^(gamma+1)) / (1 - alpha) (eq-expected-tokens); gamma + 1 at alpha = 1."""
    _check(alpha, gamma)
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def walltime_improvement(alpha, gamma, c):
    """expected_tokens / (gamma c + 1) (eq-walltime)."""
    if c < 0:
        raise ValueError("the cost coefficient c must be non-negative")
    return expected_tokens(alpha, gamma) / (gamma * c + 1.0)


def operations_factor(alpha, gamma, c_hat):
    """(gamma c_hat + gamma + 1) / expected_tokens (eq-ops)."""
    if c_hat < 0:
        raise ValueError("the arithmetic cost coefficient c_hat must be non-negative")
    return (gamma * c_hat + gamma + 1.0) / expected_tokens(alpha, gamma)
