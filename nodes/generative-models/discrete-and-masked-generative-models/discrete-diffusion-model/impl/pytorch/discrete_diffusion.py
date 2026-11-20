# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Discrete diffusion in PyTorch, the gradients by autograd.

The same pieces as the NumPy reference: the D3PM transition matrices, posterior and x0-parameterized reverse step
(eq-d3pm-forward, eq-d3pm-posterior, eq-x0-param); masked diffusion with its schedules, SUBS (eq-subs) and the
weighted cross-entropy bound (eq-mdlm) as a one-sample estimate and exactly; the ancestral sampler and its exact output
distribution; MaskGIT's decoding; score entropy (eq-score-entropy); and the denoiser network as an nn.Module trained by
the exact bound with torch.optim.Adam. Tokens are integers 0, ..., K - 1 and the mask is K. Everything is float64.
"""

import itertools
import math

import torch
from torch import nn

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------------------------------------------------
# D3PM


def uniform_q(beta: float, K: int) -> torch.Tensor:
    return (1.0 - beta) * torch.eye(K) + beta / K * torch.ones(K, K)


def absorbing_q(beta: float, K: int) -> torch.Tensor:
    Q = (1.0 - beta) * torch.eye(K + 1)
    Q[:, K] += beta
    return Q


def uniform_qbar(alpha_bar: float, K: int) -> torch.Tensor:
    return alpha_bar * torch.eye(K) + (1.0 - alpha_bar) / K * torch.ones(K, K)


def absorbing_qbar(alpha_bar: float, K: int) -> torch.Tensor:
    return absorbing_q(1.0 - alpha_bar, K)


def cumulative(Qs: list[torch.Tensor]) -> list[torch.Tensor]:
    out = [torch.eye(Qs[0].shape[0])]
    for Q in Qs:
        out.append(out[-1] @ Q)
    return out


def one_hot(z, n: int) -> torch.Tensor:
    return nn.functional.one_hot(torch.as_tensor(z), n).to(torch.get_default_dtype())


def posterior(xt: torch.Tensor, x0: torch.Tensor, Qt: torch.Tensor, Qbar_prev: torch.Tensor) -> torch.Tensor:
    """eq-d3pm-posterior with a one-hot x0, or the reverse step of eq-x0-param with a probability vector."""
    num = (xt @ Qt.T) * (x0 @ Qbar_prev)
    return num / num.sum(-1, keepdim=True)


def multinomial_posterior(xt, x0, alpha_t: float, alpha_bar_prev: float, K: int) -> torch.Tensor:
    th = (alpha_t * xt + (1.0 - alpha_t) / K) * (alpha_bar_prev * x0 + (1.0 - alpha_bar_prev) / K)
    return th / th.sum(-1, keepdim=True)


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pos = p > 0
    return torch.where(
        pos,
        p * (torch.log(torch.where(pos, p, 1.0)) - torch.log(torch.where(pos, q, 1.0))),
        0.0,
    ).sum(-1)


# ---------------------------------------------------------------------------------------------------------------------
# Masked diffusion


def alpha(t, kind: str = "linear") -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.get_default_dtype())
    if kind == "linear":
        return 1.0 - t
    if kind == "cosine":
        return 1.0 - torch.sin(0.5 * math.pi * t)  # = 1 - cos(pi (1 - t) / 2), exact at t = 0 and 1
    raise ValueError(kind)


def dalpha(t, kind: str = "linear") -> torch.Tensor:
    t = torch.as_tensor(t, dtype=torch.get_default_dtype())
    if kind == "linear":
        return -torch.ones_like(t)
    if kind == "cosine":
        return -0.5 * math.pi * torch.cos(0.5 * math.pi * t)
    raise ValueError(kind)


def weight(t, kind: str = "linear") -> torch.Tensor:
    """-alpha_t' / (1 - alpha_t), the weight of eq-mdlm."""
    return -dalpha(t, kind) / (1.0 - alpha(t, kind))


def mask_tokens(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, K: int, kind: str = "linear") -> torch.Tensor:
    return torch.where(u < (1.0 - alpha(t, kind))[:, None], torch.full_like(x, K), x)


def subs_probs(logits: torch.Tensor, z: torch.Tensor, K: int) -> torch.Tensor:
    """SUBS: softmax over the data tokens where z is masked, the one-hot of z elsewhere."""
    sm = torch.softmax(logits, -1)
    keep = one_hot(torch.where(z == K, 0, z), K)
    return torch.where((z == K)[..., None], sm, keep)


def mdlm_estimate(
    logp: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    K: int,
    kind: str = "linear",
):
    nll = -logp.gather(-1, x[..., None])[..., 0]
    return weight(t, kind) * torch.where(z == K, nll, 0.0).sum(-1)


def mask_coefficient(m: int, L: int) -> float:
    return math.factorial(L - m) * math.factorial(m - 1) / math.factorial(L)


def all_masks(L: int) -> torch.Tensor:
    return torch.tensor([m for m in itertools.product([False, True], repeat=L) if any(m)])


def exact_nelbo(denoise, x: torch.Tensor, K: int) -> torch.Tensor:
    """eq-mdlm for one sequence, exactly: the sum over masked sets weighted by mask_coefficient."""
    L = x.shape[0]
    total = torch.zeros(())
    for m in all_masks(L):
        z = torch.where(m, K, x)
        pr = denoise(z)
        total = total + mask_coefficient(int(m.sum()), L) * -torch.log(pr[m, x[m]]).sum()
    return total


def conditional_denoiser(p: torch.Tensor, K: int):
    """The exact denoiser of a joint p of shape (K,) * L."""
    L = p.dim()

    def denoise(z: torch.Tensor) -> torch.Tensor:
        idx = tuple(slice(None) if int(zi) == K else int(zi) for zi in z)
        sub = p[idx]
        free = [i for i in range(L) if int(z[i]) == K]
        out = one_hot(torch.where(z == K, 0, z), K)
        for j, i in enumerate(free):
            dims = tuple(a for a in range(len(free)) if a != j)
            marg = sub.sum(dim=dims) if dims else sub
            out[i] = marg / marg.sum()
        return out

    return denoise


def markov_data(L: int, K: int, stay: float = 0.1, step: float = 0.8) -> torch.Tensor:
    if K < 3:
        raise ValueError("markov_data needs K >= 3, so that a third token takes the remaining mass")
    T = torch.full((K, K), (1.0 - stay - step) / max(K - 2, 1))
    for a in range(K):
        T[a, a] = stay
        T[a, (a + 1) % K] = step
    p = torch.zeros((K,) * L)
    for x in itertools.product(range(K), repeat=L):
        p[x] = math.prod(float(T[x[i], x[i + 1]]) for i in range(L - 1)) / K
    return p


# ---------------------------------------------------------------------------------------------------------------------
# Sampling


def categorical(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    c = probs.cumsum(-1)
    return torch.clamp((c < u[..., None]).sum(-1), max=probs.shape[-1] - 1)


def ancestral_step(z, probs, a_s: float, a_t: float, u, g, K: int) -> torch.Tensor:
    tok = categorical(probs, g)
    unmask = (z == K) & (u < (a_s - a_t) / (1.0 - a_t))
    return torch.where(unmask, tok, z)


def sampler_distribution(denoise, L: int, K: int, T: int, kind: str = "linear") -> torch.Tensor:
    """The exact distribution of ancestral sampling with T steps, over K^L."""
    states = {tuple([K] * L): 1.0}
    for i in range(T, 0, -1):
        a_s, a_t = float(alpha((i - 1) / T, kind)), float(alpha(i / T, kind))
        r = (a_s - a_t) / (1.0 - a_t)
        nxt: dict[tuple[int, ...], float] = {}
        for z, pz in states.items():
            pr = denoise(torch.tensor(z))
            opts = [[(z[j], 1.0)] if z[j] != K else [(K, 1.0 - r)] + [(y, r * float(pr[j, y])) for y in range(K)]
                    for j in range(L)]  # fmt: skip
            for combo in itertools.product(*opts):
                w = pz * math.prod(c[1] for c in combo)
                if w > 0:
                    key = tuple(c[0] for c in combo)
                    nxt[key] = nxt.get(key, 0.0) + w
        states = nxt
    out = torch.zeros((K,) * L)
    for z, pz in states.items():
        out[z] += pz
    return out


def maskgit_counts(N: int, T: int) -> list[int]:
    out, prev = [], N
    for t in range(1, T + 1):
        n = min(math.ceil(math.cos(0.5 * math.pi * t / T) * N - 1e-9), prev - 1) if prev > 0 else 0
        n = max(n, 0)
        out.append(n)
        prev = n
    return out


def maskgit_decode(denoise_batch, B: int, L: int, K: int, T: int, u: torch.Tensor):
    z = torch.full((B, L), K)
    history = [z.clone()]
    for t, n in enumerate(maskgit_counts(L, T)):
        pr = denoise_batch(z)
        tok = categorical(pr, u[t])
        conf = torch.where(z == K, pr.gather(-1, tok[..., None])[..., 0], torch.inf)
        filled = torch.where(z == K, tok, z)
        order = torch.sort(conf, dim=-1, stable=True).indices
        remask = torch.zeros_like(z, dtype=torch.bool)
        remask.scatter_(-1, order[:, :n], True)
        z = torch.where(remask, K, filled)
        history.append(z.clone())
    return z, history


# ---------------------------------------------------------------------------------------------------------------------
# Score entropy


def concrete_score(p: torch.Tensor) -> torch.Tensor:
    return p[None, :] / p[:, None]


def score_entropy(s: torch.Tensor, p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r = concrete_score(p)
    off = ~torch.eye(p.shape[0], dtype=torch.bool)
    terms = w * (s - r * torch.log(s) + r * (torch.log(r) - 1.0))
    return (p[:, None] * torch.where(off, terms, 0.0)).sum()


def denoising_score_entropy(s: torch.Tensor, p0: torch.Tensor, kern: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    off = ~torch.eye(kern.shape[1], dtype=torch.bool)
    r = kern[:, None, :] / kern[:, :, None]  # [x0, x, y]
    inner = torch.where(off, w * (s - r * torch.log(s)), 0.0)
    return (p0[:, None, None] * kern[:, :, None] * inner).sum()


# ---------------------------------------------------------------------------------------------------------------------
# The denoiser


def config(L: int, K: int, H: int) -> dict[str, int]:
    return {"seq_len": L, "vocab": K, "hidden": H}


def param_shapes(cfg: dict[str, int]) -> dict[str, tuple[int, ...]]:
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return {"W1": (L * (K + 1), H), "b1": (H,), "W2": (H, L * K), "b2": (L * K,)}


def param_count(cfg: dict[str, int]) -> int:
    return sum(math.prod(s) for s in param_shapes(cfg).values())


def flops_forward(cfg: dict[str, int], B: int) -> int:
    L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
    return 2 * B * L * (K + 1) * H + 2 * B * H * L * K


class Denoiser(nn.Module):
    """Logits over the K data tokens at every position from the one-hot of a partly masked sequence."""

    def __init__(self, cfg: dict[str, int], generator: torch.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        L, K, H = cfg["seq_len"], cfg["vocab"], cfg["hidden"]
        self.W1 = nn.Parameter(torch.randn(L * (K + 1), H, generator=generator) / math.sqrt(L))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.randn(H, L * K, generator=generator) / math.sqrt(H))
        self.b2 = nn.Parameter(torch.zeros(L * K))

    @classmethod
    def from_params(cls, params: dict, cfg: dict[str, int]) -> "Denoiser":
        m = cls(cfg)
        with torch.no_grad():
            for k, v in params.items():
                getattr(m, k).copy_(torch.as_tensor(v))
        return m

    def named(self) -> dict[str, nn.Parameter]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        L, K = self.cfg["seq_len"], self.cfg["vocab"]
        x = one_hot(z, K + 1).reshape(z.shape[0], L * (K + 1))
        h = torch.tanh(x @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).reshape(z.shape[0], L, K)

    def denoise(self, z: torch.Tensor) -> torch.Tensor:
        """The SUBS probabilities for one (L,) sequence."""
        with torch.no_grad():
            return subs_probs(self(z[None]), z[None], self.cfg["vocab"])[0]


def weighted_ce(model: Denoiser, Z: torch.Tensor, X: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(model(Z), -1)
    return (Wt * -lp.gather(-1, X[..., None])[..., 0]).sum()


def mc_batch(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, K: int, kind: str = "linear"):
    z = mask_tokens(x, t, u, K, kind)
    return z, x, torch.where(z == K, weight(t, kind)[:, None], 0.0) / x.shape[0]


def exact_batch(data_p: torch.Tensor, K: int):
    L = data_p.dim()
    seqs = torch.tensor(list(itertools.product(range(K), repeat=L)))
    masks = all_masks(L)
    Z, X, Wt = [], [], []
    for x in seqs:
        px = float(data_p[tuple(x.tolist())])
        for m in masks:
            Z.append(torch.where(m, K, x))
            X.append(x)
            Wt.append(torch.where(m, px * mask_coefficient(int(m.sum()), L), 0.0))
    return torch.stack(Z), torch.stack(X), torch.stack(Wt)


def train_exact(model: Denoiser, data_p: torch.Tensor, steps: int, lr: float = 0.01) -> list[float]:
    Z, X, Wt = exact_batch(data_p, model.cfg["vocab"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for _ in range(steps):
        opt.zero_grad()
        loss = weighted_ce(model, Z, X, Wt)
        loss.backward()
        opt.step()
        trace.append(loss.item())
    return trace


def entropy(p: torch.Tensor) -> float:
    q = p[p > 0]
    return float(-(q * q.log()).sum())
