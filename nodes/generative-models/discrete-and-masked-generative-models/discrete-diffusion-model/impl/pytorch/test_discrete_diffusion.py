# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/discrete-diffusion-model/ (ai://NEW-discrete-diffusion-model).
"""Tests of the PyTorch version: the shared fixture with the gradient of the one-sample bound from autograd, the counts
of shapes.json, gradcheck through the weighted cross-entropy, the worked example, the D3PM posterior against Bayes'
rule and the multinomial diffusion posterior, the exact bound against the any-order chain rule, the ancestral sampler,
MaskGIT's decoding, score entropy with its autograd gradient vanishing at the true ratios, and a denoiser trained by
the exact bound with torch.optim.Adam."""

import itertools
import json
import math
import pathlib

import numpy as np
import pytest
import torch

import discrete_diffusion as dd

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c):
    k = c["config"]
    return dd.config(k["seq_len"], k["vocab"], k["hidden"])


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount: logits for B partly masked sequences."""
    cfg, k = cfg_of(c), c["config"]
    g = torch.Generator().manual_seed(0)
    z = torch.randint(0, k["vocab"] + 1, (k["B"], k["seq_len"]), generator=g)
    return dd.Denoiser(cfg, g), (z,)


def load(name):
    return {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / name).items()}


def test_shared_fixture():
    P, inp, want = (
        load("params-small.npz"),
        load("input-small.npz"),
        load("expected-small.npz"),
    )
    L, K, H = (int(v) for v in inp["cfg"])
    model = dd.Denoiser.from_params(P, dd.config(L, K, H))
    z, x, t, u = inp["z"], inp["x"], inp["t"], inp["u"]
    with torch.no_grad():
        torch.testing.assert_close(model(z), want["logits"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(dd.subs_probs(want["logits"], z, K), want["probs"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(dd.mask_tokens(x, t, u, K, "cosine"), want["z_masked"])
    Z, X, Wt = dd.mc_batch(x, t, u, K, "cosine")
    loss = dd.weighted_ce(model, Z, X, Wt)
    assert loss.item() == pytest.approx(float(want["mc_loss"]), rel=1e-12)
    loss.backward()
    for k, prm in model.named().items():
        assert prm.grad is not None
        torch.testing.assert_close(prm.grad, want[f"d{k}"], rtol=1e-10, atol=1e-13, msg=k)
    with torch.no_grad():
        lp = torch.log_softmax(model(Z), -1)
    torch.testing.assert_close(
        dd.mdlm_estimate(lp, Z, X, t, K, "cosine"),
        want["per_seq"],
        rtol=1e-12,
        atol=1e-13,
    )
    rev = dd.posterior(
        dd.one_hot(int(inp["xt"]), 4),
        inp["p0"],
        dd.uniform_q(0.3, 4),
        dd.uniform_qbar(0.6, 4),
    )
    torch.testing.assert_close(rev, want["reverse_uniform"], rtol=1e-12, atol=1e-13)
    assert float(dd.exact_nelbo(model.denoise, x[0], K)) == pytest.approx(float(want["nelbo_x0"]), rel=1e-12)
    s = inp["scores"]
    se = dd.score_entropy(s, inp["p_se"], torch.ones_like(s))
    assert float(se) == pytest.approx(float(want["score_entropy"]), rel=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    k, mod = c["config"], c["model"]
    cfg = cfg_of(c)
    assert dd.param_count(cfg) == mod["params"]
    assert dd.flops_forward(cfg, k["B"]) == mod["flops_forward"]
    model = dd.Denoiser(cfg, torch.Generator().manual_seed(0))
    assert sum(p.numel() for p in model.parameters()) == mod["params"]
    z = torch.full((k["B"], k["seq_len"]), k["vocab"])
    with torch.no_grad():
        assert model(z).shape == (k["B"], k["seq_len"], k["vocab"])


def test_gradcheck_through_the_bound():
    K, L = 3, 3
    model = dd.Denoiser(dd.config(L, K, 4), torch.Generator().manual_seed(1))
    Z, X, Wt = dd.exact_batch(dd.markov_data(L, K), K)

    def g(*ps):
        """The loss as a pure function of the parameters, so that gradcheck can perturb them."""
        x = dd.one_hot(Z, K + 1).reshape(Z.shape[0], -1)
        lg = (torch.tanh(x @ ps[0] + ps[1]) @ ps[2] + ps[3]).reshape(Z.shape[0], L, K)
        return (Wt * -torch.log_softmax(lg, -1).gather(-1, X[..., None])[..., 0]).sum()

    ps = tuple(p.detach().clone().requires_grad_(True) for p in model.named().values())
    assert g(*ps).item() == pytest.approx(dd.weighted_ce(model, Z, X, Wt).item(), rel=1e-14)
    assert torch.autograd.gradcheck(g, ps)


def test_worked_example():
    p = torch.tensor([[0.4, 0.1], [0.1, 0.4]])
    den = dd.conditional_denoiser(p, 2)
    assert float(dd.exact_nelbo(den, torch.tensor([0, 0]), 2)) == pytest.approx(-math.log(0.4), rel=1e-14)
    torch.testing.assert_close(dd.sampler_distribution(den, 2, 2, 1), torch.full((2, 2), 0.25))
    assert float(dd.sampler_distribution(den, 2, 2, 2)[0, 0]) == pytest.approx(0.325, abs=1e-12)


def test_posterior_is_bayes_rule_and_matches_multinomial_diffusion():
    K, betas = 4, [0.2, 0.3, 0.35]
    Qs = [dd.uniform_q(b, K) for b in betas]
    Qbar = dd.cumulative(Qs)
    abar = np.cumprod([1 - b for b in betas])
    for k, b in enumerate(abar):
        torch.testing.assert_close(Qbar[k + 1], dd.uniform_qbar(float(b), K))
    for x0, xt in itertools.product(range(K), repeat=2):
        joint = Qbar[1][x0, :] * Qs[1][:, xt]
        got = dd.posterior(dd.one_hot(xt, K), dd.one_hot(x0, K), Qs[1], Qbar[1])
        torch.testing.assert_close(got, joint / joint.sum(), rtol=0, atol=1e-15)
    g = torch.Generator().manual_seed(2)
    for _ in range(5):
        xt = dd.one_hot(int(torch.randint(K, (), generator=g)), K)
        p0 = torch.softmax(torch.randn(K, generator=g), -1)
        torch.testing.assert_close(dd.posterior(xt, p0, Qs[2], Qbar[2]),
                                   dd.multinomial_posterior(xt, p0, 1 - betas[2], float(abar[1]), K))  # fmt: skip
    Qa = [dd.absorbing_q(b, K) for b in betas]
    torch.testing.assert_close(dd.cumulative(Qa)[3], dd.absorbing_qbar(float(abar[2]), K))


@pytest.mark.parametrize("L", [2, 3])
def test_nelbo_is_the_any_order_chain_rule(L):
    K = 3
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)
    for x in itertools.product(range(K), repeat=L):
        assert float(dd.exact_nelbo(den, torch.tensor(x), K)) == pytest.approx(-math.log(float(p[x])), rel=1e-12)
    for m in range(1, L + 1):  # the coefficient is the integral of eq-mdlm over t for m masked tokens
        a = torch.linspace(0, 1, 200001)
        num = torch.trapezoid(a ** (L - m) * (1 - a) ** (m - 1), a)
        assert dd.mask_coefficient(m, L) == pytest.approx(float(num), rel=1e-6)


def test_ancestral_sampler_and_maskgit():
    K, L = 3, 3
    p = dd.markov_data(L, K)
    den = dd.conditional_denoiser(p, K)
    marg = [p.sum(dim=tuple(j for j in range(L) if j != i)) for i in range(L)]
    torch.testing.assert_close(dd.sampler_distribution(den, L, K, 1), torch.einsum("a,b,c->abc", *marg))
    tv = [0.5 * float((dd.sampler_distribution(den, L, K, T) - p).abs().sum()) for T in (1, 2, 4, 8)]
    assert all(a > b for a, b in itertools.pairwise(tv))
    assert dd.maskgit_counts(16, 8)[-1] == 0
    B, T = 40, 3

    def den_b(z):
        return torch.stack([den(r) for r in z])

    u = torch.rand(T, B, L, generator=torch.Generator().manual_seed(3))
    z, hist = dd.maskgit_decode(den_b, B, L, K, T, u)
    assert bool((z < K).all())
    for n, (a, b) in zip(dd.maskgit_counts(L, T), itertools.pairwise(hist), strict=True):
        assert bool(((b == K).sum(1) == n).all())
        fixed = a != K
        assert torch.equal(b[fixed], a[fixed])


def test_score_entropy_and_its_autograd_gradient():
    g = torch.Generator().manual_seed(4)
    N = 5
    p = torch.softmax(torch.randn(N, generator=g), -1)
    w = 0.5 + 1.5 * torch.rand(N, N, generator=g)
    r = dd.concrete_score(p)
    assert float(dd.score_entropy(r, p, w)) == pytest.approx(0.0, abs=1e-14)
    for _ in range(10):
        assert float(dd.score_entropy(torch.exp(torch.randn(N, N, generator=g)), p, w)) > 0
    p0 = torch.softmax(torch.randn(N, generator=g), -1)
    kern = torch.softmax(torch.randn(N, N, generator=g), -1)
    s = dd.concrete_score(p0 @ kern).clone().requires_grad_(True)
    dd.denoising_score_entropy(s, p0, kern, w).backward()
    assert s.grad is not None
    off = ~torch.eye(N, dtype=torch.bool)
    torch.testing.assert_close(s.grad[off], torch.zeros(N * N - N), rtol=0, atol=1e-12)
    s2 = torch.exp(torch.randn(N, N, generator=g))
    d1 = dd.score_entropy(s2, p0 @ kern, w) - dd.denoising_score_entropy(s2, p0, kern, w)
    d2 = dd.score_entropy(s2 * 1.7, p0 @ kern, w) - dd.denoising_score_entropy(s2 * 1.7, p0, kern, w)
    assert float(d1) == pytest.approx(float(d2), rel=1e-10)


def test_trained_denoiser_reaches_the_entropy():
    K, L = 3, 4
    p = dd.markov_data(L, K)
    torch.manual_seed(0)
    model = dd.Denoiser(dd.config(L, K, 32), torch.Generator().manual_seed(1))
    trace = dd.train_exact(model, p, 600, lr=0.01)
    H = dd.entropy(p)
    assert trace[0] > H + 1.0
    assert trace[-1] >= H - 1e-9 and trace[-1] < H + 0.01
    tv16 = 0.5 * float((dd.sampler_distribution(model.denoise, L, K, 16) - p).abs().sum())
    assert tv16 < 0.06
