# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/qk-layernorm/ (ai://NEW-qk-layernorm).
"""Tests of the PyTorch reference: the worked example, agreement with torch, the bound, the z-loss gradient."""

import math

import pytest
import torch
import torch.nn.functional as F

from qk_layernorm import (
    QKNormAttention,
    attention,
    attention_entropy,
    logit_bound,
    max_logit,
    merge_heads,
    split_heads,
    z_loss,
    z_loss_grad,
)

torch.set_default_dtype(torch.float64)

# The worked example of the page, as in the NumPy tests.
X = [[1.0, 3.0, 1.0, 3.0], [5.0, 5.0, 1.0, 1.0], [3.0, 1.0, 3.0, 1.0]]
RAW_LOGITS = [[10.0, 12.0, 6.0], [12.0, 26.0, 12.0], [6.0, 12.0, 10.0]]
QK_LOGITS = [[2.0, 0.0, -2.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 2.0]]
Y = [0.0, math.log(2.0), math.log(3.0), math.log(2.0)]
P_Y = [0.125, 0.25, 0.375, 0.25]


def t(a):
    return torch.tensor(a, dtype=torch.float64)


def test_worked_example():
    x, eye = t(X), torch.eye(4)
    raw = attention(x, eye, eye, eye)
    torch.testing.assert_close(raw["logits"][0], t(RAW_LOGITS))
    qk = attention(x, eye, eye, eye, qk_norm=True)
    torch.testing.assert_close(qk["logits"][0], t(QK_LOGITS))
    w = qk["weights"][0]
    torch.testing.assert_close(w[0], t([0.8668, 0.1173, 0.0159]), atol=1e-4, rtol=0)
    torch.testing.assert_close(attention_entropy(w), t([0.4411, 0.6656, 0.4411]), atol=1e-4, rtol=0)
    torch.testing.assert_close(qk["out"], w @ x)
    big = attention(x, 10 * eye, 10 * eye, eye)
    torch.testing.assert_close(big["logits"][0], 100 * t(RAW_LOGITS))
    torch.testing.assert_close(attention_entropy(big["weights"][0]), torch.zeros(3), atol=1e-12, rtol=0)
    big_qk = attention(x, 10 * eye, 10 * eye, eye, qk_norm=True)
    torch.testing.assert_close(big_qk["logits"][0], t(QK_LOGITS))


def test_z_loss_worked_example():
    y = t(Y)
    assert float(z_loss(y)) == pytest.approx(1e-4 * math.log(8.0) ** 2, rel=1e-12)
    torch.testing.assert_close(z_loss_grad(y), 2e-4 * math.log(8.0) * t(P_Y))
    assert float(z_loss(y - math.log(8.0))) == pytest.approx(0.0, abs=1e-30)


@pytest.mark.parametrize("causal", [False, True])
def test_agrees_with_scaled_dot_product_attention(causal):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 6, 8, generator=g)
    wq, wk, wv = (torch.randn(8, 8, generator=g) for _ in range(3))
    gq, gk = torch.randn(4, generator=g), torch.randn(4, generator=g)
    r = attention(
        x,
        wq,
        wk,
        wv,
        heads=2,
        qk_norm=True,
        gain_q=gq,
        gain_k=gk,
        eps=1e-6,
        causal=causal,
    )
    q = F.layer_norm(split_heads(x @ wq, 2), (4,), gq, None, 1e-6)
    k = F.layer_norm(split_heads(x @ wk, 2), (4,), gk, None, 1e-6)
    ref = F.scaled_dot_product_attention(q, k, split_heads(x @ wv, 2), is_causal=causal)
    torch.testing.assert_close(r["out"], merge_heads(ref))
    plain = attention(x, wq, wk, wv, heads=2, causal=causal)
    ref2 = F.scaled_dot_product_attention(
        split_heads(x @ wq, 2),
        split_heads(x @ wk, 2),
        split_heads(x @ wv, 2),
        is_causal=causal,
    )
    torch.testing.assert_close(plain["out"], merge_heads(ref2))
    if causal:
        assert torch.all(r["weights"].triu(1) == 0)


def test_bound_and_tightness():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 8, generator=g)
    w = 1e3 * torch.randn(8, 8, generator=g)
    r = attention(x, w, w, w, heads=2, qk_norm=True)
    assert max_logit(r["logits"]) <= logit_bound(4) + 1e-9
    torch.testing.assert_close(torch.diagonal(r["logits"], dim1=-2, dim2=-1), torch.full((2, 5), 2.0))
    gq, gk = torch.full((4,), 1.5), torch.full((4,), -2.0)
    r2 = attention(x, w, w, w, heads=2, qk_norm=True, gain_q=gq, gain_k=gk)
    assert max_logit(r2["logits"]) == pytest.approx(logit_bound(4, gq, gk), rel=1e-12)
    assert logit_bound(4, gq, gk) == pytest.approx(6.0)
    assert logit_bound(16) == pytest.approx(4.0)
    assert logit_bound(16, None, [0.5, -2.5]) == pytest.approx(10.0)
    raw = attention(x, w, w, w, heads=2)
    assert max_logit(raw["logits"]) > 1e3 * logit_bound(4)


def test_eps_shrinks_the_norm():
    x, eye = t([[1.0, 3.0, 1.0, 3.0]]), torch.eye(4)
    r = attention(x, eye, eye, eye, qk_norm=True, eps=1.0)
    assert float(r["logits"][0, 0, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(3))
def test_z_loss_gradient_matches_autograd(seed):
    g = torch.Generator().manual_seed(seed)
    z = (torch.randn(3, 5, generator=g) * 4 + 2).requires_grad_(True)
    loss = z_loss(z, alpha=1e-2)
    (grad,) = torch.autograd.grad(loss, z)
    torch.testing.assert_close(grad, z_loss_grad(z.detach(), alpha=1e-2))
    assert float(loss.detach()) == pytest.approx(1e-2 * float(torch.logsumexp(z.detach(), -1).pow(2).mean()), rel=1e-12)
    assert torch.autograd.gradcheck(lambda u: z_loss(u, alpha=1e-2), (z.detach().clone().requires_grad_(True),))


def test_entropy():
    torch.testing.assert_close(attention_entropy(torch.full((8,), 1 / 8)), torch.tensor(math.log(8)))
    p = t([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    torch.testing.assert_close(attention_entropy(p), t([math.log(2), 1.5 * math.log(2)]))
    torch.testing.assert_close(attention_entropy(p.T, dim=0), t([math.log(2), 1.5 * math.log(2)]))


def test_module_trains_and_keeps_the_bound():
    torch.manual_seed(0)
    m = QKNormAttention(8, 2, qk_norm=True, eps=0.0)
    assert m.gain_q is not None
    assert m.gain_q.shape == (4,)
    x = torch.randn(3, 6, 8)
    with torch.no_grad():
        m.wq.mul_(1e3)
        m.wk.mul_(1e3)
    r = m(x, causal=True)
    assert max_logit(r["logits"]) <= logit_bound(4) + 1e-9
    r["out"].pow(2).mean().backward()
    assert m.gain_q.grad is not None and m.wq.grad is not None
    plain = QKNormAttention(8, 2, qk_norm=False)
    assert plain.gain_q is None
    assert plain(x)["out"].shape == (3, 6, 8)
    with pytest.raises(ValueError):
        QKNormAttention(9, 2)
    with pytest.raises(ValueError):
        split_heads(x, 3)
