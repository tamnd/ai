# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/linear-attention/ (ai://JBSH5S).
"""Tests of the PyTorch linear attention: the shared fixture and its gradients, Algorithm 1 as an autograd Function
against autograd through the materialized states, the three causal implementations, the layer as an RNN, and a small
copy task on which linear attention runs out of room before softmax attention does."""

import pathlib

import numpy as np
import pytest
import torch

import linear_attention as M

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())


def fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIX / "input-small.npz").items()}
    return params, inp, np.load(FIX / "expected-small.npz")


@pytest.mark.parametrize("causal", [True, False])
def test_fixture(causal):
    params, inp, exp = fixture()
    tag = "causal" if causal else "full"
    model = M.LinearAttention.from_params(params, causal=causal)
    x = inp["x"].clone().requires_grad_(True)
    out = model(x)
    assert np.allclose(out.detach().numpy(), exp["out_" + tag], atol=1e-12)
    loss = 0.5 * ((out - inp["target"]) ** 2).mean()
    assert np.isclose(loss.item(), exp["loss_" + tag], atol=1e-12)
    loss.backward()
    for k, t in model.named_parameters():
        assert t.grad is not None and np.allclose(t.grad.numpy(), exp[f"d{k}_{tag}"], atol=1e-12), k
    assert x.grad is not None and np.allclose(x.grad.numpy(), exp["dx_" + tag], atol=1e-12)
    soft = M.LinearAttention.from_params(params, kind="softmax")
    assert np.allclose(soft(inp["x"]).detach().numpy(), exp["out_softmax_causal"], atol=1e-12)


def test_algorithm_1_gradient():
    """The hand-written backward of CausalProduct against autograd through the cumulative sum of the states, and
    torch.autograd.gradcheck on top."""
    g = torch.Generator().manual_seed(0)
    fQ, fK = (M.elu_plus_one(torch.randn(2, 3, 9, 4, generator=g, dtype=torch.double)) for _ in range(2))
    V = torch.randn(2, 3, 9, 5, generator=g, dtype=torch.double)
    G = torch.randn(2, 3, 9, 5, generator=g, dtype=torch.double)
    grads = []
    for f in (M.CausalProduct.apply, M.causal_product_cumsum):
        args = [t.clone().requires_grad_(True) for t in (fQ, fK, V)]
        (f(*args) * G).sum().backward()
        grads.append([a.grad for a in args])
    for a, b in zip(*grads, strict=True):
        assert torch.allclose(a, b, atol=1e-12)
    small = [t[:1, :1, :5].clone().requires_grad_(True) for t in (fQ, fK, V)]
    assert torch.autograd.gradcheck(M.CausalProduct.apply, small)


def test_algorithm_1_keeps_only_its_inputs():
    """What each causal implementation leaves for the backward pass: Algorithm 1 saves phi(Q), phi(K) and V, the
    cumulative sum keeps an (N, C, M) tensor of states per head."""
    saved = []

    def pack(t):
        saved.append(t.numel())
        return t

    N, C, Mv = 64, 8, 8
    fQ, fK, V = (torch.rand(1, 1, N, d, requires_grad=True) for d in (C, C, Mv))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        M.CausalProduct.apply(fQ, fK, V)
    assert sum(saved) == N * (C + C + Mv)
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        M.causal_product_cumsum(fQ, fK, V)
    assert max(saved) == N * C * Mv


def test_the_three_causal_forms_agree():
    g = torch.Generator().manual_seed(1)
    fQ, fK = (M.elu_plus_one(torch.randn(2, 10, 4, generator=g, dtype=torch.double)) for _ in range(2))
    V = torch.randn(2, 10, 3, generator=g, dtype=torch.double)
    a, b, c = (M.linear_attention(fQ, fK, V, True, impl) for impl in ("algorithm1", "cumsum", "quadratic"))
    assert torch.allclose(a, b) and torch.allclose(a, c)


def test_the_layer_is_an_rnn():
    """eq-rnn: one token at a time through step gives the causal forward pass."""
    torch.manual_seed(2)
    layer = M.LinearAttention(8, 2).double()
    x = torch.randn(3, 11, 8, dtype=torch.double)
    state, outs = layer.initial_state(3), []
    with torch.no_grad():
        for i in range(11):
            y, state = layer.step(state, x[:, i])
            outs.append(y)
        assert torch.allclose(torch.stack(outs, 1), layer(x))
    assert [tuple(s.shape) for s in state] == [(3, 2, 4, 4), (3, 2, 4)]


def copy_batch(g, n, L, V=10):
    """L random symbols from 1..V-1, a separator V, then the same L symbols; the loss is on the copy."""
    x = torch.randint(1, V, (n, L), generator=g)
    seq = torch.cat([x, torch.full((n, 1), V), x], 1)
    return seq[:, :-1], seq[:, 1:]


def train_copy(kind, L, d_model, n_heads, steps=800, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = M.TinyLM(11, d_model, n_heads, 2, 2 * L + 1, kind)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = copy_batch(g, 32, L)
        loss = torch.nn.functional.cross_entropy(model(x)[:, L:].reshape(-1, 11), y[:, L:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = copy_batch(torch.Generator().manual_seed(99), 256, L)
    with torch.no_grad():
        return (model(x)[:, L:].argmax(-1) == y[:, L:]).float().mean().item()


def test_linear_attention_runs_out_of_room_on_a_long_copy():
    """Copy 32 symbols from 9 with a 2-layer model of width 16 and 2 heads of width 8, 800 steps of Adam on batches of
    32. Over the seeds 0 to 2, softmax attention copied 0.9999 to 1.0 of 256 fresh sequences and linear attention 0.36
    to 0.43, chance being 1/9; with 1 head of width 16 linear got 0.40 to 0.45. Copying 8 symbols, linear got 1.0 (seed 0),
    16 symbols 0.78 to 0.89, and with width 32 and 1 head of width 32 it copied 32 symbols at 1.0."""
    torch.set_num_threads(1)
    soft, lin = train_copy("softmax", 32, 16, 2), train_copy("linear", 32, 16, 2)
    assert soft > 0.95 > 0.6 > lin > 0.2, (soft, lin)
    assert train_copy("linear", 8, 16, 2) > 0.99
    assert train_copy("linear", 32, 32, 1) > 0.95
