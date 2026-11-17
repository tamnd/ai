# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/attention-mechanism/ (ai://NEW-attention-mechanism).
"""Tests of the PyTorch RNNsearch and Luong attention: the fixtures and their gradients, the Luong layer against the
NumPy reference, and a small version of the bottleneck of Bahdanau et al.: reversing a 12-word sequence, which the
model with attention learns and the same model with a fixed context does not, with alignments on the anti-diagonal."""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

import attention as M

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE.parents[1] / "fixtures"


def numpy_reference():
    spec = importlib.util.spec_from_file_location("attention_np", HERE.parent / "numpy" / "attention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIX / "input-small.npz").items()}
    return params, inp, np.load(FIX / "expected-small.npz")


def test_fixture():
    params, inp, exp = fixture()
    model = M.RNNsearch.from_params(M.config(), params)
    logits, alpha = model(inp["src"], inp["tgt"], inp["src_mask"])
    assert np.allclose(logits.detach().numpy(), exp["logits"], atol=1e-12)
    assert np.allclose(alpha.detach().numpy(), exp["alpha"], atol=1e-12)
    loss = M.nll(logits, inp["tgt"], inp["tgt_mask"])
    assert np.isclose(loss.item(), exp["loss"], atol=1e-12)
    loss.backward()
    for k, t in model.named_parameters():
        assert np.allclose(t.grad.numpy(), exp["d" + k], atol=1e-12), k
    plain = M.RNNsearch.from_params(M.config(attention=False), params)
    assert np.allclose(plain(inp["src"], inp["tgt"], inp["src_mask"])[0].detach().numpy(), exp["logits_encdec"])


def test_parameter_count():
    cfg = M.config(30000, 30000, 620, 1000, 1000, 500)
    assert sum(int(np.prod(s)) for s in M.shapes(cfg).values()) == 80_401_000
    assert sum(p.numel() for p in M.RNNsearch(M.config()).parameters()) == numpy_reference().param_count(M.config())


@pytest.mark.parametrize("score", ["dot", "general", "concat", "location"])
@pytest.mark.parametrize("local", [None, 2])
def test_luong_layer_matches_the_numpy_reference(score, local):
    ref = numpy_reference()
    torch.manual_seed(0)
    layer = M.LuongAttention(4, score, local, n_a=5, max_len=7).double()
    h, Hs = torch.randn(3, 4, dtype=torch.double), torch.randn(3, 7, 4, dtype=torch.double)
    lengths = torch.tensor([7, 5, 2])
    out, a, p_t = layer(h, Hs, lengths)
    P = {k: v.detach().numpy() for k, v in layer.named_parameters()}
    hn, Hn, ln = h.numpy(), Hs.numpy(), lengths.numpy()
    if local is None:
        want_a, c = ref.global_attention(score, hn, Hn, np.arange(7) < ln[:, None], P.get("W_a"), P.get("v_a"))
    else:
        want_p, want_a, c = ref.local_p_attention(score, hn, Hn, ln, local, P["W_p"], P["v_p"], P.get("W_a"),
                                                  P.get("v_a"))  # fmt: skip
        assert np.allclose(p_t.detach().numpy(), want_p)
    assert np.allclose(a.detach().numpy(), want_a)
    assert np.allclose(out.detach().numpy(), ref.attentional_state(c, hn, P["W_c"]))


def test_local_p_position_gets_a_gradient():
    """Luong et al.'s footnote 10: through the Gaussian of eq-local-p, backprop reaches W_p and v_p."""
    torch.manual_seed(1)
    layer = M.LuongAttention(4, "general", local_D=2).double()
    h, Hs = torch.randn(2, 4, dtype=torch.double), torch.randn(2, 9, 4, dtype=torch.double)
    layer(h, Hs, torch.tensor([9, 6]))[0].sum().backward()
    assert layer.W_p.grad.abs().sum() > 0 and layer.v_p.grad.abs().sum() > 0


def reversal(g, n, L, V):
    """Random words 1..V-1; the target is the source reversed, then <EOS> (0)."""
    x = torch.randint(1, V, (n, L), generator=g)
    return x, torch.cat([x.flip(1), torch.zeros(n, 1, dtype=torch.long)], 1)


def train_reversal(attention, steps=300, L=12, V=10, n=16, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = M.RNNsearch(M.config(V, V, 8, n, n, n // 2, attention))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = reversal(g, 32, L, V)
        loss = M.nll(model(x, y)[0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = reversal(torch.Generator().manual_seed(99), 256, L, V)
    ys, alpha = model.greedy(x, L)
    correct = (ys == y[:, :L]).float()
    on_diagonal = (alpha.argmax(-1) == torch.arange(L - 1, -1, -1)).float().mean().item()
    return correct.mean().item(), correct.mean(0), on_diagonal


def test_attention_removes_the_bottleneck():
    """Reversing 12 words from 9, hidden width 16, 300 steps of Adam on batches of 32. Over the seeds 0 to 3 RNNsearch
    got 0.996 to 1.0 of the words right and RNNencdec, the same network with c_i fixed, 0.29 to 0.32, chance being
    1/9. RNNencdec gets 0.63 to 0.80 of the first output word, the last one it read, and 0.17 to 0.31 of each word
    from the fifth on; the argmax of alpha_i of RNNsearch sits on source position L + 1 - i for 0.89 to 0.99 of the
    output words."""
    torch.set_num_threads(1)
    att, _, diag = train_reversal(True)
    enc, per_position, _ = train_reversal(False)
    assert att > 0.95 > 0.5 > enc, (att, enc)
    assert per_position[0] > 0.55 > 0.35 > per_position[4:].max(), per_position
    assert diag > 0.8, diag
