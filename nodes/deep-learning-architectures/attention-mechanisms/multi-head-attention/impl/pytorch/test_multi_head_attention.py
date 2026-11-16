# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multi-head-attention/ (ai://NEW-multi-head-attention).
"""Tests of the PyTorch reference: the shared fixture with autograd, the masks, the heads, and torch's own attention."""

import math
import pathlib

import numpy as np
import pytest
import torch

from multi_head_attention import MultiHeadAttention, attention, causal_mask, config, padding_mask, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    cfg = config(int(inp["d_model"]), int(inp["n_heads"]))
    return MultiHeadAttention.from_params(params, cfg), inp, want


@pytest.mark.parametrize("case", ["self", "cross"])
def test_shared_fixture_forward_and_autograd(case):
    model, inp, want = fixture()
    x = inp["x"].clone().requires_grad_(True)
    mem = None if case == "self" else inp["memory"].clone().requires_grad_(True)
    mask = inp["causal"] if case == "self" else inp["padding"]
    y, a = model(x, mem, mask)
    torch.testing.assert_close(y, want[f"{case}/y"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(a, want[f"{case}/a"], rtol=1e-10, atol=1e-12)
    (y * inp["dy"]).sum().backward()
    torch.testing.assert_close(x.grad, want[f"{case}/dx"], rtol=1e-10, atol=1e-12)
    if mem is not None:
        torch.testing.assert_close(mem.grad, want["cross/dmemory"], rtol=1e-10, atol=1e-12)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        torch.testing.assert_close(getattr(model, name).grad, want[f"{case}/d{name}"], rtol=1e-10, atol=1e-12)


def test_worked_example():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    eye = {n: np.eye(2) for n in ("W_q", "W_k", "W_v", "W_o")}
    y, a = MultiHeadAttention.from_params(eye, config(2, 2))(x)
    e = math.e
    torch.testing.assert_close(y[0, 0], torch.tensor([2 * e / (2 * e + 1), 2 / 3]))
    y, a = MultiHeadAttention.from_params(eye, config(2, 2))(x, mask=causal_mask(3))
    torch.testing.assert_close(y[0, 1], torch.tensor([0.5, e / (1 + e)]))


def test_rows_sum_to_one_and_the_causal_mask_hides_the_future():
    g = torch.Generator().manual_seed(1)
    model = MultiHeadAttention(config(8, 2), generator=g)
    x = torch.randn(2, 6, 8, generator=g)
    y, a = model(x, mask=causal_mask(6))
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 2, 6))
    assert torch.all(a[..., ~causal_mask(6)] == 0)
    x2 = x.clone()
    x2[:, 3:] += 1.0
    y2, _ = model(x2, mask=causal_mask(6))
    torch.testing.assert_close(y2[:, :3], y[:, :3])


def test_permutation_equivariance():
    g = torch.Generator().manual_seed(2)
    model = MultiHeadAttention(config(8, 4), generator=g)
    x = torch.randn(2, 5, 8, generator=g)
    perm = torch.randperm(5, generator=g)
    torch.testing.assert_close(model(x[:, perm])[0], model(x)[0][:, perm])


def test_all_masked_row_gives_zero_and_finite_gradient():
    q = torch.randn(1, 1, 2, 4, requires_grad=True)
    k, v = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)
    y, a = attention(q, k, v, torch.tensor([[True, False, True], [False, False, False]]))
    assert torch.all(y[0, 0, 1] == 0) and torch.all(a[0, 0, 1] == 0)
    y.sum().backward()
    assert torch.all(torch.isfinite(q.grad))


@pytest.mark.parametrize(("M", "H"), [(8, 1), (8, 2), (16, 4)])
def test_matches_torch_multihead_attention(M, H):
    g = torch.Generator().manual_seed(3)
    ours = MultiHeadAttention(config(M, H), generator=g)
    theirs = torch.nn.MultiheadAttention(M, H, bias=False, batch_first=True)
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([ours.W_q.T, ours.W_k.T, ours.W_v.T]))
        theirs.out_proj.weight.copy_(ours.W_o.T)
    x, mem = torch.randn(2, 5, M, generator=g), torch.randn(2, 4, M, generator=g)
    # torch's boolean attn_mask means "may not attend": the opposite of ours
    want, w = theirs(x, x, x, attn_mask=~causal_mask(5), average_attn_weights=False)
    got, a = ours(x, mask=causal_mask(5))
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(a, w)
    want, _ = theirs(x, mem, mem, key_padding_mask=~padding_mask([4, 2], 4)[:, 0, 0])
    torch.testing.assert_close(ours(x, mem, padding_mask([4, 2], 4))[0], want)


def test_matches_scaled_dot_product_attention():
    g = torch.Generator().manual_seed(4)
    q, k, v = (torch.randn(2, 3, 5, 4, generator=g) for _ in range(3))
    want = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(attention(q, k, v, causal_mask(5))[0], want)


def test_param_count():
    cfg = config(12, 3, 5, 7)
    assert param_count(cfg) == sum(p.numel() for p in MultiHeadAttention(cfg).parameters())


def test_config_defaults_and_errors():
    assert config() == {"d_model": 512, "n_heads": 8, "d_k": 64, "d_v": 64}
    assert config(8, 2, d_k=3, d_v=5) == {"d_model": 8, "n_heads": 2, "d_k": 3, "d_v": 5}
    for bad in ({"d_k": 0, "d_v": 4}, {"d_k": 4, "d_v": 0}):
        with pytest.raises(ValueError):
            config(8, 2, **bad)
    with pytest.raises(ValueError):
        config(10, 3)


def test_padding_mask_values():
    got = padding_mask([2, 0, 3], 3)
    assert got.shape == (3, 1, 1, 3)
    want = torch.tensor([[True, True, False], [False, False, False], [True, True, True]])
    assert torch.equal(got[:, 0, 0, :], want)


def test_shapes_and_init_scale_when_d_k_and_d_v_differ_from_d_model_over_h():
    cfg = config(16, 2, d_k=3, d_v=5)
    m = MultiHeadAttention(cfg, generator=torch.Generator().manual_seed(0))
    assert m.W_q.shape == m.W_k.shape == (16, 6)
    assert m.W_v.shape == (16, 10) and m.W_o.shape == (10, 16)
    big = MultiHeadAttention(config(256, 4), generator=torch.Generator().manual_seed(1))
    std = big.W_q.detach().std().item()
    assert abs(std - 256**-0.5) < 0.1 * 256**-0.5
    x = torch.randn(2, 4, 16)
    y, a = m(x, memory=torch.randn(2, 7, 16))
    assert y.shape == (2, 4, 16) and a.shape == (2, 2, 4, 7)


def test_numpy_mask_and_a_huge_masked_score_are_handled():
    # The masked key has a dot product far above the others: it must get zero weight, and the rest must still sum to 1.
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)  # (1, 1, 1, 2)
    k = torch.tensor([[[[1.0, 0.0], [2.0, 0.0], [1e4, 0.0]]]], dtype=torch.float64)  # (1, 1, 3, 2)
    v = torch.eye(3, dtype=torch.float64)[None, None]  # (1, 1, 3, 3)
    mask = np.array([True, True, False])
    o, a = attention(q, k, v, mask)
    s = np.array([1.0, 2.0]) / np.sqrt(2.0)
    want = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    np.testing.assert_allclose(a[0, 0, 0].numpy(), [want[0], want[1], 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(o[0, 0, 0].numpy(), [want[0], want[1], 0.0], rtol=1e-12, atol=0)


def test_large_scores_with_a_mask_stay_finite():
    q = torch.full((1, 1, 2, 4), 30.0)  # scores of 30 * 30 * 4 / 2 = 1800, above the float64 exp limit of about 709
    k = torch.full((1, 1, 3, 4), 30.0)
    v = torch.randn(1, 1, 3, 2)
    o, a = attention(q, k, v, causal_mask(2, 3))
    assert torch.isfinite(o).all() and torch.isfinite(a).all()
    torch.testing.assert_close(a.sum(-1), torch.ones(1, 1, 2))
    torch.testing.assert_close(a[0, 0, 0], torch.tensor([0.5, 0.5, 0.0]))
