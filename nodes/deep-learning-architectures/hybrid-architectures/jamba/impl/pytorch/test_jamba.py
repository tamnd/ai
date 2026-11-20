# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Tests of the PyTorch Jamba: the shared fixture and its gradients, gradcheck, the counts of the shape contract,
and a small version of the paper's ablation in which the hybrid recalls what pure Mamba does not."""

import json
import pathlib

import numpy as np
import pytest
import torch

import jamba as M

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]
SIZES = ("vocab", "d_model", "n_blocks", "block_len", "moe_every", "n_experts", "top_k", "d_ff", "n_heads", "n_kv",
         "d_state", "d_conv", "expand", "dt_rank")  # fmt: skip


def test_fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    tokens = torch.as_tensor(np.load(FIX / "input-small.npz")["tokens"])
    exp = np.load(FIX / "expected-small.npz")
    model = M.JambaLM.from_params(M.config(), params)
    logits, aux = model(tokens[:, :-1])
    assert np.allclose(logits.detach().numpy(), exp["logits"], atol=1e-12)
    assert np.isclose(aux.item(), exp["aux"], atol=1e-12)
    loss = M.lm_loss(model, tokens)
    assert np.isclose(loss.item(), exp["loss"], atol=1e-12)
    loss.backward()
    named = model.named_like_numpy()
    assert set(named) == set(params)
    for k, t in named.items():
        assert t.grad is not None
        assert np.allclose(t.grad.numpy(), exp["d" + k], atol=1e-12), k


def test_gradcheck():
    torch.manual_seed(0)
    cfg = M.config(vocab=5, d_model=4, n_experts=3, top_k=2, d_ff=3, n_heads=2, d_state=2, d_conv=2, dt_rank=1)
    layers = M.JambaLM(cfg).double().layers
    x = torch.randn(1, 4, 4, dtype=torch.double, requires_grad=True)
    for l in layers:
        assert torch.autograd.gradcheck(lambda r, l=l: l(r)[0], (x,))


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    cfg = M.config(**{k: c["config"][k] for k in SIZES})
    if c["kind"] == "named":
        pytest.skip("the released model is too large to build here; the NumPy tests check its counts")
    model = M.JambaLM(cfg)
    assert sum(t.numel() for t in model.parameters()) == c["model"]["params"]
    tok = torch.randint(0, cfg["vocab"], (c["config"]["B"], c["config"]["T"]))
    assert list(model(tok)[0].shape) == c["blocks"]["head"]["outputs"]["logits"]


NK = 8  # keys 0..7, values 8..15


def recall(g, n, pairs=4):
    """Associative recall: 4 pairs (key, value) with distinct keys, then one of the keys; the answer is its value."""
    keys = torch.stack([torch.randperm(NK, generator=g)[:pairs] for _ in range(n)])
    vals = torch.randint(NK, 2 * NK, (n, pairs), generator=g)
    q = torch.randint(0, pairs, (n,), generator=g)
    x = torch.cat([torch.stack([keys, vals], -1).reshape(n, -1), keys[torch.arange(n), q][:, None]], 1)
    return x, vals[torch.arange(n), q]


def train_recall(kinds, seed, steps=400):
    cfg = M.config(vocab=2 * NK, d_model=32, n_experts=4, top_k=2, d_ff=64, n_heads=4, n_kv=2, d_state=4, d_conv=4,
                   expand=2, dt_rank=4)  # fmt: skip
    torch.manual_seed(seed)
    model = M.JambaLM(cfg, kinds)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = recall(g, 64)
        logits, aux = model(x)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], y) + aux
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = recall(torch.Generator().manual_seed(99), 1000)
    with torch.no_grad():
        return (model(x)[0][:, -1].argmax(-1) == y).double().mean().item()


def test_the_hybrid_recalls_where_pure_mamba_does_not():
    """A block of 4 layers, the third attention, against the same block with Mamba in all 4, after 400 steps of
    Adam: with seed 0 the hybrid answers 1.000 of 1,000 held-out queries and pure Mamba 0.798, where chance is
    1 / 8. The Experiments section of the page has seeds 0 to 2 and the all-attention block too."""
    torch.set_num_threads(1)
    hybrid, mamba = train_recall(None, 0), train_recall(["mamba"] * 4, 0)
    assert hybrid > 0.95 and mamba < hybrid - 0.1, (hybrid, mamba)
