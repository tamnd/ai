# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mamba/ (ai://NEW-mamba).
"""Tests of the PyTorch Mamba: the fixtures, the gradients, the two scans, and the selective copying task of the
paper's Section 4.1 on which the selective model beats the time-invariant ablation."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch

import mamba as M

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE.parents[1] / "fixtures"
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"]
SIZES = ("vocab", "d_model", "n_layers", "d_state", "d_conv", "expand", "dt_rank")


def numpy_reference():
    spec = importlib.util.spec_from_file_location("mamba_np", HERE.parent / "numpy" / "mamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_model(mode="loop"):
    params = dict(np.load(FIX / "params-small.npz"))
    model = M.MambaLM.from_params(M.config(), params)
    for b in model.blocks:
        b.mode = mode
    return model, torch.as_tensor(np.load(FIX / "input-small.npz")["tokens"]), np.load(FIX / "expected-small.npz")


@pytest.mark.parametrize("mode", ["loop", "quadratic"])
def test_fixture(mode):
    model, tok, exp = fixture_model(mode)
    logits = model(tok[:, :-1])
    assert np.allclose(logits.detach().numpy(), exp["logits"], atol=1e-10)
    loss = M.nll(logits, tok[:, 1:])
    assert np.isclose(loss.item(), exp["loss"], atol=1e-12)
    loss.backward()
    for k, g in model.grads().items():
        assert np.allclose(g.numpy(), exp["d" + k], atol=1e-10), k


def test_zoh_matches_the_numpy_reference():
    ref = numpy_reference()
    rng = np.random.default_rng(3)
    cfg = M.config(disc="zoh", d_state=5, d_conv=4)
    params = ref.init(rng, ref.config(disc="zoh", d_state=5, d_conv=4), scale=0.4)
    tok = rng.integers(0, cfg["vocab"], (2, 7))
    want, cache = ref.forward(params, ref.config(disc="zoh", d_state=5, d_conv=4), tok[:, :-1])
    model = M.MambaLM.from_params(cfg, params)
    got = model(torch.as_tensor(tok[:, :-1]))
    assert np.allclose(got.detach().numpy(), want, atol=1e-10)
    M.nll(got, torch.as_tensor(tok[:, 1:])).backward()
    g = ref.backward(params, ref.config(disc="zoh", d_state=5, d_conv=4), cache, want, tok[:, 1:])
    for k, v in model.grads().items():
        assert np.allclose(v.numpy(), g[k], atol=1e-10), k


def test_quadratic_scan_equals_the_loop():
    torch.manual_seed(0)
    cfg = M.config(d_model=12, d_state=6)
    loop = M.MambaLM(cfg).double()
    quad = M.MambaLM(cfg, mode="quadratic").double()
    quad.load_state_dict(loop.state_dict())
    tok = torch.randint(0, cfg["vocab"], (3, 20))
    assert torch.allclose(loop(tok), quad(tok), atol=1e-12)


def test_time_invariant_ablation():
    """With selective=False Delta, B and C are the same at every position, so the SSM is a convolution: its
    kernel K_k = C A_bar^k B_bar applied to the SSM input gives the scan's output."""
    torch.manual_seed(1)
    cfg = M.config(d_model=6, d_state=3)
    b = M.MambaBlock(cfg, selective=False).double()
    xa = torch.randn(2, 9, cfg["d_inner"], dtype=torch.double)
    dt, Bm, C = b.ssm_inputs(xa)
    assert torch.equal(dt[0, 0], dt[1, 5])
    A = -torch.exp(b.A_log)
    dA, dB = torch.exp(dt[0, 0, :, None] * A), dt[0, 0, :, None] * Bm[0, 0]
    kernel = torch.stack([(C[0, 0] * dA**k * dB).sum(-1) for k in range(9)])  # (9, I)
    conv = torch.stack([(kernel[: t + 1].flip(0) * xa[:, : t + 1]).sum(1) for t in range(9)], 1)
    h, ys = torch.zeros(2, cfg["d_inner"], cfg["d_state"], dtype=torch.double), []
    for t in range(9):
        h = dA * h + dB * xa[:, t, :, None]
        ys.append((h * C[0, 0]).sum(-1))
    assert torch.allclose(torch.stack(ys, 1), conv, atol=1e-12)


def test_parameter_count():
    for kw in ({}, {"d_model": 16, "d_state": 8, "d_conv": 4}, {"dt_rank": 3, "n_layers": 3}):
        cfg = M.config(**kw)
        assert sum(p.numel() for p in M.MambaLM(cfg).parameters()) == M.param_count(cfg)


@pytest.mark.parametrize("c", [c for c in SHAPES if c["kind"] != "named"], ids=lambda c: c["name"])
def test_counts_and_shapes(c):
    k = c["config"]
    cfg = M.config(**{n: k[n] for n in SIZES})
    model = M.MambaLM(cfg)
    assert sum(p.numel() for p in model.parameters()) == M.param_count(cfg) == c["model"]["params"]
    logits = model(torch.randint(0, cfg["vocab"], (k["B"], k["T"])))
    assert list(logits.shape) == c["blocks"]["head"]["outputs"]["logits"]


def test_default_config_is_the_example():
    assert M.config() == M.config(**{n: SHAPES[0]["config"][n] for n in SIZES})


def selective_copying(g, n, T, k):
    """Section 4.1: k data tokens from {1, ..., 4} at random positions among T noise tokens 0, then k markers 5;
    at each marker the model must output the data tokens in order. A time-invariant model cannot tell where they
    are, since their positions change from sequence to sequence."""
    data = torch.randint(1, 5, (n, k), generator=g)
    x = torch.zeros(n, T, dtype=torch.long)
    for i in range(n):
        x[i, torch.sort(torch.randperm(T, generator=g)[:k]).values] = data[i]
    return torch.cat([x, torch.full((n, k), 5)], 1), data


def train_selective_copying(selective, steps, T, k, D, N, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = M.MambaLM(M.config(vocab=6, d_model=D, n_layers=2, d_state=N, d_conv=4), selective)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = selective_copying(g, 32, T, k)
        loss = M.nll(model(x)[:, -k:], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = selective_copying(torch.Generator().manual_seed(99), 512, T, k)
        return (model(x)[:, -k:].argmax(-1) == y).float().mean().item()


def test_selection_solves_selective_copying():
    """The paper's Table 1 in small: the same Mamba architecture with the selective SSM (S6) or a time-invariant
    one (S4D with real A), trained alike on selective copying with 3 data tokens among 16 positions. Chance is 0.25.
    At 400 steps of Adam the selective model gets about 0.85 of the tokens and the time-invariant one about 0.65;
    over the seeds 0 to 3 the first ranged from 0.75 to 0.85 and the second from 0.58 to 0.70."""
    torch.set_num_threads(1)
    sel = train_selective_copying(True, 400, T=16, k=3, D=8, N=4, seed=1)
    lti = train_selective_copying(False, 400, T=16, k=3, D=8, N=4, seed=1)
    assert sel > 0.75 > lti, (sel, lti)
