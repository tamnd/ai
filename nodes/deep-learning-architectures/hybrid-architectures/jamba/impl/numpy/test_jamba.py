# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/jamba/ (ai://2V1JK1).
"""Tests of the NumPy reference of Jamba: the fixtures, the gradients, generation with a cache, the layout of the
Jamba block, the router, the counts of the shape contract, and the numbers of Table 1 of the paper."""

import json
import pathlib

import numpy as np
import pytest

import jamba as J

HERE = pathlib.Path(__file__).resolve().parent
FIX = next(p for p in (HERE / "fixtures", HERE.parents[1] / "fixtures") if p.is_dir())
# aik flopcount imports this module in a directory of its own, without the fixtures, and calls flop_case only.
SHAPES = json.loads((FIX / "shapes.json").read_text())["configs"] if (FIX / "shapes.json").exists() else []
SIZES = ("vocab", "d_model", "n_blocks", "block_len", "moe_every", "n_experts", "top_k", "d_ff", "n_heads", "n_kv",
         "d_state", "d_conv", "expand", "dt_rank")  # fmt: skip


def cfg_of(c):
    return J.config(**{k: c["config"][k] for k in SIZES})


def setup(seed=0, **kw):
    cfg = J.config(**kw)
    rng = np.random.default_rng(seed)
    p = J.init(rng, cfg, scale=0.5)
    for k in p:  # move the structured parameters off their initial values, so that their gradients are generic
        if k.endswith(("A_log", "_D", "norm", "norm1", "norm2", "norm_f")):
            p[k] = p[k] + rng.normal(0.0, 0.2, p[k].shape)
    return cfg, p, rng


def test_the_shape_contract_is_there():
    assert len(SHAPES) >= 3


def test_fixture():
    params = dict(np.load(FIX / "params-small.npz"))
    tokens = np.load(FIX / "input-small.npz")["tokens"]
    exp = np.load(FIX / "expected-small.npz")
    cfg = J.config()
    logits, aux, cache = J.forward(params, cfg, tokens[:, :-1])
    assert np.allclose(logits, exp["logits"], atol=1e-12)
    assert np.isclose(J.nll(logits, tokens[:, 1:]), exp["nll"], atol=1e-12)
    assert np.isclose(aux, exp["aux"], atol=1e-12)
    assert np.isclose(J.lm_loss(params, cfg, tokens), exp["loss"], atol=1e-12)
    for i in (1, 3):
        top = np.argsort(-cache["layers"][i]["ffn"]["probs"], axis=-1, kind="stable")[:, :2]
        assert (top == exp[f"l{i}_top"]).all()
    grads = J.backward(params, cfg, cache, logits, tokens[:, 1:])
    assert set(grads) == set(params)
    for k, v in grads.items():
        assert np.allclose(v, exp["d" + k], atol=1e-12), k


@pytest.mark.parametrize("kw", [{}, {"n_blocks": 2, "n_kv": 2, "top_k": 1}, {"block_len": 8, "moe_every": 4}])
def test_gradients_match_finite_differences(kw):
    """Central differences of the loss, the load-balancing term included, against the backward pass. The choice of
    experts is piecewise constant, so a step of 1e-6 does not change it at these random points."""
    cfg, p, rng = setup(**kw)
    tok = rng.integers(0, cfg["vocab"], (2, 7))
    logits, _, cache = J.forward(p, cfg, tok[:, :-1])
    g = J.backward(p, cfg, cache, logits, tok[:, 1:])
    for k, v in p.items():
        for _ in range(2):
            idx = tuple(rng.integers(0, s) for s in v.shape)
            old = v[idx]
            v[idx] = old + 1e-6
            a = J.lm_loss(p, cfg, tok)
            v[idx] = old - 1e-6
            b = J.lm_loss(p, cfg, tok)
            v[idx] = old
            assert abs((a - b) / 2e-6 - g[k][idx]) < 1e-6, (k, idx)


def test_generation_with_a_cache_gives_the_same_scores():
    cfg, p, rng = setup(1, n_blocks=2)
    tok = rng.integers(0, cfg["vocab"], (3, 9))
    logits, _, _ = J.forward(p, cfg, tok)
    state, outs = J.init_state(cfg, 3), []
    for t in range(9):
        o, state = J.step(p, cfg, tok[:, t], state)
        outs.append(o)
    assert np.allclose(np.stack(outs, 1), logits, atol=1e-12)


def test_only_the_attention_layers_grow_with_the_sequence():
    """eq-cache: after t tokens the attention layers keep t keys and values; the Mamba layers keep a window of
    d_conv - 1 inputs and a state of d_state per channel, whatever t."""
    cfg, p, rng = setup(2)
    state = J.init_state(cfg, 2)
    sizes = []
    for t in range(5):
        _, state = J.step(p, cfg, rng.integers(0, cfg["vocab"], 2), state)
        sizes.append([a.size + b.size for a, b in state])
        assert sum(sizes[-1]) == J.cache_numbers(cfg, 2, t + 1)
    for i, (m, _) in enumerate(J.layers(cfg)):
        grows = [s[i] for s in sizes]
        assert (len(set(grows)) == 5) == (m == "attn")


def test_the_layout_of_a_block():
    """Figure 1 and Section 3.1: in a block of l = 8 with e = 2, layer 4 is attention, the other 7 are Mamba, and
    the odd layers have the mixture of experts; Jamba-v0.1 stacks 4 such blocks."""
    cfg = J.config(d_model=32, n_heads=4, block_len=8, moe_every=2, n_blocks=4)
    lay = J.layers(cfg)
    assert [i for i, (m, _) in enumerate(lay) if m == "attn"] == [4, 12, 20, 28]
    assert [i for i, (_, f) in enumerate(lay) if f == "moe"] == list(range(1, 32, 2))
    assert sum(m == "mamba" for m, _ in lay[:8]) == 7
    with pytest.raises(ValueError):
        J.config(block_len=4, moe_every=3)
    with pytest.raises(ValueError):  # e = 1 would give the attention layer a mixture of experts too
        J.config(block_len=4, moe_every=1)


def test_each_token_runs_through_exactly_top_k_experts():
    cfg, p, rng = setup(3, n_experts=5, top_k=2)
    u = rng.normal(size=(2, 6, cfg["d_model"]))
    out, aux, c = J.moe(p, cfg, u, "l1_")
    rows = np.concatenate([r for r, *_ in c["parts"]])
    assert np.bincount(rows, minlength=12).tolist() == [2] * 12
    assert np.isclose(c["f"].sum(), 1.0)
    chosen = np.argsort(-c["probs"], axis=-1)[:, :2]
    by_hand = np.zeros((12, cfg["d_model"]))
    for t in range(12):
        for j in chosen[t]:
            y, _ = J.swiglu(u.reshape(12, -1)[t], p["l1_e_gate"][j], p["l1_e_up"][j], p["l1_e_down"][j])
            by_hand[t] += c["probs"][t, j] * y
    assert np.allclose(out.reshape(12, -1), by_hand)


def test_the_balance_term_is_smallest_for_a_uniform_router():
    """eq-aux: with a router that gives every token the same probabilities 1 / n and a choice spread evenly, the
    term is alpha n sum_j (1 / n)(1 / n) = alpha, its value at balance."""
    cfg, p, rng = setup(4, n_experts=4, top_k=1)
    p["l1_router"][:] = 0.0
    u = rng.normal(size=(1, 8, cfg["d_model"]))
    _, aux, c = J.moe(p, cfg, u, "l1_")
    assert np.allclose(c["probs"], 0.25)
    assert np.isclose(aux, cfg["aux"] * cfg["n_experts"] * (c["f"] * 0.25).sum())
    assert np.isclose(aux, cfg["aux"])


def test_causal():
    cfg, p, rng = setup(5)
    tok = rng.integers(0, cfg["vocab"], (1, 8))
    a, _, _ = J.forward(p, cfg, tok)
    tok2 = tok.copy()
    tok2[0, 5] = (tok[0, 5] + 1) % cfg["vocab"]
    b, _, _ = J.forward(p, cfg, tok2)
    assert np.allclose(a[:, :5], b[:, :5]) and not np.allclose(a[:, 5:], b[:, 5:])


def test_order_reaches_the_attention_through_the_mamba_layers():
    """There is no positional encoding: without the Mamba layers before it, attention alone is blind to the order
    of what came before, but in the block it is not."""
    cfg, p, rng = setup(6)
    u = rng.normal(size=(1, 5, cfg["d_model"]))
    perm = np.array([2, 0, 1, 3, 4])
    out, _ = J.attention(p, cfg, u, "l2_")
    out2, _ = J.attention(p, cfg, u[:, perm], "l2_")
    assert np.allclose(out[:, 3:], out2[:, 3:])
    tok = rng.integers(0, cfg["vocab"], (1, 5))
    a, _, _ = J.forward(p, cfg, tok)
    b, _, _ = J.forward(p, cfg, tok[:, perm])
    assert not np.allclose(a[:, 3:], b[:, 3:])


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    cfg, m = cfg_of(c), c["model"]
    assert J.param_count(cfg) == m["params"]
    assert J.param_count(cfg, active=True) == m["active_params"]
    assert sum(int(np.prod(s)) for s in J.param_shapes(cfg).values()) == m["params"]
    k = c["config"]
    assert J.flops(cfg, k["B"], k["T"]) == m["flops_forward"]
    assert J.cache_numbers(cfg, k["B"], k["T"]) == m["kv_cache"] + m["recurrent_state"]
    if c["kind"] != "named":
        p = J.init(np.random.default_rng(0), cfg)
        tok = np.random.default_rng(1).integers(0, cfg["vocab"], (k["B"], k["T"]))
        logits, _, cache = J.forward(p, cfg, tok)
        assert list(logits.shape) == c["blocks"]["head"]["outputs"]["logits"]
        i = J.layers(cfg).index(("attn", "mlp"))
        assert list(cache["layers"][i]["mixer"]["q"].shape) == c["blocks"]["attn"]["internals"]["q"]
        assert list(cache["layers"][i]["mixer"]["a"].shape) == c["blocks"]["attn"]["internals"]["scores"]
        assert list(cache["layers"][0]["mixer"]["hs"].shape) == c["blocks"]["mamba"]["internals"]["h"]
        assert list(cache["layers"][1]["ffn"]["probs"].shape) == c["blocks"]["moe"]["internals"]["probs"]


def test_table_1_of_the_paper():
    """Table 1: the KV cache at 256K tokens in 16-bit, 4GB for Jamba's 4 attention layers of 8 key-value heads of
    128, 32GB for 32 such layers (Mistral, Mixtral) and 128GB for 32 layers of 32 heads (Llama-2 6.7B); and the
    52B and 12B parameters of Jamba from the released sizes."""
    gib, T = 2**30, 256 * 1024
    assert J.kv_cache_bytes(4, 8, 128, T) == 4 * gib
    assert J.kv_cache_bytes(32, 8, 128, T) == 32 * gib
    assert J.kv_cache_bytes(32, 32, 128, T) == 128 * gib
    cfg = J.config(vocab=65536, d_model=4096, n_blocks=4, block_len=8, moe_every=2, n_experts=16, top_k=2, d_ff=14336,
                   n_heads=32, n_kv=8, d_state=16, d_conv=4, expand=2, dt_rank=256)  # fmt: skip
    assert round(J.param_count(cfg) / 1e9) == 52
    assert round(J.param_count(cfg, active=True) / 1e9) == 12
    assert 2 * J.cache_numbers(cfg, 1, T) < 4.01 * gib


def test_worked_example():
    assert J.worked_example() == {"params": 5654, "active": 4502, "cache": 8288, "cache_all_attention": 32000}


def flop_case(c):
    """The forward call of a config of shapes.json, for aik flopcount."""
    cfg = cfg_of(c)
    rng = np.random.default_rng(0)
    p = J.init(rng, cfg)
    return J.forward, (p, cfg, rng.integers(0, cfg["vocab"], (c["config"]["B"], c["config"]["T"])))
