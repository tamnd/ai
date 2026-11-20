# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/end-to-end-memory-network/ (ai://VDRYRT).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the gradients against finite differences,
the null rows, the clipping, and a short training run on a toy task in the form of the first bAbI task."""

import json
import pathlib

import numpy as np
import pytest
import torch

from memn2n import clip_each, config, cross_entropy, forward, init, param_count, zero_null

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
KEYS = ("vocab", "dim", "hops", "mem", "sent_len")
WHERE, IS, WENT = 1, 2, 3
PERSONS, PLACES = torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9, 10, 11])


def cfg_of(c, **kw):
    return config(**{k: c["config"][k] for k in KEYS}, **kw)


def toy_batch(gen, batch, n=5, j=4):
    who = PERSONS[torch.randint(0, 3, (batch, n), generator=gen)]
    where = PLACES[torch.randint(0, 5, (batch, n), generator=gen)]
    story = torch.zeros(batch, n, j, dtype=torch.long)
    story[:, :, 0], story[:, :, 1], story[:, :, 2] = who, WENT, where
    pick = torch.randint(0, n, (batch,), generator=gen)
    asked = who[torch.arange(batch), pick]
    query = torch.zeros(batch, j, dtype=torch.long)
    query[:, 0], query[:, 1], query[:, 2] = WHERE, IS, asked
    first = (who == asked[:, None]).double().argmax(dim=1)  # row 0 is the most recent sentence
    return story, query, where[torch.arange(batch), first]


def test_shared_fixture():
    params = {k: torch.tensor(v) for k, v in np.load(FIXTURES / "params-small.npz").items()}
    inp, want = np.load(FIXTURES / "input-small.npz"), np.load(FIXTURES / "expected-small.npz")
    cfg = cfg_of(SHAPES[0])
    logits, attn = forward(params, torch.tensor(inp["story"]), torch.tensor(inp["query"]), cfg, trace=True)
    np.testing.assert_allclose(logits.numpy(), want["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(attn.numpy(), want["attention"], rtol=1e-10, atol=1e-12)
    loss = cross_entropy(logits, torch.tensor(inp["answer"]))
    assert float(loss) == pytest.approx(float(want["loss"]), rel=1e-10)


@pytest.mark.parametrize("c", SHAPES, ids=[c["name"] for c in SHAPES])
def test_shape_contract(c):
    cfg, k = cfg_of(c), c["config"]
    params = init(torch.Generator().manual_seed(0), cfg)
    assert param_count(cfg) == c["model"]["params"] == sum(v.numel() for v in params.values())
    story = torch.zeros(k["B"], k["N"], k["J"], dtype=torch.long)
    query = torch.zeros(k["B"], k["J"], dtype=torch.long)
    with torch.no_grad():
        logits, attn = forward(params, story, query, cfg, trace=True)
    assert list(logits.shape) == c["blocks"]["answer"]["outputs"]["logits"]
    assert list(attn.shape[1:]) == c["blocks"]["hop"]["outputs"]["attention"]


@pytest.mark.parametrize("tying", ["adjacent", "layerwise"])
def test_gradients_match_finite_differences(tying):
    gen = torch.Generator().manual_seed(1)
    cfg = config(vocab=12, dim=3, hops=2, mem=5, sent_len=4, tying=tying)
    params = init(gen, cfg, sigma=0.5)
    story, query, answer = toy_batch(gen, 2)
    names = sorted(params)

    def loss(*tensors):
        return cross_entropy(forward(dict(zip(names, tensors, strict=True)), story, query, cfg), answer)

    inputs = tuple(params[n].clone().requires_grad_(True) for n in names)
    assert torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-6)


def test_null_rows_get_a_gradient_unless_reset():
    gen = torch.Generator().manual_seed(2)
    cfg = config(vocab=12, dim=4, hops=2, mem=5, sent_len=4)
    params = {k: v.requires_grad_(True) for k, v in init(gen, cfg).items()}
    story, query, answer = toy_batch(gen, 4)
    cross_entropy(forward(params, story, query, cfg), answer).backward()
    assert params["E1"].grad[0].abs().sum() > 0  # the pads are looked up, so their row learns
    with torch.no_grad():
        for p in params.values():
            p -= 0.1 * p.grad
    zero_null(params, cfg)
    assert all(not params[n][0].any() for n in ("E0", "E1", "E2"))


def test_clip_each_matrix():
    p = {"a": torch.zeros(3, requires_grad=True), "b": torch.zeros(2, requires_grad=True)}
    p["a"].grad, p["b"].grad = torch.tensor([300.0, 400.0, 0.0]), torch.tensor([3.0, 4.0])
    clip_each(p, 40.0)
    assert p["a"].grad.norm() == pytest.approx(40.0) and p["b"].grad.norm() == pytest.approx(5.0)


def test_one_hop_learns_the_toy_task():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(3)
    cfg = config(vocab=12, dim=20, hops=1, mem=5, sent_len=4)
    params = {k: v.requires_grad_(True) for k, v in init(gen, cfg).items()}
    opt = torch.optim.Adam(params.values(), lr=0.02)
    for _ in range(400):
        story, query, answer = toy_batch(gen, 32)
        opt.zero_grad()
        cross_entropy(forward(params, story, query, cfg), answer).backward()
        clip_each(params, 40.0)
        opt.step()
        zero_null(params, cfg)
    story, query, answer = toy_batch(torch.Generator().manual_seed(99), 500)
    with torch.no_grad():
        logits, attn = forward(params, story, query, cfg, trace=True)
    assert (logits.argmax(-1) == answer).double().mean() > 0.95
    first = (story[:, :, 0] == query[:, 2:3]).double().argmax(dim=1)
    assert attn[0][torch.arange(500), first].mean() > 0.8  # it learned to look at the supporting sentence
