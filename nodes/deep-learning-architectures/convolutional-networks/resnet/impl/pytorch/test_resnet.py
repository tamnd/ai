# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Tests of the PyTorch reference: the shared fixture, the shape contract, the counts and the gradient."""

import json
import pathlib

import numpy as np
import pytest
import torch

from resnet import Bottleneck, ResNet, config, param_count

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]
RUN = [c for c in SHAPES if c["kind"] != "named"]


def cfg_of(c):
    k = c["config"]
    return config(k["width"], (k["n1"], k["n2"], k["n3"], k["n4"]), k["classes"])


def block_id(s, i):
    return f"s{s + 1}_{'first' if i == 0 else 'rest'}"


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    state = dict(np.load(FIXTURES / "state-small.npz"))
    x = torch.as_tensor(np.load(FIXTURES / "input-small.npz")["x"])
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return ResNet.from_params(params, cfg_of(SHAPES[0]), state), x, want


def test_shared_fixture():
    model, x, want = fixture()
    model.train()
    logits, cache = model(x)
    torch.testing.assert_close(logits, want["logits_train"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["stem"]["y"], want["stem_train"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["blocks"][(1, 0)]["c"], want["s2_first_c_train"], rtol=1e-10, atol=1e-12)
    model.eval()
    logits, cache = model(x)
    torch.testing.assert_close(logits, want["logits_infer"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(cache["blocks"][(1, 0)]["c"], want["s2_first_c_infer"], rtol=1e-10, atol=1e-12)


def test_train_mode_leaves_the_buffers():
    model, x, _ = fixture()
    before = {k: v.clone() for k, v in model.named_buffers()}
    model.train()
    model(x)
    assert all(torch.equal(v, before[k]) for k, v in model.named_buffers())


@pytest.mark.parametrize("c", RUN, ids=lambda c: c["name"])
def test_shapes(c):
    cfg, k, blocks = cfg_of(c), c["config"], c["blocks"]
    torch.manual_seed(0)
    model = ResNet(cfg)
    logits, cache = model(torch.randn(k["B"], 3, k["R"], k["R"]))
    assert list(logits.shape) == blocks["head"]["outputs"]["logits"]
    assert list(cache["pooled"].shape) == blocks["head"]["internals"]["pooled"]
    assert list(cache["stem"]["conv"].shape) == blocks["stem"]["internals"]["conv"]
    assert list(cache["stem"]["y"].shape) == blocks["stem"]["outputs"]["y"]
    for (s, i), got in cache["blocks"].items():
        want = blocks[block_id(s, i)]
        assert list(got["y"].shape) == want["outputs"]["y"]
        for name, shape in want["internals"].items():
            assert list(got[name].shape) == shape, (s, i, name)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_param_count(c):
    cfg = cfg_of(c)
    assert param_count(cfg) == c["model"]["params"]
    if c["kind"] != "named":
        assert sum(p.numel() for p in ResNet(cfg).parameters()) == c["model"]["params"]


def test_resnet50_parameters():
    # Building ResNet-50 takes a moment and 100 MB in float64; counting its tensors checks the closed form once.
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            model = ResNet(config())
        assert sum(p.numel() for p in model.parameters()) == 25_557_032
    finally:
        torch.set_default_dtype(torch.float64)


def test_zero_gamma_makes_blocks_identity():
    torch.manual_seed(0)
    model = ResNet(config(2, (2, 2, 2, 2), 5), zero_gamma=True)
    _, cache = model(torch.randn(2, 3, 32, 32))
    for s in range(4):
        torch.testing.assert_close(cache["blocks"][(s, 1)]["y"], cache["blocks"][(s, 0)]["y"], rtol=0, atol=0)


@pytest.mark.parametrize("projection", [False, True])
def test_bottleneck_gradcheck(projection):
    torch.manual_seed(1)
    c_in = 3 if projection else 8
    block = Bottleneck(c_in, 2, 2 if projection else 1, projection)
    x = torch.randn(3, c_in, 6, 6, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: block(t)[0], (x,))


def test_identity_shortcut_carries_the_gradient():
    # c-gradient-identity: through a block with the identity shortcut, dy/dx is I plus the Jacobian of F;
    # with the last scale at 0 the Jacobian of F is 0, and on positive inputs the gradient passes unchanged.
    torch.manual_seed(2)
    block = Bottleneck(8, 2, 1, False)
    with torch.no_grad():
        block.bn3.gamma.zero_()
    x = (torch.rand(2, 8, 4, 4) + 0.1).requires_grad_()
    y, _ = block(x)
    g = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, g)
    torch.testing.assert_close(dx, g, rtol=0, atol=1e-12)
