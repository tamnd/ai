# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/vision-transformer/ (ai://NEW-vision-transformer).
"""Tests of the PyTorch version: the shared fixture and its gradients, torch.nn.TransformerEncoderLayer, the patch
embedding as a strided convolution, the resizing against torch.nn.functional.interpolate, gradcheck, the bag of
patches, and a task that needs the position embeddings."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vit import ViT, config, nll, param_count, patchify, resize_pos, unpatchify

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = next(d for d in (HERE / "fixtures", HERE.parents[1] / "fixtures") if d.is_dir())
SHAPES = json.loads((FIXTURES / "shapes.json").read_text())["configs"]


def cfg_of(c, **kw):
    k = c["config"]
    return config(k["image_size"], k["patch_size"], k["channels"], k["d_model"], k["n_heads"], k["d_ff"],
                  k["n_layers"], k["n_classes"], **kw)  # fmt: skip


def fixture():
    params = dict(np.load(FIXTURES / "params-small.npz"))
    inp = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "input-small.npz").items()}
    want = {k: torch.as_tensor(v) for k, v in np.load(FIXTURES / "expected-small.npz").items()}
    return params, inp, want


def test_shared_fixture_and_gradients():
    params, inp, want = fixture()
    model = ViT.from_params(params, config())
    images = inp["images"].clone().requires_grad_(True)
    logits = model(images)
    torch.testing.assert_close(logits, want["logits"], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(model.embed(inp["images"]), want["tokens"], rtol=1e-12, atol=1e-13)
    loss = nll(logits, inp["labels"])
    torch.testing.assert_close(loss, want["loss"], rtol=1e-12, atol=1e-13)
    loss.backward()
    for name, p in model.p.items():
        torch.testing.assert_close(p.grad, want["d" + name], rtol=1e-10, atol=1e-12, msg=name)
    torch.testing.assert_close(images.grad, want["dimages"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("c", SHAPES, ids=lambda c: c["name"])
def test_counts(c):
    assert param_count(cfg_of(c)) == c["model"]["params"]
    if c["kind"] != "named":  # the named models are too large to build in a test
        assert sum(p.numel() for p in ViT(cfg_of(c)).parameters()) == c["model"]["params"]


def test_layer_matches_torch_transformer_encoder_layer():
    g = torch.Generator().manual_seed(0)
    cfg = config(8, 2, 3, 12, 3, 20, 2, 4)
    model = ViT(cfg, scale=0.5, generator=g)
    z = torch.randn(3, 17, 12, generator=g)
    for l in range(2):
        layer = model.encoder_layer(l).train()  # train mode with dropout 0: the plain path, not the fused one
        torch.testing.assert_close(layer(z), model.layer(l, z), rtol=1e-12, atol=1e-12)
    default = torch.nn.TransformerEncoderLayer(12, 3, 20, batch_first=True)
    assert (default.norm_first, default.activation is F.relu, default.norm1.eps) == (False, True, 1e-5)


def test_patch_embedding_is_a_strided_convolution():
    g = torch.Generator().manual_seed(1)
    cfg = config(12, 3, 2, 6, 2, 8, 1, 3)
    model = ViT(cfg, scale=0.5, generator=g)
    images = torch.randn(2, 12, 12, 2, generator=g)
    w, b = model.embed_conv()
    conv = F.conv2d(images.permute(0, 3, 1, 2), w, b, stride=3).flatten(2).transpose(1, 2)
    torch.testing.assert_close(conv, patchify(images, 3) @ model.p["embed_W"] + model.p["embed_b"], rtol=1e-12,
                               atol=1e-12)  # fmt: skip
    ref = load_numpy_reference()
    np.testing.assert_array_equal(ref.embed_as_conv(model.p["embed_W"].detach().numpy(), 3, 2), w.detach().numpy())


def load_numpy_reference():
    spec = importlib.util.spec_from_file_location("vit_np", HERE.parent / "numpy" / "vit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("old,new", [(14, 24), (5, 3), (4, 2), (3, 7), (6, 6)])
def test_resize_pos_matches_torch_interpolate(old, new):
    pos = torch.randn(1 + old * old, 5, generator=torch.Generator().manual_seed(old * 31 + new))
    ref = load_numpy_reference()
    np.testing.assert_allclose(resize_pos(pos, old, new).numpy(), ref.resize_pos(pos.numpy(), old, new), rtol=1e-12,
                               atol=1e-13)  # fmt: skip


@pytest.mark.parametrize("kw", [{}, {"pool": "gap", "pos": "none", "head_hidden": 3}], ids=["cls", "gap-no-pos"])
def test_gradcheck(kw):
    g = torch.Generator().manual_seed(2)
    cfg = config(4, 2, 1, 4, 2, 6, 1, 3, **kw)
    model = ViT(cfg, scale=0.5, generator=g)
    images = torch.randn(2, 4, 4, 1, generator=g, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: model(x), (images,))
    names = list(model.p)

    # gradcheck of the parameters through functional_call, which keeps the graph
    def g_of(*ps):
        return torch.func.functional_call(model, {f"p.{n}": v for n, v in zip(names, ps)}, (images.detach(),))

    ps = tuple(model.p[n].detach().clone().requires_grad_(True) for n in names)
    assert torch.autograd.gradcheck(g_of, ps)


def test_no_positions_means_a_bag_of_patches():
    g = torch.Generator().manual_seed(3)
    cfg = config(6, 2, 2, 8, 2, 12, 2, 4, pos="none")
    model = ViT(cfg, scale=0.5, generator=g)
    images = torch.randn(4, 6, 6, 2, generator=g)
    perm = torch.randperm(9, generator=g)
    shuffled = unpatchify(patchify(images, 2)[:, perm], 2, 2)
    torch.testing.assert_close(model(shuffled), model(images), rtol=1e-11, atol=1e-13)


def position_task(g, n):
    """8 x 8 images of one channel with noise, a bright 2 x 2 patch somewhere in the 4 x 4 grid of patches; the
    label is 1 when the patch is in the left half of the grid."""
    images = 0.1 * torch.randn(n, 8, 8, 1, generator=g)
    r, c = torch.randint(0, 4, (n,), generator=g), torch.randint(0, 4, (n,), generator=g)
    for i in range(n):
        images[i, 2 * r[i] : 2 * r[i] + 2, 2 * c[i] : 2 * c[i] + 2] += 1.0
    return images, (c < 2).long()


def mirror_patches(images):
    """The patches of the image with the columns of the grid reversed; each patch itself is unchanged."""
    p = patchify(images, 2).reshape(-1, 4, 4, 4)
    return unpatchify(p.flip(2).reshape(-1, 16, 4), 2, 1)


def test_the_positions_are_needed():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(4)
    test, labels = position_task(g, 500)
    test, labels = torch.cat([test, mirror_patches(test)]), torch.cat([labels, 1 - labels])
    acc = {}
    for pos in ("learned", "none"):
        model = ViT(config(8, 2, 1, 16, 2, 32, 2, 2, pos=pos), scale=0.1, generator=torch.Generator().manual_seed(5))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(300):
            x, y = position_task(g, 64)
            opt.zero_grad()
            nll(model(x), y).backward()
            opt.step()
        with torch.no_grad():
            logits = model(test)
        acc[pos] = (logits.argmax(1) == labels).double().mean().item()
        if pos == "none":
            torch.testing.assert_close(logits[:500], logits[500:], rtol=0, atol=1e-9)
    assert acc["learned"] > 0.95
    assert acc["none"] == 0.5
