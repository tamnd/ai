# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/sparse-autoencoders/ (ai://NEW-sparse-autoencoders).
"""Tests of the PyTorch code: the encoders against their formulas, TopK as an exact L0 constraint, gradients through
the kept latents only, the start with the encoder equal to the decoder transpose, the decoder renormalization and
gradient projection, the scale symmetry, FVU and the relative reconstruction bias on the worked example, the dead-latent
tracker, the AuxK loss, and a training run on synthetic data with features in superposition, where a TopK autoencoder
finds the features without shrinkage and a ReLU autoencoder with an L1 penalty shrinks its reconstructions."""

import pytest
import torch

from sparse_autoencoders import (
    DeadLatentTracker,
    SparseAutoencoder,
    auxk_loss,
    fvu,
    l1_loss,
    mse_loss,
    relative_reconstruction_bias,
    topk_relu,
    train,
)

torch.set_default_dtype(torch.float64)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def test_constructor_checks():
    with pytest.raises(ValueError):
        SparseAutoencoder(4, 8, "gelu")
    with pytest.raises(ValueError):
        SparseAutoencoder(4, 8, "topk")
    with pytest.raises(ValueError):
        SparseAutoencoder(4, 8, "topk", k=9)
    with pytest.raises(ValueError):
        topk_relu(torch.zeros(2, 3), 0)


def test_start_has_unit_decoder_and_tied_encoder():
    sae = SparseAutoencoder(6, 20, "relu", generator=gen(0))
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(20))
    assert torch.equal(sae.W_enc, sae.W_dec.T)
    assert sae.W_enc.data_ptr() != sae.W_dec.data_ptr()  # equal at the start, not tied


def test_relu_encoder_formula():
    sae = SparseAutoencoder(5, 12, "relu", generator=gen(1))
    with torch.no_grad():
        sae.b_enc.normal_(generator=gen(2))
        sae.b_pre.normal_(generator=gen(3))
    x = torch.randn(9, 5, generator=gen(4))
    x_hat, z, pre = sae(x)
    expect_pre = (x - sae.b_pre) @ sae.W_enc.T + sae.b_enc
    assert torch.allclose(pre, expect_pre)
    assert torch.allclose(z, torch.clamp(expect_pre, min=0.0))
    assert torch.allclose(x_hat, z @ sae.W_dec.T + sae.b_pre)
    assert torch.allclose(sae.encode(x), z)


def test_topk_encoder_is_an_exact_l0_constraint():
    for k in (1, 3, 7):
        sae = SparseAutoencoder(5, 16, "topk", k=k, generator=gen(k))
        with torch.no_grad():
            sae.b_enc.fill_(100.0)  # unused by the TopK encoder
        x = torch.randn(30, 5, generator=gen(10 + k))
        _x_hat, z, pre = sae(x)
        assert torch.allclose(pre, (x - sae.b_pre) @ sae.W_enc.T)
        nnz = (z != 0).sum(dim=-1)
        assert torch.equal(nnz, torch.minimum((pre > 0).sum(dim=-1), torch.tensor(k)))
        assert (z >= 0).all()
        kept = z != 0
        assert torch.allclose(z[kept], pre[kept])
        for row_pre, row_kept in zip(pre, kept, strict=True):
            if row_kept.any() and (~row_kept).any():
                assert row_pre[row_kept].min() >= row_pre[~row_kept].max()


def test_topk_gradient_reaches_only_kept_latents():
    pre = torch.tensor([[3.0, -1.0, 2.0, 0.5]], requires_grad=True)
    topk_relu(pre, 2).sum().backward()
    assert torch.equal(pre.grad, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))


def test_normalize_and_project():
    sae = SparseAutoencoder(4, 10, "relu", generator=gen(5))
    with torch.no_grad():
        sae.W_dec.mul_(torch.rand(10, generator=gen(6)) * 5 + 0.1)
    sae.normalize_decoder_()
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(10))
    x = torch.randn(20, 4, generator=gen(7))
    x_hat, z, _ = sae(x)
    l1_loss(x, x_hat, z, 0.5).backward()
    g = sae.W_dec.grad.clone()
    sae.project_decoder_grad_()
    assert torch.allclose(
        (sae.W_dec.grad * sae.W_dec).sum(dim=0), torch.zeros(10), atol=1e-12
    )
    assert torch.allclose(g - sae.W_dec.grad, sae.W_dec * (g * sae.W_dec).sum(dim=0))


def test_scale_symmetry_lowers_l1_only():
    sae = SparseAutoencoder(4, 10, "relu", generator=gen(8))
    with torch.no_grad():
        sae.b_enc.normal_(generator=gen(9))
    x = torch.randn(25, 4, generator=gen(10))
    with torch.no_grad():
        x_hat, z, _ = sae(x)
        base = l1_loss(x, x_hat, z, 1.0)
        s = 10.0
        sae.W_enc.div_(s)
        sae.b_enc.div_(s)
        sae.W_dec.mul_(s)
        x_hat_s, z_s, _ = sae(x)
    assert torch.allclose(x_hat_s, x_hat)
    assert torch.allclose(z_s * s, z)
    assert l1_loss(x, x_hat_s, z_s, 1.0) < base


def test_losses_and_metrics_on_the_worked_example():
    x = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    expect = {
        "l1": (0.5, 1.0, 0.25, 13 / 16),
        "l1 lam=2": (1.0, 4.0, 1.0, 2 / 3),
        "topk": (None, 2.0, 0.5, 1.0),
    }
    for name, (half_lam, err, f, g) in expect.items():
        sae = SparseAutoencoder(
            2, 2, "topk" if name == "topk" else "relu", k=1 if name == "topk" else None
        )
        with torch.no_grad():
            sae.W_dec.copy_(torch.eye(2))
            sae.W_enc.copy_(torch.eye(2))
            if half_lam is not None:
                sae.b_enc.fill_(-half_lam)
        x_hat, z, _ = sae(x)
        assert ((x - x_hat) ** 2).sum().item() == pytest.approx(err)
        assert fvu(x, x_hat).item() == pytest.approx(f)
        assert relative_reconstruction_bias(x, x_hat).item() == pytest.approx(g)
        assert mse_loss(x, x_hat).item() == pytest.approx(err / 2)
        if half_lam is not None:
            assert l1_loss(x, x_hat, z, 2 * half_lam).item() == pytest.approx(
                err / 2 + 2 * half_lam * z.sum().item() / 2
            )
    with pytest.raises(ValueError):
        l1_loss(x, x, x, -0.1)
    with pytest.raises(ValueError):
        fvu(torch.ones(3, 2), torch.ones(3, 2))


def test_dead_latent_tracker():
    t = DeadLatentTracker(4, threshold=5)
    t.update(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]))
    assert torch.equal(t.since, torch.tensor([0, 2, 0, 2]))
    t.update(torch.tensor([[0.0, 0.0, 0.0, 0.0]] * 3))
    assert torch.equal(t.dead(), torch.tensor([False, True, False, True]))
    assert t.fraction() == 0.5
    t.update(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert torch.equal(t.dead(), torch.tensor([False, False, False, True]))


def test_auxk_loss():
    sae = SparseAutoencoder(5, 8, "topk", k=2, generator=gen(11))
    x = torch.randn(12, 5, generator=gen(12))
    x_hat, _z, pre = sae(x)
    none = torch.zeros(8, dtype=torch.bool)
    assert auxk_loss(x, x_hat, pre, none, sae.W_dec, 3).item() == 0.0
    dead = torch.tensor([True, False, True, False, False, True, False, False])
    loss = auxk_loss(x, x_hat, pre, dead, sae.W_dec, 2)
    e = (x - x_hat).detach()
    masked = pre.detach().clone()
    masked[:, ~dead] = float("-inf")
    v, i = masked.topk(2, dim=-1)
    zz = torch.zeros_like(masked).scatter(-1, i, v.clamp(min=0))
    assert loss.item() == pytest.approx(
        ((e - zz @ sae.W_dec.T.detach()) ** 2).sum(-1).mean().item()
    )
    loss.backward()
    # the gradient reaches the dead latents' encoder rows and decoder columns only
    assert (sae.W_enc.grad[~dead] == 0).all() and (sae.W_dec.grad[:, ~dead] == 0).all()
    assert sae.W_enc.grad[dead].abs().sum() > 0


def superposition_data(m=8, features=24, p=0.06, rows=20000, seed=0):
    """Rows are sums of a few of `features` unit directions in R^m, each on with probability p, magnitude in [0.5, 1.5]."""
    g = gen(seed)
    G = torch.randn(m, features, generator=g)
    G = G / G.norm(dim=0, keepdim=True)
    on = torch.rand(rows, features, generator=g) < p
    return (on * (0.5 + torch.rand(rows, features, generator=g))) @ G.T, G


def test_training_recovers_features_in_superposition():
    X, G = superposition_data()
    topk_sae = train(
        SparseAutoencoder(8, 48, "topk", k=3, generator=gen(1)),
        X,
        1200,
        lr=3e-3,
        generator=gen(2),
    )
    relu_sae = train(
        SparseAutoencoder(8, 48, "relu", generator=gen(1)),
        X,
        1200,
        lr=3e-3,
        lam=0.3,
        generator=gen(2),
    )
    with torch.no_grad():
        for sae in (topk_sae, relu_sae):
            assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(48))
        x_hat, z, _ = topk_sae(X)
        assert ((z != 0).sum(dim=-1) <= 3).all()
        assert fvu(X, x_hat) < 0.15
        # each of the 24 true directions has a decoder column close to it, more than 3 features per dimension
        best = (G.T @ topk_sae.W_dec).max(dim=1).values
        assert best.mean() > 0.85 and best.min() > 0.6
        assert abs(relative_reconstruction_bias(X, x_hat) - 1.0) < 0.03
        x_hat_r, _, _ = relu_sae(X)
        assert relative_reconstruction_bias(X, x_hat_r) < 0.95


def test_k_bounds_and_loss_reductions():
    pre = torch.tensor([[1.0, -2.0, 3.0]])
    assert torch.equal(topk_relu(pre, 3), torch.tensor([[1.0, 0.0, 3.0]]))
    with pytest.raises(ValueError):
        topk_relu(pre, 0)
    with pytest.raises(ValueError):
        topk_relu(pre, 4)
    assert SparseAutoencoder(2, 3, kind="topk", k=3).k == 3
    with pytest.raises(ValueError):
        SparseAutoencoder(2, 3, kind="topk", k=0)
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    x_hat = torch.zeros(2, 2)
    z = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert l1_loss(x, x_hat, z, 0.5).item() == pytest.approx(2.5 + 0.5 * 3.0)
    assert l1_loss(x, x_hat, z, 0.0).item() == pytest.approx(mse_loss(x, x_hat).item())
    x3 = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    assert fvu(x3, x3.mean(dim=0, keepdim=True).expand(2, 3)).item() == pytest.approx(
        1.0
    )
    assert fvu(x3, torch.zeros(2, 3)).item() == pytest.approx(10.0 / 4.0)
