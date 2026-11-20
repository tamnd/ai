# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/safety-training/ (ai://NEW-safety-training).
import copy

import torch

from safety_training import (
    TinyDecoder,
    ablation_hook,
    context_distillation_loss,
    gcg_step,
    orthogonalize_,
    refusal_direction,
    target_loss,
    token_gradients,
)

torch.set_default_dtype(torch.float64)


def model(seed=0, vocab=23):
    torch.manual_seed(seed)
    return TinyDecoder(vocab).eval()


def test_orthogonalized_weights_match_the_ablation_hook():
    m = model()
    ids = torch.randint(23, (2, 9))
    r_hat = torch.nn.functional.normalize(torch.randn(32), dim=0)
    hooked, stream_h = m(ids, hook=ablation_hook(r_hat), return_stream=True)
    edited = orthogonalize_(copy.deepcopy(m), r_hat)
    plain, stream_e = edited(ids, return_stream=True)
    assert torch.allclose(hooked, plain, atol=1e-10)
    for a, b in zip(stream_h, stream_e):
        assert torch.allclose(a, b, atol=1e-10)
        assert (b @ r_hat).abs().max() < 1e-10
    assert not torch.allclose(m(ids), plain)


def test_refusal_direction_and_activation_addition():
    m = model(1)
    harmful = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 5, 4]])]
    harmless = [torch.tensor([[7, 8, 9, 4]]), torch.tensor([[7, 8, 10, 4]])]
    r, r_hat = refusal_direction(m, harmful, harmless, layer=1)
    assert torch.isclose(r_hat.norm(), torch.tensor(1.0)) and torch.isclose(r @ r_hat, r.norm())
    # Adding r to the stream entering block 1 on harmless prompts moves their activation there by exactly r.
    for p in harmless:
        _, s0 = m(p, return_stream=True)
        _, s1 = m(p, add_at=(1, r), return_stream=True)
        assert torch.allclose(s0[1], s1[1])
        assert not torch.allclose(s0[2], s1[2])


def test_context_distillation_loss():
    m = model(2)
    prompt, pre, resp = torch.randint(23, (3, 5)), torch.randint(23, (3, 4)), torch.randint(23, (3, 6))
    loss = context_distillation_loss(m, prompt, pre, resp)
    assert loss.item() > 0
    # With an empty preprompt the teacher is the student and the KL is zero.
    assert context_distillation_loss(m, prompt, pre[:, :0], resp).abs().item() < 1e-12
    # Gradients flow to the student only and reduce the loss.
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    before = loss.item()
    m.train()
    for _ in range(30):
        opt.zero_grad()
        context_distillation_loss(m, prompt, pre, resp).backward()
        opt.step()
    assert context_distillation_loss(m, prompt, pre, resp).item() < before


def test_token_gradients_predict_swaps_to_first_order():
    m = model(3)
    ids = torch.randint(23, (1, 10))
    sl, target = slice(3, 7), ids[:, -3:]
    grad = token_gradients(m, ids, sl, target)
    assert grad.shape == (4, 23)
    base = target_loss(m, m.tok(ids), target)
    # Moving the embedding a small step along one token direction matches the directional derivative.
    i, v, eps = 1, 5, 1e-6
    w = m.tok.weight
    e = m.tok(ids).detach().clone()
    e[0, sl.start + i] += eps * (w[v] - w[ids[0, sl.start + i]])
    with torch.no_grad():
        num = (target_loss(m, e, target) - base) / eps
    assert torch.isclose(num, grad[i, v] - grad[i, ids[0, sl.start + i]], atol=1e-5)


def test_gcg_lowers_the_target_loss_and_keeps_the_rest():
    m = model(4)
    ids = torch.randint(23, (1, 12))
    sl, target = slice(4, 8), ids[:, -3:]
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        start = target_loss(m, m.tok(ids), target).item()
    cur, cur_loss = ids, start
    for _ in range(10):
        new, loss = gcg_step(m, cur, sl, target, k=8, batch=32, generator=g)
        assert torch.equal(new[:, : sl.start], ids[:, : sl.start]) and torch.equal(new[:, sl.stop :], ids[:, sl.stop :])
        if loss.item() < cur_loss:
            cur, cur_loss = new, loss.item()
    assert cur_loss < start
