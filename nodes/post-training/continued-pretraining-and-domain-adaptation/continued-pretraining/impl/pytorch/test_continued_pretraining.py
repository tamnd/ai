# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/continued-pretraining/ (ai://NEW-continued-pretraining).
import copy
import math

import torch
import torch.nn.functional as F
from continued_pretraining import (
    EWC,
    TinyRoPEDecoder,
    continue_pretraining,
    extend_vocab_,
    fisher_diagonal,
    lm_loss,
    rewarm_cosine_lambda,
)

torch.set_default_dtype(torch.float64)


def markov(step, n, length, vocab, g):
    """Sequences where each token is the previous one plus `step`, mod vocab, with 10% random tokens."""
    start = torch.randint(vocab, (n, 1), generator=g)
    seq = (start + step * torch.arange(length)) % vocab
    noise = torch.rand(n, length, generator=g) < 0.1
    return torch.where(noise, torch.randint(vocab, (n, length), generator=g), seq)


def test_schedule():
    f = rewarm_cosine_lambda(10, 110, 0.1)
    assert f(0) == 0 and math.isclose(f(5), 0.5) and math.isclose(f(10), 1.0)
    assert math.isclose(f(60), 0.55) and math.isclose(f(110), 0.1) and math.isclose(f(500), 0.1)


def test_rope_base_change_keeps_weights_and_relative_scores():
    torch.manual_seed(0)
    m = TinyRoPEDecoder(11)
    ids = torch.randint(11, (2, 12))
    before = m(ids)
    state = copy.deepcopy(m.state_dict())
    m.set_rope_base(1e6, max_len=128)
    after = m(ids)
    assert all(torch.equal(state[k], v) for k, v in m.state_dict().items())
    assert not torch.allclose(before, after)
    # A single token has no offsets to encode: its output does not depend on the base.
    m.set_rope_base(1e4)
    one = m(ids[:, :1])
    m.set_rope_base(1e6)
    assert torch.allclose(one, m(ids[:, :1]))


def test_extend_vocab_mean_init():
    torch.manual_seed(1)
    m = TinyRoPEDecoder(20).eval()
    ids = torch.randint(20, (3, 9))
    with torch.no_grad():
        p = F.softmax(m(ids), -1)
    extend_vocab_(m, 12)
    assert m.tok.weight.shape == (32, 32) and m.head.weight.shape == (32, 32)
    with torch.no_grad():
        q = F.softmax(m(ids), -1)
    assert (q[..., 20:].sum(-1) <= 12 / 32 + 1e-12).all()
    kl = (p * (p.log() - q[..., :20].log())).sum(-1)
    assert (kl <= math.log1p(12 / 20) + 1e-12).all()
    # New tokens can be embedded and trained.
    new_ids = torch.randint(20, 32, (4, 9))
    loss = lm_loss(m, new_ids)
    loss.backward()
    assert m.tok.weight.grad[20:].abs().sum() > 0


def test_replay_and_ewc_reduce_forgetting():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    V = 12
    old, new = markov(1, 256, 16, V, g), markov(5, 256, 16, V, g)
    old_test, new_test = markov(1, 128, 16, V, g), markov(5, 128, 16, V, g)
    base = TinyRoPEDecoder(V)
    continue_pretraining(base, old, steps=150, lr=1e-2, generator=g)
    with torch.no_grad():
        before = lm_loss(base, old_test).item()

    def run(**kw):
        torch.manual_seed(1)
        m = copy.deepcopy(base)
        continue_pretraining(m, new, steps=120, lr=1e-2, generator=torch.Generator().manual_seed(1), **kw)
        with torch.no_grad():
            return lm_loss(m, old_test).item() - before, lm_loss(m, new_test).item()

    f_plain, n_plain = run()
    f_replay, n_replay = run(old=old, replay_fraction=0.25)
    fisher = fisher_diagonal(base, [old[:64]])
    f_ewc, _ = run(ewc=EWC(base, fisher, lam=200.0))
    assert f_plain > 0.5
    assert f_replay < 0.5 * f_plain
    assert f_ewc < f_plain
    assert n_plain < 1.5 and n_replay < 1.5  # both still learn the new data
