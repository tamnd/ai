# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/group-relative-policy-optimization/ (ai://NEW-group-relative-policy-optimization).
import functools
import math

import torch
from grpo import Policy, generate, group_advantages, grpo_loss, success_rate, token_logprobs, train, verify

torch.manual_seed(0)
Q, V = 8, 6


@functools.cache
def trained(mode):
    torch.manual_seed(0)
    answers = torch.randint(0, V - 1, (Q,))
    policy = Policy(Q, V)
    before = success_rate(policy, answers)
    history = train(policy, answers, mode=mode)
    return policy, answers, before, history


def test_generation_masks_after_eos_and_verifier():
    torch.manual_seed(1)
    policy = Policy(Q, V)
    qq, tokens, mask, finished = generate(policy, torch.arange(Q), 4, 5)
    eos = V - 1
    for i in range(len(qq)):
        n = int(mask[i].sum())
        if finished[i]:
            assert tokens[i, n - 1] == eos and (tokens[i, : n - 1] != eos).all()
        else:
            assert n == 5
    tokens = torch.tensor([[3, eos, 0, 0], [3, 1, 2, 4], [eos, 0, 0, 0], [2, eos, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]]).float()
    finished = torch.tensor([True, False, True, True])
    answers = torch.tensor([3, 3, 3, 3])
    assert verify(tokens, mask, finished, answers, torch.zeros(4, dtype=torch.long)).tolist() == [1, 0, 0, 0]


def test_first_step_loss_is_minus_mean_advantage_and_reductions():
    adv = torch.tensor([1.0, -1.0, 2.0])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]).float()
    lp = torch.randn(3, 4)
    seq = grpo_loss(lp, lp, lp, adv, mask, mode="sequence")
    tok = grpo_loss(lp, lp, lp, adv, mask, mode="token")
    const = grpo_loss(lp, lp, lp, adv, mask, mode="constant", max_len=4)
    assert math.isclose(seq.item(), -adv.mean().item(), rel_tol=1e-6)
    assert math.isclose(tok.item(), -(2 * 1 - 4 * 1 + 1 * 2) / 7, abs_tol=1e-6)
    assert math.isclose(const.item(), -(2 - 4 + 2) / 12, abs_tol=1e-6)


def test_gradient_at_ratio_one_is_reinforce_with_group_baseline():
    torch.manual_seed(2)
    policy = Policy(Q, V)
    qq, tokens, mask, _ = generate(policy, torch.arange(4), 4, 4)
    adv = torch.randn(len(qq))
    logp = token_logprobs(policy, qq, tokens)
    loss = grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, beta=0.3)
    g1 = torch.autograd.grad(loss, list(policy.parameters()))
    logp = token_logprobs(policy, qq, tokens)
    pg = -((adv[:, None] * logp * mask).sum(1) / mask.sum(1)).mean()
    g2 = torch.autograd.grad(pg, list(policy.parameters()))
    for a, b in zip(g1, g2, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_group_advantages():
    r = torch.tensor([1.0, 0, 0, 0, 1, 1, 1, 1])
    a = group_advantages(r, 4)
    assert torch.allclose(a[:4].sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(a[0], torch.tensor(math.sqrt(3.0)), atol=1e-4)
    assert (a[4:] == 0).all()


def test_grpo_raises_the_success_rate():
    policy, answers, before, history = trained("sequence")
    assert before < 0.25
    assert success_rate(policy, answers) > 0.85
    assert sum(history[-10:]) / 10 > sum(history[:10]) / 10 + 0.5


def test_token_level_loss_also_learns():
    policy, answers, before, _ = trained("token")
    assert success_rate(policy, answers) > 0.85 > before
