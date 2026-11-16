# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reinforcement-learning-from-human-feedback/ (ai://NEW-reinforcement-learning-from-human-feedback).
"""Tests of the PyTorch losses: the shared fixture, agreement with torch.nn.functional, the gradients of autograd
against the closed forms, a reward model and a DPO step on toy tensors, the frozen reference, and the masks."""

import math

import pytest
import torch
import torch.nn.functional as F

from rlhf import (bt_loss, dpo_loss, implicit_reward, kl_shaped_rewards, ppo_clip_loss, ptx_loss, ranked_loss,
                  sequence_logprob)

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
RW = [1.0, -0.5, 2.0, 0.3]
RL = [0.0, 0.5, -1.0, 0.3]
BT_LOSS = 0.5920644767925333
BT_GRAD_W = [-0.06723535534249878, -0.18276464465750122, -0.011856468294391695, -0.125]
RANKED = [[2.0, 1.0, 0.0, -1.0], [0.5, 1.5, -0.5, 0.0]]
RANKED_LOSS = 0.38710392238639496
LOGP = [[-1.0, -2.0, -0.5], [-0.3, -0.7, 0.0]]
LOGP_REF = [[-1.2, -1.5, -0.5], [-0.4, -0.2, 0.0]]
MASK = [[1, 1, 1], [1, 1, 0]]
SCORE = [1.0, -0.5]
KL_REWARDS = [[-0.02, 0.05, 1.0], [-0.01, -0.45, 0.0]]
PPO_LOGP = [0.0, 0.5, -0.5, 0.1]
PPO_OLD = [0.0, 0.0, 0.0, 0.0]
PPO_ADV = [1.0, 1.0, -1.0, -2.0]
PPO_OBJ = -0.20258545903782382
DPO_ARGS = ([-1.0, -2.0], [-1.5, -1.0], [-1.2, -1.8], [-1.2, -1.4], 0.1)
DPO_LOSS = 0.6960283805447136


def t(v, grad=False):
    return torch.tensor(v, dtype=torch.float64, requires_grad=grad)


def test_fixture_and_autograd():
    rw, rl = t(RW, True), t(RL, True)
    loss = bt_loss(rw, rl)
    assert loss.item() == pytest.approx(BT_LOSS, rel=1e-12)
    loss.backward()
    torch.testing.assert_close(rw.grad, t(BT_GRAD_W), rtol=1e-12, atol=0)
    torch.testing.assert_close(rl.grad, -t(BT_GRAD_W), rtol=1e-12, atol=0)
    assert ranked_loss(t(RANKED)).item() == pytest.approx(RANKED_LOSS, rel=1e-12)
    out = kl_shaped_rewards(t(SCORE), t(LOGP), t(LOGP_REF), 0.1, t(MASK))
    torch.testing.assert_close(out, t(KL_REWARDS), rtol=0, atol=1e-12)
    assert ppo_clip_loss(t(PPO_LOGP), t(PPO_OLD), t(PPO_ADV)).item() == pytest.approx(-PPO_OBJ, rel=1e-12)
    assert dpo_loss(*[t(a) for a in DPO_ARGS[:4]], DPO_ARGS[4]).item() == pytest.approx(DPO_LOSS, rel=1e-12)


def test_bt_matches_torch_builtins():
    g = torch.Generator().manual_seed(20261115)
    rw, rl = torch.randn(64, generator=g) * 5, torch.randn(64, generator=g) * 5
    d = rw - rl
    ref = F.binary_cross_entropy_with_logits(d, torch.ones_like(d))
    torch.testing.assert_close(bt_loss(rw, rl), ref, rtol=1e-12, atol=0)
    torch.testing.assert_close(bt_loss(rw, rl), F.softplus(-d).mean(), rtol=1e-12, atol=0)
    # The two-way cross-entropy of Christiano et al. and Ziegler et al., with the preferred response as the label.
    two = torch.stack([rw, rl], dim=1)
    torch.testing.assert_close(bt_loss(rw, rl), F.cross_entropy(two, torch.zeros(64, dtype=torch.long)), rtol=1e-12,
                               atol=0)
    # Large margins stay finite.
    assert torch.isfinite(bt_loss(t([-800.0]), t([0.0])))


def test_bt_gradient_closed_form():
    g = torch.Generator().manual_seed(3)
    rw = (torch.randn(16, generator=g) * 3).requires_grad_()
    rl = (torch.randn(16, generator=g) * 3).requires_grad_()
    bt_loss(rw, rl).backward()
    expected = -torch.sigmoid(-(rw - rl)).detach() / 16
    torch.testing.assert_close(rw.grad, expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(rl.grad, -expected, rtol=1e-12, atol=0)


def test_reward_model_learns_preferences():
    # A linear reward model on features, trained on pairs labeled by a hidden linear reward, recovers its direction.
    g = torch.Generator().manual_seed(20261114)
    true_w = torch.tensor([1.0, -2.0, 0.5])
    xa, xb = torch.randn(512, 3, generator=g), torch.randn(512, 3, generator=g)
    a_wins = torch.rand(512, generator=g) < torch.sigmoid(xa @ true_w - xb @ true_w)
    xw = torch.where(a_wins[:, None], xa, xb)
    xl = torch.where(a_wins[:, None], xb, xa)
    w = torch.zeros(3, requires_grad=True)
    opt = torch.optim.SGD([w], lr=1.0)
    for _ in range(300):
        opt.zero_grad()
        loss = bt_loss(xw @ w, xl @ w)
        loss.backward()
        opt.step()
    cos = torch.dot(w, true_w) / (w.norm() * true_w.norm())
    assert cos.item() > 0.95
    assert loss.item() < math.log(2)


def test_ranked_loss_matches_pairs():
    r = t([[3.0, 1.0, 0.5, -1.0, -2.0]])
    i, j = zip(*[(a, b) for a in range(5) for b in range(a + 1, 5)])
    torch.testing.assert_close(ranked_loss(r), bt_loss(r[0, list(i)], r[0, list(j)]), rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        ranked_loss(t([1.0, 2.0]))


def test_sequence_logprob_mask():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 4, 6, generator=g)
    tokens = torch.randint(0, 6, (2, 4), generator=g)
    mask = t([[0, 1, 1, 1], [0, 0, 1, 0]])
    lp = torch.log_softmax(logits, dim=-1)
    want = torch.stack([lp[0, 1, tokens[0, 1]] + lp[0, 2, tokens[0, 2]] + lp[0, 3, tokens[0, 3]],
                        lp[1, 2, tokens[1, 2]]])
    torch.testing.assert_close(sequence_logprob(logits, tokens, mask), want, rtol=1e-12, atol=0)


def test_kl_shaped_rewards_detached_and_masked():
    logp = t([[-1.0, -1.0, -9.0]], True)
    out = kl_shaped_rewards(t([2.0], True), logp, t([[-1.0, -1.0, 0.0]]), 1.0, t([[1, 1, 0]]))
    torch.testing.assert_close(out, t([[0.0, 2.0, 0.0]]), rtol=0, atol=1e-12)
    assert not out.requires_grad
    # The sign: more mass than the reference is penalized.
    assert kl_shaped_rewards(t([0.0]), t([[-0.5]]), t([[-1.0]]), 0.2).item() == pytest.approx(-0.1)


def test_kl_penalty_matches_kl_div():
    # The expectation of log(pi / pi_ref) under pi is KL(pi || pi_ref), which F.kl_div computes from log pi_ref.
    pi = t([0.2, 0.3, 0.5])
    ref = t([0.5, 0.3, 0.2])
    kl = (pi * (pi.log() - ref.log())).sum()
    torch.testing.assert_close(kl, F.kl_div(ref.log(), pi, reduction="sum"), rtol=1e-12, atol=0)


def test_ppo_clip_gradient():
    # For A > 0 and rho above 1 + eps the gradient is zero; inside the interval it is -rho A, that of the
    # unclipped surrogate.
    logp = t([math.log(1.5), math.log(1.1), math.log(0.5), math.log(0.5)], True)
    adv = t([1.0, 1.0, -1.0, 1.0])
    ppo_clip_loss(logp, torch.zeros(4), adv).backward()
    want = -t([0.0, 1.1, 0.0, 0.5]) / 4
    torch.testing.assert_close(logp.grad, want, rtol=1e-12, atol=1e-15)


def test_ptx_loss():
    assert ptx_loss(t(1.0), t([-2.0, -4.0]), 27.8).item() == pytest.approx(1.0 + 27.8 * 3.0, rel=1e-12)
    assert ptx_loss(t(1.0), t([-2.0, -4.0]), 0.0).item() == 1.0


def test_dpo_matches_reference_code():
    # The loss in the code of Appendix B of Rafailov et al.: -logsigmoid(beta (pi_logratios - ref_logratios)).
    g = torch.Generator().manual_seed(5)
    pw, pl, rw, rl = (torch.randn(32, generator=g) - 3 for _ in range(4))
    beta = 0.1
    ref = -F.logsigmoid(beta * ((pw - pl) - (rw - rl))).mean()
    torch.testing.assert_close(dpo_loss(pw, pl, rw, rl, beta), ref, rtol=1e-12, atol=0)
    torch.testing.assert_close(implicit_reward(pw, rw, beta), beta * (pw - rw), rtol=1e-12, atol=0)


def test_dpo_reference_is_frozen():
    pw, pl = t([-1.0], True), t([-1.5], True)
    rw, rl = t([-1.2], True), t([-1.2], True)
    loss = dpo_loss(pw, pl, rw, rl, 0.1)
    loss.backward()
    assert rw.grad is None and rl.grad is None
    # The gradient is -beta sigma(r_hat_l - r_hat_w) on log pi(y_w) and the opposite on log pi(y_l).
    weight = torch.sigmoid(t(0.1 * ((-1.5 + 1.2) - (-1.0 + 1.2))))
    assert pw.grad.item() == pytest.approx(-0.1 * weight.item(), rel=1e-12)
    assert pl.grad.item() == pytest.approx(0.1 * weight.item(), rel=1e-12)


def test_dpo_step_on_toy_policy():
    # A softmax policy over 4 responses of one prompt, trained by DPO on the pair (2 preferred to 0), moves mass to
    # response 2 and away from 0, and the loss falls.
    ref_logits = t([0.5, 0.0, -0.5, 0.0])
    logits = ref_logits.clone().requires_grad_()
    ref = torch.log_softmax(ref_logits, dim=0)
    opt = torch.optim.SGD([logits], lr=1.0)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        lp = torch.log_softmax(logits, dim=0)
        loss = dpo_loss(lp[2:3], lp[0:1], ref[2:3], ref[0:1], 0.5)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    p = torch.softmax(logits, dim=0).detach()
    assert losses[0] == pytest.approx(math.log(2), rel=1e-12)
    assert losses[-1] < losses[0]
    assert p[2] > torch.softmax(ref_logits, 0)[2] and p[0] < torch.softmax(ref_logits, 0)[0]
