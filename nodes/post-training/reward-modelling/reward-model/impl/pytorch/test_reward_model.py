# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/reward-model/ (ai://NEW-reward-model).
"""Tests of the PyTorch code: the fixture shared with the NumPy tests, autograd gradients against the analytic
-sigma(-d), the noisy loss's stability and vanishing pull, the batched ranked loss against a loop, ensembles, a reward
that ignores right padding, and a training run on toy comparisons that reaches high held-out accuracy."""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F
from reward_model import TinyRewardModel, ensemble, last_token_index, pairwise_loss, ranked_loss

torch.set_default_dtype(torch.float64)

# The fixture shared with the NumPy tests.
RW = [2.0, 0.5, -1.0, 0.3]
RL = [0.5, 1.0, -1.0, -2.0]
MARGIN = [1.0, 2 / 3, 0.0, 1 / 3]
FIXTURE_PLAIN = 0.4910457268301918
FIXTURE_MARGIN = 0.6839872064402405
FIXTURE_NOISY = 0.5044775494077371
FIXTURE_BATCH_RANKED = 0.303617303750964
RANKINGS = [[2.0, 0.5, 0.0], [1.0, -1.0], [0.3, 0.2, 0.1, -0.5]]


def padded(rankings):
    k = max(len(r) for r in rankings)
    rewards = torch.tensor([r + [0.0] * (k - len(r)) for r in rankings])
    return rewards, torch.tensor([len(r) for r in rankings])


def test_fixture():
    rw, rl, m = torch.tensor(RW), torch.tensor(RL), torch.tensor(MARGIN)
    assert abs(pairwise_loss(rw, rl).item() - FIXTURE_PLAIN) < 1e-12
    assert abs(pairwise_loss(rw, rl, margin=m).item() - FIXTURE_MARGIN) < 1e-12
    assert abs(pairwise_loss(rw, rl, eps=0.1).item() - FIXTURE_NOISY) < 1e-12
    assert abs(ranked_loss(*padded(RANKINGS)).item() - FIXTURE_BATCH_RANKED) < 1e-12


def test_gradient_is_minus_sigma_of_minus_difference():
    rw = torch.tensor(RW, requires_grad=True)
    rl = torch.tensor(RL, requires_grad=True)
    pairwise_loss(rw, rl).backward()
    want = -torch.sigmoid(torch.tensor(RL) - torch.tensor(RW)) / 4
    assert torch.allclose(rw.grad, want) and torch.allclose(rl.grad, -want)


def test_noisy_loss_is_stable_and_stops_pulling():
    d = torch.tensor([-200.0, -3.0, 0.0, 200.0], requires_grad=True)
    loss = pairwise_loss(d, torch.zeros(4), eps=0.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(d.grad).all()
    assert abs(d.grad[0].item()) < 1e-15
    assert abs(4 * abs(d.grad[1].item()) - 0.438687) < 1e-6
    each = [pairwise_loss(d[i : i + 1].detach(), torch.zeros(1), eps=0.1).item() for i in range(4)]
    assert all(x <= -math.log(0.05) + 1e-12 for x in each)
    with pytest.raises(ValueError):
        pairwise_loss(d, torch.zeros(4), eps=1.0)


def test_ranked_loss_matches_loop():
    torch.manual_seed(0)
    counts = torch.tensor([4, 9, 2, 6])
    rewards = torch.randn(4, 9)
    loop = []
    for b, k in enumerate(counts.tolist()):
        pairs = list(itertools.combinations(range(k), 2))
        loop.append(sum(-F.logsigmoid(rewards[b, i] - rewards[b, j]) for i, j in pairs) / len(pairs))
    assert torch.isclose(ranked_loss(rewards, counts), torch.stack(loop).mean())
    # Padding values do not matter.
    noisy = rewards.clone()
    noisy[0, 4:] = 100.0
    assert torch.isclose(ranked_loss(noisy, counts), ranked_loss(rewards, counts))
    r = torch.tensor([[2.0, 0.5, 0.0]], requires_grad=True)
    ranked_loss(r, torch.tensor([3])).backward()
    assert torch.allclose(r.grad, torch.tensor([[-0.100543, -0.065038, 0.165581]]), atol=1e-6)
    with pytest.raises(ValueError):
        ranked_loss(rewards, torch.tensor([4, 9, 1, 6]))


def test_ensemble():
    r = torch.tensor([[1.0], [1.2], [3.0]])
    assert abs(ensemble(r).item() - 1.733333) < 1e-6
    assert ensemble(r, "min").item() == 1.0
    assert abs(ensemble(r, "uwo", lam=0.5).item() - 1.328889) < 1e-6
    with pytest.raises(ValueError):
        ensemble(r, "median")


def test_reward_ignores_right_padding():
    torch.manual_seed(1)
    model = TinyRewardModel(vocab=8, dim=16, heads=2, max_len=12).eval()
    seq = torch.tensor([[3, 1, 4, 1, 5]])
    pad = torch.tensor([[3, 1, 4, 1, 5, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    assert torch.allclose(model(seq), model(pad, mask), atol=1e-12)
    # Reading the last position instead would see the padding.
    assert not torch.allclose(model(seq), model(pad), atol=1e-4)
    assert last_token_index(mask).tolist() == [4]
    with pytest.raises(ValueError):
        last_token_index(torch.zeros(1, 3))


def _toy(n, gen):
    """Responses of 6 tokens over {0..5} after a prompt token 6; the true reward is the number of 1s minus the number
    of 2s. Returns pairs (preferred, rejected) with distinct true rewards."""
    a = torch.randint(0, 6, (n, 6), generator=gen)
    b = torch.randint(0, 6, (n, 6), generator=gen)

    def true(x):
        return (x == 1).sum(-1) - (x == 2).sum(-1)

    keep = true(a) != true(b)
    a, b = a[keep], b[keep]
    first = (true(a) > true(b))[:, None]
    w, lo = torch.where(first, a, b), torch.where(first, b, a)
    prompt = torch.full((len(w), 1), 6)
    return torch.cat([prompt, w], 1), torch.cat([prompt, lo], 1)


def test_training_reaches_high_heldout_accuracy():
    gen = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    train_w, train_l = _toy(2000, gen)
    test_w, test_l = _toy(1000, gen)
    model = TinyRewardModel(vocab=7, dim=32, heads=4, max_len=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        before = (model(test_w) > model(test_l)).double().mean().item()
    for _ in range(300):
        idx = torch.randint(0, len(train_w), (128,), generator=gen)
        opt.zero_grad()
        pairwise_loss(model(train_w[idx]), model(train_l[idx])).backward()
        opt.step()
    with torch.no_grad():
        after = (model(test_w) > model(test_l)).double().mean().item()
    assert before < 0.7
    assert after > 0.9
