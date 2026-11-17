# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://CZW0BQ).
"""Tests of the PyTorch code: the worked example and the shared fixture, agreement with a hand-built reference written
with F.logsigmoid in the indexed form of the paper's appendix, the gradient weight sigmoid(r_l - r_w) from autograd,
the detached reference and rewards, stability, the IPO loss, sequence log-probabilities with the one-token shift,
and two small training runs: DPO recovers the optimal policy of the KL-constrained objective from Bradley-Terry
preferences, and on a deterministic preference its margin keeps growing while IPO's stops at 1/(2 tau)."""

import math

import pytest
import torch
import torch.nn.functional as F

from dpo import DPOLoss, dpo_loss, ipo_loss, sequence_logps

torch.set_default_dtype(torch.float64)

# The fixture shared with the NumPy tests: four pairs, beta = 0.1.
PC = [-12.0, -30.5, -7.25, -40.0]
PR = [-15.0, -28.0, -9.0, -41.5]
RC = [-13.0, -31.0, -7.0, -42.0]
RR = [-14.0, -29.0, -9.5, -40.5]
FIXTURE = 0.65057597567255


def t(x, grad=False):
    return torch.tensor(x, requires_grad=grad)


def reference_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """The indexed form: one vector of log-probabilities and index vectors for the preferred and dispreferred."""
    pi_logratios = pi_logps[yw_idxs] - pi_logps[yl_idxs]
    ref_logratios = ref_logps[yw_idxs] - ref_logps[yl_idxs]
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios))


def test_worked_example():
    pc = t([-12.0], grad=True)
    pr = t([-15.0], grad=True)
    loss, rw, rl = dpo_loss(pc, pr, t([-13.0]), t([-14.0]), beta=0.1)
    assert loss.item() == pytest.approx(0.5981388693815918, abs=1e-12)
    assert rw.item() == pytest.approx(0.1) and rl.item() == pytest.approx(-0.1)
    loss.backward()
    assert pc.grad.item() == pytest.approx(-0.04501660026875222, abs=1e-12)
    assert pr.grad.item() == pytest.approx(0.04501660026875222, abs=1e-12)


def test_fixture():
    assert dpo_loss(t(PC), t(PR), t(RC), t(RR), beta=0.1)[0].item() == pytest.approx(FIXTURE, abs=1e-12)


def test_matches_indexed_reference():
    g = torch.Generator().manual_seed(0)
    for _ in range(100):
        n = int(torch.randint(1, 9, (1,), generator=g))
        pi = -20 + 8 * torch.randn(2 * n, generator=g)
        ref = -20 + 8 * torch.randn(2 * n, generator=g)
        perm = torch.randperm(2 * n, generator=g)
        yw, yl = perm[:n], perm[n:]
        beta = float(torch.rand(1, generator=g)) * 2 + 0.01
        want = reference_loss(pi, ref, yw, yl, beta)
        got = dpo_loss(pi[yw], pi[yl], ref[yw], ref[yl], beta, reduction="none")[0]
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dpo_loss(pi[yw], pi[yl], ref[yw], ref[yl], beta)[0], want.mean())
        torch.testing.assert_close(
            dpo_loss(pi[yw], pi[yl], ref[yw], ref[yl], beta, reduction="sum")[0],
            want.sum(),
        )


def test_matches_numpy_formula():
    m = torch.linspace(-30, 30, 61)
    z = torch.zeros_like(m)
    got = dpo_loss(m, z, z, z, beta=1.0, reduction="none")[0]
    torch.testing.assert_close(got, torch.log1p(torch.exp(-m)))


def test_gradient_weight():
    pc, pr = t(PC, grad=True), t(PR, grad=True)
    rc, rr = t(RC), t(RR)
    loss, rw, rl = dpo_loss(pc, pr, rc, rr, beta=0.1, reduction="sum")
    loss.backward()
    weight = torch.sigmoid(rl - rw)
    torch.testing.assert_close(pc.grad, -0.1 * weight)
    torch.testing.assert_close(pr.grad, 0.1 * weight)
    # The mean divides the gradient by the number of pairs.
    pc2 = t(PC, grad=True)
    dpo_loss(pc2, t(PR), rc, rr, beta=0.1)[0].backward()
    torch.testing.assert_close(pc2.grad, -0.1 * weight / 4)


def test_reference_gets_no_gradient_and_rewards_are_detached():
    pc, pr = t(PC, grad=True), t(PR, grad=True)
    rc, rr = t(RC, grad=True), t(RR, grad=True)
    loss, rw, rl = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert not rw.requires_grad and not rl.requires_grad
    loss.backward()
    assert rc.grad is None and rr.grad is None
    ipo_loss(pc, pr, rc, rr, tau=0.1).backward()
    assert rc.grad is None and rr.grad is None


def test_equal_policy_and_reference():
    x, y = t([-3.0, -50.0]), t([-4.0, -1.0])
    torch.testing.assert_close(dpo_loss(x, y, x, y, 0.5, "none")[0], torch.full((2,), math.log(2.0)))


def test_stable_for_large_margins():
    z = t([0.0], grad=True)
    loss = dpo_loss(z, t([-1e5]), t([0.0]), t([0.0]), beta=1.0)[0]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    z2 = t([-1e5], grad=True)
    loss2 = dpo_loss(z2, t([0.0]), t([0.0]), t([0.0]), beta=1.0)[0]
    loss2.backward()
    assert loss2.item() == pytest.approx(1e5) and z2.grad.item() == pytest.approx(-1.0)
    # The textbook -log(sigmoid(.)) overflows to inf in the same place.
    assert torch.isinf(-torch.log(torch.sigmoid(t([-1e5])))).all()


def test_bad_arguments_and_module():
    with pytest.raises(ValueError):
        dpo_loss(t(PC), t(PR), t(RC), t(RR), beta=-1.0)
    with pytest.raises(ValueError):
        dpo_loss(t(PC), t(PR), t(RC), t(RR), beta=0.0)
    with pytest.raises(ValueError):
        dpo_loss(t(PC), t(PR), t(RC), t(RR), reduction="avg")
    with pytest.raises(ValueError):
        ipo_loss(t(PC), t(PR), t(RC), t(RR), tau=0.0)
    with pytest.raises(ValueError):
        DPOLoss(beta=0.0)
    assert DPOLoss(0.1)(t(PC), t(PR), t(RC), t(RR)).item() == pytest.approx(FIXTURE, abs=1e-12)
    assert DPOLoss(0.1, "none")(t(PC), t(PR), t(RC), t(RR)).shape == (4,)


def test_ipo_loss():
    assert ipo_loss(t(-12.0), t(-15.0), t(-13.0), t(-14.0), tau=0.1).item() == pytest.approx(9.0)
    pc = t([5.0], grad=True)
    loss = ipo_loss(pc, t([0.0]), t([0.0]), t([0.0]), tau=0.1)
    loss.backward()
    assert loss.item() == pytest.approx(0.0) and pc.grad.item() == pytest.approx(0.0)
    assert ipo_loss(t(PC), t(PR), t(RC), t(RR), tau=0.5, reduction="none").shape == (4,)


def test_sequence_logps_shift_and_mask():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 6, 5, generator=g)
    labels = torch.randint(0, 5, (2, 6), generator=g)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]])
    got = sequence_logps(logits, labels, mask)
    for b in range(2):
        want = 0.0
        for s in range(1, 6):
            if mask[b, s]:
                want += torch.log_softmax(logits[b, s - 1], -1)[labels[b, s]].item()
        assert got[b].item() == pytest.approx(want, rel=1e-12)
    # Padding after the completion, masked out, does not change the sum.
    labels2 = labels.clone()
    labels2[0, 5] = (labels[0, 5] + 1) % 5
    assert sequence_logps(logits, labels2, mask)[0].item() == pytest.approx(got[0].item())


def test_dpo_recovers_the_optimal_policy():
    # One prompt, four responses; preferences drawn from Bradley-Terry with reward r over all ordered pairs.
    ref = torch.tensor([0.4, 0.3, 0.2, 0.1])
    r = torch.tensor([1.0, 0.5, 0.0, -1.0])
    beta = 0.5
    target = torch.softmax(torch.log(ref) + r / beta, 0)
    i, j = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    p_ij = torch.sigmoid(r[i] - r[j])  # probability that i is preferred to j
    theta = torch.zeros(4, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax(theta, 0)
        lr = torch.log(ref)
        losses = dpo_loss(logp[i], logp[j], lr[i], lr[j], beta, "none")[0]
        loss = (p_ij * losses).sum() + ((1 - p_ij) * dpo_loss(logp[j], logp[i], lr[j], lr[i], beta, "none")[0]).sum()
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    torch.testing.assert_close(torch.softmax(theta.detach(), 0), target, rtol=1e-6, atol=1e-7)


def test_deterministic_preference_dpo_diverges_ipo_does_not():
    # y1 always beats y2: DPO keeps pushing the margin up, IPO settles at h = 1 / (2 tau).
    tau = beta = 0.5
    lr = torch.log(torch.tensor([0.5, 0.5]))
    th_dpo = torch.zeros(2, requires_grad=True)
    th_ipo = torch.zeros(2, requires_grad=True)
    opt = torch.optim.SGD([{"params": [th_dpo], "lr": 1.0}, {"params": [th_ipo], "lr": 0.1}])
    margins = []
    for step in range(3000):
        opt.zero_grad()
        a = torch.log_softmax(th_dpo, 0)
        b = torch.log_softmax(th_ipo, 0)
        loss = dpo_loss(a[:1], a[1:], lr[:1], lr[1:], beta)[0] + ipo_loss(b[:1], b[1:], lr[:1], lr[1:], tau)
        loss.backward()
        opt.step()
        if step in (999, 2999):
            margins.append((th_dpo[0] - th_dpo[1]).item())
    assert margins[1] > margins[0] + 1.0 and margins[1] > 10.0
    assert torch.softmax(th_dpo.detach(), 0)[1].item() < 1e-4
    assert (th_ipo[0] - th_ipo[1]).item() == pytest.approx(1.0 / (2 * tau), abs=1e-6)
