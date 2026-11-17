# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/direct-preference-optimization/ (ai://NEW-direct-preference-optimization).
"""Tests of the NumPy code: the worked example and the shared fixture, the loss against the textbook formula, its
stability, the gradient against finite differences, symmetry and invariances, the optimal policy of the
KL-constrained objective and the cancelling partition function, the IPO loss and sequence log-probabilities."""

import numpy as np
import pytest

from dpo import (
    bradley_terry,
    dpo_loss,
    dpo_loss_grad,
    dpo_margins,
    implicit_rewards,
    ipo_loss,
    kl_regularized_objective,
    log_partition,
    log_sigmoid,
    optimal_policy,
    preference_accuracy,
    sequence_logps,
    sigmoid,
)

# The fixture shared with the PyTorch tests: four pairs, beta = 0.1.
PC = [-12.0, -30.5, -7.25, -40.0]
PR = [-15.0, -28.0, -9.0, -41.5]
RC = [-13.0, -31.0, -7.0, -42.0]
RR = [-14.0, -29.0, -9.5, -40.5]
FIXTURE = 0.65057597567255


def naive_loss(pc, pr, rc, rr, beta):
    m = beta * ((np.asarray(pc) - np.asarray(rc)) - (np.asarray(pr) - np.asarray(rr)))
    return -np.log(1.0 / (1.0 + np.exp(-m)))


def random_batch(rng, n):
    return [rng.normal(-20.0, 8.0, size=n) for _ in range(4)]


def test_worked_example():
    assert dpo_loss(-12.0, -15.0, -13.0, -14.0, beta=0.1) == pytest.approx(0.5981388693815918, abs=1e-12)
    rw = implicit_rewards(-12.0, -13.0, 0.1)
    rl = implicit_rewards(-15.0, -14.0, 0.1)
    assert rw == pytest.approx(0.1) and rl == pytest.approx(-0.1)
    gw, gl = dpo_loss_grad(
        np.array([-12.0]),
        np.array([-15.0]),
        np.array([-13.0]),
        np.array([-14.0]),
        beta=0.1,
    )
    assert gw[0] == pytest.approx(-0.04501660026875222, abs=1e-12)
    assert gl[0] == pytest.approx(0.04501660026875222, abs=1e-12)


def test_fixture():
    assert dpo_loss(PC, PR, RC, RR, beta=0.1) == pytest.approx(FIXTURE, abs=1e-12)


def test_matches_naive_formula():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        pc, pr, rc, rr = random_batch(rng, n)
        beta = float(rng.uniform(0.01, 2.0))
        want = naive_loss(pc, pr, rc, rr, beta)
        np.testing.assert_allclose(
            dpo_loss(pc, pr, rc, rr, beta, reduction="none"),
            want,
            rtol=1e-10,
            atol=1e-12,
        )
        assert dpo_loss(pc, pr, rc, rr, beta) == pytest.approx(want.mean(), rel=1e-10)
        assert dpo_loss(pc, pr, rc, rr, beta, reduction="sum") == pytest.approx(want.sum(), rel=1e-10)


def test_equal_policy_and_reference_gives_log2():
    x = np.array([-3.0, -50.0, -7.5])
    y = np.array([-4.0, -20.0, -1.0])
    np.testing.assert_allclose(dpo_loss(x, y, x, y, beta=0.7, reduction="none"), np.log(2.0), rtol=1e-12)


def test_stable_for_large_margins():
    big = dpo_loss([0.0], [-1e5], [0.0], [0.0], beta=1.0, reduction="none")
    assert np.isfinite(big).all() and big[0] < 1e-300 + 1e-12
    small = dpo_loss([-1e5], [0.0], [0.0], [0.0], beta=1.0, reduction="none")
    assert small[0] == pytest.approx(1e5, rel=1e-12)
    assert log_sigmoid(-800.0) == pytest.approx(-800.0)
    assert sigmoid(800.0) == 1.0 and sigmoid(-800.0) >= 0.0


def test_loss_is_softplus_of_negative_margin():
    m = np.linspace(-30, 30, 61)
    got = dpo_loss(
        m,
        np.zeros_like(m),
        np.zeros_like(m),
        np.zeros_like(m),
        beta=1.0,
        reduction="none",
    )
    np.testing.assert_allclose(got, np.logaddexp(0.0, -m), rtol=1e-12)
    np.testing.assert_allclose(got - got[::-1], -m, atol=1e-10)


def test_gradient_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 6))
        pc, pr, rc, rr = random_batch(rng, n)
        beta = float(rng.uniform(0.05, 1.5))
        for reduction in ("mean", "sum"):
            gw, gl = dpo_loss_grad(pc, pr, rc, rr, beta, reduction)
            eps = 1e-6
            for i in range(n):
                e = np.zeros(n)
                e[i] = eps
                fw = (dpo_loss(pc + e, pr, rc, rr, beta, reduction) - dpo_loss(pc - e, pr, rc, rr, beta, reduction)) / (
                    2 * eps
                )
                fl = (dpo_loss(pc, pr + e, rc, rr, beta, reduction) - dpo_loss(pc, pr - e, rc, rr, beta, reduction)) / (
                    2 * eps
                )
                assert gw[i] == pytest.approx(fw, rel=1e-5, abs=1e-7)
                assert gl[i] == pytest.approx(fl, rel=1e-5, abs=1e-7)


def test_gradient_weight_is_sigmoid_of_reversed_margin():
    pc, pr, rc, rr = map(np.array, (PC, PR, RC, RR))
    gw, gl = dpo_loss_grad(pc, pr, rc, rr, 0.1, reduction="sum")
    rw = implicit_rewards(pc, rc, 0.1)
    rl = implicit_rewards(pr, rr, 0.1)
    np.testing.assert_allclose(gw, -0.1 * sigmoid(rl - rw), rtol=1e-12)
    np.testing.assert_allclose(gl, -gw, rtol=1e-12)
    # A pair the implicit reward ranks wrongly gets a larger weight than one it ranks rightly.
    order = np.argsort(rw - rl)
    assert np.all(np.diff(np.abs(gw)[order]) < 0)


def test_prompt_offset_invariance():
    rng = np.random.default_rng(2)
    pc, pr, rc, rr = random_batch(rng, 5)
    c = rng.normal(0, 10, size=5)
    d = rng.normal(0, 10, size=5)
    base = dpo_loss(pc, pr, rc, rr, 0.3, reduction="none")
    np.testing.assert_allclose(dpo_loss(pc + c, pr + c, rc, rr, 0.3, reduction="none"), base, rtol=1e-10)
    np.testing.assert_allclose(dpo_loss(pc, pr, rc + d, rr + d, 0.3, reduction="none"), base, rtol=1e-10)
    # Shifting only one sequence changes the loss.
    assert not np.allclose(dpo_loss(pc + 1.0, pr, rc, rr, 0.3, reduction="none"), base)


def test_beta_scales_margin():
    pc, pr, rc, rr = map(np.array, (PC, PR, RC, RR))
    np.testing.assert_allclose(
        dpo_margins(pc, pr, rc, rr, 0.5),
        5.0 * dpo_margins(pc, pr, rc, rr, 0.1),
        rtol=1e-12,
    )


def test_bad_arguments():
    with pytest.raises(ValueError):
        dpo_loss(PC, PR, RC, RR, beta=0.0)
    with pytest.raises(ValueError):
        dpo_loss(PC, PR, RC, RR, beta=0.1, reduction="batchmean")
    with pytest.raises(ValueError):
        ipo_loss(PC, PR, RC, RR, tau=0.0)


def test_reduction_none_shape():
    assert dpo_loss(PC, PR, RC, RR, reduction="none").shape == (4,)


def test_preference_accuracy():
    assert preference_accuracy(PC, PR, RC, RR) == pytest.approx(0.5)
    assert preference_accuracy([0.0], [0.0], [0.0], [0.0]) == 0.0


def test_optimal_policy_closed_form():
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([1.0, 0.0, -1.0])
    p = optimal_policy(ref, r, 0.5)
    np.testing.assert_allclose(p, [0.91867231, 0.07459727, 0.00673043], atol=1e-8)
    assert p.sum() == pytest.approx(1.0)
    assert np.exp(log_partition(ref, r, 0.5)) == pytest.approx(4.021595106112647, rel=1e-12)


def test_optimal_policy_maximizes_objective():
    rng = np.random.default_rng(3)
    for _ in range(100):
        k = int(rng.integers(2, 7))
        ref = rng.dirichlet(np.ones(k))
        r = rng.normal(0, 2, size=k)
        beta = float(rng.uniform(0.1, 3.0))
        p = optimal_policy(ref, r, beta)
        best = kl_regularized_objective(p, ref, r, beta)
        assert best == pytest.approx(beta * log_partition(ref, r, beta), rel=1e-10, abs=1e-12)
        for _ in range(20):
            q = rng.dirichlet(np.ones(k))
            assert kl_regularized_objective(q, ref, r, beta) <= best + 1e-12
        # The gap is beta times KL(q || pi*).
        q = rng.dirichlet(np.ones(k))
        gap = best - kl_regularized_objective(q, ref, r, beta)
        assert gap == pytest.approx(beta * (q * np.log(q / p)).sum(), rel=1e-8, abs=1e-12)


def test_optimal_policy_limits():
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([1.0, 0.0, -1.0])
    np.testing.assert_allclose(optimal_policy(ref, r, 1e4), ref, atol=1e-4)
    assert optimal_policy(ref, r, 1e-3)[0] == pytest.approx(1.0)


def test_reward_recovery_and_partition_cancels():
    rng = np.random.default_rng(4)
    for _ in range(100):
        k = int(rng.integers(2, 7))
        ref = rng.dirichlet(np.ones(k))
        r = rng.normal(0, 2, size=k)
        beta = float(rng.uniform(0.1, 3.0))
        p = optimal_policy(ref, r, beta)
        implied = implicit_rewards(np.log(p), np.log(ref), beta)
        np.testing.assert_allclose(implied + beta * log_partition(ref, r, beta), r, atol=1e-9)
        i, j = rng.choice(k, size=2, replace=False)
        assert bradley_terry(implied[i], implied[j]) == pytest.approx(float(bradley_terry(r[i], r[j])), rel=1e-9)


def test_bradley_terry_form():
    assert bradley_terry(1.0, 0.0) == pytest.approx(np.e / (np.e + 1.0))
    assert bradley_terry(3.0, 3.0) == pytest.approx(0.5)


def test_ipo_loss():
    assert ipo_loss(-12.0, -15.0, -13.0, -14.0, tau=0.1) == pytest.approx(9.0)
    # Minimized, at zero, where the log-ratio gap h equals 1/(2 tau), whatever the sequences.
    assert ipo_loss(5.0, 0.0, 0.0, 0.0, tau=0.1) == pytest.approx(0.0)
    assert ipo_loss(PC, PR, RC, RR, tau=0.5, reduction="sum") == pytest.approx(
        sum(((a - b) - (c - d) - 1.0) ** 2 for a, b, c, d in zip(PC, PR, RC, RR, strict=True))
    )


def test_sequence_logps():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(2, 5, 7))
    labels = rng.integers(0, 7, size=(2, 5))
    mask = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    got = sequence_logps(logits, labels, mask)
    for b in range(2):
        want = 0.0
        for t in range(5):
            if mask[b, t]:
                z = logits[b, t]
                want += z[labels[b, t]] - np.log(np.exp(z).sum())
        assert got[b] == pytest.approx(want, rel=1e-12)
    assert (got < 0).all()


def test_objective_with_zero_probabilities():
    ref = np.array([0.5, 0.3, 0.2])
    r = np.array([1.0, 0.0, -1.0])
    # A deterministic policy: the zero entries contribute nothing to the KL term.
    got = kl_regularized_objective([1.0, 0.0, 0.0], ref, r, 0.5)
    assert got == pytest.approx(1.0 - 0.5 * np.log(2.0), rel=1e-12)


def test_sequence_logps_accepts_lists():
    logits = [[[0.0, 1.0], [2.0, 0.0]]]
    got = sequence_logps(logits, [[1, 0]], [[1, 1]])
    want = (1.0 - np.log(1.0 + np.e)) + (2.0 - np.log(np.exp(2.0) + 1.0))
    assert got[0] == pytest.approx(want, rel=1e-12)
