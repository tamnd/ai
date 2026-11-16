# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/critical-batch-size/ (ai://NEW-critical-batch-size).
"""Tests of the PyTorch code: per-example gradients against single-example backward passes; the estimators, the
quadratic model and the trade-off fit against values of the NumPy reference (impl/numpy) on the same fixed inputs; the
two-batch estimate from a real model's local and averaged gradients; and the moving-average module, its buffers and
its state_dict. The tests run apart from the NumPy code, so its outputs are written here as numbers."""

import pytest
import torch

from critical_batch_size import (
    RunningNoiseScale,
    fit_tradeoff,
    noise_scale_from_examples,
    noise_scale_from_two_batches,
    optimal_step,
    per_example_grads,
    quadratic_noise_scale,
    step_scale,
    steps_needed,
)

DT = torch.float64
gen = torch.Generator().manual_seed(4)


def _model():
    torch.manual_seed(2)
    return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)).to(DT)


def _batch(n):
    return torch.randn(n, 3, generator=gen, dtype=DT), torch.randn(n, 2, generator=gen, dtype=DT)


def _mean_grad(model, x, y):
    model.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def test_per_example_grads_match_backward():
    model, (x, y) = _model(), _batch(6)
    per = per_example_grads(model, torch.nn.functional.mse_loss, x, y)
    assert per.shape == (6, sum(p.numel() for p in model.parameters()))
    for i in range(6):
        assert torch.allclose(per[i], _mean_grad(model, x[i : i + 1], y[i : i + 1]))
    assert torch.allclose(per.mean(dim=0), _mean_grad(model, x, y))


def test_examples_estimator_matches_numpy():
    grads = torch.tensor([[0.5, 1.0, -0.2], [1.5, 0.0, 0.4], [0.9, 0.7, 0.1], [1.1, -0.3, 0.6]], dtype=DT)
    b, g2, trace = noise_scale_from_examples(grads)
    # impl/numpy gives (0.6537190082644628, 1.0083333333333333, 0.6591666666666667) on these gradients
    assert float(b) == pytest.approx(0.6537190082644628, rel=1e-12)
    assert float(g2) == pytest.approx(1.0083333333333333, rel=1e-12)
    assert float(trace) == pytest.approx(0.6591666666666667, rel=1e-12)
    b, g2, _ = noise_scale_from_examples(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=DT))
    assert g2 < 0 and torch.isinf(b)
    with pytest.raises(ValueError):
        noise_scale_from_examples(torch.ones(1, 3, dtype=DT))


def test_two_batch_from_a_data_parallel_step():
    # four workers of 8 examples: |G_small|^2 from one worker, |G_big|^2 after averaging, as before and after all-reduce
    model, (x, y) = _model(), _batch(32)
    local = _mean_grad(model, x[:8], y[:8])
    glob = _mean_grad(model, x, y)
    g2, trace = noise_scale_from_two_batches(local @ local, glob @ glob, 8, 32)
    sq_small, sq_big = float(local @ local), float(glob @ glob)
    assert float(g2) == pytest.approx((32 * sq_big - 8 * sq_small) / 24)
    assert float(trace) == pytest.approx((sq_small - sq_big) / (1 / 8 - 1 / 32))
    # impl/numpy gives (0.95, 14.4) for |G_small|^2 = 2.75 at 8 and |G_big|^2 = 1.4 at 32
    g2, trace = noise_scale_from_two_batches(torch.tensor(2.75, dtype=DT), torch.tensor(1.4, dtype=DT), 8, 32)
    assert float(g2) == pytest.approx(0.95) and float(trace) == pytest.approx(14.4)
    # with per-example gradients, B_small = 1 averaged over examples is the per-example estimator
    per = per_example_grads(model, torch.nn.functional.mse_loss, x, y)
    g2, trace = noise_scale_from_two_batches((per**2).sum(dim=1).mean(), glob @ glob, 1, 32)
    _, g2_ex, trace_ex = noise_scale_from_examples(per)
    assert torch.allclose(g2, g2_ex) and torch.allclose(trace, trace_ex)
    with pytest.raises(ValueError):
        noise_scale_from_two_batches(1.0, 1.0, 8, 8)


def test_running_noise_scale_module():
    tracker = RunningNoiseScale(decay=0.5)
    with pytest.raises(ValueError):
        tracker.value()
    for g2, tr in [(1.0, 8.0), (3.0, 4.0), (2.0, 10.0)]:
        out = tracker(
            torch.tensor(g2 + tr / 2, dtype=DT),
            torch.tensor(g2 + tr / 16, dtype=DT),
            2,
            16,
        )
    # impl/numpy gives the ratio 3.7333333333333334 and the averaged |G|^2 1.875 after the same three updates
    assert float(out) == pytest.approx(3.7333333333333334)
    assert int(tracker.steps) == 3 and float(tracker.g2) == pytest.approx(1.875)
    state = tracker.state_dict()
    assert set(state) == {"g2", "trace", "steps"}
    fresh = RunningNoiseScale(decay=0.5)
    fresh.load_state_dict(state)
    assert float(fresh.value()) == pytest.approx(float(tracker.value()))
    tracker(torch.tensor(-19.0, dtype=DT), torch.tensor(-20.0, dtype=DT), 2, 4)
    assert torch.isinf(tracker.value())
    for bad in [1.0, -0.1]:
        with pytest.raises(ValueError):
            RunningNoiseScale(bad)


def test_quadratic_matches_numpy():
    G = torch.tensor([1.0, -0.5, 0.25], dtype=DT)
    H = torch.tensor([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]], dtype=DT)
    Sigma = torch.tensor([[1.0, 0.2, 0.1], [0.2, 2.0, 0.0], [0.1, 0.0, 0.5]], dtype=DT)
    bn = quadratic_noise_scale(G, H, Sigma)
    eps_max, dl_max = step_scale(G, H)
    # impl/numpy values on the same G, H and Sigma
    assert float(bn) == pytest.approx(2.2338658146964856, rel=1e-12)
    assert float(eps_max) == pytest.approx(0.6709265175718849, rel=1e-12)
    assert float(dl_max) == pytest.approx(0.4402955271565495, rel=1e-12)
    eps = optimal_step([1.0, 10.0, 100.0], float(eps_max), float(bn))
    assert eps.tolist() == pytest.approx([0.20746887966804978, 0.548417424005014, 0.6562664066601664], rel=1e-12)
    for f in (
        lambda: quadratic_noise_scale(torch.zeros(2), torch.eye(2), torch.eye(2)),
        lambda: step_scale(torch.zeros(2), torch.eye(2)),
        lambda: optimal_step(0.0, 1.0, 1.0),
        lambda: optimal_step(1.0, 1.0, -1.0),
    ):
        with pytest.raises(ValueError):
            f()


def test_tradeoff_matches_numpy():
    B = [2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]
    S = [10050.0, 2600.0, 690.0, 210.0, 88.0, 52.0]
    # impl/numpy gives (43.686959184984694, 20603.931477043272, 471.62658746285297)
    got = [float(v) for v in fit_tradeoff(B, S)]
    assert got == pytest.approx([43.686959184984694, 20603.931477043272, 471.62658746285297], rel=1e-9)
    assert steps_needed(B, 40.0, 2.0e4).tolist() == pytest.approx([40.0 + 2.0e4 / b for b in B])
    assert float(steps_needed(2.0e4 / 40.0, 40.0, 2.0e4)) == pytest.approx(80.0)
    for bad in [
        ([1.0, 1.0], [2.0, 2.0]),
        ([1.0, 2.0], [3.0]),
        ([-1.0, 2.0], [3.0, 1.0]),
        ([1.0, 2.0], [1.0, 2.0]),
    ]:
        with pytest.raises(ValueError):
            fit_tradeoff(*bad)
    with pytest.raises(ValueError):
        steps_needed(0.0, 1.0, 1.0)


def test_boundaries():
    # g2 exactly zero gives inf; batch sizes and scales below 1 are valid
    b, g2, _ = noise_scale_from_examples(torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=DT))
    assert float(g2) == 0.0 and torch.isinf(b)
    b, g2, trace = noise_scale_from_examples(torch.tensor([[0.5, 0.0], [1.0, 0.0]], dtype=DT))
    assert [float(b), float(g2), float(trace)] == pytest.approx([0.25, 0.5, 0.125])
    G = torch.tensor([0.1, 0.0], dtype=DT)
    H = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=DT)
    Sigma = 0.5 * torch.eye(2, dtype=DT)
    assert float(quadratic_noise_scale(G, H, Sigma)) == pytest.approx(150.0)
    assert [float(v) for v in step_scale(G, H)] == pytest.approx([1.0, 0.005])
    assert optimal_step([0.5, 1.0], 1.0, 0.0).tolist() == pytest.approx([1.0, 1.0])
    assert optimal_step([0.5, 1.0], 1.0, 1.0).tolist() == pytest.approx([1 / 3, 0.5])
    assert steps_needed([0.5, 2.0], 0.5, 0.5).tolist() == pytest.approx([1.5, 0.75])
    for bad in [(1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]:
        with pytest.raises(ValueError):
            steps_needed(*bad)
    g2, trace = noise_scale_from_two_batches(1 + 8 / 0.5, 1 + 8, 0.5, 1)
    assert [float(g2), float(trace)] == pytest.approx([1.0, 8.0])
    g2, trace = noise_scale_from_two_batches(1 + 8 / 4, 1 + 8 / 2, 4, 2)
    assert [float(g2), float(trace)] == pytest.approx([1.0, 8.0])
    for bad in [(0, 4), (4, 0)]:
        with pytest.raises(ValueError):
            noise_scale_from_two_batches(1.0, 1.0, *bad)
    tracker = RunningNoiseScale(0.0)
    for k in range(1, 3):
        tracker(torch.tensor(5.0, dtype=DT), torch.tensor(3.0, dtype=DT), 1, 2)
        assert int(tracker.steps) == k
    assert float(tracker.value()) == pytest.approx(4.0)  # g2 = 1, trace = 4 with decay 0
    s_min, e_min, b_crit = fit_tradeoff([1.0, 4.0], [3.5, 3.125])  # S = 3 + 0.5 / B
    assert [float(s_min), float(e_min), float(b_crit)] == pytest.approx([3.0, 0.5, 1 / 6])
    s_min, e_min, _ = fit_tradeoff([0.5, 4.0], [4.5, 1.0])  # S = 0.5 + 2 / B
    assert [float(s_min), float(e_min)] == pytest.approx([0.5, 2.0])
    for bad in [([1.0, 2.0], [0.0, 1.0]), ([0.0, 2.0], [1.0, 1.0]), ([1.0, 2.0], [1.0, 3.0])]:
        with pytest.raises(ValueError):
            fit_tradeoff(*bad)
