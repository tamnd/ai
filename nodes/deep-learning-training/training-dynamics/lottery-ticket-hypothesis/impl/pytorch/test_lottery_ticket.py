# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lottery-ticket-hypothesis/ (ai://NEW-lottery-ticket-hypothesis).
"""Tests of the PyTorch reference: the shared fixture of the NumPy tests, one round of IMP on a module, the rounding
and tie rules, and the error barrier."""

import pytest
import torch

from lottery_ticket import error_barrier, imp_round, interpolate, magnitude_masks

torch.set_default_dtype(torch.float64)

# The shared fixture of the NumPy tests.
W1 = [[0.5, -0.1, 0.9], [-0.3, 0.7, 0.2]]
W2 = [[0.05, -0.8], [0.6, -0.15], [0.4, 0.25]]
LAYER_50 = [[[1, 0, 1], [0, 1, 0]], [[0, 1], [1, 0], [1, 0]]]
LAYER_25 = [[[1, 0, 1], [1, 1, 0]], [[0, 1], [1, 0], [1, 1]]]
GLOBAL_25 = [[[1, 0, 1], [1, 1, 1]], [[0, 1], [1, 0], [1, 1]]]
MIXED = [LAYER_50[0], LAYER_25[1]]


def t(v):
    return torch.tensor(v, dtype=torch.float64)


def ones():
    return [torch.ones(2, 3), torch.ones(3, 2)]


def same(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == torch.float64
        torch.testing.assert_close(g, t(w), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rate", "scope", "want"),
    [(0.5, "layer", LAYER_50), (0.25, "layer", LAYER_25), (0.25, "global", GLOBAL_25), ([0.5, 0.25], "layer", MIXED)],
)
def test_masks_fixture(rate, scope, want):
    same(magnitude_masks([t(W1), t(W2)], ones(), rate, scope), want)


def test_pruned_stay_pruned():
    trained = [t(W1) + 10.0 * (1 - t(LAYER_50[0])), t(W2) + 10.0 * (1 - t(LAYER_50[1]))]
    want = [[[0, 0, 1], [0, 1, 0]], [[0, 1], [1, 0], [0, 0]]]
    masks = [t(m) for m in LAYER_50]
    same(magnitude_masks(trained, masks, 1 / 3, "layer"), want)
    same(magnitude_masks(trained, masks, 1 / 3, "global"), want)


@pytest.mark.parametrize(
    ("rate", "want"), [(0.24, [1, 1]), (0.25, [1, 0]), (0.74, [1, 0]), (0.75, [0, 0]), (0.0, [1, 1])]
)
def test_rounding_half_up(rate, want):
    same(magnitude_masks([t([3.0, 1.0])], [torch.ones(2)], rate), [want])


def test_ties_and_sign():
    same(magnitude_masks([t([2.0, -2.0, 2.0, 2.0])], [torch.ones(4)], 0.5), [[0, 0, 1, 1]])
    same(magnitude_masks([t([-5.0, 1.0, -0.5, 4.0])], [torch.ones(4)], 0.5), [[1, 0, 0, 1]])


def test_masks_reject():
    w = [t(W1), t(W2)]
    with pytest.raises(ValueError):
        magnitude_masks(w, ones()[:1], 0.2)
    with pytest.raises(ValueError):
        magnitude_masks(w, [torch.ones(3, 2), torch.ones(2, 3)], 0.2)
    with pytest.raises(ValueError):
        magnitude_masks(w, ones(), [0.2], "layer")
    with pytest.raises(ValueError):
        magnitude_masks(w, ones(), [0.2, 0.2], "global")
    with pytest.raises(ValueError):
        magnitude_masks(w, ones(), 0.2, "row")
    with pytest.raises(ValueError):
        magnitude_masks(w, ones(), 1.0)
    with pytest.raises(ValueError):
        magnitude_masks(w, ones(), -0.1, "global")


def weight(net, i):
    return dict(net.named_parameters())[f"{i}.weight"]


def model():
    net = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        weight(net, 0).copy_(t(W1).T)
        weight(net, 1).copy_(t(W2).T)
    return net


def test_imp_round_reset():
    net = model()
    names = ["0.weight", "1.weight"]
    theta0 = {"0.weight": torch.full((3, 2), 0.5), "1.weight": torch.arange(6.0).reshape(2, 3)}
    masks = imp_round(net, theta0, [torch.ones(3, 2), torch.ones(2, 3)], 0.5, names)
    same(masks, [t(LAYER_50[0]).T.tolist(), t(LAYER_50[1]).T.tolist()])
    torch.testing.assert_close(weight(net, 0).detach(), 0.5 * masks[0], rtol=0, atol=0)
    torch.testing.assert_close(weight(net, 1).detach(), torch.arange(6.0).reshape(2, 3) * masks[1], rtol=0, atol=0)


def test_imp_round_global_rewind():
    net = model()
    names = ["0.weight", "1.weight"]
    theta_k = {"0.weight": -t(W1).T, "1.weight": 2.0 * t(W2).T}
    masks = imp_round(net, theta_k, [torch.ones(3, 2), torch.ones(2, 3)], 0.25, names, scope="global")
    same(masks, [t(GLOBAL_25[0]).T.tolist(), t(GLOBAL_25[1]).T.tolist()])
    torch.testing.assert_close(weight(net, 0).detach(), -t(W1).T * masks[0], rtol=0, atol=0)
    torch.testing.assert_close(weight(net, 1).detach(), 2.0 * t(W2).T * masks[1], rtol=0, atol=0)


def test_interpolate():
    out = interpolate({"w": t([1.0, 2.0])}, {"w": t([5.0, -2.0])}, 0.25)
    torch.testing.assert_close(out["w"], t([4.0, -1.0]), rtol=1e-15, atol=0)


def test_barrier_concave():
    def err(s):
        return 1.0 - float(s["w"][0]) ** 2

    h, alphas, errors = error_barrier(err, {"w": t([1.0])}, {"w": t([-1.0])}, num=3)
    assert h == 1.0
    torch.testing.assert_close(alphas, t([0.0, 0.5, 1.0]))
    torch.testing.assert_close(errors, t([0.0, 1.0, 0.0]))
    h, alphas, errors = error_barrier(err, {"w": t([1.0])}, {"w": t([-1.0])})
    assert alphas.numel() == 30 and errors.numel() == 30
    assert h == pytest.approx(1.0 - 1.0 / 841.0, rel=1e-12)


def test_barrier_mean_of_ends():
    h, _, errors = error_barrier(lambda s: float(s["w"][0]), {"w": t([3.0])}, {"w": t([1.0])}, num=5)
    assert h == pytest.approx(1.0, abs=1e-15)
    assert float(errors[0]) == 1.0 and float(errors[-1]) == 3.0


def test_barrier_rejects():
    with pytest.raises(ValueError):
        error_barrier(lambda s: 0.0, {"w": t([0.0])}, {"w": t([0.0])}, num=1)


def test_barrier_on_a_model():
    # Two networks that compute the same function with their hidden units swapped. The midpoint averages the units,
    # so the path has a bump in the squared error, and the barrier is the error at alpha = 1/2 (num = 3).
    x = t([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    a = {"w1": t([[1.0, -1.0], [0.0, 2.0]]), "w2": t([[1.0], [1.0]])}
    b = {"w1": t([[-1.0, 1.0], [2.0, 0.0]]), "w2": t([[1.0], [1.0]])}
    y = torch.relu(x @ a["w1"]) @ a["w2"]

    def err(s):
        return float(((torch.relu(x @ s["w1"]) @ s["w2"] - y) ** 2).mean())

    assert err(a) == 0.0 and err(b) == 0.0
    h, _, errors = error_barrier(err, a, b, num=3)
    mid = {k: 0.5 * a[k] + 0.5 * b[k] for k in a}
    assert h == pytest.approx(err(mid), rel=1e-12)
    assert h > 0.0


def test_barrier_two_points():
    # num = 2 is the smallest grid: only the two ends, so the barrier is half the gap between them.
    h, alphas, _ = error_barrier(lambda s: float(s["w"][0]), {"w": t([3.0])}, {"w": t([1.0])}, num=2)
    assert h == pytest.approx(1.0, abs=1e-15)
    torch.testing.assert_close(alphas, t([0.0, 1.0]))
