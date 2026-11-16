# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
"""Tests of the PyTorch reference: the worked example, the shared fixture, autograd against eq-lora-grads, the merge."""

import pytest
import torch

from lora import LoRALinear, add_lora

torch.set_default_dtype(torch.float64)

# The worked example of the page: d_out = 3, d_in = 4, r = 1, alpha = 1, so s = 1.
W0_EX = [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, -1.0], [1.0, 1.0, 0.0, 0.0]]
A_EX = [[1.0, 0.0, -1.0, 2.0]]
B_EX = [[0.5], [0.0], [-1.0]]
X_EX = [[1.0, 2.0, 0.0, 1.0]]

# The shared fixture of the NumPy tests.
W0 = [[0.5, -1.0, 2.0, 0.0, 1.5], [1.0, 0.25, -0.5, 3.0, -2.0], [-1.5, 2.0, 0.0, 1.0, 0.5], [0.0, -0.75, 1.25, -1.0, 2.0]]
A = [[0.2, -0.4, 0.1, 0.3, -0.2], [-0.1, 0.5, 0.25, -0.3, 0.4]]
B = [[1.0, -0.5], [0.0, 2.0], [-1.5, 0.5], [0.75, 1.0]]
X = [[0.5, -1.5, 2.0, 0.25, -0.75], [1.0, 3.0, -2.5, 0.0, 4.0], [-1.25, 0.5, 1.5, -3.0, 2.25]]
DY = [[1.0, 2.0, -1.0, 0.5], [-2.0, 1.0, 0.0, 4.0], [0.25, -0.5, 2.0, 1.0]]


def layer(w0, a, b, r, alpha):
    w0 = torch.tensor(w0)
    base = torch.nn.Linear(w0.shape[1], w0.shape[0], bias=False)
    with torch.no_grad():
        base.weight.copy_(w0)
    m = LoRALinear(base, r=r, alpha=alpha)
    with torch.no_grad():
        m.lora_A.copy_(torch.tensor(a))
        m.lora_B.copy_(torch.tensor(b))
    return m


def test_worked_example():
    m = layer(W0_EX, A_EX, B_EX, r=1, alpha=1.0)
    x = torch.tensor(X_EX, requires_grad=True)
    y = m(x)
    torch.testing.assert_close(y, torch.tensor([[2.5, 1.0, 0.0]]))
    y.sum().backward()
    torch.testing.assert_close(m.lora_B.grad, torch.tensor([[3.0], [3.0], [3.0]]))
    torch.testing.assert_close(m.lora_A.grad, torch.tensor([[-0.5, -1.0, 0.0, -0.5]]))
    torch.testing.assert_close(x.grad, torch.tensor([[1.5, 2.0, 2.5, -2.0]]))
    assert m.base.weight.grad is None
    assert m.num_trainable() == 7


def test_shared_fixture_against_eq_lora_grads():
    m = layer(W0, A, B, r=2, alpha=4.0)
    assert m.scale == 2.0
    x, dy = torch.tensor(X, requires_grad=True), torch.tensor(DY)
    a, b, w0 = torch.tensor(A), torch.tensor(B), torch.tensor(W0)
    y = m(x)
    torch.testing.assert_close(y, x.detach() @ (w0 + 2.0 * b @ a).T)
    (y * dy).sum().backward()
    torch.testing.assert_close(m.lora_B.grad, 2.0 * dy.T @ x.detach() @ a.T)
    torch.testing.assert_close(m.lora_A.grad, 2.0 * b.T @ dy.T @ x.detach())
    torch.testing.assert_close(x.grad, dy @ (w0 + 2.0 * b @ a))


def test_starts_at_the_pretrained_layer():
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5)
    m = LoRALinear(base, r=3)
    x = torch.randn(4, 6)
    torch.testing.assert_close(m(x), base(x))
    assert torch.all(m.lora_B == 0)
    assert not base.weight.requires_grad and not base.bias.requires_grad


def test_merged_gives_the_same_outputs():
    torch.manual_seed(1)
    m = LoRALinear(torch.nn.Linear(16, 12), r=4, alpha=8.0)
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(7, 16)
    torch.testing.assert_close(m.merged()(x), m(x))
    assert torch.linalg.matrix_rank(m.delta()).item() == 4


def test_rank_stabilized_scale():
    assert LoRALinear(torch.nn.Linear(10, 10), r=4, alpha=8.0, rank_stabilized=True).scale == 4.0
    assert LoRALinear(torch.nn.Linear(10, 10), r=4, alpha=8.0).scale == 2.0


def test_add_lora_counts():
    torch.manual_seed(2)
    model = torch.nn.Sequential()
    model.add_module("q", torch.nn.Linear(32, 32))
    model.add_module("act", torch.nn.ReLU())
    model.add_module("v", torch.nn.Linear(32, 32))
    added = add_lora(model, {"q", "v"}, r=2)
    assert added == 2 * 2 * (32 + 32)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == added


@pytest.mark.parametrize("r", [0, 5])
def test_bad_rank(r):
    with pytest.raises(ValueError):
        LoRALinear(torch.nn.Linear(4, 4), r=r)
