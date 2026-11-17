# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://NEW-residual-connection).
"""Tests of the PyTorch reference: the worked example of the NumPy tests, the identity at F = 0, the Jacobian, the gradient norm and the highway gate."""

import pytest
import torch

from residual import ResidualStack, highway

torch.set_default_dtype(torch.float64)

# The worked example of the page, with the numbers of the NumPy tests.
P = [
    {"W1": [[1.0, 1.0], [1.0, -1.0]], "b1": [0.0, 0.0], "W2": [[0.5, 0.0], [0.0, -1.0]], "b2": [0.0, 0.0]},
    {"W1": [[0.0, 1.0], [1.0, 0.0]], "b1": [0.5, 0.0], "W2": [[1.0, 0.0], [0.0, 0.5]], "b2": [0.0, 0.25]},
]
P = [{k: torch.tensor(v) for k, v in p.items()} for p in P]
EXPECTED = {
    True: ([2.5, -2.5], [1.25, 2.75], [[1.5, -0.75], [-2.0, 1.0]], [[1.5, 4.5], [1.0, 3.0]]),
    False: ([0.0, 0.5], [0.25, 0.25], [[0.5, -0.25], [0.0, 0.0]], [[0.5, 1.5], [0.0, 0.0]]),
}


def t(v):
    return torch.tensor(v, dtype=torch.float64)


@pytest.mark.parametrize("residual", [True, False])
def test_worked_example(residual):
    stack = ResidualStack.from_params(P, residual=residual)
    x = t([[2.0, -1.0]]).requires_grad_()
    y = stack(x)
    want_y, want_dx, want_dw1, want_dw2 = EXPECTED[residual]
    torch.testing.assert_close(y[0], t(want_y), rtol=1e-12, atol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad[0], t(want_dx), rtol=1e-12, atol=0)
    torch.testing.assert_close(stack.blocks[0]["fc1"].weight.grad, t(want_dw1), rtol=1e-12, atol=0)
    torch.testing.assert_close(stack.blocks[0]["fc2"].weight.grad, t(want_dw2), rtol=1e-12, atol=0)


def test_zero_branch_is_identity():
    g = torch.Generator().manual_seed(0)
    stack = ResidualStack(5, 4, hidden=7, activation="tanh", zero_last=True, generator=g)
    x = torch.randn(3, 5, generator=g)
    torch.testing.assert_close(stack(x), x, rtol=0, atol=0)
    jac = torch.autograd.functional.jacobian(stack, x[0])
    torch.testing.assert_close(jac, torch.eye(5), rtol=0, atol=0)
    with pytest.raises(ValueError):
        ResidualStack(2, 1, activation="gelu")


def test_jacobian_is_product():
    g = torch.Generator().manual_seed(1)
    stack = ResidualStack(4, 5, hidden=6, activation="tanh", gain=0.7, generator=g)
    x = torch.randn(4, generator=g)
    prod, xi = torch.eye(4), x
    for block in stack.blocks:
        j = torch.autograd.functional.jacobian(lambda v, b=block: stack.branch(b, v), xi)
        prod = (torch.eye(4) + j) @ prod
        xi = xi + stack.branch(block, xi)
    torch.testing.assert_close(torch.autograd.functional.jacobian(stack, x), prod, rtol=1e-10, atol=1e-12)


def ratio(n, residual):
    g = torch.Generator().manual_seed(1603)
    stack = ResidualStack(16, n, activation="tanh", residual=residual, gain=0.1, generator=g)
    x = torch.randn(8, 16, generator=g).requires_grad_()
    dy = torch.randn(8, 16, generator=g)
    (stack(x) * dy).sum().backward()
    return (x.grad.norm() / dy.norm()).item()


def test_gradient_norm_bounded_vs_plain():
    assert 0.8 < ratio(50, True) < 1.6
    assert ratio(50, False) < 1e-30


def test_highway_gates():
    g = torch.Generator().manual_seed(4)
    stack = ResidualStack(3, 1, hidden=5, generator=g)
    x = torch.randn(4, 3, generator=g)
    f = stack.branch(stack.blocks[0], x)
    torch.testing.assert_close(highway(stack, x, t=1.0, c=1.0), stack(x))
    torch.testing.assert_close(highway(stack, x, t=1.0), f)
    torch.testing.assert_close(highway(stack, x, t=0.0), x)
    gate = torch.nn.Linear(3, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    torch.testing.assert_close(highway(stack, x, gate=gate), 0.5 * f + 0.5 * x)
