# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/residual-connection/ (ai://NEW-residual-connection).
"""A stack of residual MLP blocks as a torch.nn.Module, the PyTorch reference.

Each block computes x + F(x) with F(x) = W2 act(W1 x + b1) + b2 (eq-residual, eq-branch). Autograd gives the
backward pass, and the tests compare it with the hand-written NumPy backward on the same numbers.
"""

import torch


class ResidualStack(torch.nn.Module):
    """n blocks of width d with branches of width hidden; residual=False drops the shortcuts and keeps the blocks."""

    def __init__(self, d, n, hidden=None, activation="relu", residual=True, gain=1.0, zero_last=False, generator=None):
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError(f"unknown activation {activation}")
        m = hidden or d
        self.activation = activation
        self.residual = residual
        self.blocks = torch.nn.ModuleList()
        for _ in range(n):
            block = torch.nn.ModuleDict({"fc1": torch.nn.Linear(d, m), "fc2": torch.nn.Linear(m, d)})
            with torch.no_grad():
                block["fc1"].weight.normal_(0.0, (2.0 / d) ** 0.5, generator=generator)
                block["fc2"].weight.normal_(0.0, gain / m**0.5, generator=generator)
                if zero_last:
                    block["fc2"].weight.zero_()
                block["fc1"].bias.zero_()
                block["fc2"].bias.zero_()
            self.blocks.append(block)

    @classmethod
    def from_params(cls, params, activation="relu", residual=True):
        """Build the stack from the NumPy parameter list: W1 (m, d), b1 (m,), W2 (d, m), b2 (d,) per block."""
        m, d = params[0]["W1"].shape
        stack = cls(d, len(params), hidden=m, activation=activation, residual=residual)
        with torch.no_grad():
            for block, p in zip(stack.blocks, params):
                block["fc1"].weight.copy_(torch.as_tensor(p["W1"]))
                block["fc1"].bias.copy_(torch.as_tensor(p["b1"]))
                block["fc2"].weight.copy_(torch.as_tensor(p["W2"]))
                block["fc2"].bias.copy_(torch.as_tensor(p["b2"]))
        return stack

    def branch(self, block, x):
        z = block["fc1"](x)  # eq: eq-branch
        h = torch.tanh(z) if self.activation == "tanh" else torch.relu(z)
        return block["fc2"](h)

    def forward(self, x):
        for block in self.blocks:
            f = self.branch(block, x)
            x = x + f if self.residual else f  # eq: eq-residual
        return x


def highway(stack, x, t=None, c=None, gate=None):
    """A highway layer y = H(x) T(x) + x C(x) (eq-highway) with H the first block of stack.

    Give t and c to fix the gates; otherwise T = sigmoid(gate(x)) for a torch.nn.Linear gate, and C = 1 - T.
    """
    h = stack.branch(stack.blocks[0], x)
    if t is None:
        t = torch.sigmoid(gate(x))
    if c is None:
        c = 1.0 - t
    return h * t + x * c
