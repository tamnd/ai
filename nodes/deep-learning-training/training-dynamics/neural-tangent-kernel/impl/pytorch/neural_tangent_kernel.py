# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://NEW-neural-tangent-kernel).
"""The neural tangent kernel of a two-layer ReLU network, in PyTorch.

The network is eq-two-layer of the page, f(x) = (1/sqrt(m)) sum_j a_j relu(w_j . x), in the NTK parametrization.
The Jacobian (eq-jacobian) comes from autograd, with torch.func.jacrev, not from the hand formula of the NumPy
reference; the tests check that the two agree. The loss is L = (1/(2n)) sum_i (f(x_i) - y_i)^2 over n points.
"""

import math

import torch
from torch.func import functional_call, jacrev


class TwoLayer(torch.nn.Module):
    """f(x) = (1/sqrt(m)) a . relu(W x), with W of shape (m, d) and a of shape (m,), all drawn from N(0, 1)."""

    def __init__(self, d, m, generator=None):
        super().__init__()
        if d < 1 or m < 1:
            raise ValueError(f"d and m must be positive, got d={d}, m={m}")
        self.W = torch.nn.Parameter(torch.randn(m, d, generator=generator, dtype=torch.float64))
        self.a = torch.nn.Parameter(torch.randn(m, generator=generator, dtype=torch.float64))

    def forward(self, X):
        # eq: eq-two-layer
        return torch.relu(X @ self.W.T) @ self.a / math.sqrt(self.a.shape[0])


def flat_params(model):
    """theta as one vector: the rows of W, then a (the order of the NumPy reference)."""
    return torch.cat([model.W.detach().reshape(-1), model.a.detach()])


def jacobian(model, X):
    """The Jacobian of f at the rows of X, shape (n, m*d + m), by autograd."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    # eq: eq-jacobian
    jac = jacrev(lambda p: functional_call(model, p, (X,)))(params)
    return torch.cat([jac["W"].reshape(X.shape[0], -1), jac["a"]], dim=1)


def empirical_ntk(model, X1, X2):
    """The empirical NTK J(X1) J(X2)^T, shape (n1, n2)."""
    # eq: eq-ntk
    return jacobian(model, X1) @ jacobian(model, X2).T


def relu_ntk(X1, X2):
    """The infinite-width NTK: x.x' (pi - phi)/(2 pi) + |x||x'| (sin phi + (pi - phi) cos phi)/(2 pi)."""
    dot = X1 @ X2.T
    norms = torch.outer(X1.norm(dim=1), X2.norm(dim=1))
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    cos = torch.clamp(torch.where(norms > 0, dot / safe, torch.ones_like(dot)), -1.0, 1.0)
    phi = torch.arccos(cos)
    # eq: eq-relu-ntk
    return dot * (math.pi - phi) / (2 * math.pi) + norms * (torch.sin(phi) + (math.pi - phi) * cos) / (2 * math.pi)


def residual_flow(K, r0, eta, t, n):
    """exp(-eta K t / n) r0, the residual of gradient flow with the constant kernel K (eq-residual-flow)."""
    lam, V = torch.linalg.eigh(K)
    # eq: eq-residual-flow
    return V @ (torch.exp(-eta * lam * t / n) * (V.T @ r0))


def train_linearized(model, X, y, eta, steps):
    """Gradient descent on f_lin(theta) = f_0 + J_0 (theta - theta_0) (eq-flin), with autograd on omega.
    Returns the residuals after each step (steps + 1, n) and the final omega."""
    n = X.shape[0]
    J = jacobian(model, X)
    with torch.no_grad():
        f0 = model(X)
    omega = torch.zeros(J.shape[1], dtype=J.dtype, requires_grad=True)
    out = [(f0 - y).clone()]
    for _ in range(steps):
        # eq: eq-flin
        r = f0 + J @ omega - y
        loss = 0.5 * (r**2).sum() / n
        (g,) = torch.autograd.grad(loss, omega)
        with torch.no_grad():
            # eq: eq-gd-step
            omega -= eta * g
            out.append(f0 + J @ omega - y)
    return torch.stack(out), omega.detach()


def train_network(model, X, y, eta, steps):
    """Full-batch gradient descent on the network with torch.optim.SGD. Changes model in place and returns it."""
    opt = torch.optim.SGD(model.parameters(), lr=eta)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * ((model(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return model


def relative_change(theta0, theta1):
    """||theta_1 - theta_0|| / ||theta_0|| (eq-relative-change)."""
    # eq: eq-relative-change
    return (theta1 - theta0).norm() / theta0.norm()
