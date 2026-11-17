# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://NEW-edge-of-stability).
"""The edge of stability in PyTorch: the sharpness of the training loss by power iteration on Hessian-vector
products from autograd, full-batch gradient descent that logs it, and gradient descent on a quadratic.

hvp computes H v by differentiating the gradient a second time (double backward); power_iteration repeats
q <- H q / |H q| and returns the Rayleigh quotient (eq-power); TanhMLP is a small network with tanh hidden layers,
and train_gd runs full-batch gradient descent on the mean squared error and records the loss and the sharpness;
quadratic_gd and mode_factors are gradient descent on 0.5 x^T H x, where each eigen-coordinate is multiplied by
1 - eta * lambda_i per step (eq-mode-factor); demo returns the numbers of the run described on the page."""

import itertools

import torch
from torch import nn

torch.set_default_dtype(torch.float64)


def mode_factors(eigenvalues, eta):
    """The factor 1 - eta * lambda_i of each eigen-coordinate per step (eq-mode-factor)."""
    return 1.0 - eta * torch.as_tensor(eigenvalues, dtype=torch.float64)


def quadratic_gd(H, x0, eta, steps):
    """Gradient descent on 0.5 x^T H x from x0; returns the iterates, shape (steps + 1, d)."""
    x = torch.as_tensor(x0, dtype=torch.float64).clone()
    out = [x.clone()]
    for _ in range(steps):
        x = x - eta * (H @ x)
        out.append(x.clone())
    return torch.stack(out)


def hvp(loss_fn, params, vec):
    """H v for the loss loss_fn() at the current params, by double backward; vec is a flat tensor."""
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    hv = torch.autograd.grad(flat @ vec, params)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def power_iteration(matvec, dim, iters, generator):
    """The eigenvalue of largest modulus of matvec and its unit eigenvector (eq-power): iters steps of
    q <- H q / |H q| from a random unit vector, returning the last Rayleigh quotient q . H q."""
    q = torch.randn(dim, generator=generator, dtype=torch.float64)
    q = q / q.norm()
    lam = 0.0
    for _ in range(iters):
        hq = matvec(q)
        lam = float(q @ hq)
        norm = hq.norm()
        if norm == 0:
            break
        q = hq / norm
    return lam, q


class TanhMLP(nn.Module):
    """Linear layers of the given widths with tanh between them and a linear output."""

    def __init__(self, sizes):
        super().__init__()
        layers = [nn.Linear(a, b) for a, b in itertools.pairwise(sizes)]
        for layer in layers:
            nn.init.normal_(layer.weight, std=layer.in_features**-0.5)
            nn.init.zeros_(layer.bias)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


def mse(model, X, Y):
    """0.5 * mean over examples of |f(x) - y|^2."""
    return 0.5 * ((model(X) - Y) ** 2).sum() / X.shape[0]


def sharpness(model, X, Y, iters, seed):
    """The top Hessian eigenvalue of the training loss of model, by power iteration on autograd HVPs."""
    params = [p for p in model.parameters()]
    dim = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(seed)
    lam, _ = power_iteration(lambda v: hvp(lambda: mse(model, X, Y), params, v), dim, iters, gen)
    return lam


def train_gd(model, X, Y, eta, steps, every, sharp_iters):
    """Full-batch gradient descent at step size eta; returns (losses, steps_measured, sharpnesses)."""
    losses, at, sharp = [], [], []
    for t in range(steps + 1):
        if t % every == 0:
            at.append(t)
            sharp.append(sharpness(model, X, Y, sharp_iters, 0))
        model.zero_grad()
        loss = mse(model, X, Y)
        losses.append(float(loss.detach()))
        if t < steps:
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad
    return torch.tensor(losses), at, sharp


def make_regression(n, d, generator):
    """x ~ N(0, I_d), y = sin(2 x . w) with w ~ N(0, I_d)."""
    X = torch.randn(n, d, generator=generator)
    w = torch.randn(d, 1, generator=generator)
    return X, torch.sin(2.0 * X @ w)


def demo(seed, eta, steps):
    """The run of the page with a 5-16-16-1 tanh MLP on 64 examples; the same record as the NumPy demo."""
    gen = torch.Generator().manual_seed(seed)
    X, Y = make_regression(64, 5, gen)
    torch.manual_seed(seed)
    sizes = [5, 16, 16, 1]
    model = TanhMLP(sizes)
    losses, at, sharp = train_gd(model, X, Y, eta, steps, 100, 60)
    return {
        "threshold": 2.0 / eta,
        "sizes": sizes,
        "examples": X.shape[0],
        "measured_at": at,
        "sharpness": sharp,
        "losses": losses,
    }
