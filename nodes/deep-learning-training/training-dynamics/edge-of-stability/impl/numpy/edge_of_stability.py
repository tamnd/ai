# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/edge-of-stability/ (ai://17Q1PD).
"""The edge of stability in NumPy: gradient descent on a quadratic, the sharpness of a small network by power
iteration on Hessian-vector products, and full-batch gradient descent that logs the sharpness.

mode_factors and quadratic_gd are gradient descent on 0.5 x^T H x, where each eigen-coordinate is multiplied by
1 - eta * lambda_i per step (eq-mode-factor); mss_polyak and mss_nesterov are the largest stable sharpness with
momentum (eq-mss); hvp_central is the central-difference Hessian-vector product (eq-hvp) and power_iteration the
top-eigenvalue estimate built on it (eq-power); init_mlp, loss_and_grad, sharpness and train_gd are a tanh MLP with
a hand-written backward pass, its sharpness, and full-batch gradient descent on the mean squared error that records
the loss and the sharpness; catapult_step and catapult_closed_form are the two-layer linear model of Lewkowycz et
al. (2020) (eq-catapult-f, eq-catapult-lambda); eos_map and conserved_g are the two-variable model of Damian et al.
(2023) (eq-eos-map, eq-conserved); demo returns the numbers of the run described on the page."""

import itertools

import numpy as np

# Quadratics


def mode_factors(eigenvalues, eta):
    """The factor 1 - eta * lambda_i that multiplies each eigen-coordinate per step (eq-mode-factor)."""
    return 1.0 - eta * np.asarray(eigenvalues, dtype=np.float64)


def is_stable(eigenvalues, eta):
    """True when every factor has modulus below one, so gradient descent on the quadratic converges."""
    return bool(np.max(np.abs(mode_factors(eigenvalues, eta))) < 1.0)


def quadratic_gd(H, x0, eta, steps):
    """Gradient descent on 0.5 x^T H x from x0; returns the iterates, shape (steps + 1, d)."""
    x = np.array(x0, dtype=np.float64)
    out = [x.copy()]
    for _ in range(steps):
        x = x - eta * (H @ x)
        out.append(x.copy())
    return np.array(out)


def mss_polyak(eta, beta):
    """The largest stable sharpness of gradient descent with Polyak momentum: (2 + 2 beta) / eta (eq-mss)."""
    return (2.0 + 2.0 * beta) / eta


def mss_nesterov(eta, beta):
    """The largest stable sharpness with Nesterov momentum: (2 + 2 beta) / (eta (1 + 2 beta)) (eq-mss)."""
    return (2.0 + 2.0 * beta) / (eta * (1.0 + 2.0 * beta))


def momentum_gd_1d(a, eta, beta, steps, nesterov=False, x0=1.0):
    """Momentum gradient descent on 0.5 a x^2 from x_0 = x_{-1} = x0; returns the iterates."""
    prev, x = x0, x0
    out = [x]
    for _ in range(steps):
        if nesterov:
            y = x + beta * (x - prev)
            new = y - eta * a * y
        else:
            new = x - eta * a * x + beta * (x - prev)
        prev, x = x, new
        out.append(x)
    return np.array(out)


# Hessian-vector products and power iteration


def hvp_central(grad_fn, theta, v, eps):
    """(grad(theta + eps v) - grad(theta - eps v)) / (2 eps), which approximates H v (eq-hvp)."""
    return (grad_fn(theta + eps * v) - grad_fn(theta - eps * v)) / (2.0 * eps)


def power_iteration(hvp, dim, iters, rng):
    """The eigenvalue of largest modulus of the operator hvp and its unit eigenvector (eq-power).

    Starts from a random unit vector and repeats q <- H q / |H q| iters times; returns the Rayleigh quotient q . H q
    of the last q before the update. For a positive semi-definite Hessian this is the sharpness."""
    q = rng.standard_normal(dim)
    q /= np.linalg.norm(q)
    lam = 0.0
    for _ in range(iters):
        hq = hvp(q)
        lam = float(q @ hq)
        norm = np.linalg.norm(hq)
        if norm == 0.0:
            break
        q = hq / norm
    return lam, q


# A tanh MLP with a hand-written backward pass


def init_mlp(sizes, rng):
    """Flat parameters of an MLP with layer widths `sizes`: W ~ N(0, 1/fan_in), b = 0."""
    parts = []
    for a, b in itertools.pairwise(sizes):
        parts.append((rng.standard_normal((a, b)) / np.sqrt(a)).ravel())
        parts.append(np.zeros(b))
    return np.concatenate(parts)


def unflatten(theta, sizes):
    """The list of (W, b) of each layer, as views into theta."""
    layers, i = [], 0
    for a, b in itertools.pairwise(sizes):
        W = theta[i : i + a * b].reshape(a, b)
        i += a * b
        layers.append((W, theta[i : i + b]))
        i += b
    if i != theta.size:
        raise ValueError("theta does not match sizes")
    return layers


def forward(theta, sizes, X):
    """The outputs of the MLP: tanh on every hidden layer, linear output."""
    h = X
    layers = unflatten(theta, sizes)
    for k, (W, b) in enumerate(layers):
        z = h @ W + b
        h = np.tanh(z) if k < len(layers) - 1 else z
    return h


def loss_and_grad(theta, sizes, X, Y):
    """The mean squared error 0.5 * mean over examples of |f(x) - y|^2 and its gradient in theta."""
    layers = unflatten(theta, sizes)
    acts = [X]
    h = X
    for k, (W, b) in enumerate(layers):
        z = h @ W + b
        h = np.tanh(z) if k < len(layers) - 1 else z
        acts.append(h)
    r = h - Y
    n = X.shape[0]
    loss = 0.5 * float(np.sum(r * r)) / n
    g = r / n
    grads = []
    for k in range(len(layers) - 1, -1, -1):
        W, _ = layers[k]
        grads.append((acts[k].T @ g).ravel())
        grads.append(g.sum(axis=0))
        g = (g @ W.T) * (1.0 - acts[k] ** 2)  # at k = 0 this is the unused gradient of the inputs
    # grads were collected last layer first as (W, b) pairs; restore the layout of theta
    pairs = [(grads[i], grads[i + 1]) for i in range(0, len(grads), 2)][::-1]
    return loss, np.concatenate([p for pair in pairs for p in pair])


def sharpness(theta, sizes, X, Y, iters, eps, seed):
    """The top Hessian eigenvalue of the training loss at theta, by power iteration on central-difference HVPs."""

    def grad(t):
        return loss_and_grad(t, sizes, X, Y)[1]

    lam, _ = power_iteration(
        lambda v: hvp_central(grad, theta, v, eps),
        theta.size,
        iters,
        np.random.default_rng(seed),
    )
    return lam


def train_gd(theta, sizes, X, Y, eta, steps, every, sharp_iters):
    """Full-batch gradient descent theta <- theta - eta grad L; records the loss at every step and the sharpness at
    steps 0, every, 2 every, ...; returns (theta, losses, steps_measured, sharpnesses)."""
    theta = np.array(theta, dtype=np.float64)
    losses, at, sharp = [], [], []
    for t in range(steps + 1):
        loss, g = loss_and_grad(theta, sizes, X, Y)
        losses.append(loss)
        if t % every == 0:
            at.append(t)
            sharp.append(sharpness(theta, sizes, X, Y, sharp_iters, 1e-4, 0))
        if t < steps:
            theta -= eta * g
    return theta, np.array(losses), np.array(at), np.array(sharp)


def make_regression(n, d, rng):
    """A small regression task: x ~ N(0, I_d), y = sin(2 x . w) with w ~ N(0, I_d)."""
    X = rng.standard_normal((n, d))
    Y = np.sin(2.0 * X @ rng.standard_normal((d, 1)))
    return X, Y


# The catapult model and the self-stabilization model


def catapult_f_lambda(u, v):
    """f = n^{-1/2} v . u and lambda = (|u|^2 + |v|^2) / n for the model of Lewkowycz et al. with x = 1, y = 0."""
    n = u.size
    return float(v @ u) / np.sqrt(n), float(u @ u + v @ v) / n


def catapult_step(u, v, eta):
    """One gradient step on L = f^2 / 2: u <- u - eta n^{-1/2} f v, v <- v - eta n^{-1/2} f u."""
    n = u.size
    f, _ = catapult_f_lambda(u, v)
    return u - eta * f * v / np.sqrt(n), v - eta * f * u / np.sqrt(n)


def catapult_closed_form(f, lam, eta, n):
    """The exact updates of f and lambda (eq-catapult-f, eq-catapult-lambda)."""
    f_next = (1.0 - eta * lam + eta * eta * f * f / n) * f
    lam_next = lam + eta * f * f / n * (eta * lam - 4.0)
    return f_next, lam_next


def eos_map(x, y, eta, alpha, beta):
    """The two-variable model of Damian et al.: x <- -(1 + eta y) x, y <- y + eta (alpha - beta x^2 / 2) (eq-eos-map)."""
    return -(1.0 + eta * y) * x, y + eta * (alpha - beta * x * x / 2.0)


def conserved_g(X, Y, alpha, beta):
    """g = h(beta X^2 / (2 alpha)) + Y^2 / alpha with h(z) = z - log z - 1, conserved by X' = XY,
    Y' = alpha - beta X^2 / 2 (eq-conserved)."""
    z = beta * X * X / (2.0 * alpha)
    return z - np.log(z) - 1.0 + Y * Y / alpha


def demo(seed, eta, steps):
    """The run of the page: a 5-16-16-1 tanh MLP on 64 examples, full-batch gradient descent at eta.

    Returns the threshold 2 / eta, the layer widths, the number of examples, the steps at which the sharpness was
    measured (every 100), the sharpness there, and the loss at every step. The page runs demo(0, 0.25, 1600)."""
    rng = np.random.default_rng(seed)
    X, Y = make_regression(64, 5, rng)
    sizes = [5, 16, 16, 1]
    theta = init_mlp(sizes, rng)
    _, losses, at, sharp = train_gd(theta, sizes, X, Y, eta, steps, 100, 60)
    return {
        "threshold": 2.0 / eta,
        "sizes": sizes,
        "examples": X.shape[0],
        "measured_at": [int(t) for t in at],
        "sharpness": [float(s) for s in sharp],
        "losses": losses,
    }
