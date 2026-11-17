# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-tangent-kernel/ (ai://3SJCNH).
"""The neural tangent kernel of a two-layer ReLU network, the NumPy reference.

The network is eq-two-layer of the page, f(x) = (1/sqrt(m)) sum_j a_j relu(w_j . x), in the NTK parametrization.
The parameters are theta = (W, a): W has shape (m, d) and a has shape (m,), all drawn from N(0, 1).
The functions compute the Jacobian (eq-jacobian), the empirical NTK (eq-ntk), the infinite-width kernel
(eq-relu-ntk), the linearized model (eq-flin), its training (eq-gd-step, eq-residual-flow), and the relative
change of the weights (eq-relative-change). The loss is L = (1/(2n)) sum_i (f(x_i) - y_i)^2 over n points.
"""

import numpy as np


def relu(u):
    return np.maximum(u, 0.0)


def relu_grad(u):
    """The derivative of relu, with the value 0 at u = 0."""
    return (u > 0).astype(np.float64)


def init(rng, d, m):
    """Draw W of shape (m, d) and a of shape (m,) from N(0, 1)."""
    if d < 1 or m < 1:
        raise ValueError(f"d and m must be positive, got d={d}, m={m}")
    return {"W": rng.standard_normal((m, d)), "a": rng.standard_normal(m)}


def flatten(params):
    """theta as one vector: the rows of W, then a."""
    return np.concatenate([params["W"].ravel(), params["a"]])


def unflatten(theta, d, m):
    return {"W": theta[: m * d].reshape(m, d), "a": theta[m * d :]}


def forward(params, X):
    """f(x) for each row of X (n, d); returns (n,)."""
    W, a = params["W"], params["a"]
    m = a.shape[0]
    # eq: eq-two-layer
    return relu(X @ W.T) @ a / np.sqrt(m)


def jacobian(params, X):
    """The Jacobian of f at the rows of X, shape (n, m*d + m), columns in the order of flatten."""
    W, a = params["W"], params["a"]
    m = a.shape[0]
    U = X @ W.T  # (n, m), the preactivations w_j . x_i
    # eq: eq-jacobian
    dW = (relu_grad(U) * a)[:, :, None] * X[:, None, :] / np.sqrt(m)  # (n, m, d)
    da = relu(U) / np.sqrt(m)  # (n, m)
    return np.concatenate([dW.reshape(X.shape[0], -1), da], axis=1)


def empirical_ntk(params, X1, X2):
    """The empirical NTK J(X1) J(X2)^T, shape (n1, n2)."""
    # eq: eq-ntk
    return jacobian(params, X1) @ jacobian(params, X2).T


def empirical_ntk_terms(params, X1, X2):
    """The two summands of the empirical NTK: the first-layer term (from W) and the second-layer term (from a)."""
    W, a = params["W"], params["a"]
    m = a.shape[0]
    U1, U2 = X1 @ W.T, X2 @ W.T
    first = ((relu_grad(U1) * a**2) @ relu_grad(U2).T) * (X1 @ X2.T) / m
    second = relu(U1) @ relu(U2).T / m
    return first, second


def relu_ntk_terms(X1, X2):
    """The two summands of the infinite-width kernel (eq-relu-ntk): x.x' (pi - phi)/(2 pi) and
    |x||x'| (sin phi + (pi - phi) cos phi)/(2 pi), where phi is the angle between x and x'."""
    n1 = np.linalg.norm(X1, axis=1)
    n2 = np.linalg.norm(X2, axis=1)
    dot = X1 @ X2.T
    norms = np.outer(n1, n2)
    safe = np.where(norms > 0, norms, 1.0)
    cos = np.clip(np.where(norms > 0, dot / safe, 1.0), -1.0, 1.0)
    phi = np.arccos(cos)
    # eq: eq-relu-ntk
    first = dot * (np.pi - phi) / (2 * np.pi)
    second = norms * (np.sin(phi) + (np.pi - phi) * cos) / (2 * np.pi)
    return first, second


def relu_ntk(X1, X2):
    """The infinite-width NTK of the two-layer ReLU network, shape (n1, n2)."""
    first, second = relu_ntk_terms(X1, X2)
    return first + second


def residual_flow(K, r0, eta, t, n):
    """The residual of gradient flow with a constant kernel K (n, n): exp(-eta K t / n) r0 (eq-residual-flow)."""
    lam, V = np.linalg.eigh(K)
    # eq: eq-residual-flow
    return V @ (np.exp(-eta * lam * t / n) * (V.T @ r0))


def train_linearized(params, X, y, eta, steps):
    """Gradient descent on the linearized model f_lin(theta) = f_0 + J_0 (theta - theta_0) (eq-flin).
    Returns the residuals f_lin(X) - y after each step (steps + 1, n) and the final omega = theta - theta_0."""
    n = X.shape[0]
    J = jacobian(params, X)
    f0 = forward(params, X)
    omega = np.zeros(J.shape[1])
    out = [f0 - y]
    for _ in range(steps):
        # eq: eq-flin
        r = f0 + J @ omega - y
        # eq: eq-gd-step
        omega = omega - eta * J.T @ r / n
        out.append(f0 + J @ omega - y)
    return np.array(out), omega


def grad(params, X, y):
    """The gradient of L = (1/(2n)) sum (f - y)^2 with respect to W and a."""
    n = X.shape[0]
    r = forward(params, X) - y
    return unflatten(jacobian(params, X).T @ r / n, X.shape[1], params["a"].shape[0])


def train_network(params, X, y, eta, steps):
    """Gradient descent on the network itself. Returns the final parameters; the input is not changed."""
    p = {k: v.copy() for k, v in params.items()}
    for _ in range(steps):
        g = grad(p, X, y)
        p = {k: p[k] - eta * g[k] for k in p}
    return p


def relative_change(p0, p1):
    """||theta_1 - theta_0|| / ||theta_0|| (eq-relative-change)."""
    t0, t1 = flatten(p0), flatten(p1)
    # eq: eq-relative-change
    return np.linalg.norm(t1 - t0) / np.linalg.norm(t0)
