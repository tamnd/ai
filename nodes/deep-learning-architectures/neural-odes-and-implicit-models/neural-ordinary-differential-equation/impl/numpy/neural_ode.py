# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/neural-ordinary-differential-equation/ (ai://NEW-neural-ordinary-differential-equation).
"""The neural ordinary differential equation, the NumPy reference: solvers, the forward pass and two backward passes.

The hidden state h(t) of width D = d + p follows eq-ode, dh/dt = f(h, t), from h(0) = [x, 0] to h(T); the p zeros
are the augmentation of eq-augment, and p = 0 is the plain model. The vector field is a one-hidden-layer network,
eq-field: f(h, t) = W2 tanh(W1 h + t u + b1) + b2, with W1 of the shape (H, D) and W2 of the shape (D, H), as on
the page; a batch has one example per row, so the field computes tanh(h @ W1.T + t u + b1) @ W2.T + b2.

There are three solvers: Euler (one evaluation of f per step), the classical Runge-Kutta method of order 4
(four per step) and the adaptive Dormand-Prince pair of orders 5 and 4, which picks its own steps to meet a
tolerance and counts its evaluations. The backward pass comes in two kinds. adjoint_backward solves the adjoint
system eq-adjoint and eq-adjoint-theta backwards in time with the same fixed steps (optimize, then discretize),
and stores nothing but the final state. backprop differentiates the steps of the solver exactly (discretize, then
optimize), and stores every stage of the forward pass.
"""

import numpy as np

STAGES = {"euler": 1, "rk4": 4}
NAMES = ("W1", "u", "b1", "W2", "b2")


def config(d=2, d_hidden=8, d_state=None, n_steps=4, T=1.0, method="rk4"):
    """d_state is the width D = d + p of the state; it defaults to d, the plain model with no augmentation."""
    d_state = d if d_state is None else d_state
    if min(d, d_hidden, n_steps) < 1 or d_state < d or T <= 0 or method not in STAGES:
        raise ValueError("d, d_hidden and n_steps must be at least 1, d_state at least d, T positive, method euler or rk4")
    return {"d": d, "d_hidden": d_hidden, "d_state": d_state, "n_steps": n_steps, "T": float(T), "method": method}


def width(cfg):
    return cfg["d_state"]


def param_count(cfg):
    D, H = width(cfg), cfg["d_hidden"]
    return 2 * D * H + 2 * H + D  # eq: eq-params


def nfe(cfg):
    """The evaluations of f in one forward solve with fixed steps."""
    return STAGES[cfg["method"]] * cfg["n_steps"]


def flops(cfg, B):
    """FLOPs of the forward solve over a batch of B, a multiply-add counted as 2 (eq-flops)."""
    return nfe(cfg) * 2 * B * 2 * width(cfg) * cfg["d_hidden"]


def init(rng, cfg, scale=1.0):
    D, H = width(cfg), cfg["d_hidden"]
    return {
        "W1": rng.normal(0.0, scale / np.sqrt(D), (H, D)),
        "u": rng.normal(0.0, scale, H),
        "b1": np.zeros(H),
        "W2": rng.normal(0.0, scale / np.sqrt(H), (D, H)),
        "b2": np.zeros(D),
    }


def field(params, h, t):
    """eq-field on a batch h of the shape (B, D) at the time t."""
    s = np.tanh(h @ params["W1"].T + t * params["u"] + params["b1"])  # (B, H)
    return s @ params["W2"].T + params["b2"]  # (B, D)


def field_vjp(params, h, t, a):
    """The vector-Jacobian products a^T df/dh and a^T df/dtheta of eq-adjoint and eq-adjoint-theta, summed over the batch."""
    s = np.tanh(h @ params["W1"].T + t * params["u"] + params["b1"])  # (B, H)
    g = (a @ params["W2"]) * (1.0 - s * s)  # (B, H)
    grads = {"W1": g.T @ h, "u": t * g.sum(0), "b1": g.sum(0), "W2": a.T @ s, "b2": a.sum(0)}
    return g @ params["W1"], grads  # (B, D)


def trace_jacobian(params, h, t):
    """tr(df/dh) for each row of h, at a cost linear in H: tr(W2 diag(1 - s^2) W1) = sum_j (1 - s_j^2) (W1 W2)_jj."""
    s = np.tanh(h @ params["W1"].T + t * params["u"] + params["b1"])
    c = np.einsum("ji,ij->j", params["W1"], params["W2"])
    return (1.0 - s * s) @ c  # (B,)


def euler(f, h, t0, t1, n):
    dt = (t1 - t0) / n
    for k in range(n):
        h = h + dt * f(h, t0 + k * dt)  # eq: eq-euler
    return h


def rk4(f, h, t0, t1, n):
    dt = (t1 - t0) / n
    for k in range(n):
        t = t0 + k * dt
        k1 = f(h, t)
        k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(h + dt * k3, t + dt)
        h = h + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return h


SOLVERS = {"euler": euler, "rk4": rk4}

# The Dormand-Prince pair: the nodes c, the matrix A, the weights of order 5 (b5) and of order 4 (b4).
DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
DP_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
DP_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def dopri5(f, h, t0, t1, rtol=1e-6, atol=1e-8, max_steps=100000):
    """Solve from t0 to t1 > t0 with steps that keep the local error estimate below the tolerance.

    It returns the state at t1 and the number of evaluations of f. The last stage of an accepted step is the
    first stage of the next one, so a step costs six evaluations.
    """
    t, dt = t0, (t1 - t0) / 10.0
    k1, n_eval = f(h, t), 1
    for _ in range(max_steps):
        if t >= t1:
            return h, n_eval
        dt = min(dt, t1 - t)
        ks = [k1]
        for i in range(1, 7):
            y = h + dt * sum(a * k for a, k in zip(DP_A[i], ks))
            ks.append(f(y, t + DP_C[i] * dt))
        n_eval += 6
        h5 = h + dt * sum(b * k for b, k in zip(DP_B5, ks))
        err = dt * sum((b5 - b4) * k for b5, b4, k in zip(DP_B5, DP_B4, ks))
        scale = atol + rtol * np.maximum(np.abs(h), np.abs(h5))
        e = float(np.sqrt(np.mean((err / scale) ** 2)))
        if e <= 1.0:
            t, h, k1 = t + dt, h5, ks[6]
        dt *= min(5.0, max(0.2, 0.9 * (e if e > 0 else 1e-10) ** -0.2))
    raise RuntimeError("dopri5 did not reach t1 in max_steps steps")


def augment(x, cfg):
    return np.hstack([x, np.zeros((x.shape[0], cfg["d_state"] - cfg["d"]))])  # eq: eq-augment


def forward(params, x, cfg):
    """h(T) for the inputs x of the shape (B, d): the state of width D at the end of eq-ode."""
    h0 = augment(x, cfg)  # (B, D)
    return SOLVERS[cfg["method"]](lambda h, t: field(params, h, t), h0, 0.0, cfg["T"], cfg["n_steps"])  # (B, D)


def adjoint_backward(params, y, dy, cfg):
    """dL/dx and dL/dtheta from y = h(T) and dy = dL/dh(T), by the adjoint method (Algorithm 1 of Chen et al.).

    It solves the augmented system [h, a, g] backwards from T to 0 with the solver and steps of the forward pass:
    dh/dt = f(h, t), da/dt = -a^T df/dh (eq-adjoint), dg/dt = -a^T df/dtheta (eq-adjoint-theta), with
    h(T) = y, a(T) = dy and g(T) = 0. The state h is recomputed backwards rather than stored, so the state of the
    backward solve is one vector of 2 B D + P numbers, whatever the number of steps.
    """
    shapes = {k: params[k].shape for k in NAMES}
    sizes = [int(np.prod(shapes[k])) for k in NAMES]
    B, D = y.shape
    n = B * D

    def aug(z, t):
        h, a = z[:n].reshape(B, D), z[n:2 * n].reshape(B, D)
        s = np.tanh(h @ params["W1"].T + t * params["u"] + params["b1"])  # (B, H), shared by f and its products
        f = s @ params["W2"].T + params["b2"]  # (B, D)
        g = (a @ params["W2"]) * (1.0 - s * s)  # (B, H)
        grads = {"W1": g.T @ h, "u": t * g.sum(0), "b1": g.sum(0), "W2": a.T @ s, "b2": a.sum(0)}
        return np.concatenate([f.ravel(), -(g @ params["W1"]).ravel()] + [-grads[k].ravel() for k in NAMES])

    z = np.concatenate([y.ravel(), dy.ravel(), np.zeros(sum(sizes))])  # (2 B D + P,)
    z = SOLVERS[cfg["method"]](aug, z, cfg["T"], 0.0, cfg["n_steps"])
    a0, g = z[n:2 * n].reshape(B, D), z[2 * n:]
    grads, i = {}, 0
    for k, m in zip(NAMES, sizes):
        grads[k] = g[i:i + m].reshape(shapes[k])
        i += m
    return a0[:, :cfg["d"]], grads


def backprop(params, x, dy, cfg):
    """dL/dx and dL/dtheta of the discrete forward pass, by the chain rule through every stage of the solver."""
    method, n, T = cfg["method"], cfg["n_steps"], cfg["T"]
    dt = T / n
    h = augment(x, cfg)
    tape = []
    for k in range(n):
        t = k * dt
        if method == "euler":
            stages = [(h, t)]
            h = h + dt * field(params, h, t)
        else:
            k1 = field(params, h, t)
            k2 = field(params, h + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = field(params, h + 0.5 * dt * k2, t + 0.5 * dt)
            stages = [(h, t), (h + 0.5 * dt * k1, t + 0.5 * dt), (h + 0.5 * dt * k2, t + 0.5 * dt), (h + dt * k3, t + dt)]
            k4 = field(params, stages[3][0], t + dt)
            h = h + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tape.append(stages)
    grads = {k: np.zeros_like(v) for k, v in params.items()}

    def add(gs):
        for k in grads:
            grads[k] += gs[k]

    a = dy.copy()
    for stages in reversed(tape):
        if method == "euler":
            vh, gs = field_vjp(params, *stages[0], dt * a)
            add(gs)
            a = a + vh
            continue
        # h' = h + dt/6 (k1 + 2 k2 + 2 k3 + k4), with k_i = f(y_i) and y_2 = h + dt/2 k1, y_3 = h + dt/2 k2, y_4 = h + dt k3.
        w = [dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0]
        ak = [wi * a for wi in w]  # dL/dk_i from the update alone
        ay = [None] * 4
        for i, back in ((3, (2, dt)), (2, (1, 0.5 * dt)), (1, (0, 0.5 * dt)), (0, None)):
            vh, gs = field_vjp(params, *stages[i], ak[i])
            add(gs)
            ay[i] = vh
            if back is not None:
                j, c = back
                ak[j] = ak[j] + c * vh  # y_i depends on k_j
        a = a + sum(ay)
    return a[:, :cfg["d"]], grads


def log_det_flow(params, x, cfg):
    """h(T) and the change of log density log p(h(T)) - log p(h(0)) of eq-cnf, integrated next to the state."""
    D = width(cfg)

    def aug(z, t):
        h = z[:, :D]
        return np.hstack([field(params, h, t), -trace_jacobian(params, h, t)[:, None]])

    z = np.hstack([augment(x, cfg), np.zeros((x.shape[0], 1))])
    z = SOLVERS[cfg["method"]](aug, z, 0.0, cfg["T"], cfg["n_steps"])
    return z[:, :D], z[:, D]


def mse_loss(y, t):
    """Half the mean over the batch of the squared error, and its gradient."""
    B = y.shape[0]
    return 0.5 * float(((y - t) ** 2).sum()) / B, (y - t) / B


def train(params, x, t, cfg, lr=0.1, steps=100):
    """Plain gradient descent with the adjoint gradient on the loss of the first d coordinates of h(T)."""
    losses = []
    for _ in range(steps):
        y = forward(params, x, cfg)
        loss, dyd = mse_loss(y[:, :cfg["d"]], t)
        dy = np.zeros_like(y)
        dy[:, :cfg["d"]] = dyd
        _, grads = adjoint_backward(params, y, dy, cfg)
        params = {k: params[k] - lr * grads[k] for k in params}
        losses.append(loss)
    return params, losses
