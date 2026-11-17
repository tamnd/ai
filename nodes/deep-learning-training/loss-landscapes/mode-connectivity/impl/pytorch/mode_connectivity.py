# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://MDXMTA).
"""Mode connectivity in PyTorch: the same objects as the NumPy reference, on tensors, with gradients from autograd.

linear_path, bezier_path and polychain_path are the paths of the page (eq-linear, eq-bezier); loss_barrier gives the
barrier with the mean or the linear reference (eq-barrier-mean, eq-barrier-linear); init_mlp, forward, loss, error
and train_gd are a one-hidden-layer ReLU MLP trained by full-batch gradient descent; permute_hidden reorders its
hidden units (eq-permute); linear_assignment is a Hungarian solver on CPU lists and weight_matching aligns network
B to network A (eq-matching); fit_bend trains the bend of a Bezier curve with t ~ U(0, 1) (eq-curve-loss); demo
returns the numbers of the run described on the page."""

import math

import torch

DTYPE = torch.float64

# Paths


def linear_path(w1, w2, t):
    """The point (1 - t) w1 + t w2 (eq-linear)."""
    return (1.0 - t) * w1 + t * w2


def bezier_path(w1, theta, w2, t):
    """The quadratic Bezier curve (1 - t)^2 w1 + 2 t (1 - t) theta + t^2 w2 (eq-bezier)."""
    return (1.0 - t) ** 2 * w1 + 2.0 * t * (1.0 - t) * theta + t * t * w2


def polychain_path(w1, theta, w2, t):
    """The two-segment path w1 -> theta -> w2, with theta reached at t = 1/2."""
    if t <= 0.5:
        return linear_path(w1, theta, 2.0 * t)
    return linear_path(theta, w2, 2.0 * t - 1.0)


def loss_barrier(loss_fn, path, ts, reference="mean"):
    """The barrier along path(t) for the grid ts, which must start at 0 and end at 1.

    reference="mean": max_t L(t) - (L(0) + L(1)) / 2 (eq-barrier-mean); reference="linear":
    max_t [L(t) - ((1 - t) L(0) + t L(1))] (eq-barrier-linear). Returns the barrier, its t, and the losses."""
    ts = torch.as_tensor(ts, dtype=DTYPE)
    if ts[0].item() != 0.0 or ts[-1].item() != 1.0:
        raise ValueError("ts must start at 0 and end at 1")
    with torch.no_grad():
        losses = torch.tensor([float(loss_fn(path(float(t)))) for t in ts], dtype=DTYPE)
    if reference == "mean":
        gap = losses - 0.5 * (losses[0] + losses[-1])
    elif reference == "linear":
        gap = losses - ((1.0 - ts) * losses[0] + ts * losses[-1])
    else:
        raise ValueError(f"unknown reference {reference!r}")
    k = int(torch.argmax(gap))
    return float(gap[k]), float(ts[k]), losses


# A one-hidden-layer ReLU MLP; params is a dict with W1 (h, d), b1 (h,), W2 (k, h), b2 (k,)

NAMES = ("W1", "b1", "W2", "b2")


def init_mlp(d, h, k, generator):
    """He-scaled Gaussian weights and zero biases."""
    return {
        "W1": torch.randn(h, d, generator=generator, dtype=DTYPE) * math.sqrt(2.0 / d),
        "b1": torch.zeros(h, dtype=DTYPE),
        "W2": torch.randn(k, h, generator=generator, dtype=DTYPE) * math.sqrt(2.0 / h),
        "b2": torch.zeros(k, dtype=DTYPE),
    }


def flatten(params):
    """The parameters as one vector, in the order W1, b1, W2, b2."""
    return torch.cat([params[n].reshape(-1) for n in NAMES])


def unflatten(vec, d, h, k):
    """The inverse of flatten for a d-h-k network."""
    shapes = ((h, d), (h,), (k, h), (k,))
    sizes = [math.prod(s) for s in shapes]
    if sum(sizes) != vec.numel():
        raise ValueError("vector size does not match the shapes")
    parts = torch.split(vec, sizes)
    return {n: p.reshape(s) for n, p, s in zip(NAMES, parts, shapes, strict=True)}


def forward(params, X):
    """The logits W2 relu(W1 x + b1) + b2 for each row x of X."""
    return torch.relu(X @ params["W1"].T + params["b1"]) @ params["W2"].T + params["b2"]


def loss(params, X, y):
    """Mean softmax cross-entropy against integer labels y."""
    return torch.nn.functional.cross_entropy(forward(params, X), y)


def error(params, X, y):
    """The fraction of rows whose largest logit is not the label."""
    return float((forward(params, X).argmax(dim=1) != y).to(DTYPE).mean())


def train_gd(params, X, y, lr, steps):
    """Full-batch gradient descent on loss, with gradients from autograd; returns new parameters."""
    p = {n: v.detach().clone().requires_grad_(True) for n, v in params.items()}
    for _ in range(steps):
        grads = torch.autograd.grad(loss(p, X, y), [p[n] for n in NAMES])
        with torch.no_grad():
            for n, g in zip(NAMES, grads, strict=True):
                p[n] -= lr * g
    return {n: v.detach() for n, v in p.items()}


# Permutations and weight matching


def permute_hidden(params, perm):
    """Unit i of the result is unit perm[i] of params (eq-permute); forward is unchanged."""
    perm = torch.as_tensor(perm, dtype=torch.long)
    return {"W1": params["W1"][perm], "b1": params["b1"][perm], "W2": params["W2"][:, perm], "b2": params["b2"].clone()}


def linear_assignment(score):
    """The permutation perm maximizing sum_i score[i, perm[i]], by the O(n^3) Hungarian method on Python lists."""
    c = [[-float(x) for x in row] for row in torch.as_tensor(score).tolist()]
    n = len(c)
    if any(len(row) != n for row in c):
        raise ValueError("score must be square")
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    perm = [0] * n
    for j in range(1, n + 1):
        perm[p[j] - 1] = j - 1
    return torch.tensor(perm, dtype=torch.long)


def weight_matching(params_a, params_b):
    """The hidden-unit permutation of B that best matches A, and the permuted B (eq-matching).

    One linear assignment on C = W1_A W1_B^T + b1_A b1_B^T + W2_A^T W2_B."""
    score = (
        params_a["W1"] @ params_b["W1"].T
        + torch.outer(params_a["b1"], params_b["b1"])
        + params_a["W2"].T @ params_b["W2"]
    )
    perm = linear_assignment(score)
    return perm, permute_hidden(params_b, perm)


# Curve finding


def fit_bend(loss_fn, w1, w2, steps, lr, generator, theta0=None):
    """Train the bend theta of a Bezier curve by SGD on E_{t ~ U(0, 1)} L(phi_theta(t)) (eq-curve-loss).

    loss_fn maps a flat weight tensor to a scalar tensor; autograd gives the gradient in theta."""
    theta = (0.5 * (w1 + w2) if theta0 is None else torch.as_tensor(theta0, dtype=DTYPE)).clone().requires_grad_(True)
    for _ in range(steps):
        t = float(torch.rand((), generator=generator, dtype=DTYPE))
        (g,) = torch.autograd.grad(loss_fn(bezier_path(w1, theta, w2, t)), [theta])
        with torch.no_grad():
            theta -= lr * g
    return theta.detach()


# The run of the page


def make_data(n, generator):
    """Two interleaved half-moons in 2D with Gaussian noise of standard deviation 0.1; labels 0 and 1."""
    m = n // 2
    s = torch.rand(n, generator=generator, dtype=DTYPE) * math.pi
    top = torch.stack([torch.cos(s[:m]), torch.sin(s[:m])], dim=1)
    bottom = torch.stack([1.0 - torch.cos(s[m:]), 0.5 - torch.sin(s[m:])], dim=1)
    X = torch.cat([top, bottom]) + 0.1 * torch.randn(n, 2, generator=generator, dtype=DTYPE)
    y = torch.cat([torch.zeros(m, dtype=torch.long), torch.ones(n - m, dtype=torch.long)])
    return X, y


def demo(seed_a, seed_b, width=64, steps=2000, lr=0.5, points=25, bend_steps=2000):
    """Two 2-width-2 MLPs trained from two seeds on the same data (seed 0); the barrier (mean reference) on the
    naive segment, on a trained Bezier curve, and on the segment after weight matching. Same record as NumPy."""
    X, y = make_data(256, torch.Generator().manual_seed(0))
    a = train_gd(init_mlp(2, width, 2, torch.Generator().manual_seed(seed_a)), X, y, lr, steps)
    b = train_gd(init_mlp(2, width, 2, torch.Generator().manual_seed(seed_b)), X, y, lr, steps)
    perm, b_matched = weight_matching(a, b)
    ts = torch.linspace(0.0, 1.0, points, dtype=DTYPE)
    wa, wb, wm = flatten(a), flatten(b), flatten(b_matched)

    def f(w):
        return loss(unflatten(w, 2, width, 2), X, y)

    naive, t_naive, _ = loss_barrier(f, lambda t: linear_path(wa, wb, t), ts)
    matched, t_matched, _ = loss_barrier(f, lambda t: linear_path(wa, wm, t), ts)
    theta = fit_bend(f, wa, wb, bend_steps, lr, torch.Generator().manual_seed(seed_a))
    curved, _, _ = loss_barrier(f, lambda t: bezier_path(wa, theta, wb, t), ts)
    return {
        "loss_a": float(loss(a, X, y)),
        "loss_b": float(loss(b, X, y)),
        "error_a": error(a, X, y),
        "error_b": error(b, X, y),
        "barrier_naive": naive,
        "t_naive": t_naive,
        "barrier_bezier": curved,
        "barrier_matched": matched,
        "t_matched": t_matched,
        "perm": perm,
    }
