# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/mode-connectivity/ (ai://MDXMTA).
"""Mode connectivity in NumPy: paths between two weight vectors, the loss barrier along a path, the hidden-unit
permutation of a one-hidden-layer MLP, and weight matching that aligns one network to another.

linear_path is the segment (1 - t) w1 + t w2 (eq-linear); bezier_path is the quadratic Bezier curve with bend theta
(eq-bezier) and polychain_path the two-segment path through theta; bend_weight is the factor by which a loss
gradient at a point of the curve reaches theta. loss_barrier evaluates a loss along a path and returns the barrier
in either of the two conventions of the literature (eq-barrier-mean, eq-barrier-linear). init_mlp, forward, loss
and train_gd are a ReLU MLP with one hidden layer and softmax cross-entropy, trained by full-batch gradient descent;
permute_hidden applies a hidden-unit permutation (eq-permute), linear_assignment is a Hungarian solver, and
weight_matching finds the permutation of network B that best matches the weights of network A (eq-matching).
fit_bend trains the bend of a Bezier curve by sampling t uniformly (eq-curve-loss). demo returns the numbers of the
run described on the page."""

import numpy as np

# Paths


def linear_path(w1, w2, t):
    """The point (1 - t) w1 + t w2 of the segment from w1 to w2 (eq-linear)."""
    return (1.0 - t) * np.asarray(w1, dtype=np.float64) + t * np.asarray(w2, dtype=np.float64)


def bezier_path(w1, theta, w2, t):
    """The quadratic Bezier curve (1 - t)^2 w1 + 2 t (1 - t) theta + t^2 w2 (eq-bezier)."""
    w1, theta, w2 = (np.asarray(a, dtype=np.float64) for a in (w1, theta, w2))
    return (1.0 - t) ** 2 * w1 + 2.0 * t * (1.0 - t) * theta + t * t * w2


def polychain_path(w1, theta, w2, t):
    """The two-segment path w1 -> theta -> w2, with theta reached at t = 1/2."""
    if t <= 0.5:
        return linear_path(w1, theta, 2.0 * t)
    return linear_path(theta, w2, 2.0 * t - 1.0)


def bend_weight(kind, t):
    """d phi(t) / d theta: 2 t (1 - t) for the Bezier curve; 2 t, then 2 (1 - t), for the polychain."""
    if kind == "bezier":
        return 2.0 * t * (1.0 - t)
    if kind == "polychain":
        return 2.0 * t if t <= 0.5 else 2.0 * (1.0 - t)
    raise ValueError(f"unknown path kind {kind!r}")


# The loss barrier


def loss_barrier(loss_fn, path, ts, reference="mean"):
    """The loss barrier along path(t) for t in ts, which must start at 0 and end at 1.

    reference="mean" gives max_t L(t) - (L(0) + L(1)) / 2 (eq-barrier-mean), as in Frankle et al. (2020) and
    Ainsworth et al. (2023); reference="linear" gives max_t [L(t) - ((1 - t) L(0) + t L(1))] (eq-barrier-linear),
    as in Entezari et al. (2022). Returns the barrier, the t at which it is reached, and the losses."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts[0] != 0.0 or ts[-1] != 1.0:
        raise ValueError("ts must start at 0 and end at 1")
    losses = np.array([loss_fn(path(t)) for t in ts])
    if reference == "mean":
        gap = losses - 0.5 * (losses[0] + losses[-1])
    elif reference == "linear":
        gap = losses - ((1.0 - ts) * losses[0] + ts * losses[-1])
    else:
        raise ValueError(f"unknown reference {reference!r}")
    k = int(np.argmax(gap))
    return float(gap[k]), float(ts[k]), losses


# A one-hidden-layer ReLU MLP; params is a dict with W1 (h, d), b1 (h,), W2 (k, h), b2 (k,)


def init_mlp(d, h, k, rng):
    """He-scaled Gaussian weights and zero biases."""
    return {
        "W1": rng.standard_normal((h, d)) * np.sqrt(2.0 / d),
        "b1": np.zeros(h),
        "W2": rng.standard_normal((k, h)) * np.sqrt(2.0 / h),
        "b2": np.zeros(k),
    }


def flatten(params):
    """The parameters as one vector, in the order W1, b1, W2, b2."""
    return np.concatenate([params[n].ravel() for n in ("W1", "b1", "W2", "b2")])


def unflatten(vec, d, h, k):
    """The inverse of flatten for a d-h-k network."""
    vec = np.asarray(vec, dtype=np.float64)
    shapes = (("W1", (h, d)), ("b1", (h,)), ("W2", (k, h)), ("b2", (k,)))
    out, i = {}, 0
    for name, shape in shapes:
        n = int(np.prod(shape))
        out[name] = vec[i : i + n].reshape(shape)
        i += n
    if i != vec.size:
        raise ValueError("vector size does not match the shapes")
    return out


def forward(params, X):
    """The logits W2 relu(W1 x + b1) + b2 for each row x of X."""
    return np.maximum(X @ params["W1"].T + params["b1"], 0.0) @ params["W2"].T + params["b2"]


def loss(params, X, y):
    """Mean softmax cross-entropy of the logits against integer labels y."""
    z = forward(params, X)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def error(params, X, y):
    """The fraction of rows whose largest logit is not the label."""
    return float(np.mean(np.argmax(forward(params, X), axis=1) != y))


def grad(params, X, y):
    """The gradient of loss with respect to each parameter, by a hand-written backward pass."""
    a = X @ params["W1"].T + params["b1"]
    r = np.maximum(a, 0.0)
    z = r @ params["W2"].T + params["b2"]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    dz = p / len(y)
    da = (dz @ params["W2"]) * (a > 0.0)
    return {"W1": da.T @ X, "b1": da.sum(axis=0), "W2": dz.T @ r, "b2": dz.sum(axis=0)}


def train_gd(params, X, y, lr, steps):
    """Full-batch gradient descent on loss; returns new parameters."""
    p = {n: v.copy() for n, v in params.items()}
    for _ in range(steps):
        g = grad(p, X, y)
        for n in p:
            p[n] -= lr * g[n]
    return p


# Permutations and weight matching


def permute_hidden(params, perm):
    """Reorder the hidden units: unit i of the result is unit perm[i] of params (eq-permute).

    W1 and b1 have their rows permuted and W2 its columns, so forward is unchanged."""
    perm = np.asarray(perm)
    return {"W1": params["W1"][perm], "b1": params["b1"][perm], "W2": params["W2"][:, perm], "b2": params["b2"].copy()}


def linear_assignment(score):
    """The permutation perm that maximizes sum_i score[i, perm[i]] for a square matrix, by the Hungarian method.

    Potentials u, v and the augmenting-path bookkeeping follow the O(n^3) shortest-path form of the algorithm."""
    c = -np.asarray(score, dtype=np.float64)
    n = c.shape[0]
    if c.shape != (n, n):
        raise ValueError("score must be square")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    perm = np.empty(n, dtype=int)
    perm[p[1:] - 1] = np.arange(n)
    return perm


def weight_matching(params_a, params_b):
    """The hidden-unit permutation of B whose weights best match A, and the permuted B (eq-matching).

    With one hidden layer, weight matching is a single linear assignment: maximize sum_i C[i, perm[i]] with
    C = W1_A W1_B^T + b1_A b1_B^T + W2_A^T W2_B."""
    score = (
        params_a["W1"] @ params_b["W1"].T
        + np.outer(params_a["b1"], params_b["b1"])
        + params_a["W2"].T @ params_b["W2"]
    )
    perm = linear_assignment(score)
    return perm, permute_hidden(params_b, perm)


# Curve finding


def fit_bend(loss_grad, w1, w2, steps, lr, rng, theta0=None):
    """Train the bend of a Bezier curve by stochastic gradient descent on E_{t ~ U(0, 1)} L(phi(t)) (eq-curve-loss).

    loss_grad(w) returns the gradient of the loss at the flat weight vector w. Each step samples one t."""
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    theta = 0.5 * (w1 + w2) if theta0 is None else np.array(theta0, dtype=np.float64)
    for _ in range(steps):
        t = rng.uniform()
        theta = theta - lr * bend_weight("bezier", t) * loss_grad(bezier_path(w1, theta, w2, t))
    return theta


# The run of the page


def make_data(n, rng):
    """Two interleaved half-moons in 2D with Gaussian noise of standard deviation 0.1; labels 0 and 1."""
    m = n // 2
    s = rng.uniform(0.0, np.pi, size=n)
    X = np.empty((n, 2))
    X[:m] = np.stack([np.cos(s[:m]), np.sin(s[:m])], axis=1)
    X[m:] = np.stack([1.0 - np.cos(s[m:]), 0.5 - np.sin(s[m:])], axis=1)
    X += 0.1 * rng.standard_normal(X.shape)
    y = np.concatenate([np.zeros(m, dtype=int), np.ones(n - m, dtype=int)])
    return X, y


def demo(seed_a, seed_b, width=64, steps=2000, lr=0.5, points=25, bend_steps=2000):
    """Train two 2-width-2 MLPs on the same data from two initialization seeds and compare paths between them.

    The data come from seed 0. Returns the train losses and errors at the ends, the barrier (mean reference) on
    the naive segment A -> B, on a Bezier curve A -> B whose bend was trained for bend_steps steps, and on the
    segment A -> P(B) after weight matching, and the matched permutation. The page runs demo(2s + 1, 2s + 2) for
    s = 0, ..., 9."""
    X, y = make_data(256, np.random.default_rng(0))
    a = train_gd(init_mlp(2, width, 2, np.random.default_rng(seed_a)), X, y, lr, steps)
    b = train_gd(init_mlp(2, width, 2, np.random.default_rng(seed_b)), X, y, lr, steps)
    perm, b_matched = weight_matching(a, b)
    ts = np.linspace(0.0, 1.0, points)
    wa, wb, wm = flatten(a), flatten(b), flatten(b_matched)

    def f(w):
        return loss(unflatten(w, 2, width, 2), X, y)

    naive, t_naive, _ = loss_barrier(f, lambda t: linear_path(wa, wb, t), ts)
    matched, t_matched, _ = loss_barrier(f, lambda t: linear_path(wa, wm, t), ts)

    def g(w):
        return flatten(grad(unflatten(w, 2, width, 2), X, y))

    theta = fit_bend(g, wa, wb, bend_steps, lr, np.random.default_rng(seed_a))
    curved, _, _ = loss_barrier(f, lambda t: bezier_path(wa, theta, wb, t), ts)
    return {
        "loss_a": loss(a, X, y),
        "loss_b": loss(b, X, y),
        "error_a": error(a, X, y),
        "error_b": error(b, X, y),
        "barrier_naive": naive,
        "t_naive": t_naive,
        "barrier_bezier": curved,
        "barrier_matched": matched,
        "t_matched": t_matched,
        "perm": perm,
    }
