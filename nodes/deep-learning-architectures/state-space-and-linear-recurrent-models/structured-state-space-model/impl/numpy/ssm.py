# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/structured-state-space-model/ (ai://NEW-structured-state-space-model).
"""A diagonal state space layer, the NumPy reference.

Each of the H channels is a single-input single-output SSM with N complex diagonal entries kept in conjugate
pairs, so the output is twice a real part (eq-diagonal-kernel). It is discretized by eq-zoh or eq-bilinear and
run as the recurrence eq-recurrence or the causal convolution with the kernel eq-kernel. The dense helpers
work on one general real SSM (A, B, C) for the tests and the derivation checks.
"""

import math

import numpy as np

METHODS = ("zoh", "bilinear")


def config(h=2, n=4, method="zoh", dt_min=1e-3, dt_max=1e-1):
    """Return the configuration: h channels, n complex state entries per channel, the discretization rule."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if h < 1 or n < 1:
        raise ValueError("h and n must be positive")
    if not 0 < dt_min <= dt_max:
        raise ValueError("need 0 < dt_min <= dt_max")
    return {"h": int(h), "n": int(n), "method": method, "dt_min": float(dt_min), "dt_max": float(dt_max)}


def s4d_lin(n):
    """The S4D-Lin initialization: A_k = -1/2 + i pi k."""
    k = np.arange(n)
    return -0.5 + 1j * np.pi * k


def hippo_legs(n):
    """The HiPPO-LegS state matrix as S4 uses it (equation 2 of Gu, Goel and Re), with its sign: -A_nk."""
    k = np.arange(n)
    a = np.sqrt(np.outer(2 * k + 1, 2 * k + 1))
    return -(np.tril(a, -1) + np.diag(k + 1.0))


def init(rng, cfg):
    """Draw the parameters: S4D-Lin for A, log step sizes uniform in [log dt_min, log dt_max], B = 1, C complex normal."""
    h, n = cfg["h"], cfg["n"]
    a = s4d_lin(n)
    log_dt = rng.uniform(math.log(cfg["dt_min"]), math.log(cfg["dt_max"]), size=h)
    c = (rng.standard_normal((h, n)) + 1j * rng.standard_normal((h, n))) * math.sqrt(0.5)
    return {
        "log_dt": log_dt,
        "log_a_re": np.log(-a.real) * np.ones((h, 1)),
        "a_im": a.imag * np.ones((h, 1)),
        "b": np.ones((h, n), dtype=np.complex128),
        "c": c,
        "d": rng.standard_normal(h),
    }


def continuous(params):
    """Return (A, B, dt): A = -exp(log_a_re) + i a_im has a negative real part for every parameter value."""
    a = -np.exp(params["log_a_re"]) + 1j * params["a_im"]
    return a, params["b"], np.exp(params["log_dt"])


def discretize(a, b, dt, method):
    """Discretize a diagonal SSM entrywise. a, b are (..., N) complex arrays and dt broadcasts against them."""
    dt = np.asarray(dt)
    if method == "zoh":
        ab = np.exp(dt * a)
        bb = (ab - 1.0) / a * b
    elif method == "bilinear":
        den = 1.0 - dt * a / 2.0
        ab = (1.0 + dt * a / 2.0) / den
        bb = dt * b / den
    else:
        raise ValueError(f"unknown method {method!r}")
    return ab, bb


def _discrete(params, cfg):
    a, b, dt = continuous(params)
    return discretize(a, b, dt[:, None], cfg["method"])


def kernel(params, length, cfg):
    """The convolution kernel K of shape (H, L): K[h, l] = 2 Re sum_n C[h, n] Bb[h, n] Ab[h, n]^l."""
    ab, bb = _discrete(params, cfg)
    powers = ab[:, :, None] ** np.arange(length)  # (H, N, L), the Vandermonde matrix of each channel
    return 2.0 * np.einsum("hn,hnl->hl", params["c"] * bb, powers).real


def causal_conv(u, k):
    """y[t] = sum_{j <= t} k[t - j] u[j] along axis -2 of u (B, L, H), k of shape (H, L); FFT of size 2L."""
    length = u.shape[-2]
    n = 2 * length
    uf = np.fft.rfft(u, n=n, axis=-2)
    kf = np.fft.rfft(k.T, n=n, axis=0)
    return np.fft.irfft(uf * kf, n=n, axis=-2)[..., :length, :]


def forward_conv(params, u, cfg):
    """The convolutional mode: u (B, L, H) to y (B, L, H) through the kernel and the FFT."""
    u = np.asarray(u, dtype=np.float64)
    k = kernel(params, u.shape[-2], cfg)
    return causal_conv(u, k) + params["d"] * u


def init_state(cfg, batch):
    """The zero state of shape (B, H, N)."""
    return np.zeros((batch, cfg["h"], cfg["n"]), dtype=np.complex128)


def step(params, x, u_t, cfg):
    """One step of the recurrence for inference: x (B, H, N), u_t (B, H) to the new state and y_t (B, H)."""
    ab, bb = _discrete(params, cfg)
    x = ab * x + bb * u_t[..., None]
    y_t = 2.0 * np.einsum("hn,bhn->bh", params["c"], x).real + params["d"] * u_t
    return x, y_t


def forward_scan(params, u, cfg):
    """The recurrent mode: run step over the sequence, O(L N) work per channel and O(N) state."""
    u = np.asarray(u, dtype=np.float64)
    x = init_state(cfg, u.shape[0])
    ys = []
    for t in range(u.shape[-2]):
        x, y_t = step(params, x, u[:, t, :], cfg)
        ys.append(y_t)
    return np.stack(ys, axis=1)


# The dense helpers: one real SSM x' = A x + B u, y = C x with A (N, N), B (N,), C (N,).


def expm(m):
    """The matrix exponential by scaling and squaring with a Taylor series of degree 18."""
    m = np.asarray(m, dtype=np.float64)
    norm = np.linalg.norm(m, 1)
    s = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0 else 0
    x = m / 2.0**s
    term = np.eye(len(m))
    out = np.eye(len(m))
    for k in range(1, 19):
        term = term @ x / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def discretize_dense(a, b, dt, method):
    """Discretize a dense SSM. ZOH uses the exponential of the block matrix [[A, B], [0, 0]] (Van Loan)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = len(a)
    if method == "zoh":
        big = np.zeros((n + 1, n + 1))
        big[:n, :n] = a
        big[:n, n] = b
        e = expm(dt * big)
        return e[:n, :n], e[:n, n]
    if method == "bilinear":
        eye = np.eye(n)
        lhs = eye - dt / 2.0 * a
        return np.linalg.solve(lhs, eye + dt / 2.0 * a), np.linalg.solve(lhs, dt * b)
    raise ValueError(f"unknown method {method!r}")


def kernel_dense(ab, bb, c, length):
    """K = (C Bb, C Ab Bb, ..., C Ab^(L-1) Bb), by L - 1 matrix-vector products."""
    out = np.empty(length)
    v = np.asarray(bb, dtype=np.float64).copy()
    for i in range(length):
        out[i] = c @ v
        v = ab @ v
    return out


def scan_dense(ab, bb, c, u):
    """y_k = C x_k with x_k = Ab x_(k-1) + Bb u_k and x_(-1) = 0, for a 1-D input u."""
    x = np.zeros(len(ab))
    y = np.empty(len(u))
    for k, u_k in enumerate(u):
        x = ab @ x + bb * u_k
        y[k] = c @ x
    return y


def selective_scan(u, delta, a, b, c):
    """A time-varying diagonal SSM, the selective form of Mamba, discretized with ZOH at each step.

    u, delta: (L,); a: (N,) real negative; b, c: (L, N). Returns y (L,). There is no single kernel:
    the step k uses its own Ab_k = exp(delta_k a) and Bb_k = (Ab_k - 1) / a * b_k, so only the scan applies.
    """
    x = np.zeros(len(a))
    y = np.empty(len(u))
    for k in range(len(u)):
        ab = np.exp(delta[k] * a)
        x = ab * x + (ab - 1.0) / a * b[k] * u[k]
        y[k] = c[k] @ x
    return y
