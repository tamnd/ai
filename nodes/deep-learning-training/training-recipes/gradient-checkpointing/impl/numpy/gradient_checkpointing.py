# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
"""Gradient checkpointing on the chain eq-layer, the NumPy reference: segments, recursive bisection, the BPTT-HSM
dynamic program eq-hsm of Gruslys et al., and the per-layer count eq-transformer-memory of Korthikanti et al. A Meter
counts the layer forwards and the saved activations."""

import itertools

import numpy as np

POLICIES = ("full", "segments", "recursive")


class Meter:
    """Counts the layer forwards and the saved activations; one unit is one array of shape (B, D)."""

    def __init__(self):
        self.forwards = 0
        self.live = 0
        self.peak = 0

    def save(self, count=1):
        self.live += count
        self.peak = max(self.peak, self.live)

    def free(self, count=1):
        self.live -= count


def config(n_layers=8, width=4, policy="segments", segments=None):
    """n_layers layers of width D; policy is full, segments (k = segments) or recursive."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if n_layers < 1 or width < 1:
        raise ValueError("n_layers and width must be at least 1")
    k = optimal_segments(n_layers) if segments is None else int(segments)
    if not 1 <= k <= n_layers:
        raise ValueError(f"segments must be in [1, {n_layers}], got {k}")
    return {"n_layers": int(n_layers), "width": int(width), "policy": policy, "segments": k}


def init(rng, cfg):
    n, d = cfg["n_layers"], cfg["width"]
    return {"W": rng.normal(0.0, d**-0.5, (n, d, d)), "b": rng.normal(0.0, 0.1, (n, d))}  # (L, D, D), (L, D)


def layer_forward(params, i, h, meter):
    meter.forwards += 1
    return np.tanh(h @ params["W"][i].T + params["b"][i])  # eq: eq-layer  (B, D)


def layer_backward(params, i, h_in, h_out, dy, grads):
    """Backward of layer i from its input and its output; adds dW and db to grads, returns dh_in."""
    dz = dy * (1.0 - h_out**2)  # (B, D)
    grads["W"][i] += dz.T @ h_in
    grads["b"][i] += dz.sum(axis=0)
    return dz @ params["W"][i]


def segment_sizes(n, k):
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r)


def peak_memory(n, k):
    return k + n // k  # eq: eq-segment-memory


def recompute_count(n, k):
    return 2 * n - k  # eq: eq-segment-compute


def optimal_segments(n):
    """The k in [1, n] with the least peak memory; among ties, the largest k, which recomputes the least."""
    return min(range(1, n + 1), key=lambda k: (peak_memory(n, k), -k))


def forward(params, x, cfg):
    """Run the chain on x of shape (B, D). The cache keeps only what the policy saves."""
    meter, n = Meter(), cfg["n_layers"]
    if cfg["policy"] == "full":
        keep = set(range(n + 1))
    elif cfg["policy"] == "segments":
        keep = set(np.cumsum([0] + segment_sizes(n, cfg["segments"])).tolist())
    else:
        keep = {0, n}
    saved, h = {0: x}, x
    meter.save()
    for i in range(n):
        h = layer_forward(params, i, h, meter)
        if i + 1 in keep:
            saved[i + 1] = h
            meter.save()
    return h, {"saved": saved, "cfg": cfg, "meter": meter}


def backward(params, cache, dy):
    """Return (dx, grads, meter). Dropped activations are recomputed from the saved ones."""
    saved, cfg, meter = cache["saved"], cache["cfg"], cache["meter"]
    grads = {"W": np.zeros_like(params["W"]), "b": np.zeros_like(params["b"])}
    if cfg["policy"] == "recursive":
        dx = _bisect(params, dict(saved), 0, cfg["n_layers"], dy, grads, meter)
        meter.free(2)  # the input and the output
        return dx, grads, meter
    bounds = sorted(saved)
    g = dy
    for lo, hi in reversed(list(itertools.pairwise(bounds))):
        hs = {lo: saved[lo], hi: saved[hi]}
        for i in range(lo, hi - 1):  # recompute the inside of the segment
            hs[i + 1] = layer_forward(params, i, hs[i], meter)
            meter.save()
        for i in reversed(range(lo, hi)):
            g = layer_backward(params, i, hs[i], hs[i + 1], g, grads)
        meter.free(hi - lo)  # the inside of the segment and its output checkpoint
    meter.free()
    return g, grads, meter


def _bisect(params, saved, lo, hi, g, grads, meter):
    """Backward over layers lo..hi-1, given saved[lo] and saved[hi]; keeps one checkpoint per level."""
    if hi - lo == 1:
        return layer_backward(params, lo, saved[lo], saved[hi], g, grads)
    mid = (lo + hi) // 2
    h = saved[lo]
    for i in range(lo, mid):
        h = layer_forward(params, i, h, meter)  # eq: eq-recursive-compute
    saved[mid] = h
    meter.save()
    g = _bisect(params, saved, mid, hi, g, grads, meter)
    g = _bisect(params, saved, lo, mid, g, grads, meter)
    del saved[mid]
    meter.free()
    return g


def recursive_forwards(n):
    """The layer forwards of the recursive policy, by its recurrence: n, plus R(n) = m + R(n - m) + R(m), m = n // 2."""
    r = [0, 0]
    for t in range(2, n + 1):
        r.append(t // 2 + r[t - t // 2] + r[t // 2])
    return n + r[n]


def hsm_table(t_max, m_max):
    """C[t][m] and D[t][m] of BPTT-HSM: the least forwards for t steps with m hidden-state slots, and the first push."""
    C = np.zeros((t_max + 1, m_max + 1), dtype=np.int64)
    D = np.zeros((t_max + 1, m_max + 1), dtype=np.int64)
    for t in range(1, t_max + 1):
        C[t][1] = t * (t + 1) // 2
        for m in range(2, m_max + 1):
            if m >= t:
                C[t][m], D[t][m] = 2 * t - 1, 1
                continue
            costs = [y + C[t - y][m - 1] + C[y][m] for y in range(1, t)]  # eq: eq-hsm
            D[t][m] = 1 + int(np.argmin(costs))
            C[t][m] = min(costs)
    return C, D


def bptt_hsm(params, x, dloss, m, D):
    """Forward and backward with at most m saved hidden states, following D. dloss maps h_n to dL/dh_n."""
    grads = {"W": np.zeros_like(params["W"]), "b": np.zeros_like(params["b"])}
    meter, out = Meter(), {}
    meter.save()  # the input, the first slot of the stack

    def run(s, h_s, t, m, g):
        if t == 1:
            h_out = layer_forward(params, s, h_s, meter)
            if g is None:
                out["y"], g = h_out, dloss(h_out)
            return layer_backward(params, s, h_s, h_out, g, grads)
        if m == 1:  # no free slot: rebuild each input from h_s
            for j in reversed(range(t)):
                h = h_s
                for i in range(s, s + j):
                    h = layer_forward(params, i, h, meter)
                g = run(s + j, h, 1, 1, g)
            return g
        y, h = int(D[t][m]), h_s
        for i in range(s, s + y):
            h = layer_forward(params, i, h, meter)
        meter.save()
        g = run(s + y, h, t - y, m - 1, g)
        meter.free()
        return run(s, h_s, y, m, g)

    dx = run(0, x, params["W"].shape[0], m, None)
    return out["y"], dx, grads, meter


def layer_activation_bytes(s, b, h, a):
    """Bytes saved for the backward pass by one transformer layer, term by term (16-bit values, 1-byte masks)."""
    sbh, as2b = s * b * h, a * s * s * b
    attention = {"qkv input": 2 * sbh, "Q and K": 4 * sbh, "softmax output": 2 * as2b, "softmax dropout mask": as2b}
    attention |= {"softmax dropout output": 2 * as2b, "V": 2 * sbh, "projection input": 2 * sbh, "attn mask": sbh}
    mlp = {"MLP input": 2 * sbh, "MLP hidden input": 8 * sbh, "GeLU input": 8 * sbh, "MLP dropout mask": sbh}
    return attention | mlp | {"layer norm inputs": 4 * sbh}  # eq: eq-transformer-memory


SELECTIVE = ("softmax output", "softmax dropout mask", "softmax dropout output")
