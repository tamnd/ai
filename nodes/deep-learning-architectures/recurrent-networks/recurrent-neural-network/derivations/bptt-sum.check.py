# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Numeric check of the derivation bptt-sum: the backward recursion, the sum over copies and the cost."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import rnn  # noqa: E402

rng = np.random.default_rng(7)


def loss(params, x, h0, r, nl, copies=None):
    """sum_t h_t . r_t, with an optional per-step override of U to build the unrolled network."""
    h, out = h0, 0.0
    for t in range(x.shape[1]):
        p = params if copies is None else params | {"U": copies[t]}
        h, _ = rnn.cell(p, x[:, t], h, nl)
        out += np.sum(h * r[:, t])
    return out


for nl in rnn.NONLINEARITIES:
    for _ in range(10):
        n_in, H, T = (int(v) for v in rng.integers(1, 5, size=3))
        params = rnn.init(rng, rnn.config(n_in, H))
        x, h0, r = rng.normal(size=(2, T, n_in)), rng.normal(size=(2, H)), rng.normal(size=(2, T, H))
        _, cache = rnn.forward(params, x, h0, nl)
        if np.min(np.abs(cache["a"])) < 1e-4:
            continue  # away from the kink of relu
        dx, grads, dh0 = rnn.backward(params, cache, r)
        eps = 1e-6
        for name, arr in list(params.items()) + [("x", x), ("h0", h0)]:
            want = {"x": dx, "h0": dh0}.get(name, grads.get(name))
            num = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + eps
                up = loss(params, x, h0, r, nl)
                arr[idx] = old - eps
                down = loss(params, x, h0, r, nl)
                arr[idx] = old
                num[idx] = (up - down) / (2 * eps)
            assert np.allclose(want, num, rtol=1e-6, atol=1e-8), (nl, name)
        # The sum over the copies of U.
        total = np.zeros_like(params["U"])
        for s in range(T):
            for idx in np.ndindex(params["U"].shape):
                cu = [params["U"].copy() for _ in range(T)]
                cd = [params["U"].copy() for _ in range(T)]
                cu[s][idx] += eps
                cd[s][idx] -= eps
                total[idx] += (loss(params, x, h0, r, nl, cu) - loss(params, x, h0, r, nl, cd)) / (2 * eps)
        assert np.allclose(grads["U"], total, rtol=1e-6, atol=1e-8), nl

# The cost of the backward pass: four products per step, twice the two of the forward pass.
for _ in range(20):
    n_in, H, B, T = (int(v) for v in rng.integers(1, 9, size=4))
    per_step = 2 * B * H * H + 2 * B * H * n_in + 2 * H * B * H + 2 * n_in * B * H  # dU^T, dW^T, h^T d, x^T d
    assert T * per_step == 2 * rnn.flops(rnn.config(n_in, H), B, T) == 4 * B * T * H * (n_in + H)
    assert T * (per_step - 2 * B * H * n_in) == 2 * B * T * H * (2 * H + n_in)
print("bptt-sum: ok")
