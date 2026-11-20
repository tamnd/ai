# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/recurrent-neural-network/ (ai://NEW-recurrent-neural-network).
"""Numeric check of the derivation gradient-bound: the product of Jacobians and the bound of Pascanu et al."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import rnn  # noqa: E402


def numeric_dhT_dh0(params, x, h0, nl, eps=1e-6):
    H = h0.shape[-1]
    J = np.zeros((H, H))
    for j in range(H):
        up, down = h0.copy(), h0.copy()
        up[0, j] += eps
        down[0, j] -= eps
        J[:, j] = (rnn.forward(params, x, up, nl)[1]["h_T"] - rnn.forward(params, x, down, nl)[1]["h_T"])[0] / (2 * eps)
    return J


rng = np.random.default_rng(5)
for nl in rnn.NONLINEARITIES:
    for _ in range(40):
        n_in, H, T = (int(v) for v in rng.integers(1, 6, size=3))
        params = rnn.init(rng, rnn.config(n_in, H))
        params["U"] = rng.uniform(0.2, 3.0) * params["U"]
        x, h0 = rng.normal(size=(1, T, n_in)), rng.normal(size=(1, H))
        _, cache = rnn.forward(params, x, h0, nl)
        P = np.eye(H)
        for t in range(T):
            P = rnn.step_jacobian(params, cache["a"][0, t], cache["h"][0, t], nl) @ P
        if np.min(np.abs(cache["a"])) > 1e-4:  # finite differences away from the kink of relu
            assert np.allclose(P, numeric_dhT_dh0(params, x, h0, nl), atol=1e-7), nl
        assert np.linalg.norm(P, 2) <= np.linalg.norm(params["U"], 2) ** T * (1 + 1e-12)

# Statement 4: a spectral norm below 1 makes the gradient vanish for any input.
Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
params = {"W": rng.normal(size=(2, 6)), "U": 0.8 * Q, "b_i": np.zeros(6), "b_h": np.zeros(6)}
_, cache = rnn.forward(params, 3.0 * rng.normal(size=(1, 60, 2)), rng.normal(size=(1, 6)), "tanh")
P = np.eye(6)
for t in range(60):
    P = rnn.step_jacobian(params, cache["a"][0, t], cache["h"][0, t], "tanh") @ P
assert np.linalg.norm(P, 2) <= 0.8**60

# A non-normal U: largest absolute eigenvalue 0.5, largest singular value above 4; the powers grow before they decay.
U = np.array([[0.5, 4.0], [0.0, 0.5]])
assert np.max(np.abs(np.linalg.eigvals(U))) == 0.5 and np.linalg.norm(U, 2) > 4.0
norms = [np.linalg.norm(np.linalg.matrix_power(U.T, k), 2) for k in range(1, 40)]
assert max(norms) > 2.0 and norms[-1] < 1e-6
# For a symmetric U the two numbers agree.
S = rng.normal(size=(5, 5))
S = S + S.T
assert np.isclose(np.max(np.abs(np.linalg.eigvalsh(S))), np.linalg.norm(S, 2))

# Statement 5: the largest slopes of tanh, relu and the logistic sigmoid.
a = np.linspace(-10, 10, 200_001)
assert np.isclose(np.max(1 - np.tanh(a) ** 2), 1.0)
s = 1 / (1 + np.exp(-a))
assert np.isclose(np.max(s * (1 - s)), 0.25)
print("gradient-bound: ok")
