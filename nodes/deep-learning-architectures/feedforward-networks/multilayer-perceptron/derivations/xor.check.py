# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/multilayer-perceptron/ (ai://YA59M7).
"""Numeric check of the derivation xor."""

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import multilayer_perceptron as mlp  # noqa: E402

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
T = np.array([0.0, 1.0, 1.0, 0.0])

# Statement 1: the two pairs of points have the same sum of f, so no grid point separates them.
grid = np.linspace(-3.0, 3.0, 61)
for w1, w2, c in itertools.product(grid, grid, grid):
    f = X @ np.array([w1, w2]) + c
    assert not (f[1] > 0 and f[2] > 0 and f[0] <= 0 and f[3] <= 0)
    assert abs((f[1] + f[2]) - (f[0] + f[3])) < 1e-9

# Statement 2: the normal equations and their unique solution.
A = np.hstack([X, np.ones((4, 1))])
np.testing.assert_array_equal(A.T @ A, [[2, 1, 2], [1, 2, 2], [2, 2, 4]])
np.testing.assert_array_equal(A.T @ T, [1, 1, 2])
assert np.linalg.matrix_rank(A) == 3
theta = np.linalg.solve(A.T @ A, A.T @ T)
np.testing.assert_allclose(theta, [0.0, 0.0, 0.5], atol=1e-14)

# Statement 3: the 2-2-1 ReLU network of the worked example.
y, cache = mlp.forward(mlp.xor_params(), X, mlp.config(2, 2, 1, 1, "relu"))
np.testing.assert_array_equal(cache["z"][0], [[0, -1], [1, 0], [1, 0], [2, 1]])
np.testing.assert_array_equal(cache["h"][1], [[0, 0], [1, 0], [1, 0], [2, 1]])
np.testing.assert_array_equal(y.ravel(), T)
print("xor: ok")
