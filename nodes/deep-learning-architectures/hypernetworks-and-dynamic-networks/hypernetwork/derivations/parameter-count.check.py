# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/hypernetwork/ (ai://NEW-hypernetwork).
"""Numeric check of the derivation parameter-count: the formula, the reference, the MNIST layer and the WRNs."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import hypernetwork as H  # noqa: E402


def n_hyper(Nz, d, N, f, E):
    return Nz * E + d * (Nz + 1) * N + f * N * f * (d + 1)


# 1. the formula against the reference, for random configs
rng = np.random.default_rng(0)
for _ in range(200):
    cfg = H.config(c_in=int(rng.integers(1, 4)), n_basic=int(rng.integers(1, 6)), mult=int(rng.integers(1, 4)),
                   fsize=int(rng.choice([1, 3, 5])), depth=int(rng.integers(1, 4)), n_z=int(rng.integers(1, 6)))  # fmt: skip
    shapes = H.param_shapes(cfg)
    hyper = sum(math.prod(shapes[k]) for k in ("z", "W_in", "B_in", "W_out", "B_out"))
    N, f, Nz, E = cfg["n_basic"], cfg["fsize"], cfg["n_z"], H.n_embeddings(cfg)
    assert hyper == n_hyper(Nz, Nz, N, f, E) == H.hyper_param_count(Nz, Nz, N, N, f, E)
    assert sum(math.prod(s) for s in shapes.values()) == H.param_count(cfg)

# 2. the MNIST layer of Section 4.1 and Table 7
assert n_hyper(4, 4, 16, 7, 1) == 4244 and n_hyper(4, 4, 16, 7, 1) - 4 == 4240
assert 7 * 7 * 16 * 16 == 12544 and round(12544 / 4244, 2) == 2.96
assert H.one_layer_param_count(4, 16, 16, 7) == 50176 == 4 * 12544

# 3. the FLOPs of generate, with every einsum counted
cfg = H.config(n_basic=3, mult=2, fsize=3, depth=2, n_z=4)
p = H.init(np.random.default_rng(1), cfg)
E, N, f, Nz, d = H.n_embeddings(cfg), 3, 3, 4, 4
count = 2 * E * N * d * Nz + 2 * E * N * f * N * f * d  # "idz,ez->eid" and "uwd,eid->eiuw"
assert count == 2 * E * N * d * (Nz + f * f * N)
a, K = H.generate(p, p["z"])
assert a.shape == (E, N, d) and K.shape == (E, N, f, N * f)
assert np.allclose(K, np.einsum("uwd,eid->eiuw", p["W_out"], np.einsum("idz,ez->eid", p["W_in"], p["z"]) + p["B_in"])
                   + p["B_out"])  # fmt: skip


# 4. and 5. the wide residual networks: 36 convolutions of 3 x 3, tiled from basic kernels of 16 channels
def wrn_embeddings(k, n=6):
    E, c = 0, 16
    for width in (16 * k, 32 * k, 64 * k):
        for _ in range(2 * n):
            E += (c // 16) * (width // 16)
            c = width
    return E


assert wrn_embeddings(1) == 242 and wrn_embeddings(2) == 966
assert 64 * 65 * 16 == 66560 and 3 * 16 * 3 * 65 == 9360
assert n_hyper(64, 64, 16, 3, 242) == 91408 and n_hyper(64, 64, 16, 3, 966) == 137744
# the counts fit inside the totals of Table 2, 0.097M and 0.148M, with the layers that are not generated
assert 91408 < 97000 and 137744 < 148000

print("parameter-count: ok")
