# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/variational-autoencoder/ (ai://NEW-variational-autoencoder).
"""Numeric check of the derivation parameter-and-flop-count."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import vae  # noqa: E402

COUNT = [0]


class Counted(np.ndarray):
    def __matmul__(self, other):
        a, b = np.asarray(self), np.asarray(other)
        COUNT[0] += 2 * a.size * b.shape[-1]
        return np.matmul(a, b).view(Counted)

    def __rmatmul__(self, other):
        a, b = np.asarray(other), np.asarray(self)
        COUNT[0] += 2 * a.size * b.shape[-1]
        return np.matmul(a, b).view(Counted)


def n_params(D, H, J, gaussian=False):
    return (D + 1) * H + 2 * (H + 1) * J + (J + 1) * H + (H + 1) * D + gaussian * (H + 1) * D


def n_flops(B, S, D, H, J, gaussian=False):
    return 2 * B * D * H + 4 * B * H * J + S * (2 * B * J * H + 2 * B * H * D) + gaussian * 2 * S * B * H * D


rng = np.random.default_rng(0)
for _ in range(40):
    D, H, J, B, S = (int(v) for v in rng.integers([1, 1, 1, 1, 1], [9, 9, 5, 6, 5]))
    for dec in vae.DECODERS:
        g = dec == "gaussian"
        cfg = vae.config(D, H, J, dec)
        p = vae.init(rng, cfg)
        assert sum(v.size for v in p.values()) == n_params(D, H, J, g) == vae.param_count(cfg)
        COUNT[0] = 0
        vae.forward({k: v.view(Counted) for k, v in p.items()}, cfg, rng.uniform(size=(B, D)).view(Counted),
                    rng.standard_normal((S, B, J)))  # fmt: skip
        assert COUNT[0] == n_flops(B, S, D, H, J, g) == vae.flops_forward(cfg, B, S), (COUNT[0], D, H, J, B, S, dec)

assert 785 * 500 + 2 * 501 * 20 + 21 * 500 + 501 * 784 == n_params(784, 500, 20) == 815_824
assert n_params(560, 200, 20) == 237_000 and n_params(560, 200, 20, True) == 349_560
assert 2 * 784 * 500 + 4 * 500 * 20 == 824_000 and 2 * 20 * 500 + 2 * 500 * 784 == 804_000
assert n_flops(1, 1, 784, 500, 20) == 1_628_000 and n_flops(1, 2, 784, 500, 20) - n_flops(1, 1, 784, 500, 20) == 804_000
print("parameter-and-flop-count: ok")
