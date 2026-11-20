# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/bidirectional-recurrent-neural-network/ (ai://NEW-bidirectional-recurrent-neural-network).
"""Numeric check of the derivation parameter-count: random configs, the two networks of Graves et al., torch.nn.LSTM."""

import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import birnn  # noqa: E402

COUNT = [0]


def counted_matmul(a, b):
    COUNT[0] += 2 * a.size * b.shape[-1]  # (..., k) @ (k, n): 2 m k n
    return np.matmul(a, b)


class Counted(np.ndarray):
    def __matmul__(self, other):
        return counted_matmul(np.asarray(self), np.asarray(other)).view(Counted)

    def __rmatmul__(self, other):
        return counted_matmul(np.asarray(other), np.asarray(self)).view(Counted)


rng = np.random.default_rng(0)
for _ in range(40):
    I, H, L, K, B, T = (int(v) for v in rng.integers([1, 1, 1, 2, 1, 1], [6, 6, 4, 7, 4, 6]))
    cell, bi = ("lstm", "tanh")[rng.integers(2)], bool(rng.integers(2))
    cfg = birnn.config(I, H, L, K, cell, bi)
    G, D = (4 if cell == "lstm" else 1), (2 if bi else 1)
    N = D * (G * H * (I + H) + G * H) + (L - 1) * D * (G * H * (D * H + H) + G * H) + D * H * K + K
    params = birnn.init(rng, cfg)
    assert sum(a.size for a in params.values()) == birnn.param_count(cfg) == N
    COUNT[0] = 0
    birnn.forward({k: v.view(Counted) for k, v in params.items()}, cfg, rng.normal(size=(B, T, I)))
    C = 2 * D * B * T * G * H * (I + H) + 2 * (L - 1) * D * B * T * G * H * (D * H + H) + 2 * B * T * D * H * K
    assert COUNT[0] == C == birnn.flops(cfg, B, T), (COUNT[0], C)

bi250 = birnn.config(123, 250, 3, 62, "lstm", True)
uni421 = birnn.config(123, 421, 3, 62, "lstm", False)
assert birnn.param_count(bi250) == 748_000 + 2 * 1_502_000 + 31_062 == 3_783_062
assert birnn.param_count(uni421) == 917_780 + 2 * 1_419_612 + 26_164 == 3_783_168
assert birnn.param_count(uni421) - birnn.param_count(bi250) == 106
assert round((3_783_062 + 3 * 2 * 3 * 250) / 1e6, 2) == round((3_783_168 + 3 * 3 * 421) / 1e6, 2) == 3.79
example = birnn.config()
assert birnn.param_count(example) == 717 and birnn.flops(example, 3, 6) == 23_328

if importlib.util.find_spec("torch") is not None:
    import torch

    for I, H, L, bi in [(3, 4, 2, True), (123, 250, 3, True), (123, 421, 3, False)]:
        net = torch.nn.LSTM(I, H, num_layers=L, bidirectional=bi)
        D = 2 if bi else 1
        ours = birnn.param_count(birnn.config(I, H, L, 2, "lstm", bi)) - (D * H * 2 + 2)  # without the output layer
        assert sum(p.numel() for p in net.parameters()) == ours + 4 * L * D * H
print("ok")
