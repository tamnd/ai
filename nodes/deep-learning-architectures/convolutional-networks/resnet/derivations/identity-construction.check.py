# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/resnet/ (ai://NEW-resnet).
"""Numeric check of the derivation identity-construction: a block with zero gamma and beta is the identity on x >= 0."""

import copy
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import resnet  # noqa: E402

rng = np.random.default_rng(20151210)

# One block that is not the first of its stage, with random weights and gamma = beta = 0 in its last norm.
deep = resnet.config(width=2, blocks=(1, 2, 1, 1), classes=5)
params = resnet.init(rng, deep)
state = resnet.init_state(params)
blk, blk_state = params["stages"][1][1], state["stages"][1][1]
blk["bn3"]["gamma"][:] = 0
blk["bn3"]["beta"][:] = 0
x = np.abs(rng.normal(size=(3, 16, 4, 4)))  # (B, C, H, W), C = 4 * m_1 = 16
for st in (None, blk_state):  # training mode, then inference mode
    y, _ = resnet.bottleneck(blk, x, 1, st=st)
    assert np.array_equal(y, x)
y, _ = resnet.bottleneck(blk, x - 1.0, 1)  # negative entries are clipped: the identity needs x >= 0
assert not np.allclose(y, x - 1.0)

# Insert that block into a shallower network: the logits do not change, in both modes.
shallow = resnet.config(width=2, blocks=(1, 1, 1, 1), classes=5)
small = copy.deepcopy(params)
small["stages"][1] = small["stages"][1][:1]
small_state = copy.deepcopy(state)
small_state["stages"][1] = small_state["stages"][1][:1]
img = rng.normal(size=(2, 3, 32, 32))  # (B, 3, R, R)
for s_small, s_deep in ((None, None), (small_state, state)):
    a, _ = resnet.forward(small, img, shallow, s_small)
    b, _ = resnet.forward(params, img, deep, s_deep)
    assert np.allclose(a, b), np.abs(a - b).max()
print(f"identity-construction: the zero-gamma block is the identity on x >= 0; inserting it keeps the logits ({a.shape})")
