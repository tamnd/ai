# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/generative-adversarial-network/ (ai://NEW-generative-adversarial-network).
"""Numeric check of the derivation saturation."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import gan  # noqa: E402

a = np.linspace(-30.0, 30.0, 601)
e = 1e-6
for f, df in ((lambda t: -gan.softplus(t), lambda t: -gan.sigmoid(t)),
              (lambda t: gan.softplus(-t), lambda t: gan.sigmoid(t) - 1.0)):  # fmt: skip
    num = (f(a + e) - f(a - e)) / (2 * e)
    assert np.max(np.abs(num - df(a))) < 1e-8
assert gan.sigmoid(-30.0) < 1e-12  # flat against slope -1

# statement 3: the generator's gradients of the two losses are the same backward pass with the weights D and 1 - D
rng = np.random.default_rng(0)
cfg = gan.config()
p = gan.init(rng, cfg)
z = rng.standard_normal((7, 2))
fake = gan.generate(p, z)[0]
d = gan.sigmoid(gan.critic(p, fake)[0])
_, cache = gan.gen_loss(p, z, "minimax")
gm = gan.gen_backward(p, cache)
_, cache = gan.gen_loss(p, z, "nonsat")
gn = gan.gen_backward(p, cache)
zz, t, hf, _ = cache
for w, g in ((d, gm), (1.0 - d, gn)):
    ref = gan.gen_backward(p, (zz, t, hf, -w / len(z)))
    for k in gan.GEN:
        assert np.allclose(ref[k], g[k], rtol=1e-12, atol=1e-15)
dd = 1e-3
assert abs((1 - dd) / dd - 999.0) < 1e-9
print("saturation: ok")
