# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/energy-based-model/ (ai://NEW-energy-based-model).
"""Numeric check of the derivation jem-logit-shift."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "impl" / "numpy"))
import energy_based as eb  # noqa: E402

rng = np.random.default_rng(7)
f = 3 * rng.standard_normal((50, 7))
s = rng.uniform(-5, 5, (50, 1))
np.testing.assert_allclose(eb.jem_energy(f + s), eb.jem_energy(f) - s[:, 0], rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(eb.jem_class_probs(f + s), eb.jem_class_probs(f), rtol=1e-12, atol=1e-15)
np.testing.assert_allclose(eb.jem_class_probs(f).sum(1), 1.0, rtol=1e-12)

# the joint model p(x, y) = exp(f(x)[y]) / Z on 12 inputs and 7 classes: its marginal and conditional
f12 = f[:12]
joint = np.exp(f12) / np.exp(f12).sum()
marg = joint.sum(1)
np.testing.assert_allclose(marg, np.exp(-eb.jem_energy(f12)) / np.exp(-eb.jem_energy(f12)).sum(), rtol=1e-12)
np.testing.assert_allclose(joint / marg[:, None], eb.jem_class_probs(f12), rtol=1e-12)

# large logits: the energy and the probabilities stay finite
big = f + 800.0
assert np.isfinite(eb.jem_energy(big)).all() and np.isfinite(eb.jem_class_probs(big)).all()
np.testing.assert_allclose(eb.jem_energy(big), eb.jem_energy(f) - 800.0, rtol=1e-12)
