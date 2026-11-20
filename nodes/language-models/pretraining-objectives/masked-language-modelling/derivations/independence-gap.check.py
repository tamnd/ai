# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Check of independence-gap: the best masked LM loss for two masked tokens is the sum of their entropies, which
exceeds the joint entropy by their mutual information; XLNet's New York example."""

import importlib.util
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def reference():
    path = HERE.parent / "impl" / "numpy" / "masked_language_modelling.py"
    spec = importlib.util.spec_from_file_location("mlm_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def best_pair_loss(joint: np.ndarray, steps: int = 3000, lr: float = 1.0) -> float:
    """Minimise E[-log qa(a) - log qb(b)] over softmax-parametrised qa and qb."""
    la, lb = np.zeros(joint.shape[0]), np.zeros(joint.shape[1])
    pa, pb = joint.sum(1), joint.sum(0)
    for _ in range(steps):
        qa, qb = np.exp(la - la.max()), np.exp(lb - lb.max())
        qa, qb = qa / qa.sum(), qb / qb.sum()
        la -= lr * (qa - pa)
        lb -= lr * (qb - pb)
    qa, qb = np.exp(la - la.max()), np.exp(lb - lb.max())
    qa, qb = qa / qa.sum(), qb / qb.sum()
    return float(-(joint * (np.log(qa)[:, None] + np.log(qb)[None, :])).sum())


def main():
    ml = reference()
    rng = np.random.default_rng(2)
    for _ in range(20):
        joint = rng.dirichlet(np.full(12, 0.5)).reshape(3, 4)
        mlm, h, gap = ml.independence_gap(joint)
        assert abs(best_pair_loss(joint) - mlm) < 1e-8
        pa, pb = joint.sum(1), joint.sum(0)
        mi = float((joint * np.log(joint / np.outer(pa, pb))).sum())
        assert abs(gap - mi) < 1e-12 and gap > 0
    mlm, h, gap = ml.independence_gap(np.array([[0.5, 0.0], [0.0, 0.5]]))
    assert abs(mlm - 2 * math.log(2)) < 1e-15 and abs(h - math.log(2)) < 1e-15 and abs(gap - math.log(2)) < 1e-15
    product = np.outer([0.5, 0.5], [0.5, 0.5])
    assert product[0, 1] == product[1, 0] == 0.25
    indep = np.outer([0.2, 0.8], [0.5, 0.25, 0.25])
    assert abs(ml.independence_gap(indep)[2]) < 1e-15
    print(f"independence-gap: New York example, MLM {mlm:.4f} nats vs joint {h:.4f}, gap {gap:.4f} = log 2")


if __name__ == "__main__":
    main()
