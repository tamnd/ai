# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Check of corruption-fractions: the shares of [MASK], wrong and unchanged tokens under BERT's 80/10/10 rule, exactly
and from the NumPy reference, and the number of loss terms per sequence for the masked LM and for ELECTRA."""

import importlib.util
import pathlib
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def reference():
    path = HERE.parent / "impl" / "numpy" / "masked_language_modelling.py"
    spec = importlib.util.spec_from_file_location("mlm_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fractions(r: Fraction, Vp: int) -> tuple[Fraction, Fraction, Fraction]:
    masked = Fraction(8, 10) * r
    wrong = Fraction(1, 10) * r * (1 - Fraction(1, Vp))
    same = Fraction(1, 10) * r * (1 + Fraction(1, Vp))
    return masked, wrong, same


def main():
    r = Fraction(15, 100)
    for Vp in (2, 10, 1000, 30_000):
        masked, wrong, same = fractions(r, Vp)
        assert masked + wrong + same == r
        assert masked + wrong == Fraction(135, 1000) - Fraction(15, 1000) / Vp
        assert same / r == Fraction(1, 10) * (1 + Fraction(1, Vp))
    assert fractions(r, 30_000)[0] == Fraction(12, 100)

    ml = reference()
    rng = np.random.default_rng(0)
    n, Vp = 400_000, 20
    ids = rng.integers(ml.FIRST, ml.FIRST + Vp, size=n)
    sel = rng.random(n) < 0.15
    out = ml.bert_corrupt(ids, sel, rng.random(n), rng.integers(ml.FIRST, ml.FIRST + Vp, size=n))
    got = (np.mean(out == ml.MASK), np.mean((out != ids) & (out != ml.MASK)), np.mean(sel & (out == ids)))
    want = [float(f) for f in fractions(r, Vp)]
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) < 2e-3, (g, w)

    per_seq = float(r * 512)
    assert per_seq == 76.8 and round(512 / per_seq, 2) == 6.67
    print(
        f"corruption-fractions: [MASK] {float(fractions(r, 30_000)[0]):.3f}, changed {float(r * Fraction(9, 10)):.3f}"
        f" - 0.015/V', loss terms per 512 tokens {per_seq} (MLM) vs 512 (ELECTRA)"
    )


if __name__ == "__main__":
    main()
