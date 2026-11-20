# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Check of span-length: the mean of SpanBERT's geometric span lengths truncated at 10, against capping and against
no truncation."""

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


def truncated_mean(p: Fraction, L: int) -> Fraction:
    q = 1 - p
    return (1 - (L + 1) * q**L + L * q ** (L + 1)) / (p * (1 - q**L))


def main():
    for p in (Fraction(1, 10), Fraction(1, 5), Fraction(2, 5)):
        for L in (1, 5, 10, 30):
            q = 1 - p
            direct = sum(l * p * q ** (l - 1) for l in range(1, L + 1)) / sum(p * q ** (l - 1) for l in range(1, L + 1))
            assert truncated_mean(p, L) == direct
            # min(l, L) is l below L and L with the remaining probability q^(L - 1)
            assert sum(l * p * q ** (l - 1) for l in range(1, L)) + L * q ** (L - 1) == (1 - q**L) / p
    p, L = Fraction(1, 5), 10
    mean = truncated_mean(p, L)
    capped = (1 - (1 - p) ** L) / p
    assert round(float(mean), 1) == 3.8 and round(float(mean), 3) == 3.797
    assert round(float(capped), 3) == 4.463 and 1 / p == 5
    assert round(float((1 - p) ** L), 3) == 0.107
    pmf = reference().truncated_geometric(0.2, 10)
    assert abs(float(np.arange(1, 11) @ pmf) - float(mean)) < 1e-12
    print(f"span-length: truncated mean {float(mean):.4f}, capped {float(capped):.4f}, untruncated {float(1 / p)}")


if __name__ == "__main__":
    main()
