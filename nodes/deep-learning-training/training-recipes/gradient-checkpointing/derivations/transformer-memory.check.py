# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/gradient-checkpointing/ (ai://C4TQW1).
# Check of derivations/transformer-memory.md: the term-by-term bytes of one transformer layer add up to
# sbh(34 + 5as/h), and the attention-score share is 70% for GPT-3 and 65% for MT-NLG.
from fractions import Fraction


def terms(s, b, h, a):
    sbh, as2b = s * b * h, a * s * s * b
    attention = [2 * sbh, 4 * sbh, 2 * as2b, as2b, 2 * as2b + 2 * sbh, 2 * sbh, sbh]
    mlp = [2 * sbh, 8 * sbh, 8 * sbh, sbh]
    norms = [2 * sbh, 2 * sbh]
    return attention, mlp, norms


for name, s, b, h, a in [("22B", 2048, 4, 6144, 64), ("GPT-3", 2048, 1, 12288, 96), ("MT-NLG", 2048, 1, 20480, 128),
                         ("1T", 2048, 1, 25600, 160), ("tiny", 4, 2, 6, 3)]:
    attention, mlp, norms = terms(s, b, h, a)
    assert sum(attention) == 11 * s * b * h + 5 * a * s * s * b
    assert sum(mlp) == 19 * s * b * h
    assert sum(norms) == 4 * s * b * h
    total = sum(attention + mlp + norms)
    assert Fraction(total) == s * b * h * (34 + Fraction(5 * a * s, h))
    share = Fraction(5 * a * s, h) / (34 + Fraction(5 * a * s, h))
    print(f"{name}: 5as/h = {float(Fraction(5 * a * s, h)):.2f}, bytes per layer = {total:,}, "
          f"attention-score share = {float(share):.4f}, selective keeps {34 * s * b * h:,}, full keeps {2 * s * b * h:,}")

assert round(80 / 114, 2) == 0.70 and round(64 / 98, 2) == 0.65
print("ok")
