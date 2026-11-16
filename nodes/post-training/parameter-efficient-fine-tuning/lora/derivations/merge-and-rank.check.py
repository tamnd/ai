# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/lora/ (ai://NEW-lora).
# Check of derivations/merge-and-rank.md: the merged layer equals the adapted layer, the merge can be undone, and
# the update has rank at most r.
import numpy as np


def main():
    rng = np.random.default_rng(20261114)
    for d, k, r in [(4, 5, 1), (7, 3, 2), (16, 16, 4), (30, 20, 8)]:
        w0 = rng.normal(size=(d, k))
        b = rng.normal(size=(d, r))
        a = rng.normal(size=(r, k))
        s = float(rng.uniform(0.5, 4.0))
        x = rng.normal(size=(k, 6))
        lhs = w0 @ x + s * (b @ (a @ x))
        w = w0 + s * (b @ a)
        rhs = w @ x
        assert np.linalg.norm(lhs - rhs) <= 1e-12 * np.linalg.norm(lhs), (d, k, r)
        assert np.linalg.norm((w - s * (b @ a)) - w0) <= 1e-12 * np.linalg.norm(w0), (d, k, r)
        rank = np.linalg.matrix_rank(s * (b @ a))
        assert rank <= r, (d, k, r, rank)
    print("ok: merge, unmerge and rank bound hold")


if __name__ == "__main__":
    main()
