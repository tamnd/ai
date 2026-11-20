# SPDX-License-Identifier: MIT
# Part of https://ai.tamnd.com/masked-language-modelling/ (ai://NEW-masked-language-modelling).
"""Check of span-corruption-lengths: the input and target lengths of T5's span corruption, Raffel et al.'s example of
500 tokens and 25 spans, and the expected number of runs under i.i.d. noise."""

import importlib.util
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


def main():
    ml = reference()
    rng = np.random.default_rng(1)
    for n in (8, 50, 500):
        ids = np.arange(2, n + 2)
        for _ in range(200):
            noise = rng.random(n) < rng.uniform(0.05, 0.6)
            k, s = int(noise.sum()), ml.run_count(noise) if noise.any() else 0
            inp, tgt = ml.span_corrupt(ids, noise, np.arange(10**6, 10**6 + s + 1))
            assert len(inp) == n - k + s and len(tgt) == k + s + 1
    n, r = 500, 0.15
    noise = ml.random_spans_noise_mask(n, r, 3.0, rng)
    inp, tgt = ml.span_corrupt(np.arange(2, n + 2), noise, np.arange(10**6, 10**6 + 26))
    assert noise.sum() == 75 and ml.run_count(noise) == 25 and len(inp) == 450 and len(tgt) == 101
    runs = r + (n - 1) * r * (1 - r)
    assert abs(runs - 63.7725) < 1e-12
    sim = np.mean([ml.run_count(rng.random(n) < r) for _ in range(20_000)])
    assert abs(sim - runs) < 0.15, sim
    exp_inp, exp_tgt = n - r * n + runs, r * n + runs + 1
    assert abs(exp_inp - 488.7725) < 1e-9 and abs(exp_tgt - 139.7725) < 1e-9
    assert round(1 - 101 / exp_tgt, 3) == 0.277 and round(1 - 450 / exp_inp, 3) == 0.079
    assert round(r * n / runs, 2) == 1.18
    print(
        f"span-corruption-lengths: spans of 3 give 450/101; i.i.d. gives {exp_inp:.2f}/{exp_tgt:.2f} ({sim:.2f} runs)"
    )


if __name__ == "__main__":
    main()
