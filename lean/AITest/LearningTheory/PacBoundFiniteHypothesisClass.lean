/-
  Tests of https://ai.tamnd.com/pac-bound-finite-hypothesis-class/ (ai://XSZ95D).
  SPDX-License-Identifier: Apache-2.0
-/
import AI.LearningTheory.PacBoundFiniteHypothesisClass

open AI.LearningTheory.PacBoundFiniteHypothesisClass

/-! The thresholds of the page's illustration: points `0, 1, 2, 3`, the hypothesis `h_t` says `true` from `t` on. -/

def thr (t : Fin 5) : Fin 4 → Bool := fun x => decide (t.val ≤ x.val)

-- The target is the threshold at 2. The sample 0, 3 does not tell the thresholds 1, 2 and 3 apart.
#guard Consistent (thr 2) (thr 1) ![(0 : Fin 4), 3]
#guard Consistent (thr 2) (thr 2) ![(0 : Fin 4), 3]
#guard Consistent (thr 2) (thr 3) ![(0 : Fin 4), 3]
#guard !Consistent (thr 2) (thr 0) ![(0 : Fin 4), 3]
#guard !Consistent (thr 2) (thr 4) ![(0 : Fin 4), 3]

-- The points 1 and 2 leave only the target.
#guard (List.finRange 5).filter (fun t => Consistent (thr 2) (thr t) ![(1 : Fin 4), 2]) = [2]

-- The empty sample is consistent with every hypothesis.
#guard (List.finRange 5).all fun t => Consistent (thr 2) (thr t) (![] : Fin 0 → Fin 4)

/-! The sample size of the page's worked example: `|H| = 1000`, `ε = 0.05`, `δ = 0.01` gives
`m ≥ log(10^5) / 0.05 ≈ 230.3`, so 231 draws. -/

#guard (Float.log (1000 / 0.01) / 0.05).ceil == 231
#guard Float.ofNat 1000 * Float.exp (-(0.05 * 231)) ≤ 0.01
#guard !(Float.ofNat 1000 * Float.exp (-(0.05 * 230)) ≤ 0.01)
