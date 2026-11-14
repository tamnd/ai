/-
  Part of https://ai.tamnd.com/pac-bound-finite-hypothesis-class/ (ai://XSZ95D).
  SPDX-License-Identifier: Apache-2.0
-/
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.UniformOn

/-!
# The PAC bound for a finite hypothesis class

The realizable case of the sample complexity bound for a finite class (Shalev-Shwartz and Ben-David 2014,
Corollary 2.3). The steps follow `derivations/proof.md` of the node; each `-- step:` comment names a step there.

The setting: a probability measure `μ` on a measurable space `X`, a measurable target `f : X → Bool`, a finite
class `H` of measurable hypotheses, and a sample of `m` points drawn independently from `μ`, that is, a point of
`Fin m → X` under the product measure.
-/

namespace AI.LearningTheory.PacBoundFiniteHypothesisClass

open MeasureTheory Real
open scoped ENNReal

variable {X : Type*} [MeasurableSpace X]

/-- The true error of `h` against the target `f`: the mass of the points where they disagree. -/
noncomputable def err (μ : Measure X) (f h : X → Bool) : ℝ≥0∞ := μ {x | h x ≠ f x}

/-- `h` agrees with the target `f` on every point of the sample `S`. -/
def Consistent (f h : X → Bool) {m : ℕ} (S : Fin m → X) : Prop := ∀ i, h (S i) = f (S i)

instance (f h : X → Bool) {m : ℕ} (S : Fin m → X) : Decidable (Consistent f h S) :=
  inferInstanceAs (Decidable (∀ i, h (S i) = f (S i)))

/-- The product measure of `m` independent draws from `μ`. -/
noncomputable abbrev sample (μ : Measure X) (m : ℕ) : Measure (Fin m → X) := Measure.pi fun _ => μ

-- step: one-minus-le-exp
/-- `1 - ε ≤ e^{-ε}`, and so `(1 - ε)^m ≤ e^{-εm}` for `ε ≤ 1`. -/
theorem one_sub_pow_le_exp {ε : ℝ} (hε : ε ≤ 1) (m : ℕ) : (1 - ε) ^ m ≤ exp (-(ε * m)) := by
  have h1 : 1 - ε ≤ exp (-ε) := by linarith [add_one_le_exp (-ε)]
  calc (1 - ε) ^ m ≤ exp (-ε) ^ m := pow_le_pow_left₀ (by linarith) h1 m
    _ = exp (-(ε * m)) := by rw [← exp_nat_mul]; ring_nf

lemma measurableSet_agree {f h : X → Bool} (hf : Measurable f) (hh : Measurable h) :
    MeasurableSet {x | h x = f x} :=
  measurableSet_eq_fun hh hf

lemma measurableSet_disagree {f h : X → Bool} (hf : Measurable f) (hh : Measurable h) :
    MeasurableSet {x | h x ≠ f x} :=
  (measurableSet_agree hf hh).compl

-- step: one-draw
/-- A bad hypothesis, one with error above `ε`, agrees with the target on one draw with probability at most
`e^{-ε}`. -/
theorem measure_agree_le {μ : Measure X} [IsProbabilityMeasure μ] {f h : X → Bool} (hf : Measurable f)
    (hh : Measurable h) {ε : ℝ} (hε : 0 ≤ ε) (hbad : ENNReal.ofReal ε < err μ f h) :
    μ {x | h x = f x} ≤ ENNReal.ofReal (exp (-ε)) := by
  have hc : {x | h x = f x} = {x | h x ≠ f x}ᶜ := by ext x; simp
  rw [hc, prob_compl_eq_one_sub (measurableSet_disagree hf hh)]
  calc 1 - μ {x | h x ≠ f x} ≤ 1 - ENNReal.ofReal ε := tsub_le_tsub_left hbad.le 1
    _ = ENNReal.ofReal (1 - ε) := by rw [ENNReal.ofReal_sub 1 hε, ENNReal.ofReal_one]
    _ ≤ ENNReal.ofReal (exp (-ε)) := ENNReal.ofReal_le_ofReal (by linarith [add_one_le_exp (-ε)])

-- step: independence
/-- The draws are independent, so the probability that `h` is consistent with the whole sample is the `m`-th
power of the probability that it agrees on one draw. -/
theorem sample_consistent_eq (μ : Measure X) [IsProbabilityMeasure μ] (f h : X → Bool) (m : ℕ) :
    sample μ m {S | Consistent f h S} = μ {x | h x = f x} ^ m := by
  have hs : {S : Fin m → X | Consistent f h S} = Set.univ.pi fun _ => {x | h x = f x} := by
    ext S; simp [Consistent, Set.mem_pi]
  rw [hs, Measure.pi_pi, Finset.prod_const, Finset.card_univ, Fintype.card_fin]

-- step: one-bad-hypothesis
/-- One bad hypothesis survives `m` draws with probability at most `e^{-εm}`. -/
theorem sample_consistent_le {μ : Measure X} [IsProbabilityMeasure μ] {f h : X → Bool} (hf : Measurable f)
    (hh : Measurable h) {ε : ℝ} (hε : 0 ≤ ε) (hbad : ENNReal.ofReal ε < err μ f h) (m : ℕ) :
    sample μ m {S | Consistent f h S} ≤ ENNReal.ofReal (exp (-(ε * m))) := by
  rw [sample_consistent_eq]
  calc μ {x | h x = f x} ^ m ≤ ENNReal.ofReal (exp (-ε)) ^ m := pow_le_pow_left' (measure_agree_le hf hh hε hbad) m
    _ = ENNReal.ofReal (exp (-ε) ^ m) := (ENNReal.ofReal_pow (exp_pos _).le m).symm
    _ = ENNReal.ofReal (exp (-(ε * m))) := by rw [← exp_nat_mul]; ring_nf

/-- The misleading samples: those on which some bad hypothesis of `H` is consistent. -/
def Misleading (μ : Measure X) (f : X → Bool) (H : Finset (X → Bool)) (ε : ℝ) (m : ℕ) : Set (Fin m → X) :=
  {S | ∃ h ∈ H, ENNReal.ofReal ε < err μ f h ∧ Consistent f h S}

-- step: union-bound
/-- The union bound over the finite class: the misleading samples have probability at most `|H| e^{-εm}`. -/
theorem sample_misleading_le {μ : Measure X} [IsProbabilityMeasure μ] {f : X → Bool} (hf : Measurable f)
    {H : Finset (X → Bool)} (hH : ∀ h ∈ H, Measurable h) {ε : ℝ} (hε : 0 ≤ ε) (m : ℕ) :
    sample μ m (Misleading μ f H ε m) ≤ H.card * ENNReal.ofReal (exp (-(ε * m))) := by
  have hsub : Misleading μ f H ε m ⊆
      ⋃ h ∈ H, {S | ENNReal.ofReal ε < err μ f h ∧ Consistent f h S} := by
    intro S ⟨h, hmem, hS⟩
    exact Set.mem_biUnion hmem hS
  calc sample μ m (Misleading μ f H ε m)
      ≤ sample μ m (⋃ h ∈ H, {S | ENNReal.ofReal ε < err μ f h ∧ Consistent f h S}) := measure_mono hsub
    _ ≤ ∑ h ∈ H, sample μ m {S | ENNReal.ofReal ε < err μ f h ∧ Consistent f h S} :=
        measure_biUnion_finset_le H _
    _ ≤ ∑ _h ∈ H, ENNReal.ofReal (exp (-(ε * m))) := by
        refine Finset.sum_le_sum fun h hmem => ?_
        by_cases hbad : ENNReal.ofReal ε < err μ f h
        · exact (measure_mono fun S hS => hS.2).trans (sample_consistent_le hf (hH h hmem) hε hbad m)
        · have he : {S : Fin m → X | ENNReal.ofReal ε < err μ f h ∧ Consistent f h S} = ∅ := by
            ext S; simp [hbad]
          simp [he]
    _ = H.card * ENNReal.ofReal (exp (-(ε * m))) := by rw [Finset.sum_const, nsmul_eq_mul]

-- step: solve-for-m
/-- With `m ≥ log(|H|/δ)/ε` draws, `|H| e^{-εm} ≤ δ`. -/
theorem card_mul_exp_le {n : ℕ} {ε δ : ℝ} (hε : 0 < ε) (hδ : 0 < δ) {m : ℕ}
    (hm : log (n / δ) / ε ≤ m) : (n : ℝ) * exp (-(ε * m)) ≤ δ := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simpa using hδ.le
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hq : 0 < (n : ℝ) / δ := div_pos hn' hδ
  have hlog : log (n / δ) ≤ ε * m := by rwa [div_le_iff₀ hε, mul_comm] at hm
  calc (n : ℝ) * exp (-(ε * m)) ≤ n * exp (-log (n / δ)) :=
        mul_le_mul_of_nonneg_left (exp_le_exp.mpr (neg_le_neg hlog)) hn'.le
    _ = δ := by rw [exp_neg, exp_log hq]; field_simp

/-- **The PAC bound, realizable case, misleading samples.** With `m ≥ log(|H|/δ)/ε` draws, the probability that
some hypothesis of `H` with error above `ε` is consistent with the sample is at most `δ`. -/
theorem sample_misleading_le_delta {μ : Measure X} [IsProbabilityMeasure μ] {f : X → Bool} (hf : Measurable f)
    {H : Finset (X → Bool)} (hH : ∀ h ∈ H, Measurable h) {ε δ : ℝ} (hε : 0 < ε) (hδ : 0 < δ) {m : ℕ}
    (hm : log (H.card / δ) / ε ≤ m) :
    sample μ m (Misleading μ f H ε m) ≤ ENNReal.ofReal δ := by
  calc sample μ m (Misleading μ f H ε m) ≤ H.card * ENNReal.ofReal (exp (-(ε * m))) :=
        sample_misleading_le hf hH hε.le m
    _ = ENNReal.ofReal (H.card * exp (-(ε * m))) := by
        rw [ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
    _ ≤ ENNReal.ofReal δ := ENNReal.ofReal_le_ofReal (card_mul_exp_le hε hδ hm)

/-- A learner that returns a hypothesis of `H` consistent with the sample whenever `H` has one: an ERM rule in the
realizable case. -/
def IsConsistentLearner (H : Finset (X → Bool)) (f : X → Bool) {m : ℕ} (A : (Fin m → X) → X → Bool) : Prop :=
  ∀ S, A S ∈ H ∧ ((∃ h ∈ H, Consistent f h S) → Consistent f (A S) S)

-- step: realizable
/-- Realizability: if some `h⋆ ∈ H` has error zero, almost every sample is consistent with `h⋆`. -/
theorem ae_consistent_of_err_eq_zero {μ : Measure X} [IsProbabilityMeasure μ] {f h : X → Bool}
    (hf : Measurable f) (hh : Measurable h) (h0 : err μ f h = 0) (m : ℕ) :
    ∀ᵐ S ∂(sample μ m), Consistent f h S := by
  have hone : μ {x | h x = f x} = 1 := by
    have hc : {x | h x = f x} = {x | h x ≠ f x}ᶜ := by ext x; simp
    rw [hc, prob_compl_eq_one_sub (measurableSet_disagree hf hh)]
    have h0' : μ {x | h x ≠ f x} = 0 := h0
    rw [h0', tsub_zero]
  have hmeas : MeasurableSet {S : Fin m → X | Consistent f h S} := by
    have hs : {S : Fin m → X | Consistent f h S} = Set.univ.pi fun _ => {x | h x = f x} := by
      ext S; simp [Consistent, Set.mem_pi]
    rw [hs]; exact MeasurableSet.univ_pi fun _ => measurableSet_agree hf hh
  rw [ae_iff, ← Set.compl_ofPred, prob_compl_eq_zero_iff hmeas, sample_consistent_eq, hone, one_pow]

/-- **The PAC bound for a finite class, realizable case** (Corollary 2.3 of Shalev-Shwartz and Ben-David 2014).
If `H` is finite, some `h⋆ ∈ H` has error zero, and `A` returns a hypothesis of `H` consistent with the sample
whenever there is one, then with `m ≥ log(|H|/δ)/ε` draws the error of `A` is above `ε` with probability at
most `δ`. -/
theorem pac_bound_realizable {μ : Measure X} [IsProbabilityMeasure μ] {f : X → Bool} (hf : Measurable f)
    {H : Finset (X → Bool)} (hH : ∀ h ∈ H, Measurable h) (hreal : ∃ h ∈ H, err μ f h = 0)
    {ε δ : ℝ} (hε : 0 < ε) (hδ : 0 < δ) {m : ℕ} (hm : log (H.card / δ) / ε ≤ m)
    {A : (Fin m → X) → X → Bool} (hA : IsConsistentLearner H f A) :
    sample μ m {S | ENNReal.ofReal ε < err μ f (A S)} ≤ ENNReal.ofReal δ := by
  obtain ⟨hs, hsH, hs0⟩ := hreal
  -- step: reduce-to-misleading
  have hae : {S | ENNReal.ofReal ε < err μ f (A S)} ≤ᵐ[sample μ m] Misleading μ f H ε m := by
    filter_upwards [ae_consistent_of_err_eq_zero hf (hH hs hsH) hs0 m] with S hS hbad
    exact ⟨A S, (hA S).1, hbad, (hA S).2 ⟨hs, hsH, hS⟩⟩
  exact (measure_mono_ae hae).trans (sample_misleading_le_delta hf hH hε hδ hm)

/-- The hypotheses of `pac_bound_realizable` can all be met (rule F05): two points with equal mass, the identity
as the target, the class `{id, not}`, `ε = δ = 1/2`, three draws, and the learner that always returns `id`. -/
example : (sample (ProbabilityTheory.uniformOn (Set.univ : Set Bool)) 3)
    {S | ENNReal.ofReal (1 / 2) < err (ProbabilityTheory.uniformOn (Set.univ : Set Bool)) id ((fun _ => id) S)} ≤
      ENNReal.ofReal (1 / 2) := by
  have hcard : ({id, fun b => !b} : Finset (Bool → Bool)).card = 2 := by decide
  refine pac_bound_realizable (H := {id, fun b => !b}) measurable_id ?_ ⟨id, by simp, by simp [err]⟩
    (ε := 1 / 2) (δ := 1 / 2) (by norm_num) (by norm_num) ?_ ?_
  · intro h _; exact measurable_of_countable h
  · rw [hcard]
    have h4 : log (4 : ℝ) = 2 * log 2 := by
      rw [show (4 : ℝ) = 2 ^ 2 by norm_num, log_pow]; norm_num
    norm_num [h4]
    linarith [log_two_lt_d9]
  · intro S
    exact ⟨by simp, fun _ i => rfl⟩

end AI.LearningTheory.PacBoundFiniteHypothesisClass
