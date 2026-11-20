# The arithmetic of fine-tuning

This derivation backs the mathematical claims of the fine-tuning page. The check file
`fine-tuning-arithmetic.check.py` tests every statement numerically.

## 1. Where fine-tuning moves the features

Take the two-layer linear network f(x) = vᵀBx with features B ∈ R^{k×d} and head v ∈ R^k, inputs X ∈ R^{n×d}
and labels Y ∈ R^n. The mean squared loss is L = (1/2n)‖r‖² with residuals r = XBᵀv − Y. Its gradients are

∇_v L = B Xᵀ r / n,   ∇_B L = v rᵀ X / n.

Every row of ∇_B L is a multiple of rᵀX, a combination of the rows of X. So if u is orthogonal to every
training input (Xu = 0), then ∇_B L · u = v rᵀ(Xu)/n = 0, and gradient descent never changes Bu: the
features of directions the training data does not cover stay at their pretrained values B₀u, while the
features along the data move. This is the feature distortion of Kumar et al.

If the linear probe, the head fitted with B = B₀ frozen, already reaches zero training loss, then r = 0, both
gradients vanish, and fine-tuning from the probed head does not move at all. LP-FT starts at a point where
the features have no reason to change, which is why it distorts them less.

## 2. A conserved quantity

Under gradient flow, dv/dt = −B Xᵀ r/n and dB/dt = −v rᵀX/n. Then

d(BBᵀ)/dt = Ḃ Bᵀ + B Ḃᵀ = −(v rᵀXBᵀ + B Xᵀ r vᵀ)/n,
d(vvᵀ)/dt = v̇ vᵀ + v v̇ᵀ = −(B Xᵀ r vᵀ + v rᵀ X Bᵀ)/n,

which are equal, so BBᵀ − vvᵀ keeps its initial value. With a zero head at the start it equals B₀B₀ᵀ.
Discrete gradient descent with a small step conserves it approximately.

## 3. A worked example: the golden ratio

Let k = 1, d = 2, B₀ = (1, 1), v₀ = 0, one training input x = (1, 0) with label 1, and suppose the truth is
y = vᵀB₀x with v = 1, so an input x' = (0, 1) outside the training data should get label 1.

The linear probe solves v·(B₀x) = v·1 = 1, so v = 1 and the prediction on x' is v·B₀x' = 1, exact.

Full fine-tuning updates only the first entry b of B, because the second entry multiplies x₂ = 0 (section 1).
The conserved quantity gives b² + 1 − v² = 2, that is b² − v² = 1. At convergence the training loss is zero,
so vb = 1. Substituting v = 1/b gives b⁴ − b² − 1 = 0, so b² = (1 + √5)/2 = φ, the golden ratio. Then
b = √φ ≈ 1.272, v = 1/√φ ≈ 0.786, and the prediction on x' is v·1 ≈ 0.786. Fine-tuning fitted the training
point by growing both b and v, and the growth of v, applied to the untouched second feature, is wrong off the
training data by about 21%.

## 4. Learning rate schedules

**Slanted triangular.** With cut = ⌊T·cut_frac⌋, p = t/cut for t < cut and
p = 1 − (t − cut)/(cut(1/cut_frac − 1)) after, the rate is η_t = η_max(1 + p(ratio − 1))/ratio.
At t = 0, p = 0 and η = η_max/ratio. At t = cut, p = 1 and η = η_max. If T is a multiple of 1/cut_frac,
then cut = T·cut_frac, cut(1/cut_frac − 1) = T − cut, so at t = T, p = 0 again and η = η_max/ratio.
With the defaults (0.1, 32, 0.01) the rate climbs from 0.0003125 to 0.01 over the first tenth of training and
falls back over the remaining nine tenths.

**Discriminative rates.** With η^{l−1} = η^l/2.6, the bottom of L layers has rate η^L/2.6^{L−1}.
2.6³ = 17.576 and 2.6⁴ = 45.6976.

**Warmup and linear decay.** The rate rises linearly from 0 to the peak over the first wT steps and falls
linearly to 0 at T. The area under it is a triangle of height peak and base T, so its mean is peak/2,
whatever w is.

## 5. Adam without bias correction

Adam keeps m_t = β₁m_{t−1} + (1 − β₁)g_t and v_t = β₂v_{t−1} + (1 − β₂)g_t², starting from 0, and steps by
α m̂_t/√v̂_t with m̂_t = m_t/(1 − β₁^t), v̂_t = v_t/(1 − β₂^t). With a constant gradient g,
m_t = (1 − β₁^t)g and v_t = (1 − β₂^t)g², so the corrected step is exactly α·sign(g) (ignoring ε), and the
uncorrected step m_t/√v_t is (1 − β₁^t)/√(1 − β₂^t) times larger. With β₁ = 0.9 and β₂ = 0.999:

| t | ratio |
|---|---|
| 1 | 0.1/√0.001 = √10 ≈ 3.162 |
| 12 | ≈ 6.569 (the maximum over integer t) |
| 100 | ≈ 3.241 |
| 1000 | ≈ 1.258 |

The ratio tends to 1 as t grows. Because β₂ is much closer to 1 than β₁, the second moment is the more
underestimated, and omitting the correction makes early steps larger, not smaller. Bias correction reduces
the early step relative to BERTAdam, the effect Mosbach et al. compare to warmup.

## 6. The EWC optimum

For one parameter, take the new task's loss to be (a/2)(θ − b)² near its minimum b, and add the EWC penalty
(λF/2)(θ − θ_old)². The derivative a(θ − b) + λF(θ − θ_old) vanishes at

θ = (ab + λFθ_old)/(a + λF),

a weighted average of the two optima with weights equal to their curvatures (precisions). λF → 0 gives b,
the new task's optimum; λF → ∞ gives θ_old. Fisher merging of two models uses the same precision weighting.
