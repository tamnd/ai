# The arithmetic of model merging

This derivation backs the mathematical claims of the model-merging page. The check file
`model-merging-arithmetic.check.py` tests every statement numerically.

## 1. Task arithmetic contains the averages

Let θ_pre be the pre-trained weights and θ_1, ..., θ_n models fine-tuned from it, with task vectors
τ_t = θ_t − θ_pre. Then

θ_pre + λ Σ_t τ_t = θ_pre + λ Σ_t θ_t − nλ θ_pre = (1 − nλ) θ_pre + λ Σ_t θ_t.

At λ = 1/n the first term vanishes and the result is (1/n) Σ_t θ_t, the uniform average of the fine-tuned
models. With n = 1 and λ = α the result is (1 − α) θ_pre + α θ_1, which is WiSE-FT's interpolation between
the zero-shot and the fine-tuned model. Task arithmetic with one scaling coefficient therefore moves along
a line through θ_pre and the uniform soup, and λ > 1/n extrapolates past the soup.

## 2. Fisher merging

Approximate the posterior of model i by a Gaussian with mean θ_i and diagonal precision F_i, and maximize
Σ_i λ_i log N(θ; θ_i, diag(F_i)⁻¹). Up to a constant this objective is −½ Σ_i λ_i Σ_j F_ij (θ_j − θ_ij)²,
which separates over coordinates j. Its derivative in θ_j is Σ_i λ_i F_ij (θ_ij − θ_j), which is zero at

θ_j = Σ_i λ_i F_ij θ_ij / Σ_i λ_i F_ij,

provided the denominator is positive. The second derivative −Σ_i λ_i F_ij is negative, so this is the
maximum. If every F_i is the same vector f, the f_j cancel and θ_j = Σ_i λ_i θ_ij / Σ_i λ_i, which is the
weighted average Σ_i λ_i θ_i when the λ_i sum to 1. Fisher merging differs from averaging only where the
models disagree about which coordinates matter.

Example: θ_1 = (1, 0), θ_2 = (0, 1), F_1 = (3, 1), F_2 = (1, 1) and λ_1 = λ_2 = 1/2. The first coordinate
is (3·1 + 1·0)/(3 + 1) = 0.75 and the second (1·0 + 1·1)/(1 + 1) = 0.5, against the plain average
(0.5, 0.5). The first model is three times as sure of its first coordinate, so it gets three quarters of
the weight there.

## 3. Weight averaging and output ensembling

Let f(x; θ) be linear in θ, as a linear classifier on fixed features is: f(x; θ) = W φ(x) + b with
θ = (W, b). Then for weights λ_i summing to 1,

f(x; Σ_i λ_i θ_i) = (Σ_i λ_i W_i) φ(x) + Σ_i λ_i b_i = Σ_i λ_i f(x; θ_i),

so averaging the weights gives the same logits as averaging the outputs. When θ enters through a
nonlinearity this fails: for f(x; w) = tanh(w x), tanh(½(w_1 + w_2)x) differs from
½(tanh(w_1 x) + tanh(w_2 x)) in general, for example at w_1 = 0, w_2 = 4, x = 1, where the first is
tanh 2 ≈ 0.964 and the second ½ tanh 4 ≈ 0.500.

## 4. Mean and variance of DARE

DARE replaces each delta entry δ by m δ/(1 − p), where m is Bernoulli with P(m = 0) = p. The value is
δ/(1 − p) with probability 1 − p and 0 otherwise, so

E = (1 − p) δ/(1 − p) = δ,
E[·²] = (1 − p) δ²/(1 − p)² = δ²/(1 − p),
Var = δ²/(1 − p) − δ² = δ² p/(1 − p).

At p = 0.9 the variance is 9δ² and at p = 0.99 it is 99δ². Without the rescaling the mean would be
(1 − p)δ, which is why the rescale matters. The entry is unbiased, but its standard deviation is 3|δ| at
p = 0.9: DARE works when the network's output depends on many delta entries at once, so that the errors
average out, and when the deltas are small.

## 5. The running average of SWA

Suppose w̄_n = (1/n) Σ_{k ≤ n} w_k. Then

(w̄_n · n + w_{n+1})/(n + 1) = (Σ_{k ≤ n} w_k + w_{n+1})/(n + 1) = w̄_{n+1},

so the update keeps the exact mean. Only w̄ and the current weights are stored, one extra copy however
many snapshots are averaged.

## 6. The disjoint mean

Fix a parameter, let its trimmed entries across n models be a_1, ..., a_n, and suppose the elected sign is
positive. Let P be the set of entries that are positive, assumed non-empty, with |P| = m ≤ n. The other
entries are zero or negative, so Σ_i a_i ≤ Σ_{i∈P} a_i, and

(1/n) Σ_i a_i ≤ (1/n) Σ_{i∈P} a_i ≤ (1/m) Σ_{i∈P} a_i,

the last because the sum over P is positive and m ≤ n. The disjoint mean is at least the plain mean.
With the signs reversed the same argument shows it is at most the plain mean. Either way the disjoint mean
has at least the magnitude of the plain mean in the elected direction: conflicting signs and zeros no longer
shrink the update.

## 7. The loss barrier

The barrier is B = max_{λ∈[0,1]} L((1 − λ)Θ_A + λΘ_B) − ½(L_A + L_B). The maximum includes λ = 0 and
λ = 1, so it is at least max(L_A, L_B), and

B ≥ max(L_A, L_B) − ½(L_A + L_B) = ½|L_A − L_B| ≥ 0.

If L is convex along the segment, L((1 − λ)Θ_A + λΘ_B) ≤ (1 − λ)L_A + λL_B ≤ max(L_A, L_B), so the maximum
is attained at an end and B = ½|L_A − L_B|, which is 0 when the end losses are equal. A positive barrier
between two equal-loss models therefore means the loss rises somewhere on the line between them, and
the midpoint of the two weights is a worse model than either.

## 8. Permutations

Let f(x) = W_2 σ(W_1 x) with σ applied elementwise, and let P be a permutation matrix. Since σ(P z) = P σ(z)
and Pᵀ P = I, the network with W_1' = P W_1 and W_2' = W_2 Pᵀ computes W_2 Pᵀ σ(P W_1 x) = W_2 σ(W_1 x), the
same function. The average of the two weight sets is ½(W_1 + P W_1) and ½(W_2 + W_2 Pᵀ), which in general
computes a different function: hidden unit k of the average mixes unit k of one copy with a different unit
of the other. If P is undone first, mapping the second copy back by Pᵀ, both copies are W_1, W_2 and the
average is the network itself. Weight matching estimates such a P between two independently trained networks,
where it is not known.

## 9. A worked TIES example

Take three task vectors over five parameters,

τ_1 = (0.5, −0.2, 0.05, 0.8, −0.6),
τ_2 = (−0.45, −0.4, 0.02, 0.6, 0.1),
τ_3 = (0.4, 0.1, −0.01, −0.05, −0.7),

and keep the top 60%, three of the five entries, of each by magnitude:

τ̂_1 = (0.5, 0, 0, 0.8, −0.6),
τ̂_2 = (−0.45, −0.4, 0, 0.6, 0),
τ̂_3 = (0.4, 0.1, 0, 0, −0.7).

The column sums are (0.45, −0.3, 0, 1.4, −1.3), so the elected signs are (+, −, 0, +, −). The disjoint means
are (0.5 + 0.4)/2 = 0.45, −0.4/1 = −0.4, 0 (no entry survives), (0.8 + 0.6)/2 = 0.7 and (−0.6 − 0.7)/2 =
−0.65, so τ_m = (0.45, −0.4, 0, 0.7, −0.65). The plain mean of the untrimmed vectors is
(0.15, −0.5/3, 0.02, 0.45, −0.4) ≈ (0.15, −0.167, 0.02, 0.45, −0.4). On the first parameter the two
models that agree are no longer cancelled by the one that disagrees, and on the third the small values,
which TIES treats as noise, are removed.
