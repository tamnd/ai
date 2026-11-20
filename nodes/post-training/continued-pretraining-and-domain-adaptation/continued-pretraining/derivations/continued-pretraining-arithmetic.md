# The arithmetic of continued pretraining

This derivation backs the mathematical claims of the continued-pretraining page. The check file
`continued-pretraining-arithmetic.check.py` tests every statement numerically.

## 1. Continuing costs the new data only

For a dense model with N parameters trained on D tokens, training compute is about 6ND. Re-training from scratch on
the union of the old data D0 and the new data D1 costs 6N(|D0| + |D1|). A model already trained on D0 only needs the
new tokens, 6N|D1|. The ratio is |D1|/(|D0| + |D1|): for Ibrahim et al.'s 300B tokens of the Pile followed by 300B
tokens of SlimPajama it is 300B against 600B, one half. Each further dataset makes the gap larger, because
re-training pays for all earlier data again while continuing pays only for the new data.

## 2. The re-decayed cosine schedule and where it crosses a fifth

Write s ∈ [0, 1] for the fraction of the decay done. The cosine schedule is

η(s) = ηmin + (ηmax − ηmin)(1 + cos πs)/2.

At s = 0, cos 0 = 1 and η = ηmax. At s = 1, cos π = −1 and η = ηmin. At s = 1/2, cos(π/2) = 0 and
η = (ηmax + ηmin)/2. Its derivative −(ηmax − ηmin)(π/2) sin πs is zero at both ends and steepest in the middle, so
the schedule spends the start and the end near its extremes.

Parmar et al. continue a model whose pretraining ended at 4.5e-5 by decaying from 4.5e-5 to a hundredth of it,
4.5e-7, and switch blends when the learning rate reaches a fifth of its start, 9e-6. Solving η(s) = 9e-6:

(1 + cos πs)/2 = (9e-6 − 4.5e-7)/(4.5e-5 − 4.5e-7) = (0.2 − 0.01)/0.99 ≈ 0.1919,

so cos πs ≈ −0.6162 and s = arccos(−0.6162)/π ≈ 0.711. If tokens are spread evenly over the decay, the first blend
gets about 71% of the tokens and the second blend about the last 29%.

## 3. What replay costs and what it optimizes

With replay fraction r, a fraction r of the tokens of every batch comes from the old data. A run of T tokens sees
(1 − r)T new tokens, so to see the same number of new tokens as a run without replay it needs T/(1 − r) tokens:
1/0.95 ≈ 1.053 times as many for 5%, 1/0.75 = 4/3 for 25%, and 2 for 50%. Ibrahim et al.'s compute-equivalent
replay instead holds the total fixed and gives up the new tokens: 5% of 100B is 5B old tokens and 95B new ones.

If each sequence of a batch is drawn from the old distribution with probability r and from the new one otherwise,
and the batch loss is the mean of the per-sequence losses, then by linearity of expectation

E[batch loss] = r L_old(θ) + (1 − r) L_new(θ),

and the same holds for the gradient, since differentiation is linear. Replay therefore runs stochastic gradient
descent on a weighted sum of the two losses: r = 0 ignores the old data, and r is the weight the old loss gets.

## 4. EWC pulls towards the old value in proportion to its Fisher information

Take one parameter, a new-task loss (a/2)(θ − θ_B)² with a > 0, and the EWC penalty (λ/2)F(θ − θ_A)² with λF ≥ 0.
The objective is a sum of two quadratics; its derivative a(θ − θ_B) + λF(θ − θ_A) is zero at

θ* = (a θ_B + λF θ_A)/(a + λF),

and the second derivative a + λF > 0 makes this the minimum. It is an average of θ_B and θ_A weighted by the
curvatures a and λF: λF = 0 gives θ_B, and as λF → ∞ it tends to θ_A. With a diagonal Fisher matrix the objective
separates into one such problem per parameter, so parameters with large F_i stay near their old values and the others
move to the new optimum.

## 5. Extending the vocabulary

Chinese LLaMA grows the vocabulary from V = 32,000 to V′ = 49,953, so k = V′ − V = 17,953 rows are added to the
input embedding and k to the output head. At hidden size 4,096 that is 2 × 17,953 × 4,096 = 147,070,976 ≈ 147
million new parameters, each one untrained.

How the new output rows start matters for what the model predicts before any training. Let h be the final hidden
state and w_1, ..., w_V the old output rows, with logits z_i = w_i·h and Z = Σ_i e^{z_i}. Set every new row to the mean
w̄ = (1/V) Σ_i w_i. Each new logit is then w̄·h = (1/V) Σ_i z_i = z̄, and by Jensen's inequality, since exp is convex,

e^{z̄} ≤ (1/V) Σ_i e^{z_i} = Z/V.

The k new tokens together get probability k e^{z̄}/(Z + k e^{z̄}), which increases with e^{z̄} and so is at most
(k Z/V)/(Z + k Z/V) = k/(V + k). The old tokens keep their ratios; each old probability is divided by
(Z + k e^{z̄})/Z, so the KL divergence from the old distribution p to the new one q is

KL(p ‖ q) = Σ_i p_i log((Z + k e^{z̄})/Z) = log(1 + k e^{z̄}/Z) ≤ log(1 + k/V).

For Chinese LLaMA's numbers the new tokens take at most 17,953/49,953 ≈ 0.36 of the mass and the KL divergence is at
most log(1.561) ≈ 0.45 nats. Random new rows give no such bound: a new row with a large inner product with h can take
almost all of the probability.

## 6. What the RoPE base does to wavelengths

RoPE rotates the pair i = 0, ..., d/2 − 1 of a query or key at position m by the angle m θ_i with θ_i = b^(−2i/d). The
pair repeats after 2π/θ_i = 2π b^(2i/d) positions. The fastest pair, i = 0, has wavelength 2π for every b. The slowest,
i = d/2 − 1, has wavelength 2π b^((d−2)/d). For d = 128 the exponent is 126/128:

- b = 10,000: 2π × 10^(4 × 126/128) ≈ 54,410 positions;
- b = 1,000,000: 2π × 10^(6 × 126/128) ≈ 5,063,256 positions.

The ratio is 100^(126/128) ≈ 93.06. Raising the base slows every pair except the first, and the slow pairs most. For
Xiong et al.'s b = 500,000 the slowest wavelength is about 2.56 million positions, 50^(126/128) ≈ 47 times longer.

## 7. Token counts

Code Llama's 500B tokens at 85% code hold 0.85 × 500B = 425B tokens of code, and 425/80 ≈ 5.3 times the 80B tokens of
code in Llama 2's pretraining. Llemma 7B's 200B tokens at 95% Proof-Pile-2 are 190B tokens; over a 55B-token corpus
that is 190/55 ≈ 3.45 passes. DeepSeekMath-Base's 500B tokens at 56% from its corpus are 280B tokens; over 120B that is
280/120 ≈ 2.33 passes. All three take the mixture weights as fractions of tokens.
