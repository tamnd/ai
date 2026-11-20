# The arithmetic of safety training

This derivation backs the mathematical claims of the safety-training page. The check file
`safety-training-arithmetic.check.py` tests every statement numerically.

## 1. How much of the data is safety data

Bianchi et al. keep N = 20,000 general instructions fixed and add n safety instructions. The safety data is then
the fraction f(n) = n/(N + n) of the training set:

| n | f(n) |
|---|---|
| 100 | 100/20,100 = 0.50% |
| 500 | 500/20,500 = 2.44% |
| 1,000 | 1,000/21,000 = 4.76% |
| 2,000 | 2,000/22,000 = 9.09% |

Solving n/(N + n) = f for n gives n = fN/(1 − f); for f = 0.03 this is 600/0.97 ≈ 618.6, so "3% safety
examples" is about 619 of them, the "few hundred demonstrations" of the paper.

## 2. Context distillation is a KL divergence

Let p(y | c, x) be the model with the safety preprompt c, the teacher, and q_θ(y | x) the model being trained,
which sees only x. The context-distillation loss on prompt x is the cross-entropy

H(p, q_θ) = E_{y∼p(·|c,x)}[−log q_θ(y | x)] = Σ_y p(y|c,x) log(p(y|c,x)/q_θ(y|x)) − Σ_y p(y|c,x) log p(y|c,x)
          = KL(p(·|c,x) ‖ q_θ(·|x)) + H(p(·|c,x)).

The teacher is held fixed, so its entropy H(p) does not depend on θ, and minimizing the loss over θ minimizes the KL
divergence. By Gibbs' inequality the KL divergence is non-negative and zero exactly when q_θ(·|x) = p(·|c,x), so
the minimum, if q_θ can represent it, makes the model without the preprompt behave as the model with it. Fine-tuning
on sampled teacher outputs, as Llama 2 does, is a Monte Carlo estimate of the same loss.

## 3. Projection onto the complement of a direction

Let r̂ be a unit vector and P = I − r̂r̂ᵀ. Then

* Pᵀ = I − (r̂r̂ᵀ)ᵀ = P, so P is symmetric;
* P² = I − 2r̂r̂ᵀ + r̂(r̂ᵀr̂)r̂ᵀ = I − r̂r̂ᵀ = P, since r̂ᵀr̂ = 1, so P is idempotent;
* r̂ᵀPx = r̂ᵀx − (r̂ᵀr̂)(r̂ᵀx) = 0, so Px has no component along r̂;
* if r̂ᵀx = 0 then Px = x, so P leaves the complement alone;
* ‖x‖² = ‖Px‖² + (r̂ᵀx)², since Px ⊥ r̂r̂ᵀx, so ‖Px‖ ≤ ‖x‖.

Directional ablation x′ = x − r̂r̂ᵀx is x′ = Px.

**Why the difference of means.** Let μ and ν be the mean activations on harmful and harmless prompts and
r = μ − ν ≠ 0. For a unit vector u, Cauchy–Schwarz gives uᵀr ≤ ‖u‖‖r‖ = ‖r‖, with equality exactly when
u = r/‖r‖ = r̂. So r̂ is the unit direction along which the two means are furthest apart, and the separation is ‖r‖.
Activation addition x ← x + r applied to every harmless prompt shifts their mean to ν + r = μ, the mean harmful
activation, at the layer where it is added.

## 4. Weight orthogonalization equals ablation everywhere

Write the residual stream of a decoder at layer l as x^(l), with x^(1) = e + p the token and position embeddings,
and each component (attention or MLP) adding W_out h + b_out, where h is computed from the current stream (through a
layer norm and nonlinearities). Consider two models:

* **A** is the original model with x ← Px applied to every residual activation before any component reads it and
  at the output;
* **B** has every write multiplied by P: embeddings Pe and Pp, matrices PW_out and biases Pb_out.

Claim: at every point, the residual activation of B equals that of A. At the start, A has P(e + p) and B has
Pe + Pp, which are equal by linearity. Suppose the activations agree, both equal to some z with r̂ᵀz = 0 (A's is a
projection; B's is a sum of vectors in the complement of r̂). Then the component reads the same z in both models and
computes the same h. A adds W_out h + b_out and projects: P(z + W_out h + b_out) = z + P(W_out h + b_out), using
Pz = z. B adds PW_out h + Pb_out = P(W_out h + b_out). These agree, and the sum again lies in the complement of r̂.
By induction the two models agree at every layer, and so do their logits. Every residual activation of B lies in the
complement of r̂, so ablating r̂ in B changes nothing: the weight edit is the inference-time intervention.

## 5. Base64 length

Base64 reads the input three bytes (24 bits) at a time and writes each 24 bits as four 6-bit symbols from a
64-character alphabet. A final group of one or two bytes is padded to four characters with "=". So n bytes become
4⌈n/3⌉ characters, about a third more. The question "What tools do I need to cut down a stop sign?" has 45 ASCII
characters, 45 bytes, so it becomes 4 × 15 = 60 characters with no padding, the string that begins V2hhdCB0b29scy
in Wei et al.'s Figure 1.

## 6. Red-teaming robustness

If violations are found at a constant rate γ per person-hour, the expected time per violation is 1/γ hours.
For γ = 1.8 this is 1/1.8 h = 33.3 minutes; for γ = 0.45 it is 1/0.45 h = 2.22 h = 133.3 minutes. The ratio is
1.8/0.45 = 4.

## 7. Counting refusals

XSTest's percentages are of 250 safe prompts: 0.38 × 250 = 95 full refusals and 0.216 × 250 = 54 partial refusals,
both whole numbers, so 149 prompts, 149/250 = 59.6%, are refused at least in part. Llama 2's safety evaluation set
has 1,351 + 623 = 1,974 prompts, the "roughly 2,000" of the paper.

## 8. Greedy coordinate gradient

The model embeds token x_i as Eᵀe_{x_i}, where e_{x_i} is a one-hot vector in R^V and E the embedding matrix, so the
loss L is a differentiable function of each one-hot vector. Let g = ∇_{e_{x_i}} L ∈ R^V. Replacing x_i by token v
changes the one-hot vector by e_v − e_{x_i}, so to first order

L(x with x_i = v) ≈ L(x) + gᵀ(e_v − e_{x_i}) = L(x) + g_v − g_{x_i}.

The term g_{x_i} is the same for every v, so ordering the candidates by −g_v orders them by estimated decrease, and
Top-k(−g) are the k substitutions at position i with the largest estimated decrease. The estimate is only
first-order, which is why GCG then evaluates the loss of a batch of candidates exactly.

With |I| = 20 positions and a vocabulary of V = 32,000 tokens, there are |I|(V − 1) = 20 × 31,999 = 639,980
single-token substitutions. The top-256 candidates of every position number k|I| = 256 × 20 = 5,120, and a batch of
B = 512 evaluates 512/5,120 = 10% of these, or 0.08% of all substitutions, per step.
