# The arithmetic of process reward models

This derivation backs the mathematical claims of the process-reward-model page. The check file
`process-reward-arithmetic.check.py` tests every statement numerically.

Throughout, a solution is a sequence of steps s_1, ..., s_M. For a prefix s_1..s_t, write q_t for the probability
that a completion sampled from the prefix by a fixed completer reaches the correct final answer.

## 1. Cross-entropy learns the mean of the label

Fix a step and let its label y ∈ [0, 1] be random with mean μ, 0 < μ < 1. The expected cross-entropy of a prediction
r ∈ (0, 1) is

L(r) = −(μ log r + (1 − μ) log(1 − r)),

because the loss is linear in y. Its derivative −μ/r + (1 − μ)/(1 − r) is zero exactly at r = μ, and the second
derivative μ/r² + (1 − μ)/(1 − r)² is positive, so r = μ is the unique minimum. A PRM trained by cross-entropy on
labels therefore estimates the mean of the label at each step, to the extent that the network can fit it.

## 2. Soft and hard Monte Carlo labels

Draw N completions from the prefix, each reaching the correct answer independently with probability q_t, and let C be
the number that do. C is binomial(N, q_t).

The soft label is C/N. Its mean is q_t, so by section 1 a PRM trained on soft labels estimates q_t. Its variance is
q_t(1 − q_t)/N, which is largest at q_t = 1/2, where it is 1/(4N); for N = 8 the standard deviation is up to
√(1/32) ≈ 0.177.

The hard label is 1(C ≥ 1). Its mean is P(C ≥ 1) = 1 − (1 − q_t)^N, so a PRM trained on hard labels estimates
1 − (1 − q_t)^N rather than q_t. This is above q_t for every 0 < q_t < 1 and N ≥ 2. For N = 8 and q_t = 0.1 it is
1 − 0.9^8 ≈ 0.570: a step from which only one completion in ten succeeds is labelled correct more often than not.

## 3. Finding the first error by binary search

Suppose the first wrong step is f ∈ {1, ..., M}, or f = M + 1 if there is none, and suppose a prefix check is exact:
the prefix s_1..s_t passes (some rollout succeeds) exactly when t < f. The predicate "t < f" is monotone in t, so
f is the smallest t that fails, with t = M + 1 standing for "all prefixes pass". Binary search over the M + 1 possible
values of f halves the candidate set with every check, and so needs ⌈log2(M + 1)⌉ checks in the worst case. Checking
prefixes one by one needs up to M checks. For M = 20 this is ⌈log2 21⌉ = 5 against up to 20. With k rollouts per check
the costs are about k log M and kM rollouts.

## 4. Implicit process rewards telescope

Let the outcome reward be r(y) = β log π_θ(y)/π_ref(y) for two autoregressive models, and define for a response of
T steps

q^t = Σ_{i≤t} β log π_θ(y_i | y_<i)/π_ref(y_i | y_<i), q^0 = 0.

Because both models factor over steps, q^T = β log π_θ(y)/π_ref(y) = r(y). With process rewards r^t = q^t − q^(t−1),

Σ_{t=1}^T r^t = q^T − q^0 = r(y),

so the process rewards of a response sum to its outcome reward.

Yuan et al.'s Proposition 3.1 says more: q^t = β log E_{y∼π_ref(·|y≤t)} e^{r(y)/β}. To see it, note that
e^{r(y)/β} = π_θ(y)/π_ref(y). Split y into the first t steps and the rest. The ratio of the first t steps is
e^{q^t/β}, and the expectation over the rest under π_ref of the ratio of the rest, π_θ(rest | prefix)/π_ref(rest | prefix),
is Σ π_θ(rest | prefix) = 1. So E[e^{r/β}] = e^{q^t/β}, and taking β log of both sides gives the identity.

## 5. Aggregating step scores

Let the step scores p_1, ..., p_M lie in (0, 1]. Every factor of the product is at most 1, so the product is at most
every single score, and in particular at most the minimum; the minimum is at most the last score. Adding a step
multiplies the product by p_{M+1} ≤ 1, so the product never increases with length. Ten steps at 0.95 give
0.95^10 ≈ 0.599, below a single step at 0.7, though every one of the ten steps scores higher than that one step.
Under the product, a long solution with uniformly good steps can rank below a short one with a doubtful step; the
minimum does not have that bias, and the last score ignores all earlier steps.

## 6. The F1 of error identification

ProcessBench takes the accuracy x on solutions with an error (locating the first error) and the accuracy y on correct
solutions (flagging no error), and reports the harmonic mean 2xy/(x + y). A model that flags an error everywhere has
y = 0 and one that never flags has x = 0, so either scores 0, while the arithmetic mean would give each up to 0.5. For
x = 0.9 and y = 0.5, 2 · 0.9 · 0.5/1.4 ≈ 0.643, below the arithmetic mean 0.7: the harmonic mean is pulled towards the
weaker of the two.
