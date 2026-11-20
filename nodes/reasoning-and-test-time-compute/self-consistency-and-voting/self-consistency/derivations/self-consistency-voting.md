# The arithmetic of voting over sampled answers

This derivation backs the mathematical claims of the self-consistency page. The check file
`self-consistency-voting.check.py` tests every statement numerically.

Throughout, a model given a question samples a reasoning path r and an answer a from P(r, a). The answer
distribution is the marginal P(a) = Σ_r P(r, a), and the m samples are independent draws from it unless stated
otherwise. N_a counts the samples with answer a, so N_a/m is the empirical share of a.

## 1. The vote converges to the mode of the marginal

Each indicator 1(a_i = a) has mean P(a), so by the weak law of large numbers N_a/m → P(a) in probability for every
answer a. If the mode a* is unique, write δ = P(a*) − max_{a≠a*} P(a) > 0. Once every share is within δ/2 of its
probability, N_{a*}/m exceeds every other share, so the vote returns a*. With finitely many answers that has
probability tending to 1. The vote therefore estimates arg max_a P(a), the mode of the answer distribution with
the reasoning paths summed out, and not the answer of the most probable single path.

## 2. Weighted votes

A weighted vote returns arg max_a Σ_i w_i 1(a_i = a). With w_i = 1 for all i the weighted sum for a is N_a, so the
weighted vote is the majority vote. If the weight is the unnormalized probability of the whole output, a product of
K token probabilities each below 1, then for token probabilities bounded by some q < 1 the weight is at most q^K,
which shrinks geometrically in the length K. Long paths get small weights however sensible they are, and the vote
favours short ones. The length-normalized weight exp((1/K) Σ_k log P(t_k | ...)) is the geometric mean of the token
probabilities and does not fall with K.

## 3. Two answers: the binomial vote

Suppose there are two answers and the correct one is sampled with probability p. With m odd there are no ties, and
the vote is correct when more than m/2 samples are correct:

P(correct) = Σ_{j>m/2} C(m, j) p^j (1 − p)^(m−j).

For p = 0.6 this gives 0.6 at m = 1, 0.683 at m = 5, 0.753 at m = 11 and 0.903 at m = 41, rising to 1. Replacing p by
1 − p swaps the roles of the two answers, so the accuracy for p = 0.4 is one minus that for p = 0.6: 0.4, 0.317,
0.247 and 0.097, falling to 0. Voting amplifies whichever answer the model already prefers.

## 4. An exponential bound

Let y be the correct answer and suppose P(y) − P(a) ≥ δ > 0 for every other answer a. For one competitor a, put
X_i = 1(a_i = y) − 1(a_i = a) ∈ {−1, 0, 1}. Its mean is P(y) − P(a) ≥ δ and its range is 2. The competitor ties or
beats y only if Σ_i X_i ≤ 0, a deviation of at least mδ below the mean, so Hoeffding's inequality gives

P(N_a ≥ N_y) ≤ exp(−2(mδ)²/(m · 2²)) = exp(−mδ²/2).

The vote is wrong (counting ties as errors) only if some competitor ties or beats y; a union bound over the |A| − 1
competitors gives P(wrong) ≤ (|A| − 1) exp(−mδ²/2). For δ = 0.3 and one competitor, exp(−1.8) ≈ 0.165 < 0.17 at
m = 40 and exp(−4.5) ≈ 0.0111 < 0.012 at m = 100.

## 5. Mixtures of easy and hard questions

Section 3 says accuracy rises for questions with p > 1/2 and falls for those with p < 1/2. On a benchmark that mixes
them the average can do both. Let 60% of questions have p = 0.6 and 40% have p = 0.45. At m = 1 the accuracy is
0.6 · 0.6 + 0.4 · 0.45 = 0.54. As m grows the easy part tends to 0.6 · 1 and the hard part to 0.4 · 0, so the limit
is 0.6. In between the easy questions improve fast while the hard ones, whose margin is only 0.05, decay slowly;
over odd m the average peaks at about 0.653 at m = 71 and is down to about 0.609 at m = 401. More samples first help
and then hurt.

## 6. The Beta stopping rule

Adaptive-Consistency stops once it is confident that the leading answer is the mode. With leading count v1 and
runner-up count v2 it puts a Beta(v1 + 1, v2 + 1) posterior on q = p1/(p1 + p2) and stops when P(q > 1/2) exceeds the
threshold. For integers α = v1 + 1 and β = v2 + 1, the Beta–binomial identity gives

P(q > 1/2) = P(Bin(α + β − 1, 1/2) ≤ α − 1) = 1 − P(Bin(v1 + v2 + 1, 1/2) ≥ v1 + 1).

For counts 9 and 1 the binomial has 11 trials and P(≥ 10) = (11 + 1)/2048 = 12/2048, so the confidence is
1 − 12/2048 ≈ 0.994. A unanimous run of v samples has confidence 1 − 2^−(v+1): 1 − 1/32 ≈ 0.969 at v = 4, which
passes 0.95, and 1 − 1/16 = 0.9375 at v = 3, which does not. Counts 6 and 2 give 0.910 and do not stop.

## 7. Windows of unanimity

Early-stopping self-consistency samples windows of w answers and stops at the first window whose answers all agree.
A window is unanimous on a with probability P(a)^w, and unanimous at all with probability q = Σ_a P(a)^w.

If the mode is unique, every other answer has P(a) ≤ 1/2 (two answers above one half would sum to more than one,
and an answer at exactly one half would make the remaining mass one half, so the mode would have to share it). So the
probability that a window is unanimous on a wrong answer is

Σ_{a≠mode} P(a)^w ≤ (1/2)^(w−1) Σ_{a≠mode} P(a) ≤ 2^−(w−1),

which is 1/128 for w = 8. With no cap on the number of windows, the number of windows until the first unanimous one is
geometric with success probability q, so the expected number of samples is w/q. For P = (0.9, 0.08, 0.02) and w = 5,
q = 0.9^5 + 0.08^5 + 0.02^5 ≈ 0.590 and w/q ≈ 8.5, far below a fixed budget of 40.

## 8. Coverage against voting

If each sample is correct with probability p, the probability that at least one of k samples is correct is
1 − (1 − p)^k, which tends to 1 for any p > 0. For p = 0.2 it is 1 − 0.8^10 ≈ 0.893 at k = 10. Voting instead tends
to the mode of P (section 1): if a wrong answer has probability 0.3 and the correct one 0.2, the vote converges on
the wrong answer while coverage converges to 1. Coverage measures what a perfect verifier could pick out of the
samples; the vote does not reach it.

From n samples with c correct, the estimator 1 − C(n − c, k)/C(n, k) is unbiased for 1 − (1 − p)^k when k ≤ n.
C(n − c, k)/C(n, k) is the probability that a uniformly random k-subset of the n samples contains no correct one.
Conditioned on the samples, its expectation over the subset is that probability; averaging over the samples, a fixed
k-subset of independent samples is all wrong with probability (1 − p)^k. So E[C(n − c, k)/C(n, k)] = (1 − p)^k.
