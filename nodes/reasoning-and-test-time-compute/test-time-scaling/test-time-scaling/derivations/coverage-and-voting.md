# The arithmetic of coverage, voting and compute matching

This derivation backs the mathematical claims of the test-time-scaling page. The check file
`coverage-and-voting.check.py` tests every statement numerically.

Throughout, a problem i is solved by one independent sample with probability p_i, the per-sample success rate of the
model on that problem. Coverage at k samples, also called pass@k, is the probability that at least one of k samples
is correct.

## 1. Coverage of one problem

The k samples fail independently, each with probability 1 − p_i, so

c_i(k) = 1 − (1 − p_i)^k.

The failure probability 1 − c_i(k) falls geometrically in k, and log(1 − c_i(k)) = k log(1 − p_i) is linear in k.
The gain from the (k + 1)-th sample is c_i(k + 1) − c_i(k) = p_i (1 − p_i)^k, which is positive for 0 < p_i < 1 and
shrinks as k grows, so c_i is increasing and concave in k. A problem with p_i = 0 has c_i(k) = 0 for every k.

## 2. The unbiased pass@k estimator

Suppose N ≥ k samples are drawn and C of them are correct. The estimator

ĉ(k) = 1 − binom(N − C, k) / binom(N, k)

is the probability that a k-subset chosen uniformly from the N samples contains at least one correct sample. Choose
the subset first and draw the samples afterwards: the k chosen samples are then k independent draws, so the
probability that all of them are wrong is (1 − p)^k. Averaging over the draws, E[ĉ(k)] = 1 − (1 − p)^k, so ĉ(k) is
unbiased for c(k) at every k ≤ N. The plug-in estimate 1 − (1 − C/N)^k is biased: by Jensen's inequality,
E[(1 − C/N)^k] ≥ (1 − p)^k for k ≥ 2, so the plug-in underestimates coverage on average.

## 3. A Beta mixture of problems gives an exponentiated power law

Average section 1 over a benchmark whose per-sample success rates are spread with density f(p):

c(k) = 1 − E[(1 − p)^k].

Take p ∼ Beta(α, β). Then E[(1 − p)^k] = B(α, β + k) / B(α, β), and Stirling's formula gives, for large k,

1 − c(k) = Γ(α + β) Γ(β + k) / (Γ(β) Γ(α + β + k)) ≈ A k^{−α},  with A = Γ(α + β)/Γ(β).

So the failure rate of a benchmark falls as a power of k, while for one problem it falls geometrically (section 1).
The exponent α is set by the problems that are almost never solved: the Beta density near p = 0 behaves as p^{α−1},
so the fraction of problems with p below ε scales as ε^α.

Since log c(k) = log(1 − A k^{−α}) ≈ −A k^{−α} once A k^{−α} is small,

c(k) ≈ exp(a k^b),  with a = −A and b = −α.

This is the exponentiated power law fitted to coverage curves by Brown et al. (2024), and in this model its exponent
b is minus the Beta parameter α. With α = 0.5 and β = 2 (mean success rate 0.2), exact coverage is 0.20 at k = 1,
0.612 at k = 10, 0.868 at k = 100 and 0.9867 at k = 10,000, where the power law A k^{−α} with A ≈ 1.329 predicts the
failure rate 0.0133 almost exactly. If a fraction π_0 of problems has p = 0, coverage is at most 1 − π_0 for every k.

## 4. Majority voting converges to a limit set by the mode

Let a model's final answers to one problem follow a distribution q over answers, with q* the probability of the
correct answer y*. With n samples, majority voting returns the most frequent answer. For a competitor y, the variable
X_j = 1[answer_j = y*] − 1[answer_j = y] lies in [−1, 1] and has mean q* − q_y. If q* > q_y, Hoeffding's inequality
gives

P(count(y*) ≤ count(y)) ≤ exp(−n (q* − q_y)^2 / 2),

and a union bound over the competitors bounds the probability of a wrong or tied vote by
Σ_{y ≠ y*} exp(−n (q* − q_y)^2 / 2). So if y* is the unique mode of q, voting is correct with probability tending to 1,
exponentially fast in n. If some y has q_y > q*, the law of large numbers makes count(y) > count(y*) with probability
tending to 1, and voting is wrong in the limit. With q* = 0.4 and one competitor at 0.3, the bound at n = 100 is
exp(−0.5) ≈ 0.61, and at n = 1,000 it is about 0.0067.

Over a benchmark, voting accuracy converges to the fraction of problems whose correct answer is the mode of q, the
limit stated by Wu et al. (2024), while coverage converges to the fraction with q* > 0. Problems with
0 < q* < max_y q_y count towards the limit of coverage and not towards the limit of voting, and this is the gap
between the two curves.

## 5. Best-of-n with a verifier that can be fooled

Model a verifier that ranks samples as follows: a fraction ε of samples are wrong answers that the verifier scores
above every correct sample, a fraction p are correct, and the remaining wrong samples score below every correct one.
Best-of-n returns the top-scored of n independent samples, which is correct exactly when the n samples contain a
correct one and none of the high-scoring wrong ones:

P_n = (1 − ε)^n − (1 − ε − p)^n.

Write a = 1 − ε and b = 1 − ε − p, so a > b. The function a^n − b^n is 0 at n = 0, rises, and falls back to 0, and
setting its derivative a^n log a − b^n log b to 0 gives the best number of samples

n* = log(log b / log a) / log(a / b).

With p = 0.3 and ε = 0.001, n* ≈ 16.5, and P_n is 0.30 at n = 1, about 0.981 at n = 16, 0.905 at n = 100 and 0.368 at
n = 1,000. More samples first help, because a correct answer becomes likely, and then hurt, because a fooling answer
becomes likely; the rarer the fooling answers, the later the peak.

## 6. Spending a fixed budget across problems

Give problem i a number of samples k_i, with Σ k_i = K, to maximize the expected number of problems covered,
Σ_i (1 − (1 − p_i)^{k_i}). By section 1 the marginal gain of one more sample on problem i is p_i (1 − p_i)^{k_i},
which decreases in k_i. For a sum of concave functions of integers under one budget, adding samples one at a time to
the problem with the largest current marginal gain is optimal: an exchange argument shows that any allocation whose
smallest used marginal gain is below the largest unused one can be improved by moving one sample.

The marginal gains show where the budget goes. A problem with p = 0.9 gains 0.9 from its first sample, 0.09 from its
second and 0.009 from its third. A problem with p = 0.05 gains 0.05 from its first and still 0.05 · 0.95^9 ≈ 0.0315
from its tenth. A problem with p = 0 gains nothing. Easy problems saturate after a few samples, the budget flows to
problems that are hard but solvable, and the best allocation depends on each problem's difficulty.

## 7. Matching FLOPs between a larger model and more inference

Take the approximations of 6ND FLOPs to train a model of N parameters on D tokens and 2ND FLOPs to generate D tokens.
A model M times larger, trained on D_pretrain tokens and serving D_inference tokens, costs
6MN D_pretrain + 2MN D_inference. The smaller model with its inference multiplied by a factor m costs
6N D_pretrain + 2N m D_inference. Setting the two equal,

m = M + 3 (D_pretrain / D_inference) (M − 1),

the multiplier given by Snell et al. (2024). For M = 14, m = 53 when D_inference = D_pretrain, m = 17.9 when
D_inference = 10 D_pretrain, and m = 404 when D_inference = D_pretrain / 10. The less inference a deployment serves,
the more test-time compute per query the smaller model may spend at equal total cost.

## 8. The slope of a train-test trade-off

Jones (2021) fits log(test) = −1.2 · log(train) + 0.004 · elo + 29 for AlphaZero on Hex, where test and train are the
two compute budgets. At fixed Elo, multiplying train-time compute by 10 changes log(test) by −1.2, so test-time
compute is divided by 10^1.2 ≈ 15.8. The ratio does not depend on the base of the logarithm, because a slope between
two logarithms in the same base is the same in every base. This matches the paper's statement that each 10× of train-time compute removes
about 15× of test-time compute.
