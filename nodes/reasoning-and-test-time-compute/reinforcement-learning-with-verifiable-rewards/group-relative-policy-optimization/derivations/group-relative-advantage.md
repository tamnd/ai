# The arithmetic of group-relative advantages

This derivation backs the mathematical claims of the group-relative-policy-optimization page. The check file
`group-relative-advantage.check.py` tests every statement numerically.

Throughout, a question q has G outputs o_1, ..., o_G sampled independently from the old policy, with scalar rewards
r_1, ..., r_G. Write mean(r) and std(r) for the mean and the population standard deviation of the G rewards.

## 1. A baseline that does not depend on the output keeps the gradient unbiased

For a policy π_θ over outputs o and any number b fixed before o is drawn,

E_{o∼π_θ}[b ∇ log π_θ(o)] = b Σ_o π_θ(o) ∇ log π_θ(o) = b Σ_o ∇ π_θ(o) = b ∇ Σ_o π_θ(o) = b ∇ 1 = 0.

So E[(r(o) − b) ∇ log π_θ(o)] = E[r(o) ∇ log π_θ(o)], the policy gradient. The baseline changes the variance of the
estimate, not its mean.

## 2. The group mean is a scaled leave-one-out baseline

The group mean includes r_i itself, so it is not independent of o_i. Write m_{−i} for the mean of the other G − 1
rewards. Then

r_i − mean(r) = r_i − (r_i + (G − 1) m_{−i})/G = (G − 1)/G · (r_i − m_{−i}).

The leave-one-out baseline m_{−i} is computed from outputs drawn independently of o_i, so by section 1 the term
(r_i − m_{−i}) ∇ log π_θ(o_i) has the policy gradient as its expectation. The centred group advantage is the same
term multiplied by the constant (G − 1)/G, so before any division by std(r) the GRPO gradient is an unbiased estimate
of (G − 1)/G times the policy gradient. This is the RLOO estimator up to that constant. Dividing by std(r), which
depends on o_i, gives up exact unbiasedness.

## 3. Group advantages sum to zero

Σ_i (r_i − mean(r)) = G mean(r) − G mean(r) = 0, and dividing by a common std(r) keeps the sum at 0. If every output
gets the same reward, every centred advantage is 0 (implementations add a small constant to std(r) to avoid 0/0),
and the group contributes no policy gradient.

With binary rewards and a fraction p of correct outputs, 0 < p < 1, the mean is p and the population variance is
p(1 − p). A correct output has advantage (1 − p)/√(p(1 − p)) = √((1 − p)/p), and an incorrect output has
−p/√(p(1 − p)) = −√(p/(1 − p)). A rare correct answer (small p) gets a large positive advantage, and a rare wrong
answer (p near 1) a large negative one.

## 4. How often a group is informative

If each output is correct with probability q, independently, the group is all correct with probability q^G and all
wrong with probability (1 − q)^G. It gives a nonzero advantage with probability

1 − q^G − (1 − q)^G.

For q = 0.9 and G = 16 this is 1 − 0.185 − 10^(−16) ≈ 0.815. For q = 0.95 and G = 8 it is 1 − 0.663 − 4·10^(−11)
≈ 0.337. As a policy gets good at the training questions, most groups carry no signal, which is why dynamic sampling
filters them out.

## 5. Dividing by the standard deviation reweights questions

With binary rewards and a fraction p correct, the centred advantages are 1 − p for the Gp correct outputs and −p for
the G(1 − p) wrong ones, so their absolute values sum to

Gp(1 − p) + G(1 − p)p = 2Gp(1 − p).

After division by √(p(1 − p)) the sum is 2G√(p(1 − p)). The ratio of the normalized to the unnormalized size is
1/√(p(1 − p)): 2 at p = 1/2, and 1/√(15/256) = 16/√15 ≈ 4.13 at p = 1/16. Relative to centring alone, a question
solved by 1 output in 16 is weighted about 2.07 times as much as one solved by half the group, and the factor grows
without bound as p approaches 0 or 1. This is the difficulty bias.

## 6. Length normalization reweights tokens

Each output's advantage A_i is shared by all its |o_i| tokens. Three ways of reducing the per-token losses to one
number give each token of output i the weight

* sequence mean (GRPO): (1/G)(1/|o_i|) A_i;
* token mean (DAPO): A_i / Σ_j |o_j|;
* constant (Dr. GRPO): A_i / (G · L_max), with L_max a fixed maximum generation length.

Under the sequence mean, the total push on output i is (1/G) A_i whatever its length, so a longer output gets a
smaller push per token. For a wrong output (A_i < 0) of 1,000 tokens, each token is pushed down with weight
|A_i|/(1000G), one tenth of the |A_i|/(100G) of a 100-token wrong output. Under the other two, the per-token weight is
the same for every output in the batch, so long wrong outputs are penalized in proportion to their length.

## 7. The k3 estimator of the KL divergence

Let o_t be sampled from π_θ given a prefix, and x = π_ref(o_t)/π_θ(o_t), with π_ref positive wherever π_θ is. The
function f(x) = x − log x − 1 has f(1) = 0, f'(x) = 1 − 1/x, zero only at x = 1, and f''(x) = 1/x² > 0, so f(x) ≥ 0
for every x > 0. Its expectation is

E_{π_θ}[x] − E_{π_θ}[log x] − 1 = Σ_o π_ref(o) + E_{π_θ}[log(π_θ/π_ref)] − 1 = 1 + KL(π_θ ‖ π_ref) − 1
= KL(π_θ ‖ π_ref).

So each sample gives a nonnegative estimate whose mean is the KL divergence, while the plain estimate −log x has the
same mean but can be negative.

## 8. Sharpening can lower pass@k

For a problem solved with probability p per sample, pass@k = 1 − (1 − p)^k with independent samples. Two problems
with p = 0.2 and p = 0.05 have mean pass@1 (0.2 + 0.05)/2 = 0.125 and mean pass@64

(1 − 0.8^64 + 1 − 0.95^64)/2 ≈ (1 − 6·10^(−7) + 1 − 0.0375)/2 ≈ 0.981.

A policy that moves probability to the first problem's solutions, p = 0.6, and loses the second, p = 0, has mean
pass@1 0.3 but mean pass@64 (1 − 0.4^64 + 0)/2 ≈ 0.5. Training that sharpens the distribution on problems already
solvable raises pass@1 and can lower pass@k for large k.
