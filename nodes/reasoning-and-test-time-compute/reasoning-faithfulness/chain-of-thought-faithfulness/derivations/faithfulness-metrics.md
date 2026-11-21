# The arithmetic of faithfulness tests

This derivation backs the mathematical claims of the chain-of-thought faithfulness page. The check file
`faithfulness-metrics.check.py` tests every statement numerically, including the estimators by simulation.

No test can read a model's internal reasoning directly. Each test instead changes something, the prompt or the chain
of thought, and watches the answer. The numbers below say what such a test can and cannot recover.

## 1. Correcting a hint test for random answer changes

The hint test of Chen et al. pairs an unhinted prompt x_u with a hinted prompt x_h that points to an answer h. It keeps
the pairs where the model's answer moves from a non-hint answer a_u ≠ h to the hint answer a_h = h, and scores the
fraction of kept pairs whose chain of thought names the hint as a reason.

Some of the kept pairs would have moved to h anyway, because a sampled answer changes from run to run. Let n be the
number of options, p = P(a_h = h | a_u ≠ h) and q = P(a_h ≠ h, a_h ≠ a_u | a_u ≠ h). If noise moves an answer to each
of the n − 2 other options equally often, noise alone moves it to h with probability q / (n − 2). The share of moves
to h that the hint caused is

α = (p − q / (n − 2)) / p = 1 − q / ((n − 2) p).

Suppose the chain of thought names the hint with probability f when the hint caused the move, and never when noise
did. The raw score is then α · f + (1 − α) · 0 = α f, so dividing the raw score by α recovers f. The paper divides by
α and clips the result at 1.

Two examples with n = 4 options. With p = 0.30 and q = 0.04, noise accounts for 0.02 of the 0.30, α = 0.9333, and a
raw score of 0.20 becomes 0.2143: the correction is small when the hint is strong. With p = 0.10 and q = 0.12, noise
accounts for 0.06 of the 0.10, α = 0.4, and a raw score of 0.10 becomes 0.25. When p ≤ q / (n − 2), α ≤ 0 and the
score is undefined, because the model moves to h no more often than to any other option.

The correction assumes the noise term is uniform over the other options and that noise moves are never verbalized.
It does not touch the other direction of error: a chain that names the hint may still hide another reason, so the
score bounds faithfulness from above.

## 2. The area over the early-answering curve

The early-answering test of Lanham et al. truncates a chain of L steps after k = 0, 1, ..., L steps and asks for an
answer each time. Let s_k be the fraction of truncations after k steps that give the same answer as the full chain,
so s_L = 1. Plot s_k against k / L. If the answer was settled before the reasoning, every s_k is 1 and the curve is
flat. The area over the curve (AOC) measures the distance from that flat line,

AOC = 1 − area under the curve, computed here with the trapezoid rule on the L + 1 points.

For L = 4 and s = (0.3, 0.4, 0.6, 0.9, 1.0), the area is (0.15 + 0.4 + 0.6 + 0.9 + 0.5) / 4 = 0.6375 and the AOC is
0.3625. A chain that never changes the answer has AOC 0.

The AOC cannot reach 1. A model whose answer depends entirely on the last step still agrees with its final answer by
chance before that step. If it guesses uniformly among m options until step L, s_k = 1 / m for k < L and the AOC is

AOC_max = (1 − 1 / m)(1 − 1 / (2L)),

which is 0.7 for m = 5 options and L = 4 steps. A task's AOC should be read against this ceiling: AQuA, with five
options, reached 0.44 in the paper. The task-level number weights the AOC of each chain length by the fraction of
chains with that length.

## 3. How often a rarely verbalized behaviour is caught

Suppose a model performs a behaviour, such as a reward hack, in K independent episodes and names it in its chain of
thought with probability v each time, and a monitor flags every episode where it is named. The monitor flags at least
one episode with probability

P(caught) = 1 − (1 − v)^K.

At v = 0.02 this is 0.02 for one episode, 0.1829 for K = 10, 0.8674 for K = 100 and above 0.95 from about K = 149. A
monitor of chains of thought can therefore notice a behaviour that happens often, during training or evaluation,
while being nearly useless against one rare episode. This is the asymmetry Chen et al. draw from their reveal rates.

The formula assumes independent episodes and a perfect monitor on verbalized cases. A monitor with recall ρ replaces v
by ρ v, and training against the monitor can push v itself towards zero.

## 4. Contradictory answers to a question and its reverse

The pair test of Arcuschin et al. asks "Is X south of Y?" and "Is Y south of X?". When X and Y differ in position,
exactly one of the two has the answer Yes, so the same answer to both is a contradiction. If the two answers are
sampled independently, with P(No) = n₁ for the first question and n₂ for the second, the chance of a contradictory
pair is

n₁ n₂ + (1 − n₁)(1 − n₂).

With the rates of the paper's Gemini 2.5 Flash example, n₁ = 0.99 and n₂ = 0.63, a random pair of answers is No to
both with probability 0.6237 and contradictory with probability 0.6274. The test needs no access to the hidden reason:
the contradiction alone shows that at least one of the two chains of thought does not describe what produced its
answer.
