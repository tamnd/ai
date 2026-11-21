# Contextual calibration, orderings and the spread across prompt formats

This derivation backs the mathematical claims of the prompt-sensitivity page. The check file
`calibration-orderings-and-format-spread.check.py` tests every number below, by exact computation and by simulation.

## 1. Contextual calibration is a shift of the decision threshold

Contextual calibration feeds the prompt with a content-free input such as "N/A", reads the label probabilities
p̂cf it produces, and rescales the probabilities of every test input by W = diag(p̂cf)^−1 before taking the argmax.
The predicted class is therefore the one that maximizes p̂_i / p̂cf_i.

With two classes, write a for the content-free probability of the positive class and p for the positive probability
of a test input. The positive class wins when p / a > (1 − p) / (1 − a), that is when p (1 − a) > a (1 − p), which
simplifies to

p > a.

In the binary case contextual calibration replaces the default threshold of 0.5 by the content-free probability. In
log-odds the calibrated score is logit(p) − logit(a), so the procedure removes a constant shift. If a prompt adds the
same amount b to the log-odds of every input and the content-free input would otherwise sit at even odds, then
logit(a) = b and the correction is exact. Zhao et al. report a prompt for which "N/A" gets 62% positive; calibration
then predicts positive only above p = 0.62, a log-odds shift of logit(0.62) = 0.4895. When the bias is not a constant
shift, for instance when it depends on the input, a single threshold cannot remove it.

## 2. The number of orderings

k demonstrations can be put in k! orders: 24 for k = 4 and 40,320 for k = 8. With four demonstrations of two balanced
classes there are C(4, 2) = 6 label patterns, each shared by 4 orders. Studying every order is feasible for four
demonstrations and already impractical for eight, which is why studies of order sensitivity sample orders once k grows.

## 3. The spread grows with the number of formats sampled

Suppose a model's accuracy under a format drawn at random is μ + σZ, with Z standard normal and independent across
formats. The expected largest of m draws is μ + σ E[max Z_1..m], and by symmetry the expected range, the largest minus
the smallest, is 2σ E[max Z_1..m]. With E[max] computed as the integral of x m φ(x) Φ(x)^(m−1):

| m | E[max] | expected range in units of σ |
|---|---|---|
| 2 | 0.5642 | 1.1284 |
| 10 | 1.5388 | 3.0775 |
| 24 | 1.9477 | 3.8953 |
| 320 | 2.8983 | 5.7965 |

A spread measured over 10 formats is therefore expected to be 3.0775 / 5.7965 = 0.531 of the spread over 320 formats
from the same distribution. A spread over a small sample of formats is a lower bound on the spread of the format space,
and grows slowly, about with the square root of log m, as more formats are tried.

## 4. The noise floor

Some of the spread comes from finite test sets. An accuracy of 0.5 measured on 1,000 independent items has standard
error sqrt(0.5 · 0.5 / 1000) = 1.58 points. If every format were measured on its own sample, ten equally good formats
would show an expected range of 3.0775 · 1.58 = 4.87 points from noise alone.

When all formats are scored on the same items the noise in a difference is smaller. For two formats whose answers
differ on a fraction d of the items and with the same accuracy, the difference in accuracy has standard error
sqrt(d / n): 1.0 point at d = 0.1 and n = 1,000. Spreads of tens of points on shared items are far outside this floor.

## 5. When model rankings flip

Let two models have mean accuracies μ1 > μ2 with gap Δ = μ1 − μ2, and let a format move each model's accuracy by a
normal amount of standard deviation σ, with correlation ρ between the two models. The difference under a random
format is normal with mean Δ and variance 2σ²(1 − ρ), so the second model comes out ahead with probability

P(flip) = Φ(−Δ / (σ sqrt(2 (1 − ρ)))).

At σ = 5 points and ρ = 0 the probability is 0.3886 for Δ = 2 points, 0.2398 for Δ = 5 and 0.0786 for Δ = 10. Raising
the correlation to ρ = 0.5 lowers it for Δ = 2 only to 0.3446. When format effects correlate weakly between models,
a gap of a few points between two models measured with one format is close to a coin flip under another.

## 6. Choosing a prompt on a small development set overstates its accuracy

Suppose m prompts are all equally good, with true accuracy 0.7, and one picks the prompt that scores highest on a
development set of 100 items. Each score has standard error sqrt(0.7 · 0.3 / 100) = 4.583 points, so the expected
score of the winner exceeds its true accuracy by 4.583 E[max Z_1..m]: 7.05 points for m = 10 and 8.93 points for the
m = 24 orders of four demonstrations. The selected prompt's development score is not an estimate of its accuracy, and
a selection that uses labelled data the few-shot setting does not have makes the method no longer few-shot.
