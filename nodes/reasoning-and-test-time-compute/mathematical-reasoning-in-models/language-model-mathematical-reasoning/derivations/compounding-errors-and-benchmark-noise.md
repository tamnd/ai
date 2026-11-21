# Compounding errors, digit dependence and the noise in math benchmarks

This derivation backs the mathematical claims of the mathematical-reasoning page. The check file
`compounding-errors-and-benchmark-noise.check.py` tests every number below, by exact computation and by simulation.

## 1. Errors compound over steps

Suppose a solver gets each step of a problem right with probability p, independently of the other steps, and a
wrong step cannot be undone. A problem of T steps is then solved with probability

P(T) = p^T.

At p = 0.95 this is 0.9025 for T = 2, 0.8145 for T = 4, 0.6634 for T = 8 and 0.4401 for T = 16. GSM8K problems take
2 to 8 steps, so a per-step error of 5% already costs a third of the longest problems.

The model also fixes the shape of the decline. Each added step multiplies accuracy by the same factor p, so the
absolute drop from T to T + 1 steps is p^T (1 − p), which shrinks as T grows: 0.0451, 0.0407 and 0.0332 at T = 2, 4
and 8. The ratio between the accuracies of successive difficulty levels is the number to watch, since under the
model it stays constant.

GSM-Symbolic reports four levels with one clause removed (M1), the original (Symb) and one or two clauses added (P1,
P2). For Gemma2-9b-it the accuracies are 84.4, 79.1, 68.1 and 41.8, so the successive ratios are 0.937, 0.861 and
0.614 and the drops 5.3, 11.0 and 26.3 points. For Phi-3.5-mini-instruct, at 87.6, 82.1, 64.8 and 44.8, the ratios
are 0.937, 0.789 and 0.691. In both, each added clause removes a larger fraction of the remaining accuracy than the
one before, which a constant per-step accuracy cannot produce; the per-step accuracy itself falls as problems grow.
The pattern is not universal: for GPT-4o-mini, at 92.5, 91.7, 81.1 and 72.4, the ratios are 0.991, 0.884 and 0.893.
A clause is not exactly one reasoning step, as the GSM-Symbolic authors note, so the comparison is of shapes only.

## 2. Which digits of a product are easy

For any whole numbers a and b and any j ≥ 1,

a · b mod 10^j = (a mod 10^j) · (b mod 10^j) mod 10^j,

because a = (a mod 10^j) + 10^j · s and b = (b mod 10^j) + 10^j · t, and every product term but the first is a
multiple of 10^j. So the j lowest digits of a product depend on only the j lowest digits of each factor. The units
digit depends on 2 input digits with 10^2 = 100 combinations, the two lowest digits on 4 input digits with 10^4
combinations, and the hundreds digit of a product of two three-digit numbers on all 6 input digits, 10^6
combinations.

A network that learns each output digit as a lookup on the inputs that determine it needs to see those combinations.
In n random draws from N equally likely combinations, a given combination is missed with probability (1 − 1/N)^n, so
the fraction seen is 1 − (1 − 1/N)^n. With n = 200,000 draws this is 1 − 2.1 · 10^(−9) for N = 10^4 and 0.1813 for
N = 10^6. The units and tens digits of a three-digit product therefore have every input combination in a training set
of that size, and the hundreds digit has under a fifth of them. The PyTorch implementation trains on exactly this
setting and gets the units and tens digits right on unseen pairs while the two middle digits stay near the 10% of
chance. The leading digits are easy for a different reason: they depend mainly on the sizes of a and b, a smooth
function of the leading input digits.

Long multiplication of two n-digit numbers forms n^2 one-digit products. If each is right with probability 0.99 and
nothing else can go wrong, all are right with probability 0.99^(n^2): 0.9606 for n = 2, 0.9135 for n = 3 and 0.7778
for n = 5. Adding the partial products only lowers these.

## 3. The spread of accuracy across re-instantiated test sets

GSM-Symbolic draws 50 sets, each holding one fresh instance of each of 100 templates. Let template i be solved with
probability p_i over the draw of its names and numbers. With greedy decoding the draw is the only randomness, and the
accuracy of a set is the mean of 100 independent Bernoulli variables, so its standard deviation is

sd = sqrt(Σ_i p_i (1 − p_i)) / 100.

Two cases bound it. If the outcome on each template did not depend on the instance, every p_i would be 0 or 1 and sd
would be 0. If every p_i equalled the mean accuracy p̄, sd would be sqrt(p̄ (1 − p̄) / 100); since p (1 − p) is
concave, this is the largest value for a given mean. At p̄ = 0.791, the GSM-Symbolic mean of Gemma2-9b-it, the
bound is 4.07 points. The reported spread of 3.0 points gives Σ p_i (1 − p_i) = 9.0, a mean of 0.09 per template.
So the spread is direct evidence that correctness depends on the instance, not only on the template: names and
numbers alone change whether a question is solved.

## 4. Is a gap between two benchmarks real?

GSM1k has 1205 problems and the GSM8K test set 1319. For a model whose true accuracy on both is p = 0.8, the
standard error of the difference between its two accuracies is

sqrt(p (1 − p) / 1205 + p (1 − p) / 1319) = 0.0159,

or 1.59 points. A drop of 8 points is 5.02 standard errors, far outside sampling noise, while drops under about 3.2
points (two standard errors) are within it.

## 5. Majority voting and pass@k

Let each sample be right with probability q. If every wrong sample gives a different wrong answer, a majority vote
over k samples is right unless no sample is right or ties go against it. At q = 0.4 the exact vote accuracy is 0.4,
0.7149, 0.9951 and 1.0000 for k = 1, 5, 15 and 41. Wrong answers that scatter cannot outvote a right one that
repeats.

If instead the wrong samples share one wrong answer with probability w, the vote converges to whichever of the two is
more likely. At q = 0.4 and w = 0.2 the vote still helps, at 0.6426, 0.8455 and 0.9543 for k = 5, 15 and 41. At
q = 0.4 and w = 0.5 it hurts: 0.4039, 0.3411 and 0.2493. A systematic error, such as subtracting a number that should
be ignored, is the case voting cannot fix.

Pass@k counts a problem as solved if any of k samples is right. From n samples of which c are right, the unbiased
estimate is 1 − C(n − c, k) / C(n, k). With n = 256 and c = 10 it is 0.0391 at k = 1 and 0.9470 at k = 64; with a
single right sample, c = 1, it is 0.25 at k = 64 and 1 at k = 256. Pass@k at large k therefore credits problems on
the strength of one sample, which is where answers reached by wrong reasoning, false positives, weigh most.
