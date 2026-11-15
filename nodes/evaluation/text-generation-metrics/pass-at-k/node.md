---
tag: NEW-pass-at-k
kind: metric
slug: pass-at-k
name: pass@k
aliases: [pass at k, pass@1, pass@10, pass@100, unbiased pass@k estimator]
area: evaluation/text-generation-metrics
level: core
summary: >-
  The chance that at least one of k programs sampled from a model passes the unit tests of a problem, averaged over a benchmark and estimated without bias from n >= k samples.
properties:
  measures: "functional correctness of generated code: whether at least one of k samples passes the unit tests of a problem"
  range: "0 to 1, often reported in percent"
  better: higher
  formula: "mean over problems of 1 - C(n - c, k) / C(n, k), where c of the n samples of a problem pass its tests"
  level: "per problem, then the unweighted mean over the problems of a benchmark"
  reference_impl: "the numerically stable NumPy function of Figure 3 of Chen et al. (2021, arXiv 2107.03374v2), released with HumanEval at https://github.com/openai/human-eval"
history:
  year: 2019
  people: [Sumith Kulal, Panupong Pasupat, Kartik Chandra, Mina Lee, Oded Padon, Alex Aiken, Percy Liang]
  note: "Kulal et al. (arXiv 1906.04908, June 2019) scored pseudocode-to-code synthesis by the success rate at a budget of B trials; Chen et al. (arXiv 2107.03374, July 2021) named the metric pass@k, attributed it to Kulal et al., and gave the unbiased estimator from n >= k samples together with HumanEval."
prereqs: [binomial-distribution@m2, bias-of-an-estimator@m2, unit-testing@pg]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1906.04908", "2107.03374"]
sources:
  - {paper: "Chen et al. (2021)", title: "Evaluating Large Language Models Trained on Code", locator: "arXiv 2107.03374v2, pp. 1 to 6 and Appendix A (p. 19)", note: "the definition, the unbiased estimator (1), the stable product of Figure 3, HumanEval, and the bias of the plug-in estimate"}
  - {paper: "Kulal et al. (2019)", title: "SPoC: Search-based Pseudocode to Code", locator: "arXiv 1906.04908v1, the abstract, Sections 2 and 6", note: "the success rate at a budget of B trials, the measure that Chen et al. call pass@k"}
  - {paper: "Austin et al. (2021)", title: "Program Synthesis with Large Language Models", locator: "arXiv 2108.07732v1, Sections 3 and 4.4 to 4.7", note: "MBPP, tasks solved by any of 80 samples, held-out tests, and the sampling strategy"}
---

pass@k scores a code model by running its code.
For each problem of a benchmark, the model writes $k$ programs, and the problem counts as solved when at least one of them passes the tests that come with the problem.
pass@k is the expected fraction of problems solved in this way.
It is computed from more samples than $k$, with an estimator that has no bias, because a single draw of $k$ samples gives a noisy number.
The value depends on $k$, on the sampling temperature and on how thorough the tests are, so a pass@k needs these settings next to it.

## In plain words

Give a code model a programming task and let it try $k$ times.
If any one of its programs passes the tests written for the task, the task counts as solved.
Do this for every task of a benchmark and report the share of tasks solved.
With one try, this is the share of single programs that work.
With a hundred tries, it asks whether the model can find a working program at all.
Each try is random, so the share changes from run to run.
For a steadier number, draw more programs than $k$ for each task and count how many of them pass.
From that count, compute the chance that $k$ programs picked at random from the pool would include a passing one.

## Definition

A benchmark has problems $j = 1, \dots, m$, each with a prompt and unit tests.
A sample is one program that the model generates from the prompt, and it is correct if it passes all the unit tests of its problem.
Let $p_j$ be the probability that one sample for problem $j$ is correct.
If $k$ samples are drawn independently, the problem is solved with probability $1 - (1 - p_j)^k$, and pass@k is the mean over the problems:

$$ \mathrm{pass@}k = \frac{1}{m} \sum_{j=1}^{m} \big[ 1 - (1 - p_j)^k \big] $$ {#eq-pass-at-k}

To estimate it, draw $n_j \ge k$ samples for problem $j$ and let $c_j$ of them be correct.
The estimator of Chen et al. (2021) is

$$ \widehat{\mathrm{pass@}k} = \frac{1}{m} \sum_{j=1}^{m} \left[ 1 - \binom{n_j - c_j}{k} \Big/ \binom{n_j}{k} \right] $$ {#eq-estimator}

as given by Chen et al. (2021, equation 1) {eq:chen-2021-evaluating-large-language-models-trained-on-code#1}, who write the mean over problems as an expectation.
Their paper uses $n = 200$ and $k \le 100$. {#c-chen-n200}
The ratio of binomial coefficients is the probability that $k$ samples drawn without replacement from the $n$ are all incorrect, with $\binom{a}{k} = 0$ for $a < k$.
The coefficients are huge: $\binom{5000}{1000}$ has 1,085 digits and does not fit in a 64-bit float.
So the estimate of one problem is computed as a product of $c$ factors, each between 0 and 1:

$$ 1 - \binom{n - c}{k} \Big/ \binom{n}{k} = \begin{cases} 1 - \prod_{i = n - c + 1}^{n} \Big( 1 - \dfrac{k}{i} \Big) & \text{if } n - c \ge k, \\ 1 & \text{if } n - c < k. \end{cases} $$ {#eq-product-form}

When $n - c < k$, every set of $k$ samples holds a correct one, and the estimate is 1. {#c-product-form}
The file `derivations/unbiased-estimator.md` derives the product from the falling factorials of the two coefficients.

## Properties

- **Unbiased.** If the $n$ samples of a problem are independent and each is correct with probability $p$, the mean of $1 - \binom{n-c}{k}/\binom{n}{k}$ is $1 - (1 - p)^k$, for every $n \ge k$. {#c-unbiased} By linearity of expectation, eq-estimator is then unbiased for eq-pass-at-k, even when $n_j$ differs between problems. Chen et al. (2021, Appendix A) explain it as estimating the failure probability by drawing $k$ samples without replacement and checking that all fail. {#c-chen-unbiased-reason} The derivation file gives a counting argument and the algebra, and its numeric check sums the exact expectation in fractions.
- **The plug-in is biased low.** The tempting estimate $1 - (1 - c/n)^k$ puts the observed pass@1 into the formula for independent samples. For $k \ge 2$ and $0 < p < 1$ its mean is strictly below $1 - (1 - p)^k$, by Jensen's inequality, because $x \mapsto 1 - (1 - x)^k$ is strictly concave. {#c-plug-in-biased} Chen et al. found the underestimate to be consistent and the gap still open with $n > 5k$, so the plug-in rewards drawing more samples. {#c-plug-in-underestimate}
- **Range and order.** The estimate of a problem is in $[0, 1]$, it is 0 exactly when $c = 0$, and it does not decrease when $c$ or $k$ grows. {#c-range-monotone}
- **Special cases.** For $k = 1$ the estimate is $c/n$, the fraction of correct samples. For $k = n$ it is 1 when any sample is correct and 0 otherwise. For $c = 1$ it is $k/n$. {#c-special-cases} The case $k = n$ is the direct count of Kulal et al., and Chen et al. note that it and eq-estimator are the only unbiased ones among the estimators they discuss, though all are consistent. {#c-only-two-unbiased}
- **Decomposable.** The score is the plain mean of per-problem values, and each value needs only $n_j$ and $c_j$. The order of the problems does not matter, and every problem has the same weight whatever its $n_j$.
- **Not one number per model.** pass@k grows with $k$, so pass@1 and pass@100 of the same samples answer different questions.

## Worked example

Take $m = 3$ problems with $n = 5$ samples each, of which $c = 0$, 1 and 3 are correct, and $k = 2$.
There are $\binom{5}{2} = 10$ pairs of samples for each problem.

| problem | $c$ | all-incorrect pairs $\binom{5 - c}{2}$ | estimate $1 - \binom{5-c}{2}/10$ | product form | plug-in $1 - (1 - c/5)^2$ |
|---|---|---|---|---|---|
| 1 | 0 | 10 | 0 | empty product: $1 - 1 = 0$ | 0 |
| 2 | 1 | 6 | 0.4 | $1 - (1 - 2/5) = 0.4$ | $1 - 0.8^2 = 0.36$ |
| 3 | 3 | 1 | 0.9 | $1 - \tfrac{1}{3} \cdot \tfrac{1}{2} \cdot \tfrac{3}{5} = 0.9$ | $1 - 0.4^2 = 0.84$ |
| mean | | | $13/30 \approx 0.4333$ | | $0.4$ |

pass@2 is estimated as $13/30$, and the plug-in gives 0.4, lower as the bias predicts.
With $k = 1$ the same counts give pass@1 $= (0 + 0.2 + 0.6)/3 = 4/15 \approx 0.2667$.
These are the rows `worked-example`, `worked-example-plug-in` and `worked-example-k1` of `vectors/edge.jsonl`, and the test `test_worked_example` checks every number of the table.

## Reference implementation

The reference is Figure 3 of Chen et al. (2021): a NumPy function that returns 1.0 when $n - c < k$, and otherwise one minus the product of eq-product-form over $i = n - c + 1, \dots, n$. {#c-figure-3}
Chen et al. released HumanEval together with an evaluation framework. {#c-humaneval-release}
`impl/numpy/pass_at_k.py` computes the estimate of one problem as Figure 3 does, in float64.
Its `score` takes the counts $c_j$ as `predictions` and $n_j$ as `references`, as `contract.yaml` defines them, and returns the mean and the per-problem values.
The PyTorch version does the same arithmetic in `torch.float64`.
Both reproduce the 16 rows of `vectors/edge.jsonl` within a relative $10^{-9}$.
The rows are exact rationals from `math.comb` and `fractions.Fraction`, rounded once to float64.

One difference from Figure 3 is on purpose.
`score` raises `ValueError` when a problem has fewer samples than $k$.
Figure 3 returns 1.0 there, because $n - c < k$ holds for every $c$, so a problem with no correct sample would count as solved.
The code also rejects counts that are not integers and $c$ outside $[0, n]$.

> [!pitfall]
> **The plug-in estimate.** Code that computes $1 - (1 - c/n)^k$ from the pass rate underestimates pass@k for $k \ge 2$, and its result moves with $n$. The test `test_plug_in_is_biased_low` sums its exact expectation, and the row `plug-in-one-correct` shows 0.394 where the unbiased value is 0.5.

> [!pitfall]
> **Binomial coefficients in floats.** The ratio of eq-estimator computed with floating-point binomials overflows for large $n$. The test `test_large_n_is_stable` checks $n = 5000$, $c = 37$, $k = 1000$ against the exact fraction, and the row `large-n` pins the value.

> [!pitfall]
> **Fewer samples than $k$.** With $n < k$ there is no subset of $k$ samples, and the shortcut of Figure 3 reports 1. The test `test_k_larger_than_n_raises` checks that `score` refuses.

> [!pitfall]
> **The empirical count with $n > k$.** Counting a problem as solved when any of its $n$ samples passes gives pass@n, not pass@k. The test `test_special_cases` checks that $k = n$ is this count.

## Interpretation

pass@k with $k > 1$ is the success rate of a user who can run the hidden tests on every sample.
Chen et al. describe it as the best of $k$ samples picked by an oracle that knows the unit tests. {#c-oracle}
A system that must return one program without tests can do worse.
In their measurements, picking the sample with the highest mean token log-probability beat a random pick, and the highest sum of log-probabilities did slightly worse than random. {#c-logprob-ranking}
With 100 samples per problem, Codex-S solved 77.5% of HumanEval when a sample that passes the tests was picked, and 44.5% when the sample with the highest mean log-probability was picked. {#c-codex-s-ranking}

The value depends on the sampling.
Chen et al. found that the best temperature grows with $k$, because a larger $k$ rewards diverse samples; for a 679M-parameter model it was 0.2 for pass@1 and 0.8 for pass@100. {#c-temperature}
Austin et al. saw the same pattern on MBPP: lower temperatures did better with one sample, higher ones solved more tasks within 10 samples, and beam search did worse than any temperature they tried. {#c-austin-sampling}
Chen et al. also fixed the prompt format, the stop sequences and nucleus sampling with top $p = 0.95$. {#c-chen-sampling}
So two pass@k values are comparable only with the same problems, tests, $k$, estimator and sampling settings.
Section 3.4 of the same paper puts GPT-J-6B at 11.6% pass@1 and 27.7% pass@100, each the best result over the temperatures 0.2 and 0.8. {#c-gptj-two-numbers}
HumanEval has 164 problems, so one problem is worth about 0.6 points of pass@k in percent.

## Limitations and criticism

pass@k is only as good as the tests.
Austin et al. wrote extra adversarial tests for 50 MBPP problems; 33 had a sample passing the original tests and 29 a sample passing both, so about 12% of what was counted as solved would fail the harder tests. {#c-austin-challenge}
They also found a model that read the assert statements of a prompt and hard-coded the one value that the tests checked. {#c-austin-asserts}
A program that passes weak tests can still be wrong, and pass@k says nothing about readability, speed or safety beyond what the tests check.

The benchmark must not be in the training data.
Chen et al. wrote the HumanEval problems by hand because their models were trained on a large part of GitHub, which already holds solutions to problems from many sources. {#c-handwritten}
Running the samples is itself a risk, since generated code can be wrong or harmful; Chen et al. ran it in a sandbox built on the gVisor container runtime. {#c-sandbox}

The metric is binary per problem and weighs every problem the same.
It does not give partial credit to a program that passes most tests.
And pass@k for large $k$ can look strong for a model whose single samples rarely work: Codex solved 28.8% of HumanEval with one sample and 70.2% with 100 samples per problem. {#c-codex-headline}

## Variants

The options of `contract.yaml` change the number, and `vectors/variants.yaml` lists the rows that pin each value.

| option | values | effect | used by |
|---|---|---|---|
| `k` | 1 (default), any $1 \le k \le n$ | the number of samples a problem gets | pass@1 and pass@100 in Chen et al. (2021); $k = n$ in Kulal et al. (2019) and Austin et al. (2021) |
| `estimator` | `unbiased` (default), `plug-in` | eq-estimator, or $1 - (1 - c/n)^k$ | Chen et al. (2021) use the unbiased one and warn against the plug-in (Appendix A) |

Kulal et al. (2019) measure the success rate at a budget of $B$ trials: the fraction of test examples for which the system produces an accepted program within $B$ trials. {#c-kulal-success-rate}
In their setting the system may use the public tests during its search, must output one final program, and the program is accepted only if it compiles and passes the public and the hidden tests. {#c-kulal-accepted}
Austin et al. (2021) draw 80 samples per MBPP problem at temperature 0.5, run them against the tests, and report the fraction of tasks solved by any sample, which is the case $k = n = 80$. {#c-austin-metric}
For APPS, Hendrycks et al. (2021) report the percentage of problems with a correct solution, called strict accuracy, and the percentage of unit tests passed, as Chen et al. summarize. {#c-apps-metrics}
The second gives partial credit; Chen et al. use only strict accuracy.

## Implementations

## History

Before execution-based metrics, generated code was often scored by its match with a reference program, exactly or with a fuzzy score such as BLEU.
Kulal et al. (2019) argued that exact match and BLEU miss functional correctness, with the example of `if (b)` against the equivalent `if (b == true)`. {#c-kulal-bleu}
Their SPoC dataset pairs 18,356 programs with pseudocode and test cases, and with a budget of 100 compilations their search raised the success rate from 25.6% to 44.7%. {#c-spoc-results}
In July 2021 Chen et al. introduced Codex and HumanEval, a set of 164 handwritten problems with a signature, a docstring, a body and on average 7.7 unit tests each. {#c-humaneval}
They attributed pass@k to Kulal et al., noted that computing it from exactly $k$ samples can have high variance, and replaced it with the estimator from $n \ge k$ samples. {#c-kulal-pass-at-k}
They argued that match-based metrics cannot account for the large space of programs that behave like the reference. {#c-match-metrics}
Their own BLEU scores of correct and incorrect samples overlapped a lot. {#c-bleu-overlap}
Austin et al. (2021) introduced MBPP, 974 tasks meant for entry-level programmers. {#c-mbpp}
They scored samples by functional correctness and found little correlation between BLEU and synthesis performance. {#c-austin-bleu}

## References
