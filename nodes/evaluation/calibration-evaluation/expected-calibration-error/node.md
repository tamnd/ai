---
tag: 01R1CF
kind: metric
slug: expected-calibration-error
name: Expected calibration error
aliases: [ECE, calibration error, binned calibration error, maximum calibration error, MCE, reliability diagram, top-label calibration error]
area: evaluation/calibration-evaluation
level: core
summary: >-
  How far a classifier's confidence is from its accuracy: predictions are binned by confidence, and the gaps between each bin's accuracy and mean confidence are averaged, weighted by bin size.
properties:
  measures: "miscalibration of the top-label confidence: the average gap between how often a prediction is right and how confident the model was"
  range: "0 to 1, often reported in percent"
  better: lower
  formula: "sum over bins m of (|B_m| / n) |acc(B_m) - conf(B_m)|, with M equal-width bins on [0, 1]"
  level: "over a whole evaluation set; one number per set, not per example"
  reference_impl: "equation 3 of Guo et al. (2017, arXiv 1706.04599v2) with M = 15 equal-width bins; this page's impl/numpy and impl/pytorch"
history:
  year: 2017
  people: [Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger]
  note: "Guo et al. (arXiv 1706.04599, ICML 2017) made the binned ECE with 15 bins and temperature scaling the standard way to measure and fix the calibration of neural networks; Kumar et al. (2019) and Minderer et al. (2021) studied its biases, and Kadavath et al. (2022) and OpenAI (2023) applied it to language models."
prereqs: [conditional-expectation@m2, bias-of-an-estimator@m2, cross-entropy@m2, maximum-likelihood-estimation@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1706.04599", "2106.07998", "1909.10155", "2207.05221"]
sources:
  - {paper: "Guo et al. (2017)", title: "On Calibration of Modern Neural Networks", locator: "arXiv 1706.04599v2, equations 1 to 9, Table 1, pp. 7 to 8", note: "perfect calibration, reliability diagrams, ECE, MCE, temperature scaling and the main measurements"}
  - {paper: "Minderer et al. (2021)", title: "Revisiting the Calibration of Modern Neural Networks", locator: "arXiv 2106.07998v2, pp. 1 to 9", note: "top-label calibration, equal-mass bins, the two biases of the binned estimate, and the dependence of rankings on the variant"}
  - {paper: "Kumar et al. (2019)", title: "Verified Uncertainty Calibration", locator: "arXiv 1909.10155v2, pp. 1 to 8", note: "binning underestimates the calibration error, more bins measure more, and the debiased estimator"}
  - {paper: "Kadavath et al. (2022)", title: "Language Models (Mostly) Know What They Know", locator: "arXiv 2207.05221v4, pp. 1 to 11", note: "calibration of language models on multiple-choice questions, and temperature for RLHF policies"}
  - {paper: "OpenAI (2023)", title: "GPT-4 Technical Report", locator: "arXiv 2303.08774v6, pp. 10 and 12", note: "the ECE of GPT-4 before and after post-training"}
  - {paper: "Müller et al. (2019)", title: "When Does Label Smoothing Help?", locator: "arXiv 1906.02629v3, pp. 4 to 5", note: "label smoothing against temperature scaling on ECE"}
---

A classifier is calibrated when its confidence matches its accuracy: among the predictions it makes with 80% confidence, about 80% are right.
The expected calibration error (ECE) measures how far a model is from this.
It sorts the predictions into bins by confidence, compares the accuracy of each bin with its mean confidence, and averages the gaps, weighting each bin by its share of the predictions.
The usual version takes the confidence of the predicted class only and uses 15 bins of equal width.
The ECE is easy to compute and to read, but it is an estimate with a bias whose sign and size depend on the number of bins and of predictions, so an ECE needs its binning and sample size next to it.

## In plain words

A weather forecaster who says "70% chance of rain" is calibrated if it rains on about 70% of those days.
A classifier makes the same kind of statement each time it gives its top class a probability.
To check it, collect many predictions, put those with confidence between 0.9 and 1 in one group, those between 0.8 and 0.9 in another, and so on.
In each group, count how often the model was right and compare that with the confidence it claimed.
A model that claims 95% and is right 75% of the time is overconfident.
The ECE is the average of these mismatches, with big groups counting more.
A plot of accuracy against confidence for the groups is a reliability diagram; a calibrated model lies on the diagonal.

## Definition

A classifier gives each input $x$ a probability vector over $K$ classes.
Its prediction is the class with the largest probability, and its confidence $C$ is that probability.
Let $A = 1$ when the prediction is correct and $A = 0$ otherwise.
The model is perfectly calibrated when the chance of being right at every confidence level equals that level {#c-guo-perfect}:

$$ P(A = 1 \mid C = p) = p \quad \text{for all } p \in [0, 1] $$ {#eq-perfect}

This is equation 1 of Guo et al. (2017) {eq:guo-2017-on-calibration-of-modern-neural-networks#1}.
It is top-label calibration: it concerns only the predicted class, not the whole vector. {#c-minderer-top-label}
With $r(c) = P(A = 1 \mid C = c)$, the calibration error is the expected gap

$$ \mathrm{CE} = \mathbb{E}\,\big|\, r(C) - C \,\big| $$ {#eq-ece}

and it is zero exactly when eq-perfect holds.
When $C$ is continuous, each event $C = p$ has probability zero, so $r$ cannot be estimated without grouping confidences.

The binned estimate groups $n$ predictions into $M$ bins.
Bin $B_m$ holds the predictions with confidence in $I_m = ((m-1)/M,\, m/M]$; the first bin also holds 0.
For bin $m$, the accuracy $\mathrm{acc}(B_m)$ is the fraction of correct predictions and $\mathrm{conf}(B_m)$ the mean confidence.
Then

$$ \widehat{\mathrm{ECE}} = \sum_{m=1}^{M} \frac{|B_m|}{n}\, \big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big| = \frac{1}{n} \sum_{m=1}^{M} \Big|\sum_{i \in B_m} (a_i - c_i)\Big| $$ {#eq-ece-hat}

as in equation 3 of Guo et al. {eq:guo-2017-on-calibration-of-modern-neural-networks#3} {#c-guo-ece}
The maximum calibration error is the largest gap over the non-empty bins {#c-guo-mce}:

$$ \widehat{\mathrm{MCE}} = \max_{m:\, |B_m| > 0} \big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big| $$ {#eq-mce}

The reliability diagram is a visual representation of calibration that plots the accuracy of each bin against its confidence, so a calibrated model lies on the diagonal. {#c-guo-reliability}

Temperature scaling is the usual fix.
Given logits $z$, it divides them by one scalar $T > 0$ before the softmax and picks $T$ by minimizing the negative log-likelihood (NLL) on held-out data {#c-guo-temperature}:

$$ \hat q = \mathrm{softmax}(z / T), \qquad \hat T = \arg\min_{T > 0} \; -\frac{1}{n} \sum_{i=1}^{n} \log \hat q_{i, y_i} $$ {#eq-temperature}

## Properties

**The ECE is in $[0, 1]$ and is at most the MCE.** Both follow from eq-ece-hat: the ECE is a weighted mean of the bin gaps and the MCE their maximum.

**Temperature does not change the accuracy.** For every $T > 0$, $z_j / T > z_k / T$ exactly when $z_j > z_k$, so the softmax keeps the order of the logits, and the predicted class and the accuracy are the same for all $T$. {#c-ts-argmax}
Raising $T$ only lowers the confidence, from 1 as $T \to 0$ towards $1/K$ as $T \to \infty$. {#c-ts-confidence}
So temperature scaling can trade overconfidence for underconfidence but cannot make a wrong prediction right.

**Fitting $T$ is a convex problem.** With $\beta = 1/T$, the mean NLL is convex in $\beta$, with first derivative the mean of $\mathbb{E}_q[z] - z_y$ and second derivative the mean variance of the logits under the fitted probabilities. {#c-ts-convex}
At an interior minimum,

$$ \frac{1}{n}\sum_{i=1}^n z_{i, y_i} = \frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \hat q_{ik}\, z_{ik} $$ {#eq-stationary}

that is, the mean true-class logit equals the mean logit averaged under the fitted probabilities. {#c-ts-stationary}
Guo et al. solve the same one-dimensional convex problem with conjugate gradients in about 10 iterations. {#c-guo-convex}
The file `derivations/temperature-scaling.md` derives these four facts and checks them numerically.

**The Brier score splits into calibration and sharpness.** For the top-label confidence,

$$ \mathbb{E}\big[(C - A)^2\big] = \mathbb{E}\big[(C - r(C))^2\big] + \mathbb{E}\big[r(C)\,(1 - r(C))\big] $$ {#eq-brier}

The first term is the squared calibration error and vanishes exactly for a calibrated model; the second depends only on how well the confidence separates right from wrong predictions. {#c-brier-population}
On a sample with bins, the same split holds up to two within-bin terms, the variance of the confidences and their covariance with correctness. {#c-brier-sample}
With the same bins, the ECE is at most the root-mean-square calibration error, by the Cauchy-Schwarz inequality. {#c-ece-le-rms}
`derivations/brier-decomposition.md` has the proofs and an exact check in rationals.

**The binned ECE is biased in both directions.** Binning hides errors that cancel within a bin: the population binned error is at most $\mathrm{CE}$. {#c-binning-lower}
On a fixed sample, splitting bins never lowers the ECE, so more bins measure more. {#c-refine}
A finite sample pulls the other way: given the confidences, the expected sample ECE is at least the ECE computed with the true accuracies $r(c_i)$, so even a perfectly calibrated model has a positive expected ECE. {#c-sampling-bias}
For the squared gap, the bias is exact: the expected squared gap of a bin of $k$ members is the true squared gap plus $\sum r(c_i)(1 - r(c_i)) / k^2$. {#c-squared-bias}
For a calibrated bin of $k$ predictions at confidence $p$, de Moivre's formula gives the expected absolute gap, about $\sqrt{2p(1-p)/(\pi k)}$; at $p = 0.8$ it is 0.0966, 0.0318 and 0.0101 for $k = 10$, 100 and 1,000. {#c-bias-size}
`derivations/ece-estimator.md` proves these and computes the expected ECE of a calibrated model exactly:

| $n$ | true CE | 15 bins | 100 bins |
| ---: | ---: | ---: | ---: |
| 100 | 0 | 0.090 | 0.229 |
| 1,000 | 0 | 0.028 | 0.070 |
| 10,000 | 0 | 0.0089 | 0.022 |

The confidences are spread evenly over $[0.5, 1]$ and each outcome is drawn with probability equal to its confidence. {#c-bias-table}
The bias falls roughly as $\sqrt{M / n}$.
An ECE of 2% from 1,000 predictions with 15 bins is within the noise of a calibrated model.

## Worked example

Eight predictions over three classes, with $M = 5$ bins of width 0.2:

| bin $I_m$ | confidences (correct?) | $\lvert B_m \rvert$ | acc | conf | gap |
| --- | --- | ---: | ---: | ---: | ---: |
| (0, 0.2] | none | 0 | | | |
| (0.2, 0.4] | 0.40 (yes) | 1 | 1.000 | 0.400 | 0.600 |
| (0.4, 0.6] | 0.45 (yes), 0.50 (no) | 2 | 0.500 | 0.475 | 0.025 |
| (0.6, 0.8] | 0.70 (yes), 0.65 (no) | 2 | 0.500 | 0.675 | 0.175 |
| (0.8, 1] | 0.90 (yes), 0.85 (yes), 0.95 (no) | 3 | 0.667 | 0.900 | 0.233 |

The ECE is $(1 \cdot 0.600 + 2 \cdot 0.025 + 2 \cdot 0.175 + 3 \cdot 0.2\overline{3}) / 8 = 1.7 / 8 = 0.2125$, and the MCE is 0.600.
The accuracy is $5/8 = 0.625$ and the mean confidence 0.675.
With a single bin, the ECE is $|0.625 - 0.675| = 0.05$; with 15 bins, every prediction sits in a bin of its own, the ECE becomes the mean of $|a_i - c_i|$, and it rises to 0.475.
The same data give three different numbers, which is why the bin count belongs next to any ECE.
The row `worked-example` of `vectors/edge.jsonl` holds the full probability vectors.

## Reference implementation

`impl/numpy/expected_calibration_error.py` follows eq-ece-hat with equal-width bins and edges computed as $m/M$ in float64.
A confidence equal to an edge goes to the bin below it, and a confidence of 0 to the first bin.
`score` returns the ECE, the MCE, the accuracy, the mean confidence and the count, accuracy and mean confidence of each bin, which are the points of a reliability diagram; `reliability_bins` returns the raw sums.
`softmax`, `nll` and `fit_temperature` implement eq-temperature.
`fit_temperature` bisects on the sign of the derivative of the NLL in $\beta = 1/T$, which is valid because the NLL is convex in $\beta$, and returns an end of $[0.05, 100]$ when the minimum lies outside.
The PyTorch version does the same in `torch.float64`.
Both reproduce the 10 rows of `vectors/edge.jsonl` within a relative $10^{-9}$; the rows are exact rationals rounded once.

> [!pitfall]
> **Edges in floating point.** $0.3$ and $3/10$ computed as `3 / 10` are the same float, but `0.1 * 3` is $0.30000000000000004$ and lands in the next bin. Binning with `floor(c * M)` also moves values that sit on an edge. The test `test_bin_edges` pins every edge for several $M$.

> [!pitfall]
> **Comparing ECEs across papers.** Guo et al. use 15 equal-width bins, Kadavath et al. 10 equal-mass bins, and Minderer et al. 100 equal-mass bins. {#c-kadavath-ece} {#c-minderer-setup} These are different estimators with different biases, and the row `many-bins` shows the worked example at 100 bins.

> [!pitfall]
> **Bins can hide miscalibration.** In the row `binning-cancels-2`, a bin holds 16 predictions whose overconfidence and underconfidence cancel, so the ECE is 0; with 8 bins it is 0.25. The test `test_properties` checks that refining bins never lowers the ECE.

> [!pitfall]
> **Fitting $T$ on the test set.** Temperature must be fitted on held-out data; fitting it on the evaluation set lowers the ECE by fitting noise. Minderer et al., for example, fit it on 20% of the ImageNet validation set and evaluate on the rest. {#c-minderer-holdout}

## Interpretation

An ECE of 0 means no measured gap at this binning and sample size, not that the model is calibrated.
A small ECE says nothing about accuracy: a model that always predicts the majority class with confidence equal to its frequency has an ECE near zero.
So the ECE is read next to the accuracy, or together with a proper score such as the NLL or the Brier score of eq-brier.

Measured values span a wide range.
Guo et al. report that most modern networks they test have an ECE of about 4 to 10% before calibration. {#c-guo-typical}
A 110-layer ResNet on CIFAR-100 goes from 16.53% to 1.26% with temperature scaling, and a 152-layer ResNet on ImageNet from 5.48% to 1.86%. {#c-guo-table1}
Reuters is the exception in their study: its ECE is 0.85% before temperature scaling and 0.91% after. {#c-guo-reuters}
Label smoothing in training can reach a similar ECE without a temperature: a ResNet-56 on CIFAR-100 goes from 0.150 to 0.021 with temperature scaling and 0.024 with label smoothing. {#c-muller-ls}

For language models, the confidence is the probability of an answer option.
Kadavath et al. find that large models are well calibrated on multiple-choice and true/false questions when the options are shown and labelled with letters. {#c-kadavath-mc}
Calibration improves with model size and with few-shot prompts. {#c-kadavath-size}
Their RLHF policies look badly calibrated, but one temperature, $T = 2.5$, mostly fixes this. {#c-kadavath-rlhf}
The GPT-4 report measures an ECE of 0.007 for the pre-trained model on a subset of MMLU and 0.074 after post-training. {#c-gpt4}

## Limitations and criticism

The ECE depends on the number and kind of bins, and the derivation above shows biases in both directions.
In Kumar et al., the binned model's calibration error is at most the model's own, and in one example binning reports 0 for a model whose error is at least 0.49. {#c-kumar-underestimate} They also measure more error as they add bins. {#c-kumar-more-bins}
They argue that scaling methods such as temperature scaling are less calibrated than reported, and that the binned estimate cannot tell how miscalibrated they are. {#c-kumar-less}
Their debiased estimator subtracts an estimate of the variance of each bin's mean label from its squared gap, and needs a number of samples proportional to the square root of the bin count, where the plugin estimate needs a number proportional to the bin count. {#c-kumar-debiased}
Minderer et al. call the two effects the binning bias, always negative, and the finite-sample bias, always positive. {#c-minderer-two-biases}
They find that rankings of models change with the ECE variant, and that different bin sizes can lead to opposite conclusions. {#c-minderer-ranking}

Top-label ECE ignores the probabilities of the other classes, so a model can have a low ECE with poorly calibrated probabilities for the second-ranked class.
It averages over the whole input distribution, and the error under distribution shift can be much larger: Kadavath et al. find their P(IK) predictor poorly calibrated on tasks it was not trained on. {#c-kadavath-pik}
Finally, the ECE can be lowered without any gain in usefulness, for example by predicting the base rate, which is why it is reported with accuracy.

## Variants

| option | values | effect | used by |
| --- | --- | --- | --- |
| bins | 15 equal-width; 10 or 100 equal-mass | equal-mass bins avoid empty bins; more bins add upward bias | Guo et al.; Kadavath et al.; Minderer et al. |
| norm | $\ell_1$ (ECE), $\ell_2$ (RMS), $\ell_\infty$ (MCE) | RMS is at least ECE; MCE is the worst bin | Guo et al. (ECE, MCE); Kadavath et al. (RMS) |
| debiasing | plugin; subtract per-bin variance | removes the $1/k$ bias of squared gaps | Kumar et al. |
| classes | top label; per class; marginal | per-class versions check all probabilities | Kumar et al.; Minderer et al. (top label) |
| fix | temperature; vector or matrix scaling; histogram binning; isotonic | temperature keeps accuracy and has one parameter | Guo et al. |

`vectors/variants.yaml` lists the rows for each bin count that the implementations support.

## Implementations

## History

Calibration of probability forecasts was studied in meteorology long before machine learning, and reliability diagrams come from that tradition.
Before 2017, neural networks were often taken to be well calibrated.
Guo et al. (2017) reported that modern networks, unlike those of a decade earlier, are poorly calibrated, and that depth, width, weight decay and batch normalization affect calibration. {#c-guo-modern}
They made the 15-bin ECE and temperature scaling standard, and found that temperature scaling beat the other post-hoc methods they tried on images, including histogram binning, isotonic regression and vector and matrix scaling. {#c-guo-ts-best}
Because vector scaling learned nearly constant entries, no different from a single temperature, they concluded that miscalibration is intrinsically low-dimensional. {#c-guo-low-dim}
Kumar et al. (2019) questioned how well the binned ECE measures calibration and proposed a debiased estimator.
Minderer et al. (2021) revisited the question for newer models and found the most recent ones, especially non-convolutional ones, among the best calibrated. {#c-minderer-best}
In their families, in-distribution calibration worsens slightly with size, but accuracy gains outweigh it, and under shift calibration improves with size. {#c-minderer-trends}
From 2022, the ECE became a standard measure for language models, with Kadavath et al. (2022) and the GPT-4 report (OpenAI, 2023) applying it to multiple-choice answers.

## References
