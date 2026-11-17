---
id: ece-estimator
derives: [c-binning-lower, c-refine, c-sampling-bias, c-squared-bias, c-bias-size, c-bias-table, eq-ece, eq-ece-hat]
kind: bound
assumes: ["the confidence C is in [0, 1] and the correctness A is 0 or 1", "the bins are fixed before the data are seen", "given the confidences, the outcomes are independent with P(A_i = 1) = r(c_i)"]
uses: [bias-of-an-estimator@m2, binomial-distribution@m2, bernoulli-distribution@m2, conditional-expectation@m2]
status: full
check: ece-estimator.check.py
lean: null
sources:
  - {work: kumar-2019-verified-uncertainty-calibration, locator: "p. 5"}
  - {work: kumar-2019-verified-uncertainty-calibration, locator: "p. 8"}
  - {work: minderer-2021-revisiting-the-calibration-of-modern-neural, locator: "p. 9"}
---

### Statement

The target is the top-label calibration error $\mathrm{CE} = \mathbb{E}\,|r(C) - C|$ of eq-ece, with $r(c) = P(A = 1 \mid C = c)$. The binned ECE of eq-ece-hat estimates it from $n$ predictions in $M$ fixed bins. Two biases pull in opposite directions.

1. Binning bias, downwards. The population binned error $\sum_m P(C \in I_m)\,\big|\mathbb{E}[r(C) - C \mid C \in I_m]\big|$ is at most $\mathrm{CE}$. On a sample, splitting a bin into smaller bins never lowers the sample ECE.
2. Sampling bias, upwards. Given the confidences, the expected sample ECE is at least $\frac{1}{n}\sum_m \big|\sum_{i \in B_m} (r(c_i) - c_i)\big|$. In particular a perfectly calibrated model, $r(c) = c$, has a positive expected sample ECE as soon as one confidence lies strictly between 0 and 1.
3. For the squared gap of one bin with $k$ members, the bias is exact: $\mathbb{E}\big[(\bar a - \bar c)^2\big] = (\bar r - \bar c)^2 + \frac{1}{k^2}\sum_{i \in B} r(c_i)(1 - r(c_i))$.
4. Size. For a calibrated bin whose $k$ members all have confidence $p$, the expected gap is $\mathbb{E}|X/k - p|$ with $X \sim \mathrm{Binomial}(k, p)$, which de Moivre's formula gives as $\frac{2v}{k}\binom{k}{v} p^v (1 - p)^{k - v + 1}$ with $v = \lfloor kp \rfloor + 1$, close to $\sqrt{2p(1-p)/(\pi k)}$ for large $k$. More bins mean smaller $k$ and a larger upward bias.

### Assumptions

1. $C \in [0, 1]$ and $A \in \{0, 1\}$. Used in steps 3 to 5.
2. The bins are chosen without looking at the outcomes. Used in step 3, where the bin of each prediction is fixed.
3. Given the confidences, the outcomes $A_i$ are independent with $P(A_i = 1) = r(c_i)$, as when the evaluation examples are drawn independently. Used in steps 3 to 5.

### Steps

1. In bin $I_m$, $\big|\mathbb{E}[r(C) - C \mid C \in I_m]\big| \le \mathbb{E}\big[|r(C) - C| \mid C \in I_m\big]$ by Jensen's inequality for the absolute value. Multiplying by $P(C \in I_m)$ and summing over the bins gives $\mathbb{E}|r(C) - C|$ on the right, by the law of total expectation. This is the first half of part 1, Proposition 3.3 of Kumar et al. (2019). Errors of opposite sign inside a bin cancel: confidence 5/8 that is right 7/8 of the time and confidence 7/8 that is right 5/8 of the time, in equal shares, have $\mathrm{CE} = 1/4$ but a binned error of 0 when both share a bin.
2. On a sample, if bin $B$ is split into $B'$ and $B''$, then $\big|\sum_{B} (a_i - c_i)\big| \le \big|\sum_{B'} (a_i - c_i)\big| + \big|\sum_{B''} (a_i - c_i)\big|$ by the triangle inequality. Since the sample ECE is $\frac{1}{n}\sum_m \big|\sum_{i \in B_m} (a_i - c_i)\big|$, refining the bins never lowers it. With equal-width bins, $M$ bins are refined by $fM$ bins for every integer $f \ge 2$. This is the second half of part 1.
3. Fix the confidences. Bin $m$ contributes $\frac{1}{n}|S_m|$ with $S_m = \sum_{i \in B_m} (a_i - c_i)$, and $\mathbb{E}[S_m] = \sum_{i \in B_m} (r(c_i) - c_i)$ by assumption 3. Since $|\cdot|$ is convex, $\mathbb{E}|S_m| \ge |\mathbb{E} S_m|$ (Jensen). Summing gives part 2. When $r = c$, $\mathbb{E} S_m = 0$, but $S_m$ is not constant when some $c_i \in (0, 1)$, so $\mathbb{E}|S_m| > 0$.
4. For part 3, $\bar a - \bar c = (\bar r - \bar c) + (\bar a - \bar r)$, where $\bar a - \bar r$ has mean 0 and variance $\frac{1}{k^2}\sum r(c_i)(1 - r(c_i))$ by independence; the cross term has mean 0. This is the per-bin bias that Kumar et al. (2019) subtract in their debiased estimator, and that Minderer et al. (2021) write for the squared gap.
5. For part 4, $kp$ successes are expected and $X - kp$ has mean 0, so $\mathbb{E}|X - kp| = 2\,\mathbb{E}[(X - kp)^+] = 2\sum_{j \ge v} (j - kp)\binom{k}{j}p^j(1-p)^{k-j}$; the sum telescopes, using $j\binom{k}{j} = k\binom{k-1}{j-1}$, to $v\binom{k}{v}p^v(1-p)^{k-v+1}$. The normal approximation to $X$ gives $\mathbb{E}|X - kp| \approx \sqrt{2kp(1-p)/\pi}$. Dividing by $k$ gives the stated size.
6. Numerically, for a calibrated model with $n$ confidences spread evenly over $[0.5, 1]$ and outcomes drawn with $P(A_i = 1) = c_i$, summing $\mathbb{E}|S_m|$ over the bins exactly (by convolving the Bernoulli distributions of each bin) gives an expected ECE of 0.090 with 15 bins and 0.229 with 100 bins at $n = 100$; 0.028 and 0.070 at $n = 1{,}000$; and 0.0089 and 0.022 at $n = 10{,}000$. The true calibration error is 0 in every case. The bias shrinks roughly as $\sqrt{M/n}$.

### Result and check

The binned ECE is a biased estimator in both directions: binning hides errors that cancel inside a bin, and finite samples add noise that the absolute value turns into a positive bias, larger with more bins and fewer predictions. Neither bias is fixed by the choice of $M$ alone, which is why rankings of models can depend on the number of bins.
The check `ece-estimator.check.py` verifies part 1 on 500 random discrete distributions in exact rationals, the refinement on 500 random samples, part 2 by enumerating all outcomes of up to 7 predictions, part 3 exactly for bins of up to 6 predictions, de Moivre's formula against the direct sum for 300 random $(k, p)$ and at $p = 0.8$, and the table of step 6 both exactly and by 4,000 simulated evaluations at $n = 1{,}000$; it prints ok.
