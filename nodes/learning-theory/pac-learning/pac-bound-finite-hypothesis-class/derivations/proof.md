---
id: proof
derives: [c-realizable-bound, c-sample-complexity-realizable, c-union-bound-step, eq-realizable-m]
kind: proof
assumes: ["H is a finite class of functions from X to {0, 1}", "some hypothesis of H has true error 0", "the sample is m points drawn i.i.d. from D and labeled by f", "f and every hypothesis of H are measurable", "epsilon > 0 and delta > 0", "the learner returns a hypothesis of H that fits the sample whenever one exists"]
uses: [booles-inequality@m2, probability-of-the-complement@m2, mutual-independence-of-events@m2]
status: full
lean: AI.LearningTheory.PacBoundFiniteHypothesisClass.pac_bound_realizable
sources:
  - {work: shalev-shwartz-ben-david-2014-understanding-machine-learning, locator: "pp. 39 to 41"}
---

### Statement

Let $\mathcal{D}$ be a probability distribution on a measurable space $\mathcal{X}$, let $f : \mathcal{X} \to \{0, 1\}$ be a measurable labeling function, and let $\mathcal{H}$ be a finite class of measurable functions from $\mathcal{X}$ to $\{0, 1\}$.
Write $L(h) = \mathcal{D}(\{x : h(x) \neq f(x)\})$ for the true error of $h$.
A sample is a point $S = (x_1, \dots, x_m)$ of $\mathcal{X}^m$, drawn from the product measure $\mathcal{D}^m$, and $h$ fits $S$ when $h(x_i) = f(x_i)$ for $i = 1, \dots, m$.
Let $\epsilon > 0$ and $\delta > 0$, and let $m$ be a natural number with $m \ge \log(|\mathcal{H}|/\delta)/\epsilon$, as in eq-realizable-m.
Assume that some $h^\star \in \mathcal{H}$ has $L(h^\star) = 0$, and that the learner $A$ maps each sample to a hypothesis $A(S) \in \mathcal{H}$ that fits $S$ whenever some hypothesis of $\mathcal{H}$ fits $S$.
Then

$$ \mathcal{D}^m(\{S : L(A(S)) > \epsilon\}) \le \delta . $$

An ERM learner in the realizable case is such a learner: a hypothesis that fits $S$ has empirical error 0, the smallest value, so every ERM hypothesis fits $S$ whenever one exists.
So the result is c-realizable-bound, with the complement event $L(h_S) \le \epsilon$ of probability at least $1 - \delta$, and taking the smallest integer $m$ allowed gives c-sample-complexity-realizable.

### Assumptions

1. $\mathcal{H}$ is finite. It is used in step 5, where the union bound adds one term for each hypothesis, and the sum is $|\mathcal{H}|$ times a constant.
2. Some $h^\star \in \mathcal{H}$ has $L(h^\star) = 0$ (realizability). It is used in step 7.
3. The points $x_1, \dots, x_m$ are independent and each has the distribution $\mathcal{D}$, that is, $S$ has the product measure $\mathcal{D}^m$. It is used in step 3.
4. $f$ and every $h \in \mathcal{H}$ are measurable. It is used in steps 2, 3 and 7, so that the sets $\{x : h(x) = f(x)\}$ and $\{S : h \text{ fits } S\}$ have probabilities. On a finite or countable domain with all subsets measurable it holds for every function.
5. $\epsilon > 0$ and $\delta > 0$. They are used in step 6, which divides by $\epsilon$ and takes the logarithm of $|\mathcal{H}|/\delta$.
6. The learner returns a hypothesis of $\mathcal{H}$ that fits the sample whenever one exists. It is used in step 8.

Call $h$ bad when $L(h) > \epsilon$, and call a sample misleading when some bad $h \in \mathcal{H}$ fits it:

$$ M = \{S : \exists h \in \mathcal{H},\ L(h) > \epsilon \text{ and } h \text{ fits } S\} . $$

### Steps

1. For every real $t$, $1 + t \le e^t$, because $e^t$ is convex and $1 + t$ is its tangent line at $t = 0$. With $t = -\epsilon$ this gives $1 - \epsilon \le e^{-\epsilon}$. When $\epsilon \le 1$, both sides are at least 0, and raising both sides of an inequality between nonnegative numbers to the power $m$ keeps it, so $(1 - \epsilon)^m \le (e^{-\epsilon})^m = e^{-\epsilon m}$. {#step-one-minus-le-exp}
2. Fix a bad $h$. The set $A_h = \{x : h(x) = f(x)\}$ is the complement of $\{x : h(x) \neq f(x)\}$, which is measurable by assumption 4 and has probability $L(h)$. By the rule for the complement, $\mathcal{D}(A_h) = 1 - L(h)$. Since $L(h) > \epsilon$, $\mathcal{D}(A_h) < 1 - \epsilon$, and by step 1, $\mathcal{D}(A_h) \le 1 - \epsilon \le e^{-\epsilon}$. So a bad hypothesis agrees with $f$ on one random point with probability at most $e^{-\epsilon}$. {#step-one-draw}
3. The samples that $h$ fits form the product set $A_h \times A_h \times \dots \times A_h$, with $m$ factors, because $h$ fits $S$ exactly when each $x_i$ is in $A_h$. Under the product measure of assumption 3, the measure of a product of measurable sets is the product of their measures, so $\mathcal{D}^m(\{S : h \text{ fits } S\}) = \mathcal{D}(A_h)^m$. This holds for every $h$, bad or not. {#step-independence}
4. For a bad $h$, steps 2 and 3 give $\mathcal{D}^m(\{S : h \text{ fits } S\}) = \mathcal{D}(A_h)^m \le (e^{-\epsilon})^m = e^{-\epsilon m}$, because $0 \le \mathcal{D}(A_h) \le e^{-\epsilon}$ and the power $m$ keeps the order of nonnegative numbers. {#step-one-bad-hypothesis}
5. The misleading set is the union, over the bad $h \in \mathcal{H}$, of the sets $\{S : h \text{ fits } S\}$. A measure of a finite union is at most the sum of the measures (the union bound), and the bad hypotheses are at most $|\mathcal{H}|$ in number by assumption 1. Each term is at most $e^{-\epsilon m}$ by step 4, and a hypothesis that is not bad adds no term. So $\mathcal{D}^m(M) \le |\mathcal{H}|\, e^{-\epsilon m}$. This is c-union-bound-step. {#step-union-bound}
6. If $\mathcal{H}$ is empty the bound of step 5 is $0 \le \delta$. Otherwise $|\mathcal{H}|/\delta > 0$ by assumption 5, so its logarithm is defined. Multiplying $m \ge \log(|\mathcal{H}|/\delta)/\epsilon$ by $\epsilon > 0$ gives $\epsilon m \ge \log(|\mathcal{H}|/\delta)$, so $e^{-\epsilon m} \le e^{-\log(|\mathcal{H}|/\delta)} = \delta / |\mathcal{H}|$, since the exponential is increasing. Multiplying by $|\mathcal{H}| > 0$ gives $|\mathcal{H}|\, e^{-\epsilon m} \le \delta$. With step 5, $\mathcal{D}^m(M) \le \delta$. {#step-solve-for-m}
7. By assumption 2, $L(h^\star) = 0$, so $\mathcal{D}(A_{h^\star}) = 1 - 0 = 1$ by the rule for the complement, as in step 2. By step 3, $\mathcal{D}^m(\{S : h^\star \text{ fits } S\}) = 1^m = 1$. So the samples that $h^\star$ does not fit have probability 0, and almost every sample is fitted by $h^\star$. {#step-realizable}
8. Take a sample $S$ that $h^\star$ fits and on which $L(A(S)) > \epsilon$. Some hypothesis of $\mathcal{H}$ fits $S$, namely $h^\star$, so by assumption 6, $A(S) \in \mathcal{H}$ fits $S$. Since $L(A(S)) > \epsilon$, $A(S)$ is bad, so $S \in M$. So $\{S : L(A(S)) > \epsilon\}$ is contained in $M$ together with the set of samples that $h^\star$ does not fit, which has probability 0 by step 7. By monotonicity and the union bound, $\mathcal{D}^m(\{S : L(A(S)) > \epsilon\}) \le \mathcal{D}^m(M) + 0 \le \delta$, by step 6. {#step-reduce-to-misleading}

### Result and check

With probability at least $1 - \delta$ over the sample, the hypothesis that the learner returns has true error at most $\epsilon$, once $m \ge \log(|\mathcal{H}|/\delta)/\epsilon$.
This is Corollary 2.3 of Shalev-Shwartz and Ben-David (2014), pp. 40 to 41, whose proof follows the same steps: their Eq. 2.7 is the union bound of step 5, their Eq. 2.8 is the product of step 3, and their Eq. 2.9 is the bound of step 4.
The proof never uses $\delta < 1$; for $\delta \ge 1$ the conclusion holds for every $m$.
The proof also never uses the distribution beyond its product form, so the bound holds for every $\mathcal{D}$ and every $f$ that $\mathcal{H}$ realizes, which is what PAC learnability asks.

The Lean module `AI.LearningTheory.PacBoundFiniteHypothesisClass` proves the statement as `pac_bound_realizable`, with each step above marked by a `-- step:` comment in the same order.
In Lean the true error is a value in $[0, \infty]$, a sample is a function `Fin m → X` with the measure `Measure.pi`, and the learner is any `A` with `IsConsistentLearner H f A`.
The module also has an `example` that meets every hypothesis, with two equally likely points, the class $\{\mathrm{id}, \mathrm{not}\}$, $\epsilon = \delta = 1/2$ and $m = 3$, so the hypotheses are not vacuous.
The tests in `AITest` check the numbers of the illustration: $\lceil \log(1000/0.01)/0.05 \rceil = 231$, $1000\, e^{-0.05 \cdot 231} \le 0.01$ and $1000\, e^{-0.05 \cdot 230} > 0.01$, and the consistency facts of the thresholds on four points.
