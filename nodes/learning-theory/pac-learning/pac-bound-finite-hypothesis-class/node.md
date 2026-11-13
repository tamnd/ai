---
tag: XSZ95D
kind: theorem
slug: pac-bound-finite-hypothesis-class
name: The PAC bound for a finite hypothesis class
aliases: [sample complexity of a finite hypothesis class, finite class learning bound, the union bound argument for finite classes]
area: learning-theory/pac-learning
level: core
summary: >-
  A finite hypothesis class is PAC learnable by empirical risk minimization: log(|H|/δ)/ε examples suffice when some hypothesis has zero error, and 2 log(2|H|/δ)/ε² in general.
uses: [pac-learnability, empirical-risk-minimization@al, booles-inequality@m2, hoeffdings-inequality@m2, mutual-independence-of-events@m2]
related_to: [occams-razor@al, vc-dimension@al, fundamental-theorem-of-statistical-learning@al, agnostic-pac-learning@al]
history:
  year: 1987
  people: [Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, Manfred K. Warmuth]
  note: "Valiant (1984), Section 4, bounds the number of examples with a lemma on independent trials for the classes of that paper; Blumer, Ehrenfeucht, Haussler and Warmuth (1987) state the bound for any learner that outputs a consistent hypothesis from a small class."
prereqs: [pac-learnability, hypothesis-class@al, empirical-risk-minimization@al, booles-inequality@m2, probability-of-the-complement@m2, mutual-independence-of-events@m2]
same_as:
  al: pac-learning
lean:
  module: AI.LearningTheory.PacBoundFiniteHypothesisClass
  statement: AI.LearningTheory.PacBoundFiniteHypothesisClass.pac_bound_realizable
sources:
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Section 2.3.1 and Corollary 2.3, pp. 37 to 41", note: "the realizable case, with the misleading samples and the union bound"}
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Corollary 3.2, p. 44; Section 4.2, Corollary 4.6 and Remark 4.1, pp. 55 to 57", note: "the sample complexity in the PAC definition, and the agnostic case through uniform convergence and Hoeffding's inequality"}
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Section 6.3.4 and Theorem 6.8, pp. 72 to 73", note: "the comparison with the VC dimension and the lower bound"}
  - {book: "Mohri, Rostamizadeh and Talwalkar (2018), Foundations of Machine Learning, second edition", locator: "Theorems 2.5 and 2.13", note: "the same bounds as learning bounds for the consistent and the inconsistent case"}
  - {paper: "Valiant (1984)", title: "A Theory of the Learnable", locator: "Communications of the ACM 27(11), Section 4", note: "the combinatorial bound behind the first learnability results"}
  - {paper: "Blumer, Ehrenfeucht, Haussler and Warmuth (1987)", title: "Occam's Razor", locator: "Information Processing Letters 24(6), 377 to 380", note: "the bound for consistent learners, stated in the length of the hypothesis"}
  - {book: "Kearns and Vazirani (1994), An Introduction to Computational Learning Theory", locator: "Chapter 2", note: "Occam's razor and the finite class bound"}
---

The PAC bound for a finite hypothesis class says how many examples a learner needs when it chooses from a finite list of rules.
If one rule makes no mistakes, any rule that fits the sample is probably approximately correct after about $\log(|\mathcal{H}|/\delta)/\epsilon$ examples.
The bound grows with the logarithm of the size of the list, not with the size itself.
It is the base case of sample complexity theory, which the VC dimension extends to infinite classes.

## Statement

Let $\mathcal{H}$ be a finite class of functions from a domain $\mathcal{X}$ to $\{0, 1\}$.
Let $\mathcal{D}$ be a distribution on $\mathcal{X}$ and $f : \mathcal{X} \to \{0, 1\}$ a labeling function.
The true error of $h$ is $L_{\mathcal{D}, f}(h) = \Pr_{x \sim \mathcal{D}}[h(x) \neq f(x)]$, and the empirical error $L_S(h)$ is the fraction of the sample $S$ on which $h$ and $f$ disagree.
An empirical risk minimization (ERM) hypothesis $h_S$ is any $h \in \mathcal{H}$ with the smallest $L_S(h)$.

**Realizable case.**
Assume that some $h^\star \in \mathcal{H}$ has $L_{\mathcal{D}, f}(h^\star) = 0$.
Let $\delta \in (0, 1)$, $\epsilon > 0$, and let $m$ be an integer with

$$ m \ge \frac{\log(|\mathcal{H}| / \delta)}{\epsilon} . $$ {#eq-realizable-m}

Then, with probability at least $1 - \delta$ over an i.i.d. sample $S$ of $m$ points drawn from $\mathcal{D}$ and labeled by $f$, every ERM hypothesis $h_S$ has $L_{\mathcal{D}, f}(h_S) \le \epsilon$. {#c-realizable-bound}
So a finite class is PAC learnable, with sample complexity $m_{\mathcal{H}}(\epsilon, \delta) \le \lceil \log(|\mathcal{H}|/\delta) / \epsilon \rceil$. {#c-sample-complexity-realizable}

**Agnostic case.**
Let $\mathcal{D}$ be any distribution on $\mathcal{X} \times \{0, 1\}$, with no labeling function, and let the loss take values in $[0, 1]$.
If

$$ m \ge \frac{\log(2 |\mathcal{H}| / \delta)}{2 \epsilon^2} , $$ {#eq-uniform-m}

then, with probability at least $1 - \delta$, $|L_S(h) - L_{\mathcal{D}}(h)| \le \epsilon$ holds for every $h \in \mathcal{H}$ at once. {#c-uniform-convergence}
With twice the accuracy, every ERM hypothesis then satisfies $L_{\mathcal{D}}(h_S) \le \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon$ once

$$ m \ge \frac{2 \log(2 |\mathcal{H}| / \delta)}{\epsilon^2} . $$ {#eq-agnostic-m}

So a finite class is agnostic PAC learnable, with sample complexity at most $\lceil 2 \log(2|\mathcal{H}|/\delta) / \epsilon^2 \rceil$. {#c-agnostic-bound}

Here $\log$ is the natural logarithm.

> [!note]
> The statement follows Shalev-Shwartz and Ben-David (2014), Corollaries 2.3, 3.2 and 4.6. They write $L_{(\mathcal{D}, f)}$ for the true error in the realizable case. Mohri et al. (2018) state the same results as Theorems 2.5 and 2.13, with the bound solved for the error instead of for $m$: in the realizable case, $L_{\mathcal{D}, f}(h_S) \le \frac{1}{m}\left(\log |\mathcal{H}| + \log \frac{1}{\delta}\right)$.

## In plain words

Call a rule bad if it is wrong on more than a small part of the inputs.
A bad rule that fits one random example is lucky, and a bad rule that fits many independent examples is lucky many times in a row, which gets rare fast.
There are only finitely many rules, so the chance that any bad rule fits the whole sample is at most the number of rules times that small chance.
Make the sample large enough and this total falls below the failure rate that you accept.
Then any rule that fits the sample is good, with high confidence.
The number of examples you need grows with the logarithm of the number of rules, so a list a thousand times longer costs only a few more examples.

## Proof

The full proof of the realizable case is in [derivations/proof.md](derivations/proof.md), with every step justified.
It has three parts.

1. **One bad hypothesis.** Fix $h$ with $L_{\mathcal{D}, f}(h) > \epsilon$. It agrees with $f$ on one draw with probability $1 - L_{\mathcal{D}, f}(h) < 1 - \epsilon$. The draws are independent, so it agrees on all $m$ of them with probability less than $(1 - \epsilon)^m \le e^{-\epsilon m}$, by $1 - \epsilon \le e^{-\epsilon}$.
2. **All bad hypotheses at once.** The samples on which some bad hypothesis fits the sample are the union of the samples on which each one does. By the union bound, they have probability at most $|\mathcal{H}| e^{-\epsilon m}$, and eq-realizable-m makes this at most $\delta$. {#c-union-bound-step}
3. **The learner.** Under realizability, $h^\star$ fits almost every sample, so the ERM hypothesis $h_S$ has $L_S(h_S) = 0$. If $h_S$ had error above $\epsilon$, the sample would be one of the samples of part 2. So the learner fails with probability at most $\delta$.

The agnostic case is in [derivations/agnostic.md](derivations/agnostic.md).
It replaces part 1 with Hoeffding's inequality, which bounds the chance that one empirical error is more than $\epsilon$ away from the true error by $2 e^{-2 m \epsilon^2}$, and keeps the union bound of part 2.

## Assumptions

1. **The class is finite.** The union bound of part 2 adds one term for each hypothesis. Without finiteness the sum can be infinite, and the conclusion can fail: the class of all functions on an infinite domain is not PAC learnable, by the no-free-lunch theorem (Shalev-Shwartz and Ben-David 2014, Corollary 5.2). Some infinite classes are learnable, as the thresholds on the real line, and the VC dimension decides which.
2. **The sample is i.i.d. from the distribution of the test.** Part 1 multiplies the probabilities of the $m$ draws. If the sample were $m$ copies of one point, a bad hypothesis would survive with probability $1 - L_{\mathcal{D}, f}(h)$, not $(1 - L_{\mathcal{D}, f}(h))^m$, and no sample size would help.
3. **Realizability, for the first bound.** It gives a hypothesis with zero empirical error, so that the ERM hypothesis fits the sample. Without it the first bound says nothing, and the agnostic bound pays $1/\epsilon^2$ in place of $1/\epsilon$.
4. **The learner returns a hypothesis of $\mathcal{H}$ that fits the sample.** The union bound covers only the hypotheses of $\mathcal{H}$. A learner that may return a rule outside $\mathcal{H}$ needs another argument.
5. **Measurability.** The events of the proof must have probabilities. This holds when each hypothesis and the labeling function are measurable, which is automatic on a finite or countable domain. The Lean proof states it as a hypothesis.

## Tightness

The step for one hypothesis is almost exact.
A hypothesis with error exactly $\epsilon$ fits the whole sample with probability exactly $(1 - \epsilon)^m$, and $e^{-\epsilon m}$ is close to it for small $\epsilon$.
For $\epsilon = 0.05$ and $m = 231$, the two are $7.1 \times 10^{-6}$ and $9.6 \times 10^{-6}$.

The union bound is where the bound can be loose.
It counts a sample once for each bad hypothesis that fits it, so it loses most when the bad hypotheses fail on the same points.
The thresholds on $k$ points form a class of size about $k$ but have VC dimension 1, and their sample complexity does not grow with $k$, while the bound grows as $\log k$ (Shalev-Shwartz and Ben-David 2014, Section 6.3.4).

For the classes that are as rich as their size allows, the bound is tight up to a constant.
The VC dimension $d$ of a finite class is at most $\log_2 |\mathcal{H}|$, and the fundamental theorem of PAC learning gives the lower bound $m_{\mathcal{H}}(\epsilon, \delta) \ge C_1 (d + \log(1/\delta))/\epsilon$ in the realizable case. {#c-lower-bound}
The class of all $2^d$ labelings of $d$ points has $\log_2 |\mathcal{H}| = d$, so its sample complexity is $\Theta((\log |\mathcal{H}| + \log(1/\delta))/\epsilon)$.
In the agnostic case the same theorem gives $\Theta((d + \log(1/\delta))/\epsilon^2)$, so the $1/\epsilon^2$ rate cannot be improved either.

## Consequences

The bound gives Occam's razor in the form of Blumer et al. (1987): a hypothesis that can be written in $b$ bits comes from a class of at most $2^b$ hypotheses, so $(b \log 2 + \log(1/\delta))/\epsilon$ examples are enough for a consistent learner that outputs it.
It also turns into a bound on the error for a fixed sample: in the realizable case, with probability at least $1 - \delta$, every consistent $h \in \mathcal{H}$ has error at most $(\log |\mathcal{H}| + \log(1/\delta))/m$.
Over an infinite class indexed by $n$ real numbers stored in 64-bit floats, the class has at most $2^{64 n}$ members, which gives a crude bound that grows linearly in $n$ (Shalev-Shwartz and Ben-David 2014, Remark 4.1).

## Illustration

Take a class of $|\mathcal{H}| = 1000$ hypotheses, the accuracy $\epsilon = 0.05$ and the confidence $\delta = 0.01$.

| Quantity | Formula | Value |
|---|---|---|
| realizable sample size | $\log(1000 / 0.01) / 0.05 = \log(10^5) / 0.05 = 11.513 / 0.05$ | 230.3, so 231 examples |
| check at 231 | $1000 \, e^{-0.05 \cdot 231}$ | 0.0096, at most 0.01 |
| check at 230 | $1000 \, e^{-0.05 \cdot 230}$ | 0.0101, above 0.01 |
| doubling the class | adds $\log 2 / 0.05$ | 13.9 more examples |
| dividing $\delta$ by 10 | adds $\log 10 / 0.05$ | 46.1 more examples |
| uniform convergence | $\log(2000 / 0.01) / (2 \cdot 0.05^2)$ | 2441.2 |
| agnostic sample size | $2 \log(2000 / 0.01) / 0.05^2$ | 9764.9, so 9765 examples |

The realizable learner needs about forty times fewer examples than the agnostic one here, because the bound goes as $1/\epsilon$ in place of $1/\epsilon^2$.

A small case shows the events of the proof.
Let $\mathcal{X} = \{0, 1, 2, 3\}$ with the uniform distribution, and let $\mathcal{H}$ hold the five thresholds $h_t(x) = 1$ when $x \ge t$, for $t = 0, \dots, 4$.
The target is $f = h_2$.
The sample $(0, 3)$ fits $h_1$, $h_2$ and $h_3$, so an ERM learner may return $h_1$, whose error is $1/4$, the mass of the point 1.
The sample $(1, 2)$ fits only $h_2$.
For $\epsilon = 0.2$, the bad hypotheses are $h_0$, $h_1$, $h_3$ and $h_4$, with errors $1/2$, $1/4$, $1/4$ and $1/2$.
A sample of two points is misleading when it fits one of them, that is, when it avoids the point 1 or avoids the point 2.
Of the 16 equally likely samples, 9 avoid the point 1, 9 avoid the point 2 and 4 avoid both, so 14 are misleading and the probability is $7/8$.
The union bound adds the four chances $1/4 + 9/16 + 9/16 + 1/4 = 26/16$, which is above 1: it counts the samples that fit $h_0$ and $h_1$ twice, and a sample of two points is far below the bound of eq-realizable-m.
The tests of the Lean module check these consistency facts on the same instance.

## History

Leslie Valiant defined learnability from examples in 1984, asking for a program found in polynomial time that, with high probability, never accepts a negative example and misses only a small share of the positive ones under the distribution of positive examples. {#c-valiant-definition}
His Section 4 bounds the number of examples by a lemma on independent trials, and his Section 5 uses it to show that, for each fixed $k$, the conjunctive normal form expressions with at most $k$ literals in each clause are learnable from examples. {#c-valiant-cnf}
Anselm Blumer, Andrzej Ehrenfeucht, David Haussler and Manfred Warmuth stated the counting argument for any learner that returns a short consistent hypothesis, in "Occam's Razor" (1987), and extended it to infinite classes through the VC dimension in 1989.
David Haussler gave the agnostic setting in 1992, and the bound through uniform convergence and Hoeffding's inequality became the form of the textbooks of Kearns and Vazirani (1994), Shalev-Shwartz and Ben-David (2014) and Mohri et al. (2018).
