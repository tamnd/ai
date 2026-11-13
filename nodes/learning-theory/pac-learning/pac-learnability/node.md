---
tag: 2NT5ZY
kind: concept
slug: pac-learnability
name: PAC learnability
aliases: [PAC learning, probably approximately correct learning, the PAC model, PAC learnable]
area: learning-theory/pac-learning
level: core
summary: >-
  A hypothesis class is PAC learnable when some algorithm, from a number of examples that depends only on ε and δ, returns with probability at least 1 − δ a hypothesis with error at most ε.
uses: [hypothesis-class@al, generalization-error@al, probability-measure@m2, independence-of-random-variables@m2]
related_to: [agnostic-pac-learning@al, vc-dimension@al, empirical-risk-minimization@al, no-free-lunch-theorem-learning@al, fundamental-theorem-of-statistical-learning@al, occams-razor@al]
history:
  year: 1984
  people: [Leslie Valiant]
  note: "Valiant (1984), A Theory of the Learnable, Communications of the ACM 27(11); the agnostic form is from Haussler (1992)."
prereqs: [hypothesis-class@al, generalization-error@al, probability-measure@m2, independence-of-random-variables@m2]
same_as:
  wikipedia: Probably_approximately_correct_learning
  al: pac-learning
sources:
  - {paper: "Valiant (1984)", title: "A Theory of the Learnable", locator: "Communications of the ACM 27(11), 1134 to 1142, Sections 1 to 5", note: "the original definition, with polynomial time and learning from examples"}
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Chapter 3, pp. 43 to 50", note: "the definitions of PAC and agnostic PAC learnability used here, the sample complexity and the bibliographic remarks"}
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Chapters 5 and 6, pp. 61 to 78", note: "the no-free-lunch theorem, the thresholds and the fundamental theorem"}
  - {book: "Shalev-Shwartz and Ben-David (2014), Understanding Machine Learning", locator: "Sections 8.2.4 and 8.3, pp. 107 to 108", note: "3-term DNF, proper learning and representation independent learning"}
  - {book: "Kearns and Vazirani (1994), An Introduction to Computational Learning Theory", locator: "Chapter 1", note: "the PAC model with the computational requirement"}
  - {paper: "Haussler (1992)", title: "Decision Theoretic Generalizations of the PAC Model for Neural Net and Other Learning Applications", locator: "Information and Computation 100(1), 78 to 150", note: "the agnostic setting with general losses"}
  - {paper: "Blumer, Ehrenfeucht, Haussler and Warmuth (1989)", title: "Learnability and the Vapnik-Chervonenkis Dimension", locator: "Journal of the ACM 36(4), 929 to 965", note: "learnability of infinite classes through the VC dimension"}
  - {paper: "Vapnik and Chervonenkis (1971)", title: "On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities", locator: "Theory of Probability and Its Applications 16(2), 264 to 280", note: "uniform convergence, the statistical root of the theory"}
---

PAC learnability is the standard definition of what it means to learn a class of rules from random examples.
A class is PAC learnable, for probably approximately correct, when some algorithm, given enough examples, returns a rule whose error is at most $\epsilon$ with probability at least $1 - \delta$, for every distribution of the data.
The definition turns the question "can this be learned" into a question about counting examples, which the VC dimension answers.

## In plain words

You want to learn a rule, such as which emails are spam, from examples that arrive at random.
You cannot hope to be exactly right, because the examples may miss rare cases, so you allow a small error $\epsilon$.
You cannot even be sure of that, because the examples may be unlucky, so you allow a small chance $\delta$ of failure.
A class of rules is PAC learnable when one fixed method meets any such $\epsilon$ and $\delta$ from a number of examples that depends only on them, whatever the distribution of the emails is.
"Approximately correct" is the $\epsilon$, and "probably" is the $\delta$.

## The idea

A learner sees a sample $S$ of $m$ points drawn independently from an unknown distribution $\mathcal{D}$ and labeled by an unknown rule $f$, and it returns a rule $h$.
Its error is the probability $L_{\mathcal{D}, f}(h)$ that $h$ disagrees with $f$ on a fresh point from $\mathcal{D}$.
Since $S$ is random, so is $h$, and the most that can be asked is that the error is small with high probability.
PAC learnability asks this for a class $\mathcal{H}$ of candidate rules, uniformly: one sample size $m_{\mathcal{H}}(\epsilon, \delta)$ must work for every distribution and every target that some rule of $\mathcal{H}$ fits.

The smallest example is a threshold on the line.
The class holds the rules $h_a(x) = 1$ when $x < a$, for every real $a$, so it is infinite.
A learner that returns a threshold consistent with the sample has error at most $\epsilon$ with probability at least $1 - \delta$ once $m \ge \lceil \log(2/\delta)/\epsilon \rceil$, whatever the distribution is. {#c-thresholds}
For $\epsilon = 0.1$ and $\delta = 0.05$, this is $\lceil \log 40 / 0.1 \rceil = \lceil 36.9 \rceil = 37$ examples.
The bound does not depend on $\mathcal{D}$, which is what the definition demands.

The field needs the definition because it separates two questions that the word "learning" mixes.
One is statistical: how many examples a class needs.
The other is computational: how long it takes to find a good rule from them.
Valiant's definition asks for both, and the textbook form of Shalev-Shwartz and Ben-David keeps only the first and treats the second apart.

## Formal statement

Let $\mathcal{X}$ be a domain and $\mathcal{H}$ a class of functions from $\mathcal{X}$ to $\{0, 1\}$.
For a distribution $\mathcal{D}$ on $\mathcal{X}$ and a labeling function $f : \mathcal{X} \to \{0, 1\}$, the true error of $h$ is $L_{\mathcal{D}, f}(h) = \Pr_{x \sim \mathcal{D}}[h(x) \neq f(x)]$.
The realizability assumption holds for $\mathcal{H}$, $\mathcal{D}$ and $f$ when some $h^\star \in \mathcal{H}$ has $L_{\mathcal{D}, f}(h^\star) = 0$.

**PAC learnability.**
$\mathcal{H}$ is PAC learnable when there exist a function $m_{\mathcal{H}} : (0, 1)^2 \to \mathbb{N}$ and a learning algorithm $A$ with this property: for every $\epsilon, \delta \in (0, 1)$, every distribution $\mathcal{D}$ on $\mathcal{X}$ and every $f$ for which the realizability assumption holds, when $A$ runs on $m \ge m_{\mathcal{H}}(\epsilon, \delta)$ examples drawn i.i.d. from $\mathcal{D}$ and labeled by $f$, it returns a hypothesis $h$ with

$$ \Pr_{S \sim \mathcal{D}^m}\left[ L_{\mathcal{D}, f}(A(S)) \le \epsilon \right] \ge 1 - \delta . $$ {#eq-pac}

In words, on every realizable task the algorithm returns, with probability at least $1 - \delta$ over the sample, a hypothesis whose error is at most $\epsilon$, once it has $m_{\mathcal{H}}(\epsilon, \delta)$ examples. {#c-pac-definition}

The sample complexity of $\mathcal{H}$ is the smallest such function: $m_{\mathcal{H}}(\epsilon, \delta)$ is the least integer for which some algorithm meets eq-pac. {#c-sample-complexity}

**Agnostic PAC learnability.**
Drop the labeling function and let $\mathcal{D}$ be a distribution on $\mathcal{X} \times \{0, 1\}$, with $L_{\mathcal{D}}(h) = \Pr_{(x, y) \sim \mathcal{D}}[h(x) \neq y]$.
$\mathcal{H}$ is agnostic PAC learnable when there exist $m_{\mathcal{H}}$ and $A$ such that for every $\epsilon, \delta \in (0, 1)$ and every $\mathcal{D}$, on $m \ge m_{\mathcal{H}}(\epsilon, \delta)$ i.i.d. examples,

$$ \Pr_{S \sim \mathcal{D}^m}\left[ L_{\mathcal{D}}(A(S)) \le \min_{h' \in \mathcal{H}} L_{\mathcal{D}}(h') + \epsilon \right] \ge 1 - \delta . $$ {#eq-agnostic-pac}

In words, on every distribution the error of the algorithm is within $\epsilon$ of the smallest error in the class, with probability at least $1 - \delta$. {#c-agnostic-definition}

With a general loss $\ell : \mathcal{H} \times Z \to \mathbb{R}_{\ge 0}$, the same definition holds with $L_{\mathcal{D}}(h) = \mathbb{E}_{z \sim \mathcal{D}}[\ell(h, z)]$.
Here $\log$ is the natural logarithm.

> [!note]
> The definitions follow Shalev-Shwartz and Ben-David (2014), Definitions 3.1, 3.3 and 3.4. Valiant (1984) writes $h^{-1}$ for both $\epsilon$ and $\delta$, asks that the algorithm run in time polynomial in $h$ and in the size of the target, and asks for a program that never outputs 1 where the target outputs 0, with a protocol of positive examples and an oracle. Kearns and Vazirani (1994) keep the computational requirement.

## Examples

**A finite class.**
Every finite class is PAC learnable by any algorithm that returns a consistent hypothesis, with $m_{\mathcal{H}}(\epsilon, \delta) \le \lceil \log(|\mathcal{H}|/\delta)/\epsilon \rceil$.
The proof, with a Lean proof of the realizable case, is [[pac-bound-finite-hypothesis-class]].
For $|\mathcal{H}| = 1000$, $\epsilon = 0.05$ and $\delta = 0.01$ this is 231 examples.

**Thresholds on the line.**
The class of thresholds is infinite but PAC learnable, with $m_{\mathcal{H}}(\epsilon, \delta) \le \lceil \log(2/\delta)/\epsilon \rceil$, as in the idea above.
So finiteness is not what decides learnability.
The thresholds have VC dimension 1, and the fundamental theorem of statistical learning says that a class of binary rules is PAC learnable exactly when its VC dimension is finite. {#c-fundamental-theorem}

**A non-example: all functions on an infinite domain.**
Let $\mathcal{X}$ be infinite and $\mathcal{H}$ the set of all functions from $\mathcal{X}$ to $\{0, 1\}$.
Then $\mathcal{H}$ is not PAC learnable. {#c-all-functions-not-learnable}
The reason is the no-free-lunch theorem: for any learner and any $m < |\mathcal{X}|/2$, some distribution with a target of zero error makes the learner's error at least $1/8$ with probability at least $1/7$. {#c-no-free-lunch}
On an infinite domain every $m$ is below $|\mathcal{X}|/2$, so no sample size meets $\epsilon < 1/8$ and $\delta < 1/7$.

**A borderline case: all functions on a finite domain.**
On a domain of $n = 100$ points, the class of all functions is finite, with $2^{100}$ members, so it is PAC learnable.
The finite class bound gives $\lceil (100 \log 2 + \log 20)/0.1 \rceil = 724$ examples for $\epsilon = 0.1$ and $\delta = 0.05$.
The no-free-lunch theorem shows that some growth with $n$ cannot be avoided: below $n/2 = 50$ examples, no learner meets $\epsilon < 1/8$ and $\delta < 1/7$ on every distribution.
The class is learnable, but only by seeing a large part of the domain, which is memorizing.

## Related ideas

- **Agnostic PAC learnability** ([[al:agnostic-pac-learning]]) drops realizability and compares the learner with the smallest error in the class, not with zero error; a class that is agnostic PAC learnable is PAC learnable, and for binary classification the converse holds too.
- **Uniform convergence** asks that the empirical error of every rule of the class be close to its true error at once; it implies agnostic PAC learnability by empirical risk minimization, with $m_{\mathcal{H}}(\epsilon, \delta) \le m^{UC}_{\mathcal{H}}(\epsilon/2, \delta)$. {#c-uc-implies-pac}
- **The VC dimension** ([[al:vc-dimension]]) is a number attached to a class, while PAC learnability is a yes or no property with a rate; the fundamental theorem says that the first is finite exactly when the second holds, and gives the rate.
- **Empirical risk minimization** ([[al:empirical-risk-minimization]]) is an algorithm, while PAC learnability is a property of a class; for binary classification a class is PAC learnable exactly when ERM learns it.
- **Occam's razor** ([[al:occams-razor]]) in the form of Blumer et al. (1987) is a sufficient condition: a learner whose consistent output can be written in few bits PAC learns.
- **Generalization error** ([[al:generalization-error]]) is the quantity $L_{\mathcal{D}}(h)$ for one rule; PAC learnability bounds it for the rule that the learner returns, for every distribution.
- **Efficient PAC learnability**, Valiant's original notion, adds that the learner runs in time polynomial in $1/\epsilon$, $1/\delta$ and the size of the problem.

## Common confusions

**PAC learnable is not the same as learnable in polynomial time.**
Valiant's definition requires an algorithm that runs in polynomial time, and the definition above has no such requirement. {#c-runtime-difference}
The two come apart: the class of 3-term DNF formulas over $n$ variables has at most $3^{3n}$ members, so ERM needs at most $(3n \log 3 + \log(1/\delta))/\epsilon$ examples, but unless RP = NP no polynomial time algorithm learns it properly. {#c-3dnf-hard}
It is learnable in polynomial time by a learner that returns a 3-CNF formula, which may lie outside the class. {#c-3dnf-improper}

**Proper and improper learning.**
Definition 3.1 lets the algorithm return any hypothesis, while Definition 3.4 asks for $h \in \mathcal{H}$.
Returning a rule outside the class is called representation independent learning, or improper learning, and the 3-term DNF case shows that it can change what is efficiently learnable. {#c-proper-improper}

**$\epsilon$ and $\delta$ are different requirements.**
$\epsilon$ bounds the error of the returned rule on fresh data, and $\delta$ bounds the probability that the sample is so unlucky that the learner fails.
Neither can be zero: a sample may hold one point drawn over and over, and a finite sample can miss a rare point. {#c-both-needed}
In the bounds, the sample size grows as $1/\epsilon$ or $1/\epsilon^2$ but only as $\log(1/\delta)$, so high confidence costs less than high accuracy.

**"For every distribution" is part of the definition.**
The sample size may depend on $\epsilon$, $\delta$ and the class, but not on $\mathcal{D}$.
A class that is learned well on one fixed distribution, such as the uniform distribution on the cube, is not shown to be PAC learnable by that alone.
The no-free-lunch theorem is a statement about this quantifier: for every learner there is a bad distribution.

**Realizable and agnostic results do not have the same rates.**
In the realizable case the sample complexity of a class of VC dimension $d$ is of order $(d + \log(1/\delta))/\epsilon$, up to a factor $\log(1/\epsilon)$ in the upper bound, while in the agnostic case it is of order $(d + \log(1/\delta))/\epsilon^2$. {#c-rates}
A bound quoted without saying which case it is in can be off by a factor of $1/\epsilon$.

## History

Leslie Valiant introduced the model in "A Theory of the Learnable", in 1984. {#c-valiant-1984}
He asked for an algorithm that, for every target of the class and every distribution, deduces with probability at least $1 - h^{-1}$ a program that errs rarely, in time polynomial in $h$ and the size of the target, and he showed that bounded CNF expressions are learnable in this sense.
The statistical side has an older root in the uniform convergence results of Vapnik and Chervonenkis (1971).
Blumer, Ehrenfeucht, Haussler and Warmuth tied the two in 1989, by showing that a class is learnable from a number of examples independent of the distribution exactly when its VC dimension is finite.
David Haussler extended the model in 1992 to the agnostic setting and to general losses, which is the form that covers regression and neural networks.
The two uses of the term survive: in computational learning theory, after Valiant, PAC learnable often includes polynomial time.
Shalev-Shwartz and Ben-David (2014) instead treat the number of examples and the computation as two separate resources, and study the first before they turn to the second. {#c-two-uses}
