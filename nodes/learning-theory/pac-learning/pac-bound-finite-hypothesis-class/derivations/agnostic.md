---
id: agnostic
derives: [c-uniform-convergence, c-agnostic-bound, eq-uniform-m, eq-agnostic-m]
kind: bound
assumes: ["H is a finite class", "the sample is m points drawn i.i.d. from a distribution D on Z", "the loss takes values in [0, 1]", "epsilon > 0 and 0 < delta"]
uses: [hoeffdings-inequality@m2, booles-inequality@m2, empirical-risk-minimization@al]
status: full
lean: null
sources:
  - {work: shalev-shwartz-ben-david-2014-understanding-machine-learning, locator: "pp. 54 to 57"}
---

### Statement

Let $\mathcal{D}$ be a probability distribution on a set $Z$ of examples, as $Z = \mathcal{X} \times \{0, 1\}$, let $\mathcal{H}$ be a finite class, and let $\ell : \mathcal{H} \times Z \to [0, 1]$ be a loss function.
Write $L_{\mathcal{D}}(h) = \mathbb{E}_{z \sim \mathcal{D}}[\ell(h, z)]$ and, for a sample $S = (z_1, \dots, z_m)$, $L_S(h) = \frac{1}{m} \sum_{i=1}^m \ell(h, z_i)$.
Let $\epsilon > 0$ and $\delta > 0$.

1. If $m \ge \log(2|\mathcal{H}|/\delta)/(2\epsilon^2)$, as in eq-uniform-m, then with probability at least $1 - \delta$ over $S \sim \mathcal{D}^m$, every $h \in \mathcal{H}$ has $|L_S(h) - L_{\mathcal{D}}(h)| \le \epsilon$. This is c-uniform-convergence.
2. If $m \ge 2\log(2|\mathcal{H}|/\delta)/\epsilon^2$, as in eq-agnostic-m, then with probability at least $1 - \delta$, every ERM hypothesis $h_S \in \operatorname{argmin}_{h \in \mathcal{H}} L_S(h)$ has $L_{\mathcal{D}}(h_S) \le \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon$. This is c-agnostic-bound.

### Assumptions

1. $\mathcal{H}$ is finite. It is used in step 3, where the union bound has one term for each hypothesis.
2. The examples $z_1, \dots, z_m$ are independent and each has the distribution $\mathcal{D}$. It is used in step 1, which needs independent variables with a common mean.
3. The loss takes values in $[0, 1]$. It is used in step 2, where the range $b - a = 1$ enters Hoeffding's inequality.
4. $\epsilon > 0$ and $\delta > 0$. They are used in step 4, which takes the logarithm of $2|\mathcal{H}|/\delta$ and divides by $\epsilon^2$.
5. The events of the proof are measurable, as in the realizable case; this holds when each $z \mapsto \ell(h, z)$ is measurable.

No assumption is made about how the labels are produced: there need not be a labeling function, and no hypothesis needs to have error 0.

### Steps

1. Fix $h \in \mathcal{H}$ before the sample is drawn, and let $\theta_i = \ell(h, z_i)$. By assumption 2 the $\theta_i$ are independent and identically distributed, since each is the same function of an independent draw. Each has mean $\mathbb{E}[\theta_i] = L_{\mathcal{D}}(h)$, and their average is $L_S(h)$. {#step-fixed-hypothesis}
2. By assumption 3, $0 \le \theta_i \le 1$. Hoeffding's inequality (Shalev-Shwartz and Ben-David 2014, Lemma 4.5) for $m$ independent variables in $[a, b]$ with mean $\mu$ gives $\Pr[|\frac{1}{m}\sum_i \theta_i - \mu| > \epsilon] \le 2\exp(-2m\epsilon^2/(b - a)^2)$. With $a = 0$ and $b = 1$, step 1 gives $\mathcal{D}^m(\{S : |L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\}) \le 2 e^{-2m\epsilon^2}$. {#step-hoeffding}
3. The samples on which some hypothesis is more than $\epsilon$ off are the union over $h \in \mathcal{H}$ of the sets of step 2. By the union bound and assumption 1, $\mathcal{D}^m(\{S : \exists h \in \mathcal{H},\ |L_S(h) - L_{\mathcal{D}}(h)| > \epsilon\}) \le \sum_{h \in \mathcal{H}} 2 e^{-2m\epsilon^2} = 2|\mathcal{H}|\, e^{-2m\epsilon^2}$. {#step-union-bound}
4. If $\mathcal{H}$ is empty the claims hold with nothing to check. Otherwise $2|\mathcal{H}|/\delta > 0$ by assumption 4. Multiplying $m \ge \log(2|\mathcal{H}|/\delta)/(2\epsilon^2)$ by $2\epsilon^2 > 0$ gives $2m\epsilon^2 \ge \log(2|\mathcal{H}|/\delta)$, so $e^{-2m\epsilon^2} \le \delta/(2|\mathcal{H}|)$ because the exponential is increasing, and $2|\mathcal{H}|\, e^{-2m\epsilon^2} \le \delta$. With step 3, the bad event has probability at most $\delta$, so its complement, on which every $h$ has $|L_S(h) - L_{\mathcal{D}}(h)| \le \epsilon$, has probability at least $1 - \delta$. This is part 1. {#step-solve-for-m}
5. Take a sample on which $|L_S(h) - L_{\mathcal{D}}(h)| \le \epsilon/2$ for every $h \in \mathcal{H}$, which Shalev-Shwartz and Ben-David call $\epsilon/2$-representative (Definition 4.1). Let $h_S$ be an ERM hypothesis and $h$ any hypothesis of $\mathcal{H}$. Then $L_{\mathcal{D}}(h_S) \le L_S(h_S) + \epsilon/2 \le L_S(h) + \epsilon/2 \le L_{\mathcal{D}}(h) + \epsilon/2 + \epsilon/2 = L_{\mathcal{D}}(h) + \epsilon$. The first and third inequalities are representativeness, and the second holds because $h_S$ minimizes $L_S$. Taking the minimum over $h$ gives $L_{\mathcal{D}}(h_S) \le \min_{h \in \mathcal{H}} L_{\mathcal{D}}(h) + \epsilon$. This is Lemma 4.2 of the same book. {#step-representative}
6. Apply part 1 with $\epsilon/2$ in place of $\epsilon$. It needs $m \ge \log(2|\mathcal{H}|/\delta)/(2(\epsilon/2)^2) = 2\log(2|\mathcal{H}|/\delta)/\epsilon^2$, which is eq-agnostic-m. Then with probability at least $1 - \delta$ the sample is $\epsilon/2$-representative, and step 5 gives the bound for every ERM hypothesis on it. This is part 2. {#step-agnostic}

### Result and check

A finite class has the uniform convergence property with $m^{UC}_{\mathcal{H}}(\epsilon, \delta) \le \lceil \log(2|\mathcal{H}|/\delta)/(2\epsilon^2) \rceil$, and ERM is an agnostic PAC learner for it with $m_{\mathcal{H}}(\epsilon, \delta) \le \lceil 2\log(2|\mathcal{H}|/\delta)/\epsilon^2 \rceil$.
This is Corollary 4.6 of Shalev-Shwartz and Ben-David (2014), p. 57, and its proof on pp. 55 to 57 follows the same steps: their Eq. 4.1 is the union bound of step 3 and their Eq. 4.2 is the bound of step 2.

The check is numeric.
For $|\mathcal{H}| = 1000$, $\epsilon = 0.05$ and $\delta = 0.01$, eq-uniform-m gives $\log(2 \cdot 10^5)/0.005 = 12.2061/0.005 = 2441.2$, and eq-agnostic-m gives four times as much, $9764.9$, as in the table of the illustration on the page.
At $m = 2442$, $2 \cdot 1000 \cdot e^{-2 \cdot 2442 \cdot 0.0025} = 2000\, e^{-12.21} = 0.00996 \le 0.01$.

There is no Lean proof of this part.
Mathlib states Hoeffding's inequality for sub-Gaussian variables, and a proof would follow steps 1 to 6 with the realizable module as a template.
