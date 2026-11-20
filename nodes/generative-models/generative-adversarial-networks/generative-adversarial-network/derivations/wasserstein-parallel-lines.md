---
id: wasserstein-parallel-lines
derives: [c-parallel-lines]
kind: closed-form
assumes: ["P_0 and P_theta uniform on the segments {0} x [0, 1] and {theta} x [0, 1] of the plane"]
uses: [kullback-leibler-divergence@m2, total-variation-distance-between-probability-measures@m2]
status: full
check: wasserstein-parallel-lines.check.py
lean: null
sources:
  - {work: arjovsky-2017-wasserstein-gan, locator: "p. 4", note: "Example 1, learning parallel lines"}
---

### Statement

Let $Z \sim U[0, 1]$, $P_0$ the distribution of $(0, Z)$ and $P_\theta$ that of $(\theta, Z)$. Then

1. $W(P_0, P_\theta) = |\theta|$;
2. for $\theta \ne 0$, $\mathrm{JS}(P_0, P_\theta) = \log 2$, $\mathrm{KL}(P_0 \,\|\, P_\theta) = \mathrm{KL}(P_\theta \,\|\, P_0) = +\infty$ and $\delta(P_0, P_\theta) = 1$; at $\theta = 0$ all four are 0.

So as $\theta \to 0$ the sequence $P_\theta$ converges to $P_0$ in the Wasserstein distance and in no other of the four, and only the Wasserstein distance has a useful gradient in $\theta$, $\operatorname{sign}\theta$; the others are constant on $\theta \ne 0$.

### Assumptions

1. $W(P, Q) = \inf_{\pi \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \pi}\|x - y\|$ over couplings with marginals $P$ and $Q$. Used in step 1.
2. $\mathrm{JS}(P, Q) = \tfrac12\mathrm{KL}(P \,\|\, M) + \tfrac12\mathrm{KL}(Q \,\|\, M)$ with $M = \tfrac12(P + Q)$, and $\delta(P, Q) = \sup_A |P(A) - Q(A)|$. Used in step 2.

### Steps

1. Every point of $P_0$ has first coordinate 0 and every point of $P_\theta$ first coordinate $\theta$, so $\|x - y\| \ge |\theta|$ for every pair and every coupling costs at least $|\theta|$. The coupling $(0, Z) \mapsto (\theta, Z)$ moves every point horizontally by $|\theta|$ and costs exactly $|\theta|$. So $W = |\theta|$.
2. For $\theta \ne 0$ the supports are disjoint. $M$ has density $\tfrac12$ with respect to each of $P_0$ and $P_\theta$ on their supports, so $\mathrm{KL}(P_0 \,\|\, M) = \log 2$ and $\mathrm{JS} = \log 2$. $P_0$ gives mass 1 to a set of $P_\theta$-measure 0, so $P_0$ is not absolutely continuous with respect to $P_\theta$ and $\mathrm{KL}(P_0 \,\|\, P_\theta) = +\infty$, and symmetrically. Taking $A$ the support of $P_0$ gives $\delta = 1$, its maximum.

### Result and check

$$ W(P_0, P_\theta) = |\theta|, \qquad \mathrm{JS}(P_0, P_\theta) = \begin{cases} \log 2 & \theta \ne 0 \\ 0 & \theta = 0 \end{cases} . $$

The check `wasserstein-parallel-lines.check.py` checks that the horizontal coupling costs $|\theta|$ on samples, that random couplings cost more, and that the 1-Lipschitz critic $f(x) = -\operatorname{sign}(\theta)\, x_1$ attains $|\theta|$ in the dual of Kantorovich and Rubinstein, so the two bounds meet; it computes the JS divergence of thin smoothed segments on a grid, which is $\log 2$ once they separate, and the total variation of the same.
