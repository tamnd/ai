---
id: codebook-mean-and-ema
derives: [c-codebook-ema]
kind: closed-form
assumes: ["code i has at least one assigned vector", "0 <= gamma < 1"]
uses: [k-means-clustering@m2, vector-quantization@m2]
status: full
check: codebook-mean-and-ema.check.py
lean: null
sources:
  - {work: oord-2017-neural-discrete-representation-learning, locator: "p. 11", note: "the optimal code is the average of its encoder outputs, and the EMA updates of the counts and sums"}
---

### Statement

Let $z_{i,1}, \ldots, z_{i,n_i}$ be the vectors assigned to code $i$ with $n_i \ge 1$. Then $\sum_j \lVert z_{i,j} - e \rVert^2$ is minimized exactly at the mean $\bar z_i = \frac{1}{n_i}\sum_j z_{i,j}$. For the EMA updates

$$ N_i \leftarrow \gamma N_i + (1 - \gamma) n_i , \qquad m_i \leftarrow \gamma m_i + (1 - \gamma) \sum_j z_{i,j} , \qquad e_i = m_i / N_i , $$

started from $N_i = m_i = 0$, the first update gives $e_i = \bar z_i$, and if every batch has the same $n_i$ and the same sum, $e_i = \bar z_i$ after every update.

### Assumptions

1. $n_i \ge 1$, so the mean and the ratio exist. Used in steps 2 and 3.
2. $0 \le \gamma < 1$, so $1 - \gamma^t > 0$. Used in step 4.

### Steps

1. Write $\sum_j \lVert z_{i,j} - e \rVert^2 = \sum_j \lVert z_{i,j} - \bar z_i \rVert^2 + n_i \lVert \bar z_i - e \rVert^2$, since the cross term $2\sum_j (z_{i,j} - \bar z_i)\cdot(\bar z_i - e)$ vanishes because $\sum_j (z_{i,j} - \bar z_i) = 0$.
2. The first sum does not depend on $e$ and the second is zero exactly at $e = \bar z_i$, the unique minimizer. This is the k-means update of the code.
3. From zero, one update gives $N_i = (1 - \gamma) n_i$ and $m_i = (1 - \gamma)\sum_j z_{i,j}$, and the factor $1 - \gamma$ cancels in $m_i/N_i = \bar z_i$.
4. With the same $n_i$ and sum $s_i$ every batch, after $t$ updates $N_i = (1 - \gamma^t) n_i$ and $m_i = (1 - \gamma^t) s_i$ by the geometric sum, and the ratio is again $s_i/n_i = \bar z_i$.

### Result and check

An EMA of the counts and sums, divided, is a running k-means update with no start-up bias: the bias factor $1 - \gamma^t$ is common to the numerator and the denominator. An EMA of the codes themselves, $e_i \leftarrow \gamma e_i + (1 - \gamma)\bar z_i$, would instead start at $(1 - \gamma)\bar z_i$ from zero.
The check `codebook-mean-and-ema.check.py` compares `kmeans_step` with the minimizer found by perturbation, checks that one `ema_update` from `ema_init` gives the k-means means, that 50 updates on the same batch keep them, and that unused codes are unchanged.
