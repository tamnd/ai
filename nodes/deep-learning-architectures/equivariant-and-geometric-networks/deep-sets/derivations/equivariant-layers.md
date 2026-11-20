---
id: equivariant-layers
derives: [c-commutant, c-lemma3-reading, c-padding]
kind: proof
assumes: ["a set of M >= 2 elements is an M x D array whose rows are the elements", "a permutation acts by a permutation matrix P on the rows and leaves the features in place", "the pools sum, average or take the maximum over the rows, feature by feature"]
uses: [permutation-matrix@m2, matrix-multiplication@m2]
status: full
check: equivariant-layers.check.py
lean: null
sources:
  - {work: zaheer-2017-deep-sets, locator: "p. 17", note: "the proof of Lemma 3 for one channel"}
  - {work: zaheer-2017-deep-sets, locator: "p. 19", note: "eq. 22 and 23, the layers with several channels"}
---

### Statement

Let $M \geq 2$. A linear map $T : \mathbb{R}^{M \times D} \to \mathbb{R}^{M \times D'}$ satisfies $T(Px) = P\,T(x)$ for every $M \times M$ permutation matrix $P$ if and only if $T(x) = x\Lambda + \mathbf{1}\mathbf{1}^\top x\,\Gamma$ for some $\Lambda, \Gamma \in \mathbb{R}^{D \times D'}$, and these maps form a space of dimension $2DD'$. For one feature, $\Theta = \lambda I + \gamma\mathbf{1}\mathbf{1}^\top$ and $\sigma(\Theta x) = \sigma(\lambda x + \gamma (\sum_m x_m)\mathbf{1})$. The layer $h\Lambda - \mathbf{1}\operatorname{pool}(h)\Gamma + \mathbf{1}\beta$ followed by an elementwise $\sigma$ is equivariant for the sum, the mean and the maximum. A pool that skips the rows of a mask gives the same output for a set alone and padded with zero rows; a sum that does not skip them changes by $M - |X|$ copies of $\phi(0)$.

### Assumptions

1. $M \geq 2$, so that there are transpositions. Used in step 2.
2. The pools act feature by feature over the rows. Used in step 4.
3. $\sigma$ acts elementwise. Used in steps 3 and 4.

### Steps

1. Write $T$ as a tensor $T_{(m,d),(n,e)}$ with $T(x)_{n e} = \sum_{m,d} x_{md} T_{(m,d),(n,e)}$. Equivariance for every $P$ says $T_{(\pi(m),d),(\pi(n),e)} = T_{(m,d),(n,e)}$ for every permutation $\pi$.
2. The permutations act transitively on the pairs $(m, m)$ and on the pairs $(m, n)$ with $m \neq n$, and these are the only two orbits of pairs when $M \geq 2$. So for each $(d, e)$, $T_{(m,d),(n,e)}$ takes one value $a_{de}$ when $m = n$ and one value $b_{de}$ when $m \neq n$: $T(x) = x(A - B) + \mathbf{1}\mathbf{1}^\top x\, B$. Set $\Lambda = A - B$ and $\Gamma = B$. Conversely every such map is equivariant, because $P\mathbf{1} = \mathbf{1}$ and $\mathbf{1}^\top P = \mathbf{1}^\top$. The free parameters are $A$ and $B$, $2DD'$ numbers, and the map from $(A, B)$ to $T$ is injective since the two orbits are disjoint and nonempty.
3. For $D = D' = 1$, $\Theta = \lambda I + \gamma \mathbf{1}\mathbf{1}^\top$ and $\Theta x = \lambda x + \gamma \mathbf{1}(\mathbf{1}^\top x)$, with $\mathbf{1}^\top x = \sum_m x_m$; $\sigma$ applied elementwise keeps the form.
4. The sum, the mean and the maximum over the rows, feature by feature, are invariant: $\operatorname{pool}(Ph) = \operatorname{pool}(h)$. So $Ph\Lambda - \mathbf{1}\operatorname{pool}(Ph)\Gamma + \mathbf{1}\beta = P(h\Lambda - \mathbf{1}\operatorname{pool}(h)\Gamma + \mathbf{1}\beta)$, using $P\mathbf{1} = \mathbf{1}$, and an elementwise $\sigma$ commutes with $P$.
5. With a mask, the pools over the rows where the mask is 1 see the same rows whatever the padding, and the padded rows are set to zero after each layer, so each real row has the same value alone and padded. A sum over all $M$ rows instead adds $\sum_{\text{pad}} \phi(0) = (M - |X|)\,\phi(0)$, and $\phi(0) = \sigma(\beta) \neq 0$ in general.

### Result and check

$$ \{T \text{ linear} : T(Px) = PT(x)\ \forall P\} = \{x \mapsto x\Lambda + \mathbf{1}\mathbf{1}^\top x\,\Gamma\}, \qquad \dim = 2DD' . $$

The check `equivariant-layers.check.py` builds the $MD \times MD'$ matrix of the condition of step 1 for every permutation of up to 4 elements and computes the dimension of its null space, which is $2DD'$ for $M \geq 2$ and $DD'$ for $M = 1$; it applies random layers of the reference to permuted inputs for each pool, compares one-channel layers with $\sigma(\Theta x)$, and compares masked and padded sets with the same sets alone.
