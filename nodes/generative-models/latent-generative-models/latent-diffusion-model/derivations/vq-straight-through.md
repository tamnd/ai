---
id: vq-straight-through
derives: [c-straight-through]
kind: gradient
assumes: ["the assignments k are held fixed, which holds almost everywhere", "the two terms are averaged over the n vectors"]
uses: [vector-quantization@m2, backpropagation@al]
status: full
check: vq-straight-through.check.py
lean: null
sources:
  - {work: oord-2017-neural-discrete-representation-learning, locator: "pp. 3 to 4", note: "the straight-through estimator and the three terms of the loss"}
---

### Statement

Let $z_{e,1}, \ldots, z_{e,n}$ be encoder outputs, $k_i$ the index of the code nearest to $z_{e,i}$, and let the decoder see $z_{q,i} = z_{e,i} + \mathrm{sg}(e_{k_i} - z_{e,i})$, whose value is $e_{k_i}$, with upstream gradient $g_i = \partial L_{\text{dec}}/\partial z_{q,i}$. For

$$ L = L_{\text{dec}}(z_q) + \frac1n \sum_i \lVert \mathrm{sg}(z_{e,i}) - e_{k_i} \rVert^2 + \frac{\beta}{n} \sum_i \lVert z_{e,i} - \mathrm{sg}(e_{k_i}) \rVert^2 , $$

$$ \frac{\partial L}{\partial z_{e,i}} = g_i + \frac{2\beta}{n}(z_{e,i} - e_{k_i}) , \qquad \frac{\partial L}{\partial e_j} = -\frac2n \sum_{i : k_i = j} (z_{e,i} - e_j) . $$

### Assumptions

1. The nearest code is unique, so $k$ is locally constant in $z_e$ and $e$; ties have measure zero. Used in steps 1 to 3.
2. Both terms are averages over the $n$ vectors. Used in steps 2 and 3.

### Steps

1. The stop-gradient $\mathrm{sg}$ is the identity in the forward pass and has zero derivative. So $\partial z_{q,i}/\partial z_{e,i} = I - 0 = I$, and $\partial z_{q,i}/\partial e_j = 0$, since $e$ enters $z_q$ only inside $\mathrm{sg}$. The decoder term contributes $g_i$ to $z_{e,i}$ and nothing to the codebook.
2. The codebook term depends on $e$ only: $z_e$ is inside $\mathrm{sg}$. Its derivative in $e_j$ is $\frac1n \sum_{i : k_i = j} 2(e_j - z_{e,i})$, by assumption 1 the indices do not move.
3. The commitment term depends on $z_e$ only, with derivative $\frac{2\beta}{n}(z_{e,i} - e_{k_i})$ in $z_{e,i}$.
4. Adding: the encoder receives $g_i + \frac{2\beta}{n}(z_{e,i} - e_{k_i})$ and code $j$ receives $-\frac2n \sum_{i : k_i = j}(z_{e,i} - e_j)$, which does not involve $g$.

### Result and check

The decoder's gradient passes to the encoder unchanged, as if quantization were the identity; the codebook learns only from the codebook term, which pulls each code toward the vectors assigned to it, and codes with no assigned vector receive no gradient at all, the origin of dead codes.
The check `vq-straight-through.check.py` compares `vq_grads` with central finite differences of the commitment term in $z_e$, with the codebook fixed, and of the codebook term in $e$, with $z_e$ fixed, checks that the codebook gradient does not change when $g$ does, and that unused codes get zero.
