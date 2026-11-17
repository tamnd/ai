---
id: permutation-symmetry
derives: [c-perm-invariance, c-perm-barrier, c-weight-matching-lap]
kind: closed-form
assumes: ["a network f(x) = W2 sigma(W1 x + b1) + b2 with one hidden layer of h units", "sigma acts elementwise", "P is an h x h permutation matrix"]
uses: []
status: full
check: permutation-symmetry.check.py
lean: null
sources:
  - {work: entezari-2022-permutation-invariance-mode-connectivity, locator: "p. 6"}
  - {work: ainsworth-2023-git-re-basin, locator: "p. 4"}
---

### Statement

1. For every permutation matrix $P$, the parameters $(P W_1, P b_1, W_2 P^\top, b_2)$ compute the same function as $(W_1, b_1, W_2, b_2)$. So each network has $h!$ parameter settings with the same outputs, and they are distinct points when the hidden units differ.
2. The segment between a network and a permuted copy of it can have a barrier. With $h = 2$, $W_1 = (1, -1)^\top$, $b_1 = 0$, $W_2 = (1, 1)$, $b_2 = 0$ and ReLU, the network computes $\lvert x\rvert$, and so does the copy with the two units swapped. Under the squared error on the inputs $x = \pm 1$ with targets $\lvert x\rvert$, the loss on the segment is $L(\alpha) = (1 - \lvert 1 - 2\alpha\rvert)^2$. It is 0 at both ends and 1 at $\alpha = \tfrac12$, where the network is the constant 0.
3. Weight matching for one hidden layer is a linear assignment problem. Minimizing $\lVert \theta_A - \pi(\theta_B)\rVert^2$ over permutations $\pi$ of the hidden units of $B$ is the same as maximizing $\sum_i C_{i, \pi(i)}$ with
$$ C = W_1^A (W_1^B)^\top + b_1^A (b_1^B)^\top + (W_2^A)^\top W_2^B . $$

### Assumptions

1. The network has one hidden layer and an elementwise activation $\sigma$, such as ReLU or tanh.
2. In part 3, $\pi(\theta_B)$ means $B$ with unit $\pi(i)$ moved to position $i$: rows of $W_1^B$ and entries of $b_1^B$ reordered by $\pi$, and columns of $W_2^B$ reordered the same way.

### Steps

1. An elementwise $\sigma$ commutes with a permutation: $\sigma(P z) = P\sigma(z)$. So $W_2 P^\top \sigma(P W_1 x + P b_1) + b_2 = W_2 P^\top P\,\sigma(W_1 x + b_1) + b_2$, and $P^\top P = I$.
2. In the example, the swap maps $W_1$ to $(-1, 1)^\top$ and leaves $W_2 = (1, 1)$ unchanged. At $\alpha$ on the segment, $W_1(\alpha) = (1 - 2\alpha, 2\alpha - 1)^\top$, so $f_\alpha(x) = \mathrm{relu}((1-2\alpha)x) + \mathrm{relu}((2\alpha-1)x) = \lvert 1 - 2\alpha\rvert\,\lvert x\rvert$.
3. With targets $\lvert x\rvert = 1$ at $x = \pm 1$, each squared error is $(\lvert 1 - 2\alpha\rvert - 1)^2$, which gives $L(\alpha)$. The end losses are 0, so the barrier is 1 under both conventions of the page.
4. $\lVert \theta_A - \pi(\theta_B)\rVert^2 = \lVert\theta_A\rVert^2 + \lVert\theta_B\rVert^2 - 2\langle\theta_A, \pi(\theta_B)\rangle$, since a permutation keeps the norm. So the problem is to maximize the inner product.
5. The inner product is a sum over hidden units. Unit $i$ of $A$ meets unit $\pi(i)$ of $B$ through its row of $W_1$, its bias in $b_1$ and its column of $W_2$. That contributes $C_{i, \pi(i)}$, and $b_2$ adds a constant. The Hungarian method solves this in $O(h^3)$ time.

### Result and check

Part 1 is the permutation invariance that Entezari et al. (2022) use to state their conjecture. Part 3 is the weight-matching objective of Ainsworth et al. (2023) in the special case of one hidden layer. With more layers, each layer's term involves two neighbouring permutations, and Ainsworth et al. show that the problem becomes NP-hard.
`permutation-symmetry.check.py` checks part 1 on random networks with ReLU and tanh and evaluates the example on a grid of $\alpha$. For part 3 it compares the linear assignment with a brute-force search over all $6! = 720$ permutations.
