---
id: path-count
derives: [c-path-count, c-path-share, c-dense-connections, eq-paths]
kind: count
assumes: ["x_{i+1} = x_i + F_i(x_i) for n units", "the expansion is of the product of the Jacobians (I + J_i), or of linear residual functions", "a dense block of L layers in which layer ℓ takes the outputs x_0 to x_{ℓ-1}"]
uses: [binomial-distribution@m2, matrix-multiplication@m2]
status: full
check: path-count.check.py
lean: null
sources:
  - {work: veit-2016-residual-networks-ensembles, locator: "Section 3"}
  - {work: veit-2016-residual-networks-ensembles, locator: "Section 5"}
  - {work: huang-2017-densely-connected-convolutional-networks, locator: "Section 3"}
---

### Statement

1. The product $\prod_{i=1}^{n} (I + A_i)$, with $A_1$ applied first, is the sum of $2^n$ terms, one for each subset $S$ of $\{1, \dots, n\}$; the term of $S$ is the product of the $A_i$, $i \in S$, in order. Exactly $\binom{n}{k}$ terms contain $k$ factors.
2. For $n = 54$, the share of the $2^{54}$ paths that go through 19 to 35 branches is $0.980$, and the share that go through 5 to 17 branches is $0.0045$.
3. A dense block of $L$ layers, in which layer $\ell$ takes the outputs $x_0, \dots, x_{\ell-1}$, has $L(L+1)/2$ direct connections.

### Assumptions

1. Each unit computes $x_{i+1} = x_i + F_i(x_i)$. The paths are counted in the expansion of the Jacobian $\prod (I + J_i)$, or of the forward map when every $F_i$ is linear, $F_i(x) = A_i x$. Used in steps 1 to 3.
2. In a dense block, layer $\ell$ for $\ell = 1, \dots, L$ receives the concatenation $[x_0, \dots, x_{\ell-1}]$, as in Huang et al. (2017, Section 3). Used in step 4.

### Steps

1. Distribute the product: from each factor $(I + A_i)$ pick either $I$ or $A_i$. A choice is a binary code $b \in \{0, 1\}^n$ with $b_i = 1$ when $A_i$ is picked, so there are $2^n$ choices, and the product is the sum of the terms of all codes. The identity factors drop out of each term, which leaves the ordered product of the picked $A_i$. This is the unraveled view of Veit et al. (2016, Section 3), where $b_i = 1$ means that the path goes through module $f_i$.
2. A path goes through $k$ branches when its code has $k$ ones. There are $\binom{n}{k}$ such codes, and $\sum_{k=0}^{n} \binom{n}{k} = 2^n$ by the binomial theorem with both arguments 1. So the length of a path drawn uniformly is binomial with parameters $n$ and $1/2$, with mean $n/2$.
3. For $n = 54$, the 54 two-layer units of a 110-layer CIFAR network, $\sum_{k=19}^{35} \binom{54}{k} / 2^{54} = 0.980$, which agrees with the statement of Veit et al. (Section 5) that more than 95% of the paths go through 19 to 35 modules. A path through $k$ two-layer units crosses $2k$ weight layers, so the paths 10 to 34 layers deep go through 5 to 17 units, and $\sum_{k=5}^{17} \binom{54}{k} / 2^{54} = 0.0045$.
4. In a dense block, layer $\ell$ has $\ell$ incoming connections, one from each of $x_0, \dots, x_{\ell-1}$. Summing over the layers gives $\sum_{\ell=1}^{L} \ell = L(L+1)/2$.

### Result and check

$$ \prod_{i=1}^{n} (I + A_i) = \sum_{S \subseteq \{1, \dots, n\}} \prod_{i \in S}^{\rightarrow} A_i , \qquad \#\{S : |S| = k\} = \binom{n}{k} . $$

The check `path-count.check.py` expands the product over all subsets for random matrices with $n$ up to 8, counts the subsets by size, computes the two shares for $n = 54$, and counts the connections of dense blocks.
