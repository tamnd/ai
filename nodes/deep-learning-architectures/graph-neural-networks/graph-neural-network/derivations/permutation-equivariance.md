---
id: permutation-equivariance
derives: [c-layer-equivariant, c-readout-invariant, c-gcn-equivariant]
kind: proof
assumes: ["a relabelling of the nodes is a permutation matrix P, which maps the node features X to PX and the adjacency A to PAP^T", "the update is applied to each node's row with the same weights", "the aggregation is the sum, the mean or the entrywise maximum over the neighbours"]
uses: [permutation-matrix@m2, matrix-multiplication@m2]
status: full
check: permutation-equivariance.check.py
lean: null
sources:
  - {work: gilmer-2017-neural-message-passing, locator: "Section 2"}
  - {work: xu-2019-how-powerful-are-gnns, locator: "Section 2"}
---

### Statement

1. A message-passing layer whose aggregation is the sum, the mean or the maximum over the neighbours is permutation equivariant: if the nodes are relabelled by a permutation matrix $P$, its output rows are relabelled by the same $P$.
2. A sum readout is permutation invariant, so a stack of such layers followed by a sum readout gives the same graph vector for every labelling of the same graph.
3. The GCN layer $\sigma(\hat A H W)$ with $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ is permutation equivariant.

### Assumptions

1. A relabelling is a bijection $\pi$ of the nodes, written as the permutation matrix $P$ with $P_{i,\pi(i)} = 1$, so that $(PH)_i = H_{\pi(i)}$, $(PAP^\top)_{ij} = A_{\pi(i)\pi(j)}$, $P^\top P = I$ and $P^\top \mathbf 1 = \mathbf 1$. Used in steps 1 to 6.
2. The update computes row $v$ of the output from row $v$ of $H$ and row $v$ of the aggregate only, with weights shared by all nodes, as in $\phi(H, M) = \operatorname{relu}(H W_{\text{self}} + M + \mathbf 1 b^\top)$. Used in step 4.
3. The aggregation is one of the three of Xu et al. (2019, Section 2): the sum, the mean or the entrywise maximum of the neighbours' states. Used in steps 1 to 3.

### Steps

1. Sum. With $H' = PH$ and $A' = PAP^\top$, $A'H' = PAP^\top P H = P(AH)$, because $P^\top P = I$.
2. Mean. The degrees are $d = A\mathbf 1$, and $d' = A'\mathbf 1 = PAP^\top \mathbf 1 = PA\mathbf 1 = Pd$. Dividing row $i$ of $A'H' = P(AH)$ by $d'_i = d_{\pi(i)}$ divides row $\pi(i)$ of $AH$ by $d_{\pi(i)}$, so the mean aggregate is $P$ times the old one.
3. Maximum. Node $i$ of the relabelled graph has the neighbours $j$ with $A_{\pi(i)\pi(j)} = 1$, whose states are $H'_j = H_{\pi(j)}$. As $j$ runs over them, $k = \pi(j)$ runs over the neighbours of $\pi(i)$ in the old graph, so the maximum over the same multiset of rows is row $\pi(i)$ of the old maximum.
4. Update. If $\phi$ acts on each row with shared weights (assumption 2), then $\phi(PH, PM) = P\phi(H, M)$, because permuting rows before or after a row-wise map gives the same rows. With steps 1 to 3, the layer maps $(PH, PAP^\top)$ to $P$ times its old output, which is statement 1. By induction on the layers, a stack of them is equivariant too.
5. Readout. $\mathbf 1^\top (PH) = (P^\top \mathbf 1)^\top H = \mathbf 1^\top H$, so the sum of the rows does not depend on the labelling, which is statement 2. The readout of Gilmer et al. (2017, Section 2) has to have this property for the network to be invariant to graph isomorphism.
6. GCN. $\tilde A' = PAP^\top + I = P(A + I)P^\top = P\tilde A P^\top$, and its row sums are $P\tilde d$ as in step 2, so $\tilde D' = P\tilde D P^\top$ and $\tilde D'^{-1/2} = P\tilde D^{-1/2}P^\top$. Then $\hat A' = P\tilde D^{-1/2}P^\top P\tilde A P^\top P\tilde D^{-1/2}P^\top = P\hat A P^\top$, and $\sigma(\hat A' P H W) = \sigma(P \hat A H W) = P\sigma(\hat A H W)$ since $\sigma$ acts entrywise. That is statement 3.

### Result and check

$$ f(PX, PAP^\top) = P f(X, A), \qquad \mathbf 1^\top P H = \mathbf 1^\top H . $$

The check `permutation-equivariance.check.py` draws random graphs and permutations and verifies both identities on the reference implementation, for the three aggregations, for the full model and for the GCN layer, and that a row-wise update that is not shared across nodes breaks equivariance.
