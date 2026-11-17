---
id: parameter-and-flop-count
derives: [c-params, c-flops]
kind: count
assumes: ["the model of the page: L message-passing layers with separate self and neighbour weights and a bias, a sum readout and a linear head", "the adjacency is stored dense, as a B x N x N array", "a multiply-add counts as 2 FLOPs and only the matrix products are counted"]
uses: [matrix-multiplication@m2]
status: full
check: parameter-and-flop-count.check.py
lean: null
sources:
  - {work: kipf-2017-semi-supervised-classification, locator: "Section 2.2"}
---

### Statement

1. The network of the page with input width $F$, hidden width $D$, $L$ layers and $K$ classes has $(2FD + D) + (L-1)(2D^2 + D) + DK + K$ parameters, independent of the number of nodes $N$.
2. Its forward pass on $B$ graphs of $N$ nodes with a dense adjacency costs $4BNFD + 2BN^2D$ FLOPs in the first layer, $4BND^2 + 2BN^2D$ in each later layer and $2BDK$ in the head, counting only the matrix products at 2 FLOPs per multiply-add.

### Assumptions

1. Layer 1 holds $W_{\text{self}}, W_{\text{nbr}} \in \mathbb R^{F \times D}$ and $b \in \mathbb R^D$; each later layer holds two $D \times D$ matrices and a bias; the head holds $W \in \mathbb R^{D \times K}$ and $b \in \mathbb R^K$; the readout is a sum and has no parameters. Used in step 1.
2. The adjacency is a dense $B \times N \times N$ array and the message is $A (H W_{\text{nbr}})$. Used in step 2.
3. A product of an $m \times k$ and a $k \times n$ matrix costs $2mkn$ FLOPs; additions of biases, the ReLU and the readout sum are not counted. Used in step 2.

### Steps

1. Layer 1 has $FD + FD + D$ entries and each of the $L - 1$ later layers $D^2 + D^2 + D$; the head has $DK + K$. None of the shapes involves $N$, so the same weights apply to graphs of any size. The sum is statement 1.
2. In layer 1, $H W_{\text{self}}$ and $H W_{\text{nbr}}$ are products of a $BN \times F$ and an $F \times D$ matrix, $2BNFD$ FLOPs each, and $A (H W_{\text{nbr}})$ is $B$ products of an $N \times N$ and an $N \times D$ matrix, $2BN^2D$. A later layer has $D$ in place of $F$. The head multiplies the $B \times D$ readout by $D \times K$: $2BDK$. That is statement 2.
3. The $N^2$ term comes from the dense adjacency. With an edge list, the aggregation costs $2|E|D$ multiply-adds per graph, $|E|$ the number of stored (directed) edges, which is the linear-in-edges cost $O(|E|FC)$ that Kipf and Welling (2017, Section 2.2) give for their layer.

### Result and check

$$ P = (2FD + D) + (L-1)(2D^2 + D) + DK + K, \qquad C = 4BNFD + (L-1)\,4BND^2 + L\cdot 2BN^2D + 2BDK . $$

For the example config ($F = 3$, $D = 8$, $L = 2$, $K = 4$, $B = 2$, $N = 6$) this gives $56 + 136 + 36 = 228$ parameters and $2304 + 4224 + 128 = 6656$ FLOPs. The check `parameter-and-flop-count.check.py` counts the arrays that `init` returns and the products that `forward` performs, for random sizes, and compares both with the formulas.
