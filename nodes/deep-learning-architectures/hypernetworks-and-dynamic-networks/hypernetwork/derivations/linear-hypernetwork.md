---
id: linear-hypernetwork
derives: [c-affine, c-subspace, c-tile-layout]
kind: proof
assumes: ["the hypernetwork of eq-static with hidden size d", "the kernel matrix K^j of a basic kernel is the stack of its N_in slices K_i^j, each f x N_out f"]
uses: [matrix-multiplication@m2]
status: full
check: linear-hypernetwork.check.py
lean: null
sources:
  - {work: ha-2016-hypernetworks, locator: "p. 3", note: "eqs. 1 and 2 and the comparison with a one-layer hypernetwork"}
  - {work: ha-2016-hypernetworks, locator: "p. 4", note: "eq. 3, the tiling of basic kernels"}
---

### Statement

The kernel that eq-static generates is an affine function of the embedding, $K^j = A z^j + c$, and every slice of every generated kernel lies in the same affine subspace
$$ K_i^j \in B_{\text{out}} + \operatorname{span}\{W_{\text{out}}[:, :, 1], \dots, W_{\text{out}}[:, :, d]\} $$
of dimension at most $d$ in $\mathbb{R}^{f \times N_{\text{out}} f}$, whatever $z^j$, $i$ and $j$. The kernel of a layer that is $m_{\text{in}}$ times wider in its input and $m_{\text{out}}$ times in its output, tiled from $m_{\text{in}} m_{\text{out}}$ basic kernels, is affine in the concatenation of their embeddings, and its block $(r, c)$, input channels $rN$ to $(r+1)N - 1$ and output channels $cN$ to $(c + 1)N - 1$, depends on one embedding only.

### Steps

1. $a_i^j = W_i z^j + B_i$ is affine in $z^j$, and $K_i^j = \sum_{k=1}^{d} a_{i,k}^j W_{\text{out}}[:, :, k] + B_{\text{out}}$ is linear in $a_i^j$ plus a constant. The composition of affine maps is affine: $K_i^j = \sum_k (W_i z^j + B_i)_k W_{\text{out}}[:, :, k] + B_{\text{out}}$.
2. In step 1, $K_i^j - B_{\text{out}}$ is a combination of the $d$ matrices $W_{\text{out}}[:, :, k]$ with coefficients $a_{i,k}^j$, so it lies in their span, of dimension at most $d$. This holds for every $i$ and $j$ because $W_{\text{out}}$ and $B_{\text{out}}$ are shared. When $W_{\text{out}}$ has rank $d$ as a map from $\mathbb{R}^d$, any point of the subspace is reached by a suitable $a_i^j$, and with $N_z \geq d$ and $W_i$ of rank $d$, by a suitable $z^j$.
3. A one-layer hypernetwork, $K^j = \langle W, z^j\rangle$ with $W \in \mathbb{R}^{N_{\text{in}} f \times N_{\text{out}} f \times N_z}$, is any linear map; it lets the slices span up to $\min(N_z, f \cdot N_{\text{out}} f)$ dimensions each, independently. The two-layer form restricts the slices to one shared subspace and pays for it with fewer parameters, the comparison of the derivation `parameter-count`.
4. In the tiling of eq-tile, block $(r, c)$ of the large kernel is the basic kernel $K^{j(r,c)}$ with $j(r,c) = r\,m_{\text{out}} + c$; by step 1 each block is affine in its own $z^{j(r,c)}$ and constant in the others, so the whole kernel is affine in their concatenation.

### Result and check

The capacity of a static hypernetwork is set by $d$, not by the size of the kernels: each generated slice has at most $d$ degrees of freedom around $B_{\text{out}}$, and an embedding of $N_z$ numbers chooses $N_{\text{in}}$ such points. The check `linear-hypernetwork.check.py` tests that the map from $z$ to the kernel passes the test of affinity with random weights, that the slices of many random embeddings have a rank of $d$ once $B_{\text{out}}$ is subtracted, that the one-layer form reaches a higher rank, and that the blocks of a tiled kernel depend only on their own embedding.
