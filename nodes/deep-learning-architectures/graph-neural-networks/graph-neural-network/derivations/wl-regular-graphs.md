---
id: wl-regular-graphs
derives: [c-regular-same-readout]
kind: proof
assumes: ["two graphs with the same number of nodes n in which every node has the same degree d", "every node starts with the same feature vector", "the aggregation is the sum, the mean or the maximum, and the update and readout are any functions of the kind the page defines"]
uses: []
status: full
check: wl-regular-graphs.check.py
lean: null
sources:
  - {work: xu-2019-how-powerful-are-gnns, locator: "Section 3"}
  - {work: xu-2019-how-powerful-are-gnns, locator: "Lemma 2"}
---

### Statement

1. Let $G_1$ and $G_2$ be two graphs with $n$ nodes each in which every node has degree $d$, and let every node of both start with the same features $c$. Then for every message-passing network of the page, with sum, mean or max aggregation and with any weights, all nodes of both graphs have the same state after every layer, and the sum readouts of the two graphs are equal.
2. The 1-WL test also gives every node of both graphs the same colour in every round, so it does not tell them apart.
3. Two disjoint triangles and the 6-cycle are such a pair ($n = 6$, $d = 2$), although they are not isomorphic: the first has two connected components and the second has one.

### Assumptions

1. Every node of $G_1$ and $G_2$ has exactly $d$ neighbours, and both have $n$ nodes. Used in steps 2 and 4.
2. The input features are $x_v = c$ for every node. Used in step 1.
3. Layer $t$ computes $h_v^{t+1} = \phi_t(h_v^t, \operatorname{agg}\{h_u^t : u \in N(v)\})$, where agg is the sum, the mean or the maximum of the multiset of neighbour states, and $\phi_t$ is the same for every node and both graphs; the readout is a function of the multiset of final states. Used in steps 2 and 3.

### Steps

1. Induction claim: after $t$ layers every node of both graphs has the same state $a_t$. For $t = 0$ this is assumption 2, with $a_0 = c$.
2. Suppose it holds for $t$. Each node has $d$ neighbours (assumption 1), all with state $a_t$, so the multiset it aggregates is $\{a_t, \dots, a_t\}$ with $d$ copies, the same at every node of both graphs. The sum is $d\,a_t$, the mean and the maximum are $a_t$ (for $d \ge 1$); in each case the aggregate is the same vector $m_t$ everywhere, and $h_v^{t+1} = \phi_t(a_t, m_t) =: a_{t+1}$ at every node. This proves the claim for $t + 1$.
3. After $T$ layers both graphs have the multiset of $n$ copies of $a_T$, so any readout of that multiset, in particular the sum $n\,a_T$, is equal for the two graphs. That is statement 1.
4. 1-WL refines a colour by the pair (own colour, multiset of the neighbours' colours). Starting from one colour, every node has the pair $(0, \{0^d\})$ in both graphs, so all nodes get the same new colour, and by the same induction they keep sharing one colour in every round. That is statement 2: the test stops after one round with equal colour histograms.
5. Two disjoint triangles have 6 nodes of degree 2, and so has the 6-cycle. The 6-cycle is connected and two triangles are not, so the graphs are not isomorphic, and by steps 3 and 4 neither 1-WL nor any network of the page with constant input features tells them apart. That is statement 3. Xu et al. (2019, Lemma 2) prove the general bound: if 1-WL does not separate two graphs, no message-passing network of this kind maps them to different vectors; Section 3 names regular graphs as an exception where 1-WL fails.

### Result and check

$$ h_v^{t}(G_1) = h_v^{t}(G_2) = a_t \ \text{for all } v, t, \qquad \textstyle\sum_v h_v^T(G_1) = \sum_v h_v^T(G_2) = n\, a_T . $$

The check `wl-regular-graphs.check.py` runs the reference implementation with random weights and all three aggregations on two triangles and the 6-cycle and on another pair of 3-regular graphs, compares the readouts and the 1-WL histograms, and verifies that the 6-path, which is not regular, is separated from the 6-cycle by both.
