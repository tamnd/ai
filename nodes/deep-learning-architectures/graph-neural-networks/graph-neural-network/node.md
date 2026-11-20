---
tag: PS4TCX
kind: architecture
slug: graph-neural-network
name: Graph neural network
aliases: [GNN, message passing neural network, MPNN]
area: deep-learning-architectures/graph-neural-networks
level: core
summary: >-
  A neural network on graphs that updates each node's state from its neighbours' states, layer by layer, with weights shared by all nodes, so that its outputs do not depend on the node order.
uses: [relu@al, backpropagation@al, matrix-multiplication@m2, permutation-matrix@m2, eigenvalue@m2, eigenvector@m2]
related_to: [transformer, spectral-clustering@al, laplacian-eigenmaps@al]
history:
  year: 2005
  people: [Marco Gori, Gabriele Monfardini, Franco Scarselli, Ah Chung Tsoi, Markus Hagenbuchner]
  note: "Gori, Monfardini and Scarselli (2005) and Scarselli et al. (2009), as cited by Kipf and Welling (2017) and Veličković et al. (2018); the message-passing form of this page is that of Gilmer et al. (2017)."
prereqs: [relu@al, backpropagation@al, matrix-multiplication@m2, eigenvalue@m2]
same_as:
  wikipedia: Graph_neural_network
  arxiv: ["1704.01212", "1609.02907"]
sources:
  - {paper: "Gilmer, Schoenholz, Riley, Vinyals and Dahl (2017)", title: "Neural Message Passing for Quantum Chemistry", locator: "ICML 2017, PMLR 70; arXiv 1704.01212v2, Sections 1 and 2", note: "the message passing framework"}
  - {paper: "Kipf and Welling (2017)", title: "Semi-Supervised Classification with Graph Convolutional Networks", locator: "ICLR 2017; arXiv 1609.02907v4, Sections 2, 4.2 and 7.2, Table 2 and Figure 5", note: "the GCN layer"}
  - {paper: "Hamilton, Ying and Leskovec (2017)", title: "Inductive Representation Learning on Large Graphs", locator: "NIPS 2017; arXiv 1706.02216v4, Sections 3.1 and 3.3", note: "GraphSAGE"}
  - {paper: "Veličković, Cucurull, Casanova, Romero, Liò and Bengio (2018)", title: "Graph Attention Networks", locator: "ICLR 2018; arXiv 1710.10903v3, Sections 1, 2.1 and 2.2", note: "GAT"}
  - {paper: "Xu, Hu, Leskovec and Jegelka (2019)", title: "How Powerful are Graph Neural Networks?", locator: "ICLR 2019; arXiv 1810.00826v3, Sections 4 and 5", note: "the Weisfeiler-Lehman bound and GIN"}
  - {paper: "Li, Han and Wu (2018)", title: "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning", locator: "AAAI 2018; arXiv 1801.07606v1, Section 3", note: "Laplacian smoothing and over-smoothing"}
  - {paper: "Scarselli, Gori, Tsoi, Hagenbuchner and Monfardini (2009)", title: "The Graph Neural Network Model", locator: "IEEE Transactions on Neural Networks 20(1)", note: "not read for this page; cited through Kipf and Welling (2017, Section 4.2) and Veličković et al. (2018, Section 1)"}
family: graph
inputs: "node features as a (B, N, F) array and adjacency matrices as a (B, N, N) array, graphs padded to N nodes"
outputs: "graph logits as a (B, K) array, and the node states as (B, N, D) arrays"
components: [relu@al, softmax-regression@al]
inductive_biases: ["permutation equivariance of the node states and invariance of the graph readout", "locality: a layer mixes each node only with its neighbours, so L layers see L hops", "weights shared by all nodes and independent of the number of nodes"]
params: "(2 F D + D) + (L - 1)(2 D^2 + D) + D K + K"
flops_per_example: "4 N F D + (L - 1) 4 N D^2 + L 2 N^2 D + 2 D K with a dense adjacency, counting a multiply-add as 2 FLOPs"
models: [MPNN, GCN, GraphSAGE, GAT, GIN]
---

A graph neural network is a neural network on a graph that updates the state of each node from the states of its neighbours, layer by layer, with the same weights at every node.
Because each update sums or otherwise pools over the neighbours, the output does not depend on how the nodes are numbered, and one network handles graphs of any size.
After k layers a node's state depends on its neighbourhood k hops out; pooling all the node states gives a vector for the whole graph.
Gori, Monfardini and Scarselli proposed the model in 2005, and Gilmer et al. described the message-passing form in 2017 that this page follows.

## In plain words

Many kinds of data are a set of things with links between them: atoms and bonds, papers and citations, users and friendships.
A graph neural network reads such a graph without first putting the things in an order.
In each layer, every node collects what its neighbours hold, combines it with what it holds itself, and updates its state.
After a few layers a node's state summarizes its neighbourhood a few hops out; adding up all node states gives a description of the whole graph.
Because the same small network runs at every node, it works for graphs of any size, and numbering the nodes differently changes nothing.

## Overview

A graph $G = (V, E)$ with $N$ nodes is given by an adjacency matrix $A \in \{0, 1\}^{N \times N}$, with $A_{vu} = 1$ when $u$ is a neighbour of $v$, and by node features $X \in \mathbb R^{N \times F}$, one row per node.
This page treats undirected graphs, so $A$ is symmetric, and writes $N(v)$ for the neighbours of $v$ and $d_v = |N(v)|$ for its degree.

Gilmer et al. (2017) gave the common form that this page calls a message passing neural network.
A forward pass first passes messages for $T$ steps, each node summing the messages $M_t$ that its neighbours send and updating its hidden state with a function $U_t$, and then reads out one vector for the entire graph with a function $R$. {#c-mpnn-phases}

$$ m_v^{t+1} = \sum_{u \in N(v)} M_t\big(h_v^t, h_u^t, e_{vu}\big) $$ {#eq-message}

$$ h_v^{t+1} = U_t\big(h_v^t, m_v^{t+1}\big) $$ {#eq-update}

$$ \hat y = R\big(\{ h_v^T : v \in V \}\big) $$ {#eq-readout}

Here $h_v^0 = x_v$, and $e_{vu}$ are optional edge features.
The readout sees the node states as a set, and Gilmer et al. point out that it has to give the same result for every ordering of them, or the network would not be invariant to graph isomorphism. {#c-readout-invariance-needed}
They describe at least eight earlier models in this framework; the GCN of Kipf and Welling, for instance, is the message $c_{vu} h_u$ with $c_{vu} = (d_v d_u)^{-1/2} A_{vu}$ and the update $\operatorname{ReLU}(W m_v)$. {#c-mpnn-unifies}
On QM9, 130k molecules with 13 properties each, their best network predicts the DFT values to within chemical accuracy on 11 of the 13 targets, and a network that sees only the molecular graph does so on 5 of 13. {#c-qm9-chemical-accuracy}

## Components

**The model of this page.** The implementations use the simplest member of the family: a message that is a linear map of the neighbour's state, summed, and an update that adds a linear map of the node's own state, a bias and a [[al:relu|ReLU]],

$$ H^{t+1} = \operatorname{relu}\big(H^t W_{\text{self}} + A H^t W_{\text{nbr}} + \mathbf 1 b^\top\big) , $$

which is [[graph-neural-network#eq-message|the message]] with $M_t(h_v, h_u) = W_{\text{nbr}}^\top h_u$ and [[graph-neural-network#eq-update|the update]] with $U_t(h, m) = \operatorname{relu}(W_{\text{self}}^\top h + m + b)$, different weights in each layer.
The readout is the sum $g = \sum_v h_v^L$, followed by a linear head with $K$ outputs.

**Aggregation.** The sum in [[graph-neural-network#eq-message|the message]] is one choice of aggregation; the mean and the entrywise maximum are the other two common ones,

$$ \operatorname{agg}_v(H) = \sum_{u \in N(v)} h_u ,\quad \frac{1}{d_v} \sum_{u \in N(v)} h_u ,\quad \max_{u \in N(v)} h_u , $$ {#eq-aggregate}

and the reference implementation gives an isolated node the aggregate 0 in all three.

**Permutation symmetry.** Numbering the nodes differently replaces $X$ by $PX$ and $A$ by $PAP^\top$ for a [[m2:permutation-matrix|permutation matrix]] $P$.
A layer with sum, mean or max aggregation and an update shared by all nodes is permutation equivariant: relabelling by $P$ permutes its output rows by the same $P$. {#c-layer-equivariant}
The sum readout is invariant, since $\mathbf 1^\top P H = \mathbf 1^\top H$, so a stack of such layers with a sum readout gives the same graph vector for every numbering. {#c-readout-invariant}
The derivation `permutation-equivariance` proves both from $P^\top P = I$ and $P^\top \mathbf 1 = \mathbf 1$.

**The GCN layer.** Kipf and Welling (2017) define the graph convolutional network by the propagation rule

$$ H^{(l+1)} = \sigma\big(\tilde D^{-1/2} \tilde A \tilde D^{-1/2} H^{(l)} W^{(l)}\big) , \qquad \tilde A = A + I_N ,\ \tilde D_{ii} = \textstyle\sum_j \tilde A_{ij} , $$ {#eq-gcn}

where $\tilde A$ is the adjacency of the undirected graph with added self-connections, $W^{(l)}$ a trainable weight matrix of the layer and $\sigma$ an activation. {#c-gcn-rule}
Writing $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$,

$$ \hat A_{vu} = \frac{\tilde A_{vu}}{\sqrt{(d_v + 1)(d_u + 1)}} . $$ {#eq-renorm}

Under a relabelling, $\hat A$ becomes $P \hat A P^\top$, so the GCN layer is equivariant too. {#c-gcn-equivariant}

They motivate the rule by a first-order approximation of localized spectral filters, starting from spectral graph convolutions approximated by a truncated expansion in Chebyshev polynomials $T_k$ up to order $K$, which is $K$-localized. {#c-spectral-motivation}
Limiting the expansion to $K = 1$ with $\lambda_{\max} \approx 2$ simplifies the filter to $\theta'_0 x - \theta'_1 D^{-1/2} A D^{-1/2} x$, and a single parameter $\theta = \theta'_0 = -\theta'_1$ gives $\theta (I_N + D^{-1/2} A D^{-1/2}) x$. {#c-gcn-first-order}
They note that this operator has eigenvalues in $[0, 2]$, so that repeating it in a deep model can lead to numerical instabilities and exploding or vanishing gradients, and they replace it by $\tilde D^{-1/2} \tilde A \tilde D^{-1/2}$, which they call the renormalization trick. {#c-renorm-trick}
The renormalized matrix is symmetric with eigenvalues in $[-1, 1]$, and $\tilde D^{1/2} \mathbf 1$ is an eigenvector for the eigenvalue 1, on every graph, with or without isolated nodes. {#c-norm-adj-spectrum}
[[al:spectral-clustering|Spectral clustering]] and [[al:laplacian-eigenmaps|Laplacian eigenmaps]] use the eigenvectors of a graph Laplacian themselves as a fixed embedding of the nodes; a GCN layer instead applies a fixed low-order polynomial of the normalized adjacency and learns only the weights $W$, so it never computes an eigenvector.
The derivation `gcn-renormalization` works through the algebra and proves the spectrum from the similarity of $\hat A$ to the row-stochastic $\tilde D^{-1} \tilde A$; Kipf and Welling state the range only for the operator before renormalization.

**GraphSAGE.** Hamilton et al. (2017) build each layer from an aggregation of the immediate neighbourhood into one vector, its concatenation with the node's own representation, and a fully connected layer with a nonlinearity, after which the representations are normalized. {#c-sage-algorithm}
To keep the cost of a batch fixed, the neighbourhood is a fixed-size uniform draw from the neighbours, a new one in each layer, and they report high performance with two layers and $S_1 \cdot S_2 \le 500$ sampled nodes. {#c-sage-sampling}
They argue that an aggregator should ideally be symmetric, invariant to the order of its inputs, and compare a mean aggregator, which they call nearly equivalent to the GCN rule, an LSTM aggregator, which is not symmetric, and a pooling aggregator. {#c-sage-aggregators}

**Graph attention.** In a graph attention layer of Veličković et al. (2018), node $i$ computes coefficients $e_{ij} = a(W h_i, W h_j)$ only for its first-order neighbours $j$ and itself, and normalizes them with a softmax over those neighbours; in their experiments $a$ is a single-layer feedforward network with weights $\mathbf a \in \mathbb R^{2F'}$ and a LeakyReLU of negative slope 0.2. {#c-gat-coefficients}
With $K$ heads, $K$ independent attention mechanisms run and their features are concatenated, except in the final layer, where they are averaged before the final nonlinearity. {#c-gat-multihead}
This is attention restricted to the edges of the graph; self-attention in the [[transformer|Transformer]] lets every position attend to every other, as if on a complete graph, and scores the pairs by scaled dot products rather than by a small feedforward network.

**GIN.** The Graph Isomorphism Network of Xu et al. (2019) updates $h_v^{(k)} = \mathrm{MLP}^{(k)}\big((1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in N(v)} h_u^{(k-1)}\big)$ and, for graph classification, concatenates the summed node states of all layers. {#c-gin-update}

## Shapes and parameters

The shape contract is the model of this page on a batch of $B$ graphs padded to $N$ nodes, with input width $F$, hidden width $D$, $L$ layers and $K$ outputs; padded nodes have zero features and no edges.
The first layer holds two $F \times D$ matrices and a bias, each later layer two $D \times D$ matrices and a bias, and the head a $D \times K$ matrix and a bias, so the model has

{shape:params}

parameters, none of which depends on $N$: {shape:params@example} for the example configuration. {#c-params}
The derivation `parameter-and-flop-count` counts them block by block.
A GCN layer from $F$ to $D$ features has $FD$ parameters and no bias in the form above; a GraphSAGE layer with concatenation has $2FD$.

## Computation and cost

With a dense adjacency, the products $H W_{\text{self}}$ and $H W_{\text{nbr}}$ cost $2BNFD$ FLOPs each, and $A(HW_{\text{nbr}})$ costs $2BN^2D$; the head costs $2BDK$.
The first layer therefore costs $4BNFD + 2BN^2D$, each later one $4BND^2 + 2BN^2D$, and the forward pass

{shape:flops_forward}

which is {shape:flops_forward@example} FLOPs for the example. {#c-flops}
With a sparse adjacency the cost is linear in the edges instead: Kipf and Welling give $\hat A X \Theta$ a cost of $O(|E| F C)$ with a sparse $\hat A$, linear in the number of edges, for $C$ input channels and $F$ filters. {#c-gcn-cost}
One attention head of a graph attention layer costs $O(|V| F F' + |E| F')$, on par with GCN, and $K$ heads multiply the storage and parameters by $K$. {#c-gat-cost}

## Training

The networks are trained end to end by [[al:backpropagation|backpropagation]], with a loss on nodes, edges or graphs.
Kipf and Welling train on a whole graph at once with the loss on its few labelled nodes; GraphSAGE samples neighbourhoods so that a batch does not need the whole graph; for graph classification a batch is a set of graphs, here padded to a common size.

## Inductive biases

Equivariance means that the network cannot use the order in which the nodes are stored, only the structure and the features.
Locality means that a node's state after $L$ layers depends only on the nodes within $L$ hops.
Weight sharing means that the parameter count does not grow with the graph, so one network runs on graphs of any size.
The symmetry also limits what the network can tell apart, as the next section shows.

## Evidence

**Expressive power.** The Weisfeiler-Lehman test, 1-WL, refines node colours round by round: each node's new colour is a code for the pair of its old colour and the multiset of its neighbours' colours, with one shared code table,

$$ c_v^{t+1} = \operatorname{hash}\big(c_v^t, \{\!\{ c_u^t : u \in N(v) \}\!\}\big) , $$ {#eq-wl}

and two graphs whose colour histograms differ in some round are not isomorphic.
Xu et al. prove that if an aggregation-based GNN maps two non-isomorphic graphs to different embeddings, the WL test also decides that they are not isomorphic, so such a GNN is at most as powerful as the WL test. {#c-wl-upper-bound}
Conversely, with enough layers a GNN separates every pair that the WL test separates if its aggregation, update and graph-level readout are injective on multisets. {#c-wl-equal-conditions}
For a countable feature space they prove that some $f$ makes $\sum_{x \in X} f(x)$ unique for every multiset of bounded size, and that $(1 + \epsilon) f(c) + \sum_{x \in X} f(x)$ is unique for each pair of a node feature and a multiset for infinitely many $\epsilon$, including all irrational ones. {#c-sum-injective}
Mean and max-pooling aggregators are not injective on multisets and can fail on simple structures that the sum separates. {#c-mean-max-fail}
The mean maps a multiset and the multiset with $k$ copies of each element to the same embedding, so it captures the proportions of the elements but not the exact multiset. {#c-mean-learns}
The max treats a multiset as a set, which may suit tasks where identifying representative elements, the skeleton, matters more than the distribution. {#c-max-learns}
A 1-layer perceptron is not sufficient either: there are finite multisets that a summed linear map followed by a ReLU without bias can never distinguish, and even with a bias it is not a universal approximator of multiset functions, unlike an MLP. {#c-perceptron-insufficient}
The model of this page sums, but its update is a single layer with a bias, so it is not GIN and the injectivity argument does not cover it.

Xu et al. describe the WL test as working well in general, with a few exceptions such as regular graphs. {#c-wl-regular-exception}
The smallest example is a pair that the tests of this page use.
If two graphs have the same number of nodes and every node has the same degree $d$, and all nodes start with the same features, then every network of this page, with sum, mean or max aggregation and any weights, gives all their nodes the same state after every layer and the two graphs the same sum readout, and 1-WL gives all their nodes one colour; two disjoint triangles and the 6-cycle are such a pair. {#c-regular-same-readout}
The derivation `wl-regular-graphs` proves it by induction on the layers.
The two graphs are not isomorphic, since one is connected and the other is not.

**Benchmarks.** In semi-supervised node classification, the two-layer GCN of Kipf and Welling reaches 70.3% accuracy on Citeseer, 81.5% on Cora, 79.0% on Pubmed and 66.0% on NELL. {#c-gcn-citation-results}
On their datasets, 2 or 3 layers give the best results, and training deeper than 7 layers without residual connections can become difficult. {#c-gcn-depth}

## Limitations

**Over-smoothing.** Li, Han and Wu (2018) derive that the GCN's graph convolution is Laplacian smoothing of a symmetric kind: the new features of a vertex are a weighted average of its own and its neighbours' features. {#c-gcn-laplacian-smoothing}
On Zachary's karate club network, smoothing once leaves the two classes not well separated, twice separates them relatively well, and applying it again and again mixes them. {#c-oversmoothing-karate}
They prove that on a graph with no bipartite components, repeated Laplacian smoothing makes the features within each connected component converge to the same values, for symmetric smoothing to values that scale with the square root of each vertex's degree, and conclude that such over-smoothing leaves the features impossible to tell apart and lowers classification accuracy. {#c-oversmoothing-theorem}
For a connected graph the limit is easy to state: $\hat A^k x$ tends to $s\,(s^\top x) / (s^\top s)$ with $s = \tilde D^{1/2} \mathbf 1$, whatever $x$ was. {#c-smoothing-limit}
This follows from the spectrum above, since the eigenvalue 1 is simple and every other eigenvalue lies strictly inside $(-1, 1)$ on a connected graph with self-loops.
It is a statement about repeated $\hat A$ without weights or nonlinearities; how far it carries over to a trained network with weights is not something these sources measure.

**Other limits.** Kipf and Welling list three: with full-batch gradient descent the memory grows linearly with the dataset, their model does not naturally support edge features and is limited to undirected graphs, and it implicitly assumes locality and equal importance of self-connections and edges to neighbours. {#c-gcn-limits}
The expressiveness bound of the previous section is a further limit of every network of this form.

## Variants and descendants

The family differs mostly in [[graph-neural-network#eq-aggregate|the aggregation]] and the update.
GCN weights the neighbours by their degrees, GraphSAGE learns an aggregator over a sampled neighbourhood, GAT learns the weights by attention, and GIN sums and applies an MLP to reach the WL bound; the MPNN of Gilmer et al. adds edge features to the message.

## Notable models

MPNN (Gilmer et al., 2017), GCN (Kipf and Welling, 2017), GraphSAGE (Hamilton et al., 2017), GAT (Veličković et al., 2018) and GIN (Xu et al., 2019).

## Implementations

The NumPy reference `impl/numpy/graph_neural_network.py` implements the model of this page with a dense adjacency, the three aggregations of [[graph-neural-network#eq-aggregate|the aggregation]], the renormalized adjacency and the GCN layer, and the 1-WL refinement of [[graph-neural-network#eq-wl|the test]] on several graphs with one shared code table.
The PyTorch version `impl/pytorch/graph_neural_network.py` is a `torch.nn.Module` with a `from_params` constructor, and the JAX version `impl/jax/graph_neural_network.py` is a pair of pure functions `init` and `apply`; both load the parameters of the NumPy reference and are tested against the fixture of `fixtures/`, a batch of a 6-cycle with one chord and a 5-path with an isolated node.
The tests check equivariance, invariance of the readout, the spectrum of $\hat A$, dense products against loops and edge lists, gradients, and equal readouts for the triangles and the 6-cycle under random weights.

### Worked example

Take the path with edges 1-2 and 2-3 and scalar features $x = (1, 2, 3)$.
The sum aggregation gives $(2, 1 + 3, 2) = (2, 4, 2)$, the mean $(2, 2, 2)$ and the max $(2, 3, 2)$.
With self-loops the degrees are $(2, 3, 2)$, so $\hat A_{11} = 1/2$, $\hat A_{12} = 1/\sqrt 6$ and $\hat A_{22} = 1/3$, and $\hat A x = (1.3165, 2.2997, 2.3165)$ to four decimals.
One layer of this page with $W_{\text{self}} = 1$, $W_{\text{nbr}} = 0.5$ and $b = -3$ gives $\operatorname{relu}(x + 0.5\,(2, 4, 2) - 3) = \operatorname{relu}(-1, 1, 1) = (0, 1, 1)$, and the readout is 2.
These numbers are computed here and checked by `test_worked_example`.

## Experiments

This page has no experiment of its own yet.
The tests include one small training run, done here: a two-layer network of this page fits cycles against paths of 4 to 7 nodes with constant features to a cross-entropy below 0.05 within 500 Adam steps, in PyTorch and in JAX. It checks the code, not a published result.

## History

Kipf and Welling (2017) trace neural networks on graphs to Gori et al. (2005) and Scarselli et al. (2009), who introduced them as a form of recurrent neural network whose framework requires repeated application of contraction maps as propagation functions until the node representations reach a stable fixed point. {#c-gnn-origin}
Veličković et al. (2018) describe the same networks as a generalization of recursive neural networks to more general graphs, cyclic, directed and undirected, by an iterative process that propagates node states until equilibrium. {#c-gnn-origin-recursive}

## References

The sources of this page are listed in its front matter and cited by its claims; Scarselli et al. (2009) was not read for this page and is cited only through Kipf and Welling (2017) and Veličković et al. (2018).
