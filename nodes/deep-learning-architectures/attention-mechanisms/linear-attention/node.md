---
tag: JBSH5S
kind: method
slug: linear-attention
name: Linear attention
aliases: [linear transformer, linearized attention, kernelized attention, kernel attention, linear self-attention, fast weight programmer, Transformers are RNNs]
area: deep-learning-architectures/attention-mechanisms
level: core
summary: >-
  Replace the softmax of attention by a dot product of feature maps, phi(q)^T phi(k), so that by associativity the cost is linear in the length and causal attention is an RNN with a fixed-size state.
purpose: attention over long sequences in time and memory linear in their length, and autoregressive generation at a constant cost per token
uses: [matrix-multiplication@m2, softmax-regression@al, backpropagation@al]
related_to: [multi-head-attention, transformer, attention-mechanism, recurrent-neural-network, mamba, structured-state-space-model, long-short-term-memory]
hyperparameters:
  - {name: feature map, meaning: "the map phi applied to every query and key; its inner products are the similarities, and they must be non-negative", typical: "elu(x) + 1 in Katharopoulos et al.; DPFP or random features (FAVOR+) to enlarge the state"}
  - {name: C, meaning: "the size of phi(x), which sets the state of C M + C numbers per head and the number of associations the head can hold", typical: "equal to the key width, 32 or 64, for elu + 1; 2 nu times the key width for DPFP"}
  - {name: causal implementation, meaning: "how the masked sums are computed: the recurrence, the masked quadratic form, or chunks of length T that mix the two", typical: "the recurrence for generation; chunks of 64 to 256 for training on accelerators"}
  - {name: update rule, meaning: "how a key and value are written into the state", typical: "the sum rule of Katharopoulos et al.; the delta rule with a learned rate beta in Schlag et al."}
  - {name: normalisation, meaning: "whether the output is divided by phi(q)^T z", typical: "on in Katharopoulos et al.; Schlag et al. find their Delta Net better without it"}
cost:
  train: "4 N C M + 4 N C FLOPs per head for N positions, causal or not; with the backward pass of Algorithm 1 the memory per head is N (2 C + M) numbers, against the N C M of the states that automatic differentiation of the cumulative sum keeps"
  inference: "per generated token and head, C M + C state numbers updated and read in about 4 C M FLOPs, whatever the number of tokens already generated"
history:
  year: 2020
  people: [Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret]
  note: "Katharopoulos, Vyas, Pappas and Fleuret posted Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention as arXiv 2006.16236 in June 2020, published at ICML 2020. Schlag, Irie and Schmidhuber argued in 2021 (ICML 2021) that it is a Fast Weight Programmer of the kind Schmidhuber described in 1991 to 1993, and added the delta rule."
prereqs: [attention-mechanism, multi-head-attention, matrix-multiplication@m2, softmax-regression@al, backpropagation@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2006.16236", "2102.11174"]
sources:
  - {paper: "Katharopoulos, Vyas, Pappas and Fleuret (2020)", title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", locator: "arXiv 2006.16236v3, ICML 2020, Sections 2 to 5, Algorithm 1, Tables 1 to 3, Supplementary Sections A and C", note: "the kernel form, elu + 1, causal masking, the gradient in constant memory, the RNN, and the experiments"}
  - {paper: "Schlag, Irie and Schmidhuber (2021)", title: "Linear Transformers Are Secretly Fast Weight Programmers", locator: "arXiv 2102.11174v3, ICML 2021, Sections 2 to 6, Tables 1, 2 and 4, Figure 2", note: "fast weights, the capacity limit, the delta rule, DPFP and FAVOR+"}
---

Linear attention is attention in which the similarity of a query and a key is a dot product of feature maps, $\phi(q)^\top \phi(k)$, instead of the exponential of their dot product.
The change lets the products be grouped the other way: the keys and values are summed into one matrix of fixed size, and every query reads from that matrix, so the cost grows linearly with the length of the sequence instead of with its square.
With causal masking the matrix is a running sum, and the layer becomes a recurrent network whose state is that matrix.
Katharopoulos, Vyas, Pappas and Fleuret introduced it in 2020; Schlag, Irie and Schmidhuber argued in 2021 that it is a fast weight memory, which can hold no more associations than its feature size.

## In plain words

Ordinary attention lets each word compare itself with every other word, then take an average of what the other words carry, weighted by how well they matched.
With a thousand words that is a million comparisons, and when text is generated word by word, every new word compares itself with all the words before it, so the work per word grows as the text grows.
Linear attention changes how a match is scored, so that the comparisons need not be made one by one.
Each word writes what it carries into one shared notebook of fixed size, filed under its key, and a later word reads the notebook once with its query.
Reading the notebook costs the same whether it holds ten words or ten thousand.
The price is that the notebook is small: once more words are filed than it has room for, their entries blur into each other, and the model can no longer get any one of them back exactly.

## Setting

A self-attention layer receives $N$ vectors as the rows of a matrix and computes queries $Q \in \mathbb{R}^{N \times D}$, keys $K \in \mathbb{R}^{N \times D}$ and values $V \in \mathbb{R}^{N \times M}$ from them by linear maps, as in [[multi-head-attention|multi-head attention]].
Softmax attention, the form used in the [[transformer|Transformer]], gives position $i$ the output

$$ V'_i = \frac{\sum_{j=1}^{N} \exp(Q_i^\top K_j / \sqrt{D})\, V_j}{\sum_{j=1}^{N} \exp(Q_i^\top K_j / \sqrt{D})} , $$ {#eq-softmax}

the weights being those of a softmax over the positions.
Every pair of positions is scored, so the layer takes time and memory proportional to $N^2$.
In autoregressive generation the layer is run once per new token over all the earlier ones, so the $N$-th token costs $N$ times as much as the first, and the keys and values of all of them are kept.

## The method

**Any similarity.**
The generalized attention replaces the exponential of the scaled dot product by any non-negative similarity function sim, and the output of position i is the average of the values weighted by sim(Q_i, K_j), normalized over j. {#c-general}

$$ V'_i = \frac{\sum_{j=1}^{N} \operatorname{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^{N} \operatorname{sim}(Q_i, K_j)} . $$ {#eq-general}

**A kernel.**
Take $\operatorname{sim}(q, k) = \phi(q)^\top \phi(k)$ for a feature map $\phi : \mathbb{R}^D \to \mathbb{R}^C$.
Given a kernel with a feature representation phi, the generalized attention can be rewritten as phi(Q_i)^T sum_j phi(K_j) V_j^T divided by phi(Q_i)^T sum_j phi(K_j), and the feature map phi is applied row by row to the queries and the keys. {#c-kernel}

$$ V'_i = \frac{\phi(Q_i)^\top \sum_{j=1}^{N} \phi(K_j) V_j^\top}{\phi(Q_i)^\top \sum_{j=1}^{N} \phi(K_j)} , \qquad V' = \frac{\phi(Q)\,\big(\phi(K)^\top V\big)}{\phi(Q)\,\big(\phi(K)^\top \mathbf{1}\big)} , $$ {#eq-linear}

the division being row by row.
The sums $\phi(K)^\top V \in \mathbb{R}^{C \times M}$ and $z = \phi(K)^\top \mathbf 1 \in \mathbb{R}^C$ do not depend on the query, so they are computed once and read by every query.
Katharopoulos et al. express self-attention as a linear dot-product of kernel feature maps and use the associativity of matrix products to reduce its complexity from O(N^2) to O(N), where N is the sequence length. {#c-def}

**The feature map.**
For their experiments Katharopoulos et al. use the feature map phi(x) = elu(x) + 1, which gives a positive similarity, preferring elu to relu to avoid setting the gradients to 0 when x is negative. {#c-elu}

$$ \phi(x) = \operatorname{elu}(x) + 1 = \begin{cases} x + 1 & x > 0 \\ e^{x} & x \le 0 \end{cases} , $$ {#eq-elu}

applied coordinate by coordinate, so $C = D$.
With the feature map elu(x) + 1, attention requires O(N D M) multiplications and additions. {#c-elu-cost}
Katharopoulos et al. observe that linear attention converges normally if a kernel with positive similarity scores is applied to the queries and keys. {#c-positive}
There is no $1/\sqrt{D}$ in eq-linear: the scale of the scores is whatever the learned projections make it.

**Causal masking.**
In a model that predicts the next token, position $i$ may only read the positions $j \le i$:

$$ V'_i = \frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top z_i} , \qquad S_i = \sum_{j \le i} \phi(K_j) V_j^\top , \qquad z_i = \sum_{j \le i} \phi(K_j) . $$ {#eq-causal}

Katharopoulos et al. write the masked attention with S_i = sum_{j <= i} phi(K_j) V_j^T and Z_i = sum_{j <= i} phi(K_j), which can be computed from S_{i-1} and Z_{i-1} in constant time, so the computational complexity of linear transformers with causal masking is linear in the sequence length. {#c-causal-paper}

**The layer is an RNN.**
Read one position at a time, eq-causal is a recurrence:

$$ S_i = S_{i-1} + \phi(K_i) V_i^\top , \qquad z_i = z_{i-1} + \phi(K_i) , \qquad V'_i = \frac{S_i^\top \phi(Q_i)}{z_i^\top \phi(Q_i)} , \qquad S_0 = 0,\ z_0 = 0 . $$ {#eq-rnn}

Katharopoulos et al. argue that any transformer layer with causal masking can be written as a model that, given an input, modifies an internal state and then predicts an output, namely a recurrent neural network, with two hidden states, the attention memory s and the normalizer memory z, and a recurrence with respect to time, not depth. {#c-rnn}
Katharopoulos et al. note that the computations of a linear transformer can be parallelized in training, as those of a transformer can, while at inference its cost in time and memory per prediction is constant, the matrix of the phi(K_j) V_j^T being kept as an internal state and updated at every step like that of a recurrent network. {#c-train-infer}
Unlike a [[recurrent-neural-network|recurrent network]] of the usual kind, the state is updated by addition only, with no nonlinearity between steps, which is what lets training compute all the $S_i$ at once.

**Chunks.**
Between the recurrence, $N$ sequential steps of $O(CM)$ work, and the masked quadratic form $\operatorname{tril}(\phi(Q)\phi(K)^\top) V$, $O(N^2)$ work in large matrix products, lies the chunked form.
Split the positions into chunks of length $T$; for a position $i$ in chunk $c$, whose first position is $t_c$,

$$ V'_i = \frac{\phi(Q_i)^\top S_{t_c - 1} + \sum_{t_c \le j \le i} \big(\phi(Q_i)^\top \phi(K_j)\big) V_j}{\phi(Q_i)^\top z_{t_c - 1} + \sum_{t_c \le j \le i} \phi(Q_i)^\top \phi(K_j)} , $$ {#eq-chunk}

the state being carried from chunk to chunk and the products inside a chunk done as a masked $T \times T$ matrix.
Splitting the sequence into chunks of length T, the causal output of a position is phi(Q_i)^T times the state left by the earlier chunks plus the masked T by T products inside its chunk, which equals the recurrence for every T that divides N. {#c-chunk}
$T = 1$ is the recurrence and $T = N$ the quadratic form; in between, each chunk is a few matrix products of size $T$, which suits accelerators better than $N$ small sequential steps.

## Derivation

**Associativity.**
If sim(q, k) = phi(q)^T phi(k), the attention output (phi(Q) phi(K)^T) V divided row by row by phi(Q) phi(K)^T 1 equals phi(Q) (phi(K)^T V) divided by phi(Q) (phi(K)^T 1), which costs O(N C M) instead of O(N^2 max(C, M)). {#c-assoc}
With causal masking, the numerator and the denominator of position i are phi(Q_i)^T S_i and phi(Q_i)^T z_i with S_i = S_{i-1} + phi(K_i) V_i^T and z_i = z_{i-1} + phi(K_i), so each position costs O(C M) whatever i is. {#c-causal-recurrence}
The state of a causal linear attention head is C M + C numbers whatever the number of positions read, against the N (D + M) numbers of the keys and values that causal softmax attention keeps after N positions. {#c-state-size}
The derivation `associativity` proves these and the chunked form, and its check runs them on random inputs.

**Softmax is not one of them.**
Softmax attention is the generalized attention with sim(q, k) = exp(q^T k / sqrt(D)); it is linear attention only for a feature map whose inner products are this exponential, and no feature map of finite size has that property for all queries and keys. {#c-softmax-kernel}
Katharopoulos et al. note that the feature function corresponding to the exponential kernel is infinite dimensional, which makes the linearization of exact softmax attention infeasible. {#c-exp-infinite}
Linear attention with elu + 1 is therefore a different model from softmax attention, not an approximation of it; the worked example below shows the two weighting three keys in different orders.

**The gradient in constant memory.**
Automatic differentiation of eq-causal through a cumulative sum keeps $S_1, \dots, S_N$, $NCM$ numbers per head.
Katharopoulos et al. note that a naive implementation of the cumulative sum, which stores all intermediate S_i, increases the memory consumption by max(D, M) times, and derive the gradients of the numerator as cumulative sums that are computed in linear time and constant memory with respect to the sequence length. {#c-alg1-paper}
Write $\bar n_i = \phi(Q_i)^\top S_i$ for the numerator and $G_i = \partial \mathcal L / \partial \bar n_i$. Then

$$ \frac{\partial \mathcal L}{\partial \phi(Q_i)} = S_i\, G_i , $$ {#eq-grad-q}

$$ \frac{\partial \mathcal L}{\partial \phi(K_j)} = R_j V_j , \qquad \frac{\partial \mathcal L}{\partial V_j} = R_j^\top \phi(K_j) , \qquad R_j = \sum_{i \ge j} \phi(Q_i)\, G_i^\top . $$ {#eq-grad-kv}

For the causal numerator nbar_i = phi(Q_i)^T S_i, the gradient with respect to phi(Q_i) is S_i G_i, and those with respect to phi(K_j) and V_j are R_j V_j and R_j^T phi(K_j) with R_j = sum_{i >= j} phi(Q_i) G_i^T, a sum computed backwards from the last position. {#c-alg1}
Computed with one running sum forwards and one backwards, the gradient of the causal numerator needs the inputs phi(Q), phi(K), V and one C by M state, not the N states S_1 to S_N that automatic differentiation through the cumulative sum keeps. {#c-alg1-memory}
This is Algorithm 1 of Katharopoulos et al.; the derivation `causal-gradient` derives it and checks it against finite differences.
The constant-memory gradient computation is implemented in approximately 200 lines of CUDA code. {#c-cuda}

**Counting the work.**
One head of non-causal softmax attention takes 2N^2 D + 2N^2 M + 2N^2 FLOPs and linear attention with a feature map of size C takes 4NCM + 4NC, counting a multiply-add as 2 FLOPs and only the matrix and vector products. {#c-flops}
Linear attention takes fewer FLOPs than softmax attention exactly when N > 2C(M + 1) / (D + M + 1); with C = D = M = 64 the first such N is 65. {#c-crossover}
Katharopoulos et al. give the cost of softmax attention as O(N^2 max(D, M)), with D the dimension of the queries and keys and M that of the values, and that of linear attention with a feature map of C dimensions as O(N C M). {#c-cost-paper}
The derivation `cost` counts both and checks the counts against the operations of the NumPy reference.

**An exact kernel.**
The polynomial kernel has an exact finite-dimensional feature map; the computational cost of a linearized polynomial transformer of degree 2 is of order N M times the square of D, which Katharopoulos et al. call favorable when N exceeds the square of D, true in practice for sequences of tens of thousands of elements. {#c-poly-paper}
The map phi(x) = (c, sqrt(2c) x, (x_a x_b)_{a,b}) of size 1 + D + D^2 satisfies phi(q)^T phi(k) = (q^T k + c)^2, so attention with the degree-2 polynomial kernel is linear attention with this map; at D = M = 64 it takes fewer FLOPs than softmax attention from N = 4194 on. {#c-poly-exact}

**Capacity.**
Schlag et al. argue that, since information is retrieved by matrix multiplication, keys must be orthogonal to retrieve without interference, that there cannot be more than d_dot orthogonal vectors in a d_dot-dimensional space, and that a linear Transformer reading a sequence longer than d_dot may therefore be in an overcapacity regime. {#c-capacity-paper}
Their $d_{\text{dot}}$ is the feature size $C$ here.
No state S in R^{C x M} maps more than C keys to linearly independent values, so a linear attention head with a feature map of size C cannot recall more than C arbitrary associations exactly; with orthonormal mapped keys the sum rule recalls C of them exactly, and otherwise returns each value plus the crosstalk of the others. {#c-capacity-rank}
Softmax attention has no such limit, because it keeps every key and value; its memory grows with the sequence instead.
The derivation `capacity` proves the bound and the two statements about feature maps and update rules below.

## Worked example

Take one query and three keys and values in two dimensions, with $\phi = \operatorname{elu} + 1$ and no mask:

$$ K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ -1 & -1 \end{pmatrix}, \qquad V = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 4 & 4 \end{pmatrix}, \qquad q = (0.5, -0.5) . $$

The feature map gives $\phi(K) = ((2, 1), (1, 2), (0.3679, 0.3679))$, since $\operatorname{elu}(-1) + 1 = e^{-1}$, and $\phi(q) = (1.5, 0.6065)$.
The state is $\phi(K)^\top V = ((3.4715, 3.4715), (2.4715, 5.4715))$ and $z = \phi(K)^\top \mathbf 1 = (3.3679, 3.3679)$.
The query reads $\phi(q)^\top \phi(K)^\top V = (6.7063, 8.5259)$ and $\phi(q)^\top z = 7.0945$, so the output is $(0.9453, 1.2018)$.
Grouped the other way, the similarities $\phi(q)^\top \phi(K_j)$ are $3.6065$, $2.7131$ and $0.7749$, the weights $0.5084$, $0.3824$ and $0.1092$, and the weighted mean of the values is the same $(0.9453, 1.2018)$: that is associativity.
Softmax attention with the same $q$ and $K$ scores $q^\top K_j / \sqrt 2 = 0.3536, -0.3536, 0$ and gives the weights $0.4555$, $0.2246$ and $0.3199$, and the output $(1.7350, 1.7287)$.
Softmax puts the third key second, linear attention puts it last: the two are different similarities, and a model trained with one does not compute the other.
The test `test_worked_example` checks every number.

## Hyperparameters

The **feature map** decides what the layer can compute.
It must give non-negative similarities, or the denominator can approach 0 and change sign; elu + 1 is positive everywhere and keeps a gradient for negative inputs.
Schlag et al. note that elu + 1, being an element-wise function, preserves the dimension of the key (d_key = d_dot), without changing the memory capacity. {#c-elu-dim}
DPFP, the deterministic parameter-free projection of Schlag et al., maps a key of dimension d_key to 2 d_key nu products of pairs of coordinates of (ReLU(k), ReLU(-k)), with nu a hyperparameter. {#c-dpfp}
With FAVOR+ the dimension of the feature space is 2m, which raises the theoretical capacity if 2m > d_key, but the capacity equals the infinite capacity of softmax memory only as m goes to infinity, and the random sampling is the main drawback of FAVOR+. {#c-favor}
The **feature size $C$** sets both the cost, $4NCM$, and the capacity, $C$ associations per head; for a fixed model width, fewer and wider heads hold more per head.
The **chunk length $T$** changes the speed and not the result: longer chunks do more of the work as large matrix products.
The **update rule** and the **normalisation** are the subject of Variants.

## Cost

Per head, linear attention takes $4NCM + 4NC$ FLOPs, against $2N^2(D + M + 1)$ for softmax attention; with $C = D = M = 64$ linear attention is cheaper from $N = 65$, and at $N = 4096$ it takes about 63 times fewer FLOPs.
The causal form costs the same in FLOPs, and the recurrence does it in $N$ steps of $O(CM)$ each.
Training memory per head is $N(2C + M)$ inputs with the backward pass of Algorithm 1, $NCM$ with automatic differentiation through the states, and $N^2$ for the masked quadratic form, which is still the fastest for short sequences.
Generation keeps $CM + C$ numbers per head and spends about $4CM$ FLOPs per token, whatever the position, where softmax attention keeps the $N(D + M)$ keys and values of a KV cache and spends $2N(D + M)$ FLOPs on the $N$-th token.

## Evidence

**What Katharopoulos et al. measured.**
Linear transformers achieve similar performance to vanilla transformers and are up to 4000x faster on autoregressive prediction of very long sequences. {#c-headline}
On a causal copy task with sequences of up to 128 symbols from 10, a 4-layer transformer with 8 heads converged smoothly with linear attention and reached the same loss as softmax attention, lower than Reformer, whose hashing adds noise. {#c-copy}
Computing attention and its gradients for sequence lengths from 2^9 to 2^16 on a GTX 1080 Ti with 11 GB, softmax attention fit up to 4,096 elements and Reformer with 4 or 8 hashing rounds up to 16,384; linear attention was faster and used less memory than both at every length, and both it and Reformer scaled linearly. {#c-memory-bench}
On autoregressive generation of MNIST, softmax attention reached 0.621 bits/dim at 0.45 images/sec, Reformer with 1 and 4 hashing rounds 0.745 and 0.676, and linear attention 0.644 at 142.8 images/sec, 317 times the throughput of softmax. {#c-mnist}
The linear transformer could generate 10,000 MNIST images at once on a single GPU, its memory being constant in the sequence length because only s_i and z_i are stored between pixels, while softmax and Reformer need memory that grows with the length. {#c-mnist-memory}
Trained for 7 days on one GPU to generate CIFAR-10, softmax attention reached 3.47 bits/dim at 0.004 images/sec and linear attention 3.40 at 17.85 images/sec, 4,462 times the throughput; the linear transformer completed 3 times more epochs, which gave the better score. {#c-cifar}
A softmax transformer that saves the keys and values and reuses them to predict the next element, stateful-softmax, generated 7.56 MNIST images/sec and 0.32 CIFAR-10 images/sec. {#c-stateful}
The complexity of stateful-softmax is still quadratic in the sequence length, and linear attention was more than 50 times faster than it on CIFAR-10. {#c-stateful-gap}
On WSJ speech recognition with CTC, softmax attention reached 5.12 phoneme error rate at 2711 seconds per epoch, linear attention 8.08 at 824, Reformer with 4 rounds 9.33 at 2250 and a 3-layer bidirectional LSTM 10.94 at 1047. {#c-wsj}
The speed-ups of thousands of times are against a softmax transformer that recomputes attention over all earlier pixels for every new one; against the cached keys and values of stateful-softmax the gap on CIFAR-10 is fifty times, and on speech recognition, with no generation, softmax attention was clearly more accurate.

**What Schlag et al. measured.**
On associative retrieval problems with 20 to 600 unique associations, linear attention, with a capacity of 64 from its key dimension of 64, begins to accumulate errors with 60 or more associations, DPFP projections 1, 2 and 3 start to accumulate errors as they approach their limits at 128, 256 and 384, FAVOR+ fails to achieve a loss of 0 in any experiment, and softmax attention outperforms all the feature functions, although it struggles to fully converge with more than 500 keys. {#c-capacity-exp}
On the WMT14 English to German translation task, with a key dimension of 64, the Transformer got 26.6 valid and 27.7 test, the Linear Transformer 25.5 and 26.8, and the Performer 24.2 and 24.4 at 64 and 26.7 and 27.7 at 512, while DPFP got 26.2 and 26.9 at 256. {#c-wmt}
On WikiText-103 with a context of 256 tokens, the small Transformer reached 33.0 valid and 34.1 test perplexity, the linear Transformer 37.1 and 38.3, and the linear Transformer with the delta rule 34.1 and 35.5; in the medium configuration with 384 tokens the numbers were 27.9 and 29.6, 31.1 and 33.0, and 29.7 and 31.5. {#c-wt103}
Trained and evaluated without truncating the context, the linear Transformer's perplexity on WikiText-103 exceeded 260 and the Delta Network reached 27.8 valid and 29.4 test, both with a state size of 0.13 (in millions). {#c-no-truncation}
In language modelling, where the context is longer than the feature size, the sum rule loses most and the delta rule recovers most of the gap; at equal context the softmax Transformer stayed best in every comparison.

## Limitations and failure modes

**A fixed-size memory.**
The state holds at most $C$ associations per head exactly; past that, every read returns a mixture.
This is the price of the constant cost per token, and it shows in tasks that need exact recall of something far back, such as copying a long sequence or retrieving a key from many: the Experiments below show a small model that copies 8 symbols perfectly and 32 symbols at little better than chance.

**No sharp weights.**
The similarity $\phi(q)^\top \phi(k)$ of elu + 1 is a sum of products of moderate numbers and cannot be made as peaked as an exponential, so a query has a hard time putting nearly all its weight on one key among many.

**Not a drop-in replacement.**
Because elu + 1 is not an approximation of the exponential, a model trained with softmax attention does not keep its behaviour when its attention is replaced, and has to be trained again.
Katharopoulos et al. suggest that approximating the RBF kernel with random Fourier features could allow linear attention to use models pretrained with softmax attention. {#c-future-rff}

**Short sequences.**
For $N$ below a few times the head width the quadratic form is no more expensive, and it runs as a few large matrix products; the gain is for long sequences and for generation.

**Numerics.**
The denominator is a sum of positive numbers that grows with $i$; in low precision the running sums lose resolution over very long sequences, and the normalisation by $z$ divides one large number by another.

## Variants

**Fast weight programmers and the delta rule.**
Schlag et al. show the formal equivalence of linearised self-attention and the fast weight controllers of the early nineties, in which a slow network learns by gradient descent to program the fast weights of another network through additive outer products of keys and values. {#c-fwp}
Schlag et al. conclude that the core of linear Transformer variants are outer product-based Fast Weight Programmers, with normalisation. {#c-fwp-core}
In their terms eq-rnn is the sum rule: every key and value is added to the fast weights, and nothing is ever removed.
Given a new key-value pair, the update rule of Schlag et al. first retrieves from the current state of the memory the value paired with the key, then writes the new value in its place with a dynamic learning rate, effectively a delta rule, so that the model learns to correct the key to value association. {#c-delta-paper}
In the notation of this page, with a rate $\beta_i \in (0, 1)$ computed from the input $x_i$ of the step by a sigmoid,

$$ \bar v_i = S_{i-1}^\top \phi(K_i) , \qquad S_i = S_{i-1} + \beta_i\, \phi(K_i) \big(V_i - \bar v_i\big)^\top , \qquad \beta_i = \sigma(w_\beta^\top x_i) . $$ {#eq-delta}

With beta = 1 and a key of unit norm in feature space, the delta rule replaces the value stored under that key by the new one, and the sum rule stores their sum; keys orthogonal to it are untouched by either rule. {#c-delta-overwrite}
The delta rule is no longer a sum, so its states cannot be computed with one cumulative sum; it is still linear in $N$.

**Other feature maps.**
DPFP and FAVOR+, described under Hyperparameters, enlarge $C$ to raise the capacity; the polynomial map of degree 2 makes attention with the kernel $(q^\top k + c)^2$ exact at $C = 1 + D + D^2$.

**Later models.**
Gated and decaying versions of eq-rnn multiply the state by a factor below 1 at every step, so old associations fade instead of crowding the state. The [[structured-state-space-model|structured state space models]] and [[mamba|Mamba]] also carry a fixed-size state updated linearly from step to step, with a decay that in Mamba depends on the input.

## When to use it

Use linear attention when sequences are long, thousands of positions or more, and the task depends on the gist of the context more than on exact recall of single items, or when generation must run at a fixed cost per token and a fixed memory, as on a small device or for very long outputs.
Prefer softmax attention for sequences of a few hundred positions, where it is cheap and fast enough, and for tasks that need exact retrieval from a long context; prefer a delta rule or a gated state when the context is much longer than the feature size.
Mixing the two, with softmax attention in a few layers or over a local window and linear attention elsewhere, is another option.

## Pitfalls

> [!pitfall]
> **ReLU as the feature map.** A query with no positive coordinate gets $\phi(q) = 0$ and an output of $0/0$. With phi = ReLU, a query with no positive coordinate has phi(q) = 0 and an output of 0 divided by 0, while with phi = elu + 1 every coordinate is positive and the denominator is a sum of positive terms. {#c-relu-zero} In three dimensions one random Gaussian query in eight is such a query; `test_relu_features_can_divide_zero_by_zero` shows the NaN rows are exactly those queries.

> [!pitfall]
> **Calling elu + 1 an approximation of softmax.** It is a different similarity. In the worked example the two rank the same three keys in different orders, and `test_softmax_is_the_exponential_kernel` checks that only the exponential kernel gives softmax attention.

> [!pitfall]
> **Differentiating the cumulative sum.** `torch.cumsum` over the outer products $\phi(K_j) V_j^\top$ is correct and keeps $NCM$ numbers per head for the backward pass. The hand-written backward of Algorithm 1 keeps $N(2C + M)$; `test_algorithm_1_keeps_only_its_inputs` counts what each saves.

> [!pitfall]
> **Expecting the state to hold the whole context.** A head with $C = 16$ holds 16 associations exactly, not 16 times the sequence length. The PyTorch test `test_linear_attention_runs_out_of_room_on_a_long_copy` trains models that fail at 32 symbols and succeed when the head is made wider.

> [!pitfall]
> **Keeping the $1/\sqrt{D}$ of softmax attention.** It has no role in eq-linear: scaling $\phi(q)$ by a constant leaves the output unchanged, as the numerator and the denominator both scale. Scaling the query inside elu + 1 does change the output, and should be left to the learned projection.

## Implementations

The NumPy reference `impl/numpy/linear_attention.py` implements the generalized attention eq-general, linear attention eq-linear and its causal form eq-causal as the recurrence eq-rnn, the gradient of the causal numerator by the two sums of Algorithm 1, eq-grad-q and eq-grad-kv, and a multi-head layer with projections, its backward pass by hand, and the layer as an RNN one token at a time.
It has the feature maps elu + 1, ReLU, the exact polynomial map of degree 2 and DPFP, the sum and delta rules eq-delta, and the FLOP counts of the Cost section.
The PyTorch version `impl/pytorch/linear_attention.py` has the causal product as an `autograd.Function` whose backward is Algorithm 1, the cumulative-sum and quadratic forms, the layer with a `step` for generation, and a small causal transformer that takes softmax or linear attention.
The JAX version `impl/jax/linear_attention.py` is a set of pure functions with the recurrence as `jax.lax.scan`, the chunked form eq-chunk as a scan over chunks, and the quadratic form, the gradients being left to `jax.grad`.
All three load the shared fixture of `fixtures/`: a two-head layer of width 8 run causally and not, its outputs, loss and gradients with respect to the weights and the input, and the output of softmax attention with the same weights.

## Experiments

The PyTorch test `test_linear_attention_runs_out_of_room_on_a_long_copy` trains a causal transformer of 2 layers to copy a sequence of random symbols from 9 after a separator, for 800 steps of Adam at a learning rate of 3e-3 on batches of 32, and counts the copied symbols of 256 fresh sequences.
With width 16 and 2 heads of width 8, over the seeds 0 to 2, softmax attention copied 32 symbols at 0.9999 to 1.0 and linear attention at 0.36 to 0.43, chance being 1/9.
With one head of width 16, linear attention got 0.40 to 0.45; at 16 symbols it got 0.78 to 0.89 and at 8 symbols 1.0.
With width 32 and one head of width 32 it copied 32 symbols at 1.0.
A head of width 8 cannot hold 32 associations, and widening the head, not adding heads of the same width, is what restores the copy, as the capacity bound says.
The checks of the four derivations verify associativity, the causal recurrence and the chunks, Algorithm 1 against finite differences, the FLOP counts and their crossovers, the rank bound, the zero denominators of ReLU and the overwriting of the delta rule on random inputs.

## History

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas and François Fleuret published Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention at ICML 2020. {#c-hist-katharopoulos}
Imanol Schlag, Kazuki Irie and Jürgen Schmidhuber published Linear Transformers Are Secretly Fast Weight Programmers at ICML 2021. {#c-hist-schlag}
Schlag et al. write that context-dependent Fast Weight Programmers were introduced in the two-network systems of Schmidhuber of 1991 to 1993, in which a slow net learns to program the fast weights of a fast net, one of the proposed instructions being an outer product. {#c-hist-fwp}

## References

- A. Katharopoulos, A. Vyas, N. Pappas and F. Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020; arXiv 2006.16236.
- I. Schlag, K. Irie and J. Schmidhuber. Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021; arXiv 2102.11174.
- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017; arXiv 1706.03762.
- K. Choromanski et al. Rethinking Attention with Performers. ICLR 2021; arXiv 2009.14794.
- N. Kitaev, Ł. Kaiser and A. Levskaya. Reformer: The Efficient Transformer. ICLR 2020; arXiv 2001.04451.
- D.-A. Clevert, T. Unterthiner and S. Hochreiter. Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). ICLR 2016; arXiv 1511.07289.
