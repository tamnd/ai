---
tag: NEW-multi-head-attention
kind: method
slug: multi-head-attention
name: Multi-head attention
aliases: [MHA, multi-headed attention]
area: deep-learning-architectures/attention-mechanisms
level: core
summary: >-
  Run h scaled dot-product attentions in parallel, each on its own learned projections of the queries, keys and values, then concatenate the h outputs and project them back to the model width.
purpose: let each position of a sequence gather information from other positions, or from a second sequence, with several weightings at once
uses: [dot-product@m2, matrix-multiplication@m2, softmax-regression@al, backpropagation@al]
related_to: [transformer, beam-search@al]
hyperparameters:
  - {name: h, meaning: "the number of heads", typical: "8 in the base Transformer of Vaswani et al.; 1 recovers single-head attention"}
  - {name: d_k and d_v, meaning: "the width of the queries and keys, and of the values, in each head", typical: "d_model / h, so that the total cost stays that of one head of width d_model; 64 in the base Transformer"}
  - {name: mask, meaning: "which keys each query may attend to", typical: "causal in a decoder, padding over the keys of a batch of unequal lengths, none in an encoder"}
  - {name: key-value heads, meaning: "the number of distinct key and value projections, in the grouped-query variant", typical: "h for multi-head, 1 for multi-query, between them for grouped-query"}
cost:
  train: "8 B T d^2 + 4 B T^2 d FLOPs for self-attention over B sequences of length T at width d, with d_k = d_v = d / h; the weights take B h T^2 numbers if they are stored for the backward pass"
  inference: "in incremental decoding, each new token attends to the cached keys and values of all earlier tokens, 2 h d_k numbers per token and per layer for multi-head attention"
history:
  year: 2017
  people: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
  note: "Vaswani et al. (2017), arXiv 1706.03762, Section 3.2; attention in neural translation goes back to Bahdanau, Cho and Bengio (2015) and Luong, Pham and Manning (2015)."
prereqs: [dot-product@m2, matrix-multiplication@m2, softmax-regression@al, backpropagation@al, embedding@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1706.03762"]
sources:
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "arXiv 1706.03762, Sections 3.2.1 to 3.2.3, 4 and 6.2, Tables 1 and 3, footnote 4 on p. 4", note: "scaled dot-product and multi-head attention, the masks, the cost and the ablation of the heads"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "Section 3.1 and p. 12", note: "additive attention in an encoder-decoder network"}
  - {paper: "Luong, Pham and Manning (2015)", title: "Effective Approaches to Attention-based Neural Machine Translation", locator: "EMNLP 2015, Sections 3.1 and 5.3", note: "global and local attention, and the dot, general and concat scores"}
  - {paper: "Shazeer (2019)", title: "Fast Transformer Decoding: One Write-Head is All You Need", locator: "arXiv 1911.02150, Sections 2.4.1, 3, 3.1, 4.2 and 4.3", note: "multi-query attention"}
  - {paper: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón and Sanghai (2023)", title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", locator: "arXiv 2305.13245, the abstract, Sections 2.1 and 2.2", note: "grouped-query attention and uptraining"}
  - {paper: "Dao, Fu, Ermon, Rudra and Ré (2022)", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", locator: "the abstract, Sections 2.2 and 3.1", note: "the memory of the standard implementation, and exact attention in linear extra memory"}
---

Multi-head attention is the layer that lets each position of a sequence read from other positions.
Each query is compared with every key by a scaled dot product, a softmax turns the scores into weights, and the output is the weighted mean of the values.
Multi-head attention runs $h$ such attentions side by side on different learned projections and joins their outputs.
It is the attention layer of the [Transformer](/transformer/), where it serves as self-attention in the encoder and the decoder and as cross-attention from the decoder to the encoder.

## In plain words

Picture a row of words.
Each word asks a question, its query, and each word also carries a label, its key, and some content, its value.
A word compares its query with every key, and the better the match, the more of that word's value it takes.
The result is a weighted mix of the values, with weights that add up to 1.
Multi-head attention runs several comparisons at once, each with its own learned way to form queries, keys and values, so one head can follow one kind of relation and another head a different one.
The outputs of the heads are glued together and mixed back to the original width.
A mask can forbid some matches: a word may not look at later words when the model writes text left to right, and nothing may look at the padding added to make sequences the same length.

## Setting

The layer maps a sequence of $T$ query vectors and a sequence of $S$ memory vectors, all of width $d_{\text{model}}$, to $T$ output vectors of width $d_{\text{model}}$.
Write $X \in \mathbb{R}^{T \times d_{\text{model}}}$ for the rows that give the queries and $M \in \mathbb{R}^{S \times d_{\text{model}}}$ for the rows that give the keys and values.
In self-attention $M = X$, so every position attends to the positions of the same sequence.
In cross-attention $M$ is another sequence, such as the output of an encoder.
Vaswani et al. use the layer in three places: self-attention in the encoder, self-attention in the decoder, and attention from the decoder to the encoder, where the decoder supplies the queries and the encoder output supplies the keys and values. {#c-three-uses}
The layer has no recurrence and no notion of order, so a model that needs order adds it to the inputs, as the Transformer does with its positional encodings.

## The method

**Scaled dot-product attention.**
Given queries $Q \in \mathbb{R}^{T \times d_k}$, keys $K \in \mathbb{R}^{S \times d_k}$ and values $V \in \mathbb{R}^{S \times d_v}$, the scores are

$$ Z = \frac{Q K^\top}{\sqrt{d_k}} , $$ {#eq-scores}

the weights are the softmax of each row of $Z$, over the keys,

$$ A_{ts} = \frac{\exp(Z_{ts})}{\sum_{u} \exp(Z_{tu})} , $$ {#eq-softmax}

and the output is

$$ \mathrm{Attention}(Q, K, V) = A V = \mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) V . $$ {#eq-attention}

This is Eq. 1 of Vaswani et al., who call it scaled dot-product attention. {#c-attention-eq}
Row $t$ of the output is a convex combination of the rows of $V$, because the weights of a row are nonnegative and sum to 1. {#c-convex}

**Heads.**
With $h$ heads, head $r$ has its own projections $W^Q_r, W^K_r \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V_r \in \mathbb{R}^{d_{\text{model}} \times d_v}$:

$$ \mathrm{head}_r = \mathrm{Attention}(X W^Q_r,\; M W^K_r,\; M W^V_r) . $$ {#eq-heads}

The heads are concatenated and multiplied by $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$:

$$ \mathrm{MultiHead}(X, M) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O . $$ {#eq-multihead}

These are the equations of Section 3.2.2 of Vaswani et al., with the projections applied to the queries, keys and values $h$ times in parallel. {#c-multihead-def}
Splitting $W^O$ into blocks $W^O_r$ of $d_v$ rows, the concatenation followed by $W^O$ is the sum $\sum_r \mathrm{head}_r W^O_r$, so each head writes its own contribution into the output. {#c-head-sum}
Vaswani et al. say that multiple heads let the model jointly attend to information from different representation subspaces at different positions, and that with a single head, averaging inhibits this. {#c-subspaces}
The base model uses $h = 8$ and $d_k = d_v = d_{\text{model}}/h = 64$, and because each head is narrower, the total cost is similar to that of one head of full width. {#c-base-heads}
The layer has $2 d_{\text{model}} h (d_k + d_v)$ weights, which is $4 d_{\text{model}}^2$ for every $h$ when $d_k = d_v = d_{\text{model}}/h$. {#c-param-count}

**Masks.**
A mask says which keys each query may attend to.
Its entries are applied to the scores before the softmax: an allowed entry is left alone and a forbidden one is set to $-\infty$, so its weight is exactly 0 and the remaining weights still sum to 1.
In the decoder of Vaswani et al., the scaled dot-product attention masks out, by setting to $-\infty$, all the values of the softmax input that would let a position see later positions, which keeps the model auto-regressive. {#c-mask-minus-inf}
With this causal mask, row $t$ of self-attention depends only on rows 1 to $t$ of the input. {#c-causal-independence}
A padding mask forbids the keys that only pad a sequence to the length of the batch.
If it allows exactly the first $L$ keys, the output is the output of the same layer on the first $L$ keys with no mask. {#c-padding-truncation}

**Order.**
Without positional information inside the layer, self-attention is permutation equivariant: permuting the input rows permutes the output rows in the same way. {#c-equivariant}
Cross-attention does not depend on the order of the memory at all: permuting the rows of $M$ leaves the output unchanged. {#c-cross-invariant}

## Derivation

**Why divide by $\sqrt{d_k}$.**
Vaswani et al. suspect that for large $d_k$ the dot products grow large in magnitude and push the softmax into regions where it has extremely small gradients. {#c-scale-reason}
To illustrate, their footnote 4 assumes that the components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with mean 0 and variance 1, so that $\mathbf{q} \cdot \mathbf{k}$ has mean 0 and variance $d_k$. {#c-footnote}
The calculation is short.
Each term $q_c k_c$ has mean $\mathbb{E}[q_c]\mathbb{E}[k_c] = 0$ and variance $\mathbb{E}[q_c^2]\mathbb{E}[k_c^2] = 1$, and the $d_k$ terms are independent, so their sum has variance $d_k$; dividing by $\sqrt{d_k}$ brings the variance back to 1 for every $d_k$. {#c-dot-variance}
The softmax side is a bound.
The Jacobian of a row of weights $\mathbf{a}$ with respect to its scores is $\mathrm{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^\top$.
If one of $n$ scores exceeds every other score of its row by at least $\Delta$, every other weight is at most $e^{-\Delta}$ and every entry of this Jacobian has absolute value at most $(n-1)e^{-\Delta}$. {#c-softmax-saturation}
Unscaled scores have standard deviation $\sqrt{d_k}$ under the footnote's assumption, so their gaps grow with the width, and the bound on the gradient through the softmax falls exponentially in those gaps.
The derivation `scale` gives the steps and a Monte Carlo check.

**The backward pass.**
Write $\bar{U}$ for the gradient of the loss with respect to a matrix $U$, and fix one head.
With $C$ the concatenation of the heads and $\bar{O}$ the columns of $\bar{Y} W^{O\top}$ that belong to the head,

$$ \bar{W}^O = C^\top \bar{Y}, $$ {#eq-grad-out}

$$ \bar{V} = A^\top \bar{O}, $$ {#eq-grad-v}

$$ \bar{A} = \bar{O} V^\top, $$ {#eq-grad-weights}

$$ \bar{Z}_{ts} = A_{ts}\Big(\bar{A}_{ts} - \sum_u A_{tu}\bar{A}_{tu}\Big), $$ {#eq-grad-scores}

$$ \bar{Q} = \frac{\bar{Z} K}{\sqrt{d_k}}, \qquad \bar{K} = \frac{\bar{Z}^\top Q}{\sqrt{d_k}} , $$ {#eq-grad-qk}

and the projection weights get $X^\top \bar{Q}$, $M^\top \bar{K}$ and $M^\top \bar{V}$ for their columns of the head. {#c-backward}
Every row of $\bar{Z}$ sums to 0, the gradient form of the fact that adding a constant to a row of scores leaves its softmax unchanged, and a masked score gets a zero gradient. {#c-score-grad-sum}
In self-attention $X$ reaches the output through the queries, the keys and the values, and its gradient is the sum of the three paths.
The derivation `backward` gives the steps and a finite-difference check.

**Counts.**
The derivation `properties` proves the head sum, the masks, the order properties and the counts used on this page.

## Worked example

Take three tokens of width $d_{\text{model}} = 2$, $\mathbf{x}_1 = (1, 0)$, $\mathbf{x}_2 = (0, 1)$, $\mathbf{x}_3 = (1, 1)$, self-attention, and every projection the identity.
With $h = 2$ heads and $d_k = d_v = 1$, head 1 sees the first coordinate and head 2 the second, and the scale is $1/\sqrt{1} = 1$.

In head 1 the queries, keys and values are all $(1, 0, 1)$.
Token 1 has the scores $(1, 0, 1)$ and the weights $(e, 1, e)/(2e + 1) \approx (0.4223, 0.1554, 0.4223)$, so its head output is $2e/(2e+1) \approx 0.8446$.
Token 2 has the query 0, so all its scores are 0, its weights are uniform, and its output is $2/3$.
Token 3 has the same query as token 1 and the same output.
In head 2 the roles swap: the values are $(0, 1, 1)$, token 1 has uniform weights and the output $2/3$, and tokens 2 and 3 get $0.8446$.
With $W^O = I$ the outputs are $(0.8446, 0.6667)$, $(0.6667, 0.8446)$ and $(0.8446, 0.8446)$.
Token 1 spreads its attention evenly in head 2 and leans to tokens 1 and 3 in head 1: two weightings in one layer.

With one head of width $d_k = 2$, the scale is $1/\sqrt{2}$.
Token 1 has the scores $(1, 0, 1)/\sqrt{2}$, the weights $(0.4011, 0.1978, 0.4011)$ and the output $(0.8022, 0.5989)$; token 3 has the weights $(0.2483, 0.2483, 0.5035)$ and the output $(0.7517, 0.7517)$.

With the causal mask and two heads, token 1 can only see itself, and its output is $(1, 0)$.
Token 2 sees tokens 1 and 2: head 1 has the query 0 and the weights $(0.5, 0.5)$, and head 2 has the scores $(0, 1)$ and the weights $(1, e)/(1 + e) \approx (0.2689, 0.7311)$, so its output is $(0.5, 0.7311)$.
Token 3 sees everything and keeps its output $(0.8446, 0.8446)$.
These numbers were computed by hand and checked by the NumPy reference of this page; its tests assert them.

## Hyperparameters

The number of heads $h$ and the widths $d_k$ and $d_v$ are the main choices.
The usual rule, from the base Transformer, is $d_k = d_v = d_{\text{model}}/h$, which keeps the number of weights at $4 d_{\text{model}}^2$ and the cost close to that of one wide head whatever $h$ is.
The mask is part of the task rather than a tuning knob: causal for a decoder that generates left to right, padding for any batch of unequal lengths.
The variants below add one more choice, the number of distinct key-value heads.

## Cost

Let $B$ be the batch size, $T$ the number of queries, $S$ the number of keys and $d = d_{\text{model}}$.
Counting a multiply-add as 2 FLOPs and leaving out the softmax, the forward pass costs $2BTdhd_k + 2BSdh(d_k + d_v) + 2BTdhd_v + 2BhTS(d_k + d_v)$ FLOPs; for self-attention with $d_k = d_v = d/h$ this is $8BTd^2 + 4BT^2d$. {#c-flops}
The first term is the four projections and the second is the scores and the weighted sums.
By this count, the second term is the larger one once $T > 2d$.
The weights $A$ have $BhTS$ entries, so a naive implementation that keeps them for the backward pass needs memory quadratic in the sequence length.
For example, with $h = 32$ heads and $T = S = 4096$ the weights of one sequence in one layer are $32 \cdot 4096^2 \approx 5.4 \times 10^8$ numbers, 1 GiB in a 16-bit format; this is our own arithmetic.

In the complexity table of Vaswani et al., a self-attention layer costs $O(n^2 \cdot d)$ per layer with $O(1)$ sequential operations and a maximum path length of $O(1)$, against $O(n \cdot d^2)$, $O(n)$ and $O(n)$ for a recurrent layer. {#c-table1}
Self-attention layers are therefore faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$. {#c-faster-than-rnn}
Dao et al. (2022) point out that self-attention needs time and memory that grow with the square of the sequence length, and that standard implementations write the matrices of scores and weights to GPU high bandwidth memory, which takes $O(N^2)$ memory. {#c-quadratic}
Their FlashAttention computes the same exact output with $O(N^2 d)$ FLOPs and $O(N)$ additional memory beyond the inputs and the output. {#c-flash}

In incremental decoding the keys and values of earlier tokens are cached.
Each cached position of a layer holds $2 h d_k$ numbers for multi-head attention, $2 d_k$ for multi-query attention and $2 G d_k$ for grouped-query attention with $G$ groups. {#c-kv-cache}
Shazeer (2019) finds that incremental multi-head attention has a ratio of memory access to arithmetic of $\Theta(n/d + 1/b)$, for sequence length $n$ and batch size $b$, so memory bandwidth limits it when $n$ is near $d$ or $b$ is near 1. {#c-mha-ratio}
With multi-query attention the ratio becomes $\Theta(1/d + n/(dh) + 1/b)$, the $n/d$ term divided by $h$. {#c-mqa-ratio}

## Evidence

**The number of heads.**
In the ablation of Vaswani et al. on English-to-German newstest2013, with the computation held constant, $(h, d_k)$ of $(1, 512)$, $(4, 128)$, $(8, 64)$, $(16, 32)$ and $(32, 16)$ gave BLEU 24.9, 25.5, 25.8, 25.8 and 25.4. {#c-table3-heads}
Single-head attention was 0.9 BLEU worse than the best setting, and quality also dropped off with too many heads. {#c-single-head-worse}
Reducing $d_k$ alone hurt the model quality, which the authors take to suggest that determining compatibility is not easy. {#c-dk-reduction}

**Score functions.**
In the experiments of Luong et al. (2015), the dot score worked well for global attention and the general score was better for local attention. {#c-luong-dot-general}

**Sharing keys and values.**
In the translation experiments of Shazeer (2019), multi-query attention was slightly worse than the multi-head baseline, but much closer to it than the alternatives that reduce $h$, $d_k$ and $d_v$. {#c-mqa-quality}
On the test set, with beam search of width 4, the multi-query model performed similarly to the baseline and had the highest BLEU score, 28.5. {#c-mqa-beam}
On one TPUv2 with 8 cores, incremental greedy decoding took 46 µs per token in the decoder of the baseline and 3.8 µs with multi-query attention, while a training step took 433 ms and 425 ms. {#c-mqa-speed}
Ainslie et al. (2023) report that grouped-query attention, uptrained from a multi-head checkpoint, reaches quality close to multi-head attention at a speed comparable to multi-query attention. {#c-gqa-result}

## Limitations and failure modes

- **Quadratic cost.** The scores cost $O(T^2 d)$ and the weights take $O(T^2)$ memory per head, which dominates for long sequences. FlashAttention removes the quadratic memory of the weights but not the quadratic number of FLOPs.
- **Decoding is memory-bound.** Each generated token reads the whole cache of keys and values, which is what multi-query and grouped-query attention shrink.
- **No order of its own.** The layer is permutation equivariant, so it cannot tell "dog bites man" from "man bites dog" without positional information added from outside.
- **Saturation.** When one score leads its row by a wide margin the softmax is nearly one-hot and its gradient is nearly zero, by the bound above; the scale prevents this at initialization under the footnote's assumption, not throughout training.

## Variants

**Additive attention.**
Bahdanau et al. (2015) score the previous decoder state $\mathbf{s}_{i-1}$ against each encoder annotation $\mathbf{h}_j$ with $\mathbf{v}_a^\top \tanh(W_a \mathbf{s}_{i-1} + U_a \mathbf{h}_j)$, a feedforward network with one hidden layer. {#c-bahdanau-additive}
Vaswani et al. note that the two most common attention functions are additive and dot-product attention, and that dot-product attention is their algorithm without the scaling factor. {#c-additive-vs-dot}
In practice the dot-product form runs much faster and uses less space, since optimized matrix multiplication code computes it. {#c-dot-faster}
The two do about equally well when $d_k$ is small, but when $d_k$ is large the additive form beats the unscaled dot product. {#c-additive-large-dk}

**Multiplicative attention.**
Luong et al. (2015) compare three scores between the current target state $\mathbf{h}_t$ and a source state $\bar{\mathbf{h}}_s$: dot, $\mathbf{h}_t^\top \bar{\mathbf{h}}_s$; general, $\mathbf{h}_t^\top W_a \bar{\mathbf{h}}_s$; and concat, $\mathbf{v}_a^\top \tanh(W_a[\mathbf{h}_t; \bar{\mathbf{h}}_s])$. {#c-luong-scores}

**Multi-query attention.**
Shazeer (2019) keeps the $h$ query heads but lets all heads share a single set of keys and values. {#c-mqa-def}

**Grouped-query attention.**
Ainslie et al. (2023) divide the query heads into $G$ groups, each of which shares a single key head and value head; one group is multi-query attention and $h$ groups is multi-head attention. {#c-gqa-def}
Going from multi-head to multi-query attention cuts the key-value cache by a factor of $h$. {#c-gqa-cache}
To convert a multi-head checkpoint, they mean-pool the key and value projections of the heads that will share them. {#c-gqa-meanpool}
Then they continue pre-training, which they call uptraining, with 5% of the original pre-training compute. {#c-gqa-uptrain}
They do not apply it to the encoder self-attention layers, which are computed in parallel, so that memory bandwidth is generally not their bottleneck. {#c-gqa-not-encoder}

**Restricted self-attention.**
Vaswani et al. also list self-attention restricted to a neighbourhood of size $r$ around each position, which costs $O(r \cdot n \cdot d)$ per layer and raises the maximum path length to $O(n/r)$. {#c-restricted}

## When to use it

Use multi-head attention when each element of a sequence must draw on other elements whose positions are not fixed in advance: tokens of text, patches of an image, frames of audio.
Use self-attention to mix one sequence and cross-attention to let one sequence read another, as a decoder reads its encoder.
For autoregressive generation, apply the causal mask in training and cache the keys and values at inference; if the cache is the bottleneck, grouped-query or multi-query attention trade a little quality for a smaller cache.
For very long sequences, the quadratic cost is the limit; an IO-aware kernel helps memory, and restricted attention helps compute.

## Pitfalls

- **The scale is $\sqrt{d_k}$, the width of one head**, not $\sqrt{d_{\text{model}}}$.
- **Softmax over the keys.** The normalization runs over the last axis of the $T \times S$ score matrix; a softmax over the queries gives weights that do not sum to 1 per query.
- **Mask before the softmax.** Zeroing weights after the softmax leaves rows that no longer sum to 1.
- **Rows with nothing to attend to.** With every key masked, a naive softmax computes $-\infty - (-\infty)$ and returns NaN. The references of this page return a zero row and a zero gradient.
- **Finite mask values.** A large negative constant in place of $-\infty$ must be representable: the largest finite 16-bit float is 65504, as `numpy.finfo(numpy.float16).max` reports.
- **Mask conventions differ.** In this page a boolean mask is True where attention is allowed. In `torch.nn.MultiheadAttention` a True entry of a boolean `attn_mask` or `key_padding_mask` marks a position that is not allowed; the PyTorch tests of this page pass the complement and match it.
- **Splitting heads.** Reshape $(B, T, h d_k)$ to $(B, T, h, d_k)$ and then move the head axis; reshaping straight to $(B, h, T, d_k)$ mixes positions across heads.
- **Self-attention gradient.** The input feeds the queries, the keys and the values, and its gradient is the sum of all three paths.

## History

Bahdanau, Cho and Bengio (2015) added attention to an encoder-decoder network for translation.
Their context vector is a weighted sum $\mathbf{c}_i = \sum_j \alpha_{ij} \mathbf{h}_j$ of the encoder annotations, with weights from a softmax over the scores $e_{ij} = a(\mathbf{s}_{i-1}, \mathbf{h}_j)$ of an alignment model trained jointly with the rest of the network. {#c-bahdanau-context}
This frees the encoder from having to encode all the information of a source sentence into a fixed-length vector. {#c-bahdanau-fixed}
Luong, Pham and Manning (2015) computed the attention from the current target state rather than the previous one, and compared the dot, general and concat scores, where Bahdanau et al. had experimented with the concat form only. {#c-luong-vs-bahdanau}
Vaswani et al. (2017) scaled the dot product by $1/\sqrt{d_k}$, ran $h$ attentions in parallel on learned projections, and built the Transformer from attention alone.
They observed that individual heads clearly learn to perform different tasks, and that many appear to behave in ways related to the syntactic and semantic structure of the sentences. {#c-heads-tasks}
To speed up incremental decoding, Shazeer (2019) shared the keys and values across heads, and Ainslie et al. (2023) interpolated between the two with grouped-query attention.
Dao et al. (2022) kept the mathematics and changed the implementation, computing exact attention without storing the weights.
