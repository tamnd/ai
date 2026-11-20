---
tag: NEW-t5
kind: architecture
slug: t5
name: T5
aliases: [Text-to-Text Transfer Transformer, T5-Small, T5-Base, T5-Large, T5-3B, T5-11B, text-to-text Transformer]
area: language-models/encoder-decoder-models
level: core
summary: >-
  An encoder-decoder Transformer that casts every language task as text to text, with pre-norm layers, scalar relative position biases and span-corruption pre-training.
uses: [matrix-multiplication@m2, conditional-probability@m2, cross-entropy@m2, linearity-of-expectation@m2]
related_to: [transformer, sequence-to-sequence-model, bert, decoder-only-transformer, masked-language-modelling, rms-normalization, rotary-position-embedding, byte-pair-encoding, neural-scaling-laws, autoencoder]
history:
  year: 2019
  people: [Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu]
  note: "Raffel et al. posted Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer to arXiv (1910.10683) in October 2019, and it appeared in the Journal of Machine Learning Research in 2020; Lewis et al. posted BART, another pre-trained encoder-decoder, in the same month."
prereqs: [transformer, sequence-to-sequence-model, multi-head-attention, rms-normalization, masked-language-modelling]
same_as:
  wikipedia: T5_(language_model)
  arxiv: ["1910.10683"]
sources:
  - {paper: "Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li and Liu (2020)", title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", locator: "JMLR 21; arXiv 1910.10683v4, Sections 2, 3.1 to 3.3 and 3.7, Tables 2, 14 and 15", note: "the architecture, the text-to-text format, the comparison of architectures and objectives, span corruption and the five model sizes"}
  - {paper: "Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov and Zettlemoyer (2020)", title: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", locator: "ACL 2020, pp. 7871 to 7873", note: "a pre-trained encoder-decoder of the same period, with text infilling"}
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "arXiv 1706.03762v7, pp. 3, 5 and 6", note: "the original Transformer: normalization after the residual sum, tied embeddings, sinusoidal positions"}
  - {paper: "Zhang and Sennrich (2019)", title: "Root Mean Square Layer Normalization", locator: "arXiv 1910.07467v1, p. 1", note: "the normalization without centring"}
  - {paper: "Kudo and Richardson (2018)", title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", locator: "p. 1", note: "the tokenizer of the vocabulary"}
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "arXiv 1810.04805v2, p. 4", note: "the masked language model that span corruption is compared with"}
family: transformer
inputs: "a batch of source token ids as a (B, S) integer array, and for training the target token ids as a (B, T) integer array"
outputs: "a (B, T, V) array of logits: row t scores every token of the vocabulary as target token t, given the source and the target tokens before t"
components: [multi-head-attention, multilayer-perceptron, rms-normalization, residual-connection]
inductive_biases: ["an encoder that reads the whole source in both directions and a causal decoder that reads it through attention, so the input and the output are different sequences", "position only as a learned scalar per head added to each attention score, chosen by the bucket of the offset between key and query, so a layer sees offsets and not places", "buckets exact for small offsets and logarithmic up to a maximum distance, beyond which a layer cannot tell offsets apart", "one vocabulary and one embedding for the input, the decoder input and the output layer"]
params: "V M + 2 K H + L (12 M I + 4 M F + 5 M) + 2 M with I = H d_kv, for a vocabulary of V, L encoder and L decoder layers of width M with H heads of d_kv features and a feed-forward width F, and position tables of K buckets"
flops_per_example: "L [S (8 M I + 4 M F) + T (12 M I + 4 M F) + 4 S M I + 4 I (S^2 + S T + T^2)] + 2 T M V for S source and T target tokens, counting a multiply-add as 2 FLOPs and only the matrix products, with the attention computed in full"
---

T5 is an encoder-decoder [[transformer|Transformer]] trained to map an input text to an output text, which Raffel et al. used in 2019 to cast every language task they studied, translation, classification, question answering and summarization, in one format.
An encoder reads the input in both directions; a decoder writes the output one token at a time with causal self-attention and attention to the encoder.
The layers differ from the original Transformer in three ways: the normalization comes before each sublayer and has no bias, position enters only as a learned scalar added to each attention score by the bucket of the key-query offset, and the output layer is the embedding transposed.
The paper compared architectures, objectives, data and sizes in one setting and trained five models, from 60 million to 11 billion parameters, on about a trillion tokens of filtered web text with a span-corruption objective.

## In plain words

Every task becomes the same kind of task: read some text, write some text.
To translate, the model reads "translate English to German: That is good." and writes "Das ist gut."; to classify, it writes the name of the class as a word.
The model has two halves.
The encoder reads the whole input at once, each token looking at all the others; the decoder writes the answer a token at a time, each new token looking back at the tokens it has written and across at everything the encoder produced.
To learn language before any task, the model reads web text with some short stretches cut out and replaced by placeholders, and writes the missing stretches, each after its placeholder.
Instead of being told where each token is, the model is told how far apart two tokens are, and only roughly when they are far apart.

## Overview

The paper casts all tasks into a text-to-text format: the model is fed some text for context or conditioning and asked to produce some output text, which gives the same objective, maximum likelihood with teacher forcing, for pre-training and for every task, and a text prefix added to the input names the task. {#c-text-to-text}
To translate "That is good." the model is fed "translate English to German: That is good." and trained to output "Das ist gut."; for classification it predicts one word, such as "entailment" for the MNLI input "mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity." {#c-prefix-example}
The network is the [[sequence-to-sequence-model|sequence-to-sequence]] Transformer: an encoder of self-attention and feed-forward blocks, with each sublayer's input normalized by a rescaling with no additive bias and a residual connection around it, {#c-arch-blocks} and a decoder of the same blocks with a causal self-attention and an attention to the encoder output after it, whose last output goes through a dense layer with weights shared with the input embedding. {#c-arch-decoder}
The authors describe it as roughly the original Transformer except for removing the Layer Norm bias, placing the normalization outside the residual path, and a different position scheme, and leave ablating these changes for future work. {#c-vaswani-diff}
In that original, the output of each sublayer is $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$, with the normalization after the residual sum, {#c-vaswani-postnorm} and positions enter as sine and cosine functions of different frequencies added to the embeddings. {#c-vaswani-sinusoid}

Of the two halves, the encoder is the bidirectional stack of [[bert|BERT]] and the decoder the causal stack of a [[decoder-only-transformer|decoder-only language model]], joined by attention from the second to the first.
The input and the output are separate sequences, so the decoder never spends attention or a feed-forward pass on the input tokens, and the encoder reads them without a mask.
The page follows Section 2.1 of the paper with one table of position biases per stack, the reading of the paper's "share the position embedding parameters across all layers" used here; the reference has a switch pos="none" that drops the tables, for the inductive biases below.

## Components

The notation uses row vectors; the source is $s_1, \dots, s_S$, the target $t_1, \dots, t_T$, $V$ is the size of the vocabulary, $M$ the width, $H$ the number of heads with $d_{kv}$ features each, $I = H d_{kv}$ and $F$ the feed-forward width.
No projection has a bias.

**Embedding.**
One table $E \in \mathbb{R}^{V \times M}$ embeds the source tokens, embeds the decoder's input tokens, and, transposed, gives the logits.
The original Transformer also shares one matrix between its two embedding layers and the pre-softmax linear transformation, and there multiplies the embedding weights by $\sqrt{d_{\text{model}}}$. {#c-vaswani-tied}

**Normalization.**
Each token's vector is divided by its root mean square and multiplied by a learned gain $g \in \mathbb{R}^{M}$; there is no centring and no bias:

$$ \operatorname{N}(x) = g \odot \frac{x}{\sqrt{\tfrac{1}{M} \sum_{j=1}^{M} x_j^2 + \epsilon}} , \qquad \epsilon = 10^{-6} . $$ {#eq-norm}

This is the [[rms-normalization|RMS normalization]]; the paper describes it only as layer normalization in which the activations are rescaled and no bias is added.
Zhang and Sennrich proposed it as RMSNorm, on the hypothesis that the re-centring of layer normalization is dispensable; it keeps the invariance to re-scaling, and they report accuracy comparable to LayerNorm with the running time cut by 7% to 64% on different models. {#c-rmsnorm}

**Position biases.**
Each stack has a table $R \in \mathbb{R}^{K \times H}$.
Instead of a fixed embedding per position, a scalar is added to the attention logit of each head, chosen by the offset between key and query; the table is shared by the layers, each head has its own column, and 32 buckets cover offsets with ranges that grow logarithmically up to 128, beyond which all offsets share one bucket. {#c-relpos}
The paper gives only this outline.
The page uses the following bucket function: for a decoder query at $i$ and key at $j \le i$, with $n = i - j$,

$$ \beta(n) = \begin{cases} n & n < K/2 \\ \min\!\Big(K - 1,\ \tfrac{K}{2} + \Big\lfloor \tfrac{K}{2} \tfrac{\log(2n/K)}{\log(2 d_{\max}/K)} \Big\rfloor\Big) & n \ge K/2 \end{cases} $$ {#eq-bias}

and the encoder gives half the buckets to each direction: a key before the query at distance $n$ goes to $\beta(n)$ computed with $K/2$ buckets, and a key after it to $K/2$ plus that.
The bias of head $h$ is $R[\beta, h]$; attention to the encoder has none.

**Attention.**
A head of queries from $X$ on keys and values from $Y$ is

$$ \operatorname{Attn}_h(X, Y) = \operatorname{softmax}\!\Big(\frac{X W^Q_h (Y W^K_h)^\top}{\sqrt{d_{kv}}} + B_h + C\Big)\, Y W^V_h , $$ {#eq-attn}

with $W^Q_h, W^K_h, W^V_h \in \mathbb{R}^{M \times d_{kv}}$, $B_h$ the position biases, $C$ zero or $-\infty$ above the diagonal for the causal mask, and the heads concatenated and multiplied by $W^O \in \mathbb{R}^{I \times M}$.
The paper does not give the scale of the scores; the page keeps the $1/\sqrt{d_{kv}}$ of the Transformer.

**Encoder.**
With $z^0$ the embedded source, each of the $L$ layers is

$$ z' = z + \operatorname{Attn}\big(\operatorname{N}(z), \operatorname{N}(z)\big) , \qquad z \leftarrow z' + \max\big(0, \operatorname{N}(z') W_i\big) W_o , \qquad e = \operatorname{N}(z^L) , $$ {#eq-enc}

with $W_i \in \mathbb{R}^{M \times F}$ and $W_o \in \mathbb{R}^{F \times M}$, every sublayer with its own gain, and a last normalization giving the encoder output $e$.

**Decoder.**
The decoder reads the target shifted right, a start token and then $t_1, \dots, t_{T-1}$; each of its $L$ layers is

$$ y' = y + \operatorname{Attn}_{\text{causal}}\big(\operatorname{N}(y), \operatorname{N}(y)\big) , \quad y'' = y' + \operatorname{Attn}\big(\operatorname{N}(y'), e\big) , \quad y \leftarrow y'' + \max\big(0, \operatorname{N}(y'') W_i\big) W_o . $$ {#eq-dec}

**Output and loss.**
The logits are the last normalized decoder state times the embedding transposed, and the loss is the mean negative log-likelihood of the target tokens:

$$ \text{logits} = \operatorname{N}(y^L)\, E^\top , \qquad \mathcal{L} = -\frac{1}{BT} \sum_{b=1}^{B} \sum_{j=1}^{T} \log p\big(t_{b,j} \mid t_{b,<j}, s_b\big) , $$ {#eq-logits}

where $p$ is the softmax of the logits, a [[m2:cross-entropy|cross-entropy]] with the observed tokens and by the chain rule of [[m2:conditional-probability|conditional probability]] the log-likelihood of the whole target. {#eq-loss}

**Span corruption.**
Pre-training turns unlabeled text into pairs.
In the paper's example, "for", "inviting" and "last" of "Thank you for inviting me to your party last week." are chosen; each run of chosen tokens is replaced by a sentinel unique within the example, so "for inviting" becomes one sentinel, and the target is the dropped spans, each after its sentinel, and a final sentinel. {#c-objective}
The input is "Thank you <X> me to your party <Y> week ." and the target "<X> for inviting <Y> last <Z>".
For $n$ tokens with $N$ chosen in $m$ spans,

$$ |\text{input}| = n - N + m , \qquad |\text{target}| = N + m + 1 . $$ {#eq-span}

BERT's masked language model instead masks 15% of the WordPiece tokens of each sequence at random and predicts only the masked tokens, where a denoising autoencoder would reconstruct the entire input. {#c-bert-mlm}
The [[masked-language-modelling|masked language modelling]] page compares the two objectives further.

## Shapes and parameters

The embedding has $VM$ parameters and the two position tables $2KH$.
An encoder layer has $4MI$ in its attention, $2MF$ in its feed-forward network and $2M$ in two gains; a decoder layer has $8MI$ in its two attentions, $2MF$ and $3M$; two final gains add $2M$:

$$ N = VM + 2KH + L\big(12MI + 4MF + 5M\big) + 2M . $$ {#eq-params}

This is the count of the reference for any configuration. {#c-params}
In the shape contract it is

{shape:params}

and the example network, 13 tokens, width 8, 2 heads of 3, a feed-forward width of 16, 2 layers in each stack and 8 buckets, has {shape:params@example} parameters.

**The baseline and the sizes.**
The baseline has 12 blocks in each stack, a feed-forward network of $d_{\text{ff}} = 3072$ with a ReLU, 12 heads with $d_{kv} = 64$ and $d_{\text{model}} = 768$, about 220 million parameters, roughly twice BERT-Base, and dropout of 0.1. {#c-baseline}
The vocabulary is 32,000 SentencePiece wordpieces trained on English C4 with some German, French and Romanian and shared by input and output, so the model handles only a fixed set of languages. {#c-vocab}
SentencePiece trains its subword model directly from raw sentences, where earlier tools assumed text already split into words. {#c-sentencepiece}
Small has $d_{\text{model}} = 512$, $d_{\text{ff}} = 2048$, 8 heads and 6 layers in each stack, about 60 million parameters; Large $d_{\text{model}} = 1024$, $d_{\text{ff}} = 4096$, $d_{kv} = 64$, 16 heads and 24 layers each, around 770 million; 3B and 11B keep $d_{\text{model}} = 1024$ and 24 layers with $d_{kv} = 128$, and have $d_{\text{ff}} = 16{,}384$ and 32 heads, around 2.8 billion, and $d_{\text{ff}} = 65{,}536$ and 128 heads, about 11 billion. {#c-sizes}
The authors grew $d_{\text{ff}}$ because accelerators such as TPUs are most efficient on large dense matrix products like those of the feed-forward networks. {#c-scale-dff}
In 11B the feed-forward networks are 64 times as wide as the residual stream, and each head has $d_{kv} = 128$ while $I = 16{,}384$ is 16 times $M$.

Eq-params gives {shape:params@t5_small} for Small and {shape:params@t5_base} for Base, and for all five sizes 60,441,088, 222,805,248, 737,537,024, 2,851,467,264 and 11,307,190,272. {#c-params-sizes}
Small, Base and 11B round to the paper's figures; 3B rounds to 2.9 billion, not 2.8, and Large is 32.5 million short of 770 million; counting a separate output matrix would bring Large to 770,305,024 but Base to 247,381,248, so no single way of counting matches both. {#c-params-large}
The derivation `parameter-count` proves eq-params and these counts, and its check compares them with the parameters the NumPy reference creates.

**Where the parameters are.**
In Base the embedding is 24,576,000 parameters, 11.0% of the total; the encoder stack holds {shape:params_encoder@t5_base} and the decoder stack {shape:params_decoder@t5_base}, the decoder more because of its attention to the encoder.
The paper puts the parameters of the encoder-decoder attention at about 10% of the total for the sizes it considers, and assumes that an $L + L$ encoder-decoder has the parameters of a $2L$-layer language model. {#c-cross-ten}
Counted, the attention to the encoder holds $4LMI$: 12.7% of Base and from 13.6% to 14.2% of Large, 3B and 11B. {#c-cross-share}

## Computation and cost

The matrix products of a forward pass over $B$ pairs cost

$$ C = B\Big[L\big(S(8MI + 4MF) + T(12MI + 4MF) + 4SMI + 4I(S^2 + ST + T^2)\big) + 2TMV\Big] $$ {#eq-flops}

FLOPs, a multiply-add counted as two.
The $4SMI$ is the keys and values of the attention to the encoder, computed from the $S$ encoder outputs; $4IS^2$, $4IT^2$ and $4IST$ are the scores and weighted sums of the three attentions.
For Base on 512 source and 512 target tokens this is {shape:flops_forward@t5_base} FLOPs, of which the output layer is 25.2 billion.

**Against a language model.**
Raffel et al. note that an $L + L$ encoder-decoder has about the parameters of a language model of $2L$ layers but about the cost of one of $L$ layers, because the language model applies its layers to both input and output while the encoder reads only the input and the decoder only the output; the equivalences are approximate, and they observed nearly identical step times. {#c-pm-paper}
Counted, the matrices of the $L + L$ model hold $L(12MI + 4MF)$ parameters and those of the $2L$-layer model $L(8MI + 4MF)$, a ratio of 7/6 when $I = M$ and $F = 4M$. {#c-pm-params}
Without the attention products, the $L + L$ model costs $2L(S + T)(6MI + 2MF)$ FLOPs on a pair and the $L$-layer model on the same tokens $2L(S + T)(4MI + 2MF)$, the same ratio; for Base at $S = T = 512$ they cost 231.9 and 212.6 GFLOPs, a ratio of 12/11. {#c-pm-flops}
The attention products cost $4LI(S^2 + ST + T^2)$ in the encoder-decoder and $4LI(S + T)^2$ in the language model, three quarters as much in the encoder-decoder when $S = T$. {#c-pm-attention}
Both excesses of 7/6 come from the attention to the encoder; the derivation `encoder-decoder-cost` proves them and its check counts the products of the reference.

**Decoding.**
The encoder runs once per input.
Each decoder layer computes the keys and values of the attention to the encoder once, $S \times I$ numbers each, and keeps a cache of its self-attention keys and values that grows by one row per generated token, as in a decoder-only model; the bias of a new token at position $t$ is row $t$ of the decoder's bias matrix.
For Base the cross keys and values are $2 \times 12 \times 768 = 18{,}432$ numbers per source token, and the self-attention cache the same per generated token.

## Training

**Data.**
C4, the Colossal Clean Crawled Corpus, is the web text of Common Crawl from April 2019 after a set of filters, about 750 GB of mostly clean English. {#c-c4}

**The baseline.**
Every task is trained by maximum likelihood with teacher forcing and a cross-entropy loss, with AdaFactor, and decoded greedily; each baseline is pre-trained for $2^{19} = 524{,}288$ steps on C4 with sequences of at most 512 tokens and batches of 128. {#c-training}
With several examples packed into each row, a batch holds about $2^{16} = 65{,}536$ tokens and pre-training sees $2^{35}$, about 34B, tokens, far fewer than BERT's 137B or RoBERTa's 2.2T, and no example twice. {#c-tokens}
The learning rate is $1/\sqrt{\max(n, k)}$ at step $n$ with $k = 10^4$, 0.01 for the first $10^4$ steps and then decaying; fine-tuning runs $2^{18} = 262{,}144$ steps at a constant 0.001. {#c-lr}

**Span corruption.**
The objective of the final models corrupts contiguous, randomly spaced spans, set by the share of tokens corrupted and the number of spans: in 500 tokens, 15% corrupted in 25 spans is 75 tokens in spans of mean length 3. {#c-span-def}
By eq-span the 500 tokens become an input of 450 tokens and a target of 101. {#c-span-lengths}
The baseline's objective instead corrupts each token independently and merges runs; with probability $r$ per token the expected number of spans is $r + (n - 1)r(1 - r)$ and the mean span tends to $1/(1 - r)$, 1.18 tokens at 15%, so on 512 tokens the expected target is 143.1 tokens and the input 500.5, against 104 and 461 for spans of mean length 3. {#c-span-iid}
The two objectives choose the same number of tokens; spans of mean length 3 need fewer sentinels, so both sequences are shorter.
At 50% corruption with spans of mean length 3, the target of 512 tokens grows to 342 tokens, against 104 at 15%. {#c-span-rate}
The derivation `span-lengths` proves these lengths, and its check runs the span corruption of the reference.

**The final models.**
They use span corruption with mean span length 3 at 15%, 1 million steps of $2^{11}$ sequences of 512 tokens, about a trillion tokens and 32 times the baseline, and a multi-task mixture of the unsupervised task and the supervised tasks in pre-training. {#c-final-recipe}

## Inductive biases

**Two sequences, two stacks.**
The encoder sees the whole input; the decoder sees the output so far and, through the attention to the encoder, all of the input.
A decoder-only model can be given the same view of the input with a prefix mask, and the paper compares both, below.

**Offsets, not places.**
Position reaches T5 only through eq-bias.
The bias of a pair depends on the offset $j - i$ alone, so moving the whole sequence along changes no score; `test_the_bias_depends_on_the_offset_alone` checks this.
With pos="none" the encoder has no position at all, and permuting the source tokens permutes its output in the same way: the encoder sees the input as a multiset; the tests `test_without_positions_the_encoder_sees_a_set` of the three implementations check this, and that the biases break it.
The decoder still knows where it is through its causal mask, as a [[decoder-only-transformer|decoder-only]] model without position embeddings does.

**Coarse far away.**
With 32 buckets reaching 128, the decoder has one bucket for each offset from 0 to 15, and buckets 16 to 31 begin at offsets 16, 19, 21, 24, 27, 31, 35, 40, 46, 52, 59, 67, 77, 87, 99 and 113, each 1.139 times the last; each direction of the encoder has buckets beginning at 8, 12, 16, 23, 32, 46, 64 and 91, growing by $\sqrt{2}$. {#c-bucket-edges}
Every offset from 113 on shares the decoder's last bucket, earlier than the 128 of the paper's description, and in the encoder bucket 16, which would hold offset 0 of the keys after the query, never receives an offset, so 31 of its 32 rows are used. {#c-bucket-unused}
The derivation `buckets` proves both from eq-bias.
The paper notes that a layer cannot tell offsets apart beyond 128, but that later layers can build a sensitivity to larger offsets by combining local information. {#c-relpos-beyond}
[[rotary-position-embedding|Rotary position embeddings]] are another relative scheme: they rotate the queries and keys instead of adding a bias, and depend on every offset.

## Evidence

**Architectures.**
Raffel et al. compared five architectures of BERT-Base-sized stacks, with $P$ the parameters of a stack and $M$ the FLOPs of the $L + L$ encoder-decoder on a pair, each pre-trained once on $2^{35}$ tokens.

| architecture | params | cost | GLUE | CNNDM | SQuAD | SuperGLUE |
|---|---|---|---|---|---|---|
| encoder-decoder | 2P | M | 83.28 | 19.24 | 80.88 | 71.36 |
| encoder-decoder, shared | P | M | 82.81 | 18.78 | 80.63 | 70.73 |
| encoder-decoder, 6 + 6 | P | M/2 | 80.88 | 18.97 | 77.59 | 68.42 |
| language model | P | M | 74.70 | 17.93 | 61.14 | 55.02 |
| prefix LM | P | M | 81.82 | 18.61 | 78.94 | 68.11 |

With the denoising objective the scores are those of the table. {#c-table2}
With a language-modelling objective the same five score 79.56, 79.60, 78.67, 73.78 and 79.68 on GLUE and 64.29, 63.50, 64.06, 56.51 and 64.86 on SuperGLUE. {#c-table2-lm}
The full encoder-decoder with denoising was best on every task; sharing its parameters between the stacks cost little, halving its layers cost more; the shared encoder-decoder beat the prefix LM with the same parameters and cost, which the authors read as a benefit of explicit attention to the encoder; and denoising beat language modelling for every architecture. {#c-table2-findings}

**Objectives.**
With independent corruption, rates of 10%, 15%, 25% and 50% differed little except that 50% degraded GLUE and SQuAD significantly, and higher rates give longer targets. {#c-rate}
At 15%, spans of mean length 2, 3, 5 and 10 differed little, 10 slightly worse in some cases; mean length 3 slightly but significantly beat independent corruption on most benchmarks other than translation, and trained faster because its sequences are shorter. {#c-span-results}

**The final models.**
T5 gave the best results reported to date on 18 of the 24 tasks, 11B was best of the five sizes on every task, and its average GLUE score was 90.3. {#c-results}
On SuperGLUE the average went from the previous best of 84.6 to 88.9, close to the human 89.8, though humans score 100% on COPA and WSC, well above the model. {#c-superglue}
On WMT translation T5 scored below earlier systems; the authors point to its English-only pre-training data and to the back-translation those systems used. {#c-wmt}
To separate scale from the other choices, they pre-trained the baseline on a trillion tokens: on GLUE, CNNDM, SQuAD and SuperGLUE, the baseline scored 83.28, 19.24, 80.88 and 71.36, the baseline on a trillion tokens 84.80, 19.62, 83.01 and 73.90, and T5-Base 85.97, 20.90, 85.44 and 75.64. {#c-table15}

These are reported results; this page has not reproduced them.
The paper estimates the spread of its baseline across runs and, in each table, marks the scores within two standard deviations of the best.

## Limitations

**Two stacks for every task.**
Each pair of layers carries the attention to the encoder, a seventh of the layers' matrices when $I = M$ and $F = 4M$, and tasks with no input to speak of, such as open-ended generation, still pay for the encoder and its attention.
Generation needs two caches, the fixed cross keys and values and the growing self-attention cache.

**Positions far apart.**
A layer sees all offsets beyond the last bucket edge as one; the encoder at 91 or more in a direction and the decoder at 113 or more.
Longer inputs work, but only through stacking, and the paper trains on 512 tokens.

**A fixed vocabulary.**
The 32,000 wordpieces cover English, German, French and Romanian; other languages fall into rare pieces, and the vocabulary cannot change after pre-training.

**Cost.**
The authors note that larger models tend to perform better but that many applications need smaller or cheaper ones, and they ask for methods that do better with cheaper models. {#c-large-models}
The attention costs grow with $S^2$, $T^2$ and $ST$.

## Variants and descendants

- **The paper's own variants.** An encoder-decoder with the parameters shared between the stacks, one of half depth, a decoder-only language model and a prefix language model, compared in the evidence above.
- **BART.** BART is a denoising autoencoder for sequence-to-sequence models, trained by corrupting text with any noising function and reconstructing the original, on a standard Transformer translation architecture that generalizes BERT's bidirectional encoder and GPT's left-to-right decoder. {#c-bart-def}
  It keeps the Transformer's layers with GeLU activations and $\mathcal{N}(0, 0.02)$ initialization, 6 layers per stack in the base model and 12 in the large, about 10% more parameters than the equivalent BERT. {#c-bart-arch}
  Its text infilling draws span lengths from a Poisson distribution with $\lambda = 3$ and replaces each span, even an empty one, with one [MASK], so the model must predict how many tokens are missing. {#c-bart-infill}
  It matched RoBERTa on GLUE and SQuAD, improved on earlier results in abstractive dialogue, question answering and summarization by up to 3.5 ROUGE, and added 1.1 BLEU over a back-translation system with only target-language pre-training. {#c-bart-results}
  BART is an [[autoencoder]] of text whose decoder reconstructs the whole input, where T5's writes only the corrupted spans.

## Notable models

- **T5-Small, -Base, -Large, -3B and -11B**, 60 million to 11 billion parameters by the paper's figures, released with C4 and the code.
- **The baseline** of the study, Base trained on $2^{35}$ tokens with independent corruption, against which the paper's tables are measured.
- **BART-base and BART-large**, of 6 and 12 layers per stack.

## Implementations

The NumPy reference `impl/numpy/t5.py` implements eq-norm, eq-bias, eq-attn, eq-enc, eq-dec, eq-logits and eq-loss, the gradients of the loss by hand through both stacks and the bias tables, greedy decoding with and without the cache, span corruption by eq-span with random spans, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/t5.py` is an `nn.Module` with the same parameters, using `F.rms_norm` and `F.scaled_dot_product_attention` with the biases as a float mask; its tests run `torch.autograd.gradcheck` and compare the attention with an explicit softmax.
The JAX version `impl/jax/t5.py` is pure functions of a parameter dictionary with a cache of fixed size, so a decoding step compiles once under `jax.jit` and greedy decoding runs in `lax.scan`; its tests check causality through the Jacobian.
All three load the fixture of `fixtures/`: the example network on two pairs of 7 source and 5 target tokens, its encoder output, logits, loss and gradients, the logits computed in two pieces with the cache, and six greedy tokens.

### Worked example

Take the paper's sentence as 11 tokens, "Thank you for inviting me to your party last week .", with $N = 3$ tokens chosen in $m = 2$ spans.
By eq-span the input has $11 - 3 + 2 = 10$ tokens, "Thank you <X> me to your party <Y> week .", and the target $3 + 2 + 1 = 6$, "<X> for inviting <Y> last <Z>"; `test_span_corruption_of_the_paper` checks both.
With 32 buckets reaching 128, a decoder query at position 50 puts the key at 30, offset 20, in bucket 17, since bucket 17 begins at 19 and bucket 18 at 21, and every key 113 or more places back in bucket 31.
In the encoder, a key 10 places before the query goes to bucket 8, and one 10 places after it to $16 + 8 = 24$.
`test_buckets` checks these buckets, the last edge at 113 and the empty bucket 16.

### Pitfalls

> [!pitfall]
> **Not shifting the target.** The decoder's input is the start token and then $t_1, \dots, t_{T-1}$; its output at position $j$ is scored against $t_j$. Feeding the target unshifted asks the decoder to copy the token it reads, and the loss falls while the model learns nothing. `test_decoder_is_causal_and_the_target_is_shifted` checks the shift.

> [!pitfall]
> **The sign of the offset.** The page takes the offset as key minus query. In the decoder all visible keys have offset at most 0; with the sign flipped, every past key falls into bucket 0 and the decoder loses position. In the encoder a flipped sign mirrors the two halves of the table, which gives a different model from the same weights.

> [!pitfall]
> **Centring in the normalization.** A layer normalization with a mean subtracted is a different function; with weights trained without centring, it gives wrong activations though every shape is right. `test_rms_norm_has_no_centring` checks that a constant vector is rescaled, not sent to zero as a layer normalization would.

> [!pitfall]
> **The cache and the biases.** When the decoder runs with a cache, a new token at position $t$ needs the biases of row $t$, for offsets $0$ to $-t$, not those of row 0. The keys and values of the attention to the encoder are computed once and are not part of the self-attention cache. `test_cache_of_keys_and_values` checks the logits from the cache against the full pass.

> [!pitfall]
> **Counting the parameters.** The paper's figures mix two ways of counting, as c-params-large shows; compare counts with the formula and the convention stated, not with the rounded names.

> [!pitfall]
> **Running out of sentinels.** Each span takes one sentinel and the target one more; a vocabulary with fewer sentinels than spans plus one cannot express the pair. The reference raises an error, which `test_bad_configs` checks, and `test_random_spans` checks the number of spans and both lengths.

## Experiments

The fixture script `fixtures/make.py` trains two small models with the NumPy reference to reverse a sequence: the source is six symbols drawn uniformly from 8, and the target is the same six in reverse order.
Each model has width 32, 4 heads of 8, a feed-forward width of 64 and 2 layers in each stack, and is trained for 1500 steps of Adam at a learning rate of 0.003 on batches of 32.
One has the position tables of eq-bias, 8 buckets reaching 16; the other has none, so that the two differ only in position.
On 500 test pairs, the loss is the mean over the six target tokens under teacher forcing, the token accuracy the share of target tokens the most likely next token gets right, and the sequence accuracy the share of whole targets greedy decoding gets right.

| model | test loss (nats) | token accuracy | sequence accuracy |
|---|---|---|---|
| position biases | 0.0129 | 99.63% | 97.80% |
| no position | 0.9200 | 53.13% | 1.00% |

With the biases the model reverses almost every sequence.
Without them the encoder sees the source as a multiset, so the decoder can learn which symbols occur and how often but not in what order; its token accuracy, 53.1%, is above the 12.5% of a guess, and it gets 1% of whole sequences right.
These are single runs of one seed, on a toy task.
The checks of the four derivations count the parameters, FLOPs, buckets and span lengths of the models of the paper from the reference.

## History

The Transformer of Vaswani et al., in 2017, was an encoder-decoder for translation.
In 2018 GPT used its decoder alone as a language model and BERT its encoder alone with a masked-word objective, and both were fine-tuned on each task, so that many approaches to transfer learning in language used a single stack.
In October 2019 Raffel et al. returned to the encoder-decoder, cast every task as text to text, compared architectures, objectives, corpora and transfer schemes in one setting, and trained models up to 11 billion parameters on C4; in the same month Lewis et al. posted BART, an encoder-decoder pre-trained to reconstruct corrupted documents.
Most later large language models were decoder-only, and the [[neural-scaling-laws|scaling laws]] page follows how their loss falls with size; T5 remains a reference point for the encoder-decoder, its span corruption and its [[byte-pair-encoding|subword]] vocabulary.

## References

- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research 21. arXiv 1910.10683.
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. ACL 2020, pp. 7871 to 7880. arXiv 1910.13461.
- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. arXiv 1810.04805.
- Kudo, T. and Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. EMNLP 2018, System Demonstrations. arXiv 1808.06226.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017. arXiv 1706.03762.
- Zhang, B. and Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019. arXiv 1910.07467.
