---
tag: NEW-decoder-only-transformer
kind: architecture
slug: decoder-only-transformer
name: Decoder-only Transformer
aliases: [decoder-only Transformer, GPT, generative pre-trained transformer, causal Transformer language model, autoregressive Transformer, Transformer decoder language model, GPT-2, GPT-3]
area: language-models/decoder-only-models
level: core
summary: >-
  Transformer layers with self-attention masked so each token sees only earlier tokens, trained to predict the next token at every position; the architecture of GPT and most large language models.
uses: [matrix-multiplication@m2, permutation-matrix@m2, conditional-probability@m2, cross-entropy@m2]
related_to: [transformer, bert, masked-language-modelling, neural-probabilistic-language-model, n-gram-language-model, autoregressive-model, few-shot-in-context-learning, neural-scaling-laws, byte-pair-encoding, rotary-position-embedding, rms-normalization, speculative-decoding, recurrent-neural-network, sparse-mixture-of-experts]
history:
  year: 2018
  people: [Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan]
  note: "Radford et al. described GPT in June 2018 and GPT-2 in February 2019; Brown et al. posted GPT-3 to arXiv (2005.14165) in May 2020, and Kaplan et al. measured how decoder-only Transformers scale in January 2020."
prereqs: [transformer, multi-head-attention, layer-normalization, autoregressive-model, conditional-probability@m2]
same_as:
  wikipedia: Generative_pre-trained_transformer
  arxiv: ["2005.14165", "2001.08361"]
sources:
  - {paper: "Radford, Narasimhan, Salimans and Sutskever (2018)", title: "Improving Language Understanding by Generative Pre-Training", locator: "OpenAI technical report, Sections 3 and 4, Table 5", note: "GPT: the decoder-only language model, its pre-training on BooksCorpus and fine-tuning"}
  - {paper: "Radford, Wu, Child, Luan, Amodei and Sutskever (2019)", title: "Language Models are Unsupervised Multitask Learners", locator: "OpenAI technical report, Sections 2 and 3, Table 2", note: "GPT-2: WebText, byte-level BPE, pre-norm layers and zero-shot transfer"}
  - {paper: "Brown, Mann, Ryder, Subbiah, Kaplan and others (2020)", title: "Language Models are Few-Shot Learners", locator: "NeurIPS 2020; arXiv 2005.14165v4, Section 2.1, Table 2.1 and Appendix D", note: "GPT-3: eight sizes up to 175B and in-context learning"}
  - {paper: "Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (2020)", title: "Scaling Laws for Neural Language Models", locator: "arXiv 2001.08361, Sections 1 to 3, Table 1 and Figure 5", note: "the parameter and compute counts N = 12 n_layer d_model^2 and C = 6N, and the weak dependence on shape"}
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "NeurIPS 2017; arXiv 1706.03762, Sections 3.1 and 3.2.3", note: "the masked self-attention of the decoder"}
  - {paper: "Shazeer (2019)", title: "Fast Transformer Decoding: One Write-Head is All You Need", locator: "arXiv 1911.02150, Sections 2.4 and 3", note: "incremental decoding and multi-query attention"}
  - {paper: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón and Sanghai (2023)", title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", locator: "EMNLP 2023; arXiv 2305.13245v3, the abstract and Section 1", note: "grouped-query attention"}
  - {paper: "Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang and Liu (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "ICML 2020; arXiv 2002.04745v2, the abstract", note: "why pre-norm layers train without warm-up"}
  - {paper: "Press and Wolf (2017)", title: "Using the Output Embedding to Improve Language Models", locator: "EACL 2017, the abstract", note: "tying the input and output embeddings"}
  - {paper: "Touvron, Lavril, Izacard and others (2023)", title: "LLaMA: Open and Efficient Foundation Language Models", locator: "arXiv 2302.13971, Section 2.2 and 2.4", note: "RMSNorm, SwiGLU and rotary embeddings"}
  - {paper: "Chowdhery, Narang, Devlin and others (2022)", title: "PaLM: Scaling Language Modeling with Pathways", locator: "arXiv 2204.02311, Section 2", note: "parallel layers, multi-query attention, no biases"}
  - {paper: "Hendrycks and Gimpel (2016)", title: "Gaussian Error Linear Units (GELUs)", locator: "arXiv 1606.08415v5, the abstract", note: "the activation of the MLP"}
family: transformer
inputs: "a batch of sequences of token ids as a (B, n) integer array with n at most the context n_ctx"
outputs: "a (B, n, V) array of logits: row t scores every token of the vocabulary as the token at position t + 1"
components: [multi-head-attention, multilayer-perceptron, layer-normalization, residual-connection]
inductive_biases: ["the causal mask: the state at a position depends only on the tokens at that position and before it, so one pass gives a prediction of the next token at every position", "weights shared across positions: every layer applies the same maps at every position, and order enters only through the position embeddings and the mask", "learned absolute position embeddings for a fixed context of n_ctx places, with no embedding beyond it", "the output layer tied to the token embedding, so a token is read and written with the same vector"]
params: "V D + n_ctx D + L (4 D^2 + 2 D F + 9 D + F) + 2 D for a vocabulary of V, a context of n_ctx, L layers of width D with an MLP of width F, tied embeddings and learned positions"
flops_per_example: "L (8 n D^2 + 4 n^2 D + 4 n D F) + 2 n D V for a sequence of n tokens with the attention computed in full, counting a multiply-add as 2 FLOPs and only the matrix products; 2 L n(n+1) D replaces 4 L n^2 D when the masked half is skipped"
---

A decoder-only Transformer is a stack of [[transformer|Transformer]] layers whose self-attention is masked so that each token sees only itself and the tokens before it, trained to predict the next token.
It has no encoder and no cross-attention; the prompt and the text it writes are one sequence.
Radford et al. introduced it as a language model in 2018: GPT, a multi-layer Transformer decoder trained to maximize the likelihood of the next token and then fine-tuned on each task.
Its successors kept the architecture and grew it: GPT-2, with 1.5 billion parameters, was evaluated on tasks with no fine-tuning at all, and GPT-3, with 175 billion, was given new tasks as a few examples in its input.
Most large language models since then, LLaMA and PaLM among them, are decoder-only Transformers with small changes to the layer.

## In plain words

The model reads text as a row of tokens, pieces of words, and at every place in the row it guesses the next piece.
Each layer lets every token look back over the tokens before it, through attention, and mixes what it finds into its own vector; a token is never allowed to look ahead, so the guess at a place cannot cheat by reading the answer.
Training shows the model a long stretch of text and scores all its guesses at once, one per position, by how much probability it gave the token that came next.
To write, the model guesses a next token, appends it to the row and guesses again, and because earlier tokens never look at later ones, the work done for them can be kept and reused.
The same machine, made larger and trained on more text, was found to answer questions, translate and continue patterns given only a description or a few examples in its input.

## Overview

The language model estimates the probability of a sequence by the chain rule of [[m2:conditional-probability|conditional probability]], $p(x) = \prod_i p(s_i \mid s_1, \dots, s_{i-1})$; since language has a natural order, this factorization is common, and it allows tractable sampling from and estimation of $p(x)$ as well as any conditionals. {#c-gpt2-factor}
GPT trains a network with parameters $\Theta$ to maximize $L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$ over a corpus of tokens $\mathcal{U}$, with $k$ the size of the context window. {#c-gpt1-objective}
The network is a multi-layer Transformer decoder, multi-headed self-attention over the context tokens followed by position-wise feed-forward layers: $h_0 = U W_e + W_p$, $h_l = \text{transformer\_block}(h_{l-1})$ and $P(u) = \operatorname{softmax}(h_n W_e^\top)$, with $W_e$ the token embedding and $W_p$ the position embedding. {#c-gpt1-decoder}
The decoder of the original Transformer was already masked in this way: its self-attention keeps positions from attending to subsequent positions, and with the output shifted by one position, the prediction for position $i$ can depend only on the outputs at positions less than $i$. {#c-vaswani-mask}
Dropping the encoder and the cross-attention leaves a single stack that is an [[autoregressive-model|autoregressive model]] of text, the same kind of model as an [[n-gram-language-model|n-gram model]] or the [[neural-probabilistic-language-model|neural probabilistic language model]], with a context of hundreds or thousands of tokens.

The network on this page is that of GPT-2 and GPT-3: a token embedding tied to the output layer, learned position embeddings, $L$ pre-norm layers of width $D$ with causal [[multi-head-attention|multi-head self-attention]] and an MLP of width $F$, and a final [[layer-normalization|layer normalization]].
The reference implementation has two switches: an untied output matrix (tied=False) and no position embeddings (pos="none"), the latter for the inductive biases below.
GPT-3 also alternated dense and locally banded sparse attention, which the reference does not implement.

## Components

The notation uses row vectors; a sequence is $x_1, \dots, x_n$ with $n \leq n_{\text{ctx}}$, and $V$ is the size of the vocabulary.

**Embedding.**
Each token is looked up in the token embedding $W_e \in \mathbb{R}^{V \times D}$ and the position embedding $W_p \in \mathbb{R}^{n_{\text{ctx}} \times D}$ of its place is added:

$$ \mathbf{z}_0[t] = W_e[x_t] + W_p[t], \qquad t = 1, \dots, n . $$ {#eq-embed}

GPT used learned position embeddings instead of the sinusoidal version of the original Transformer, a BPE vocabulary of 40,000 merges and the GELU activation. {#c-gpt1-details}
GPT-2 moved to a [[byte-pair-encoding|byte-level BPE]], whose base vocabulary is the 256 bytes, so that the model can assign a probability to any Unicode string and be evaluated on any dataset whatever its tokenization. {#c-gpt2-bpe}

**Causal self-attention.**
In a head of width $d_{\text{head}} = D/H$, the queries, keys and values of the $n$ tokens are compared, the scores of the pairs in which a token would see a later one are set to $-\infty$, and each token averages the values it may see:

$$ M_{tj} = \begin{cases} 0 & j \leq t \\ -\infty & j > t \end{cases}, \qquad A = \operatorname{softmax}\Big(\frac{\mathbf{q}\mathbf{k}^\top}{\sqrt{d_{\text{head}}}} + M\Big), \qquad \operatorname{head}(\mathbf{z}) = A\mathbf{v} . $$ {#eq-mask}

Vaswani et al. implement the mask in this way, inside scaled dot-product attention, setting to $-\infty$ the inputs of the softmax that correspond to illegal connections, to preserve the auto-regressive property. {#c-vaswani-inf}
$H$ heads run in parallel on one fused projection, and their outputs are concatenated and projected back:

$$ [\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z} W_{qkv} + \mathbf{b}_{qkv}, \qquad \operatorname{MSA}(\mathbf{z}) = [\operatorname{head}_1; \dots; \operatorname{head}_H]\, W_o + \mathbf{b}_o . $$ {#eq-attn}

After the softmax, row $t$ of $A$ is zero beyond column $t$, so the output at $t$ is a function of $\mathbf{z}[1], \dots, \mathbf{z}[t]$ alone; every layer keeps this property, and so does the stack.

**Layer.**
Layer $\ell$ applies the attention and an [[multilayer-perceptron|MLP]] to normalized states, each inside a [[residual-connection|residual connection]]:

$$ \mathbf{z}'_\ell = \mathbf{z}_{\ell-1} + \operatorname{MSA}\big(\operatorname{LN}(\mathbf{z}_{\ell-1})\big), \qquad \mathbf{z}_\ell = \mathbf{z}'_\ell + \operatorname{GELU}\big(\operatorname{LN}(\mathbf{z}'_\ell) W_1 + \mathbf{b}_1\big) W_2 + \mathbf{b}_2 . $$ {#eq-layer}

GPT-2 made this change from GPT: layer normalization was moved to the input of each sub-block, as in a pre-activation residual network, and a final layer normalization was added after the last block. {#c-gpt2-model}
The GELU activation function is $x\Phi(x)$, where $\Phi$ is the standard Gaussian cumulative distribution function. {#c-gelu}
Every LN has its own gain and bias and normalizes each token over its $D$ features, with $\epsilon = 10^{-5}$ in the reference.

**Output.**
The final state is normalized and multiplied by the token embedding transposed, which gives a score for every token of the vocabulary at every position:

$$ \text{logits} = \operatorname{LN}(\mathbf{z}_L)\, W_e^\top, \qquad p(x_{t+1} = v \mid x_1, \dots, x_t) = \operatorname{softmax}(\text{logits}[t])_v . $$ {#eq-logits}

Using one matrix for both ends follows Press and Wolf, who found that the topmost weight matrix of a neural language model is a valid word embedding, recommended tying it to the input embedding, and reported a significant reduction in perplexity. {#c-press}

**Loss.**
Training minimizes the [[m2:cross-entropy|cross-entropy]] of the next token, averaged over the $B(n-1)$ positions of a batch that have one:

$$ \mathcal{L} = -\frac{1}{B(n-1)} \sum_{b=1}^{B} \sum_{t=1}^{n-1} \log p\big(x^{(b)}_{t+1} \mid x^{(b)}_1, \dots, x^{(b)}_t\big) . $$ {#eq-loss}

One forward pass scores all $n - 1$ predictions of a sequence at once, in parallel, which is what makes the training of a Transformer language model cheap compared with a [[recurrent-neural-network|recurrent network]], which must run through the sequence one step at a time.

### The layer in PyTorch

torch.nn.TransformerEncoderLayer with norm_first=True, activation "gelu", dropout 0 and a causal mask computes eq-layer; `impl/pytorch/decoder_lm.py` builds it with the weights of a layer, and `test_layer_matches_torch_transformer_encoder_layer_with_a_causal_mask` checks that the two agree.
The name is a reminder that a decoder-only Transformer has no decoder in the sense of the original paper: it is an encoder layer with a mask, and without cross-attention.

## Shapes and parameters

The embeddings have $VD + n_{\text{ctx}} D$ parameters.
A layer has $4D^2 + 4D$ in the attention, $2DF + F + D$ in the MLP and $4D$ in its two normalizations, and the final normalization has $2D$, so the network has

$$ N_{\text{total}} = VD + n_{\text{ctx}} D + L\big(4D^2 + 2DF + 9D + F\big) + 2D $$ {#eq-params}

parameters. {#c-params}
In the shape contract this is

{shape:params}

and the example network, a vocabulary of 11, a context of 8, width 8 with 2 heads, an MLP of 16 and 2 layers, has {shape:params@example} parameters.

**GPT and GPT-2.**
GPT is a 12-layer decoder-only Transformer with masked self-attention heads, 768-dimensional states and 12 heads, and inner states of 3072 in the MLP. {#c-gpt1-spec}
GPT-2 has four sizes, 117M, 345M, 762M and 1542M parameters with 12, 24, 36 and 48 layers of width 768, 1024, 1280 and 1600; the smallest is equivalent to GPT, and the second to the largest BERT. {#c-gpt2-sizes}
Its vocabulary is 50,257 tokens and its context 1024. {#c-gpt2-vocab}
Eq-params gives {shape:params@gpt2_117m} for the smallest and {shape:params@gpt2_1542m} for the largest, and 354,823,168 and 774,030,080 for the two between, all more than the table's figures. {#c-params-gpt2}
The table does not say how it counted; the gap of the smallest, 7.4 million, is close to the $1024 \times 768 = 786{,}432$ of the position embeddings plus a vocabulary of about 40,000 at width 768, which suggests that the numbers were carried over from GPT, but the paper does not say so.

**GPT-3.**
The eight GPT-3 models range from Small, with 125M parameters, 12 layers of width 768 and 12 heads of 64, to 175B, with 96 layers of width 12,288 and 96 heads of 128; all have $d_{\text{ff}} = 4\,d_{\text{model}}$, a context of 2048 tokens, and were trained on 300 billion tokens. {#c-gpt3-table}
Eq-params gives {shape:params@gpt3_small} for Small and {shape:params@gpt3_175b} for 175B, and the counts of all eight round to those of the paper's Table D.1. {#c-params-gpt3}
Two rows of Table 2.1 are inconsistent with their own heads: XL is listed with width 2048 and 24 heads of 128, and 13B with width 5140 and 40 heads of 128. {#c-gpt3-printed}
Forty heads of 128 make 5120, and with 5120 eq-params gives the 12,850M of Table D.1, while 5140 would give 12,952,938,780, about 100 million more. {#c-params-13b}
The shape contract uses 2048 for XL and 5120 for 13B.

**Where the parameters are.**
Kaplan et al. call $N$ the parameters other than the embeddings and count $N \approx 2 d_{\text{model}} n_{\text{layer}}(2 d_{\text{attn}} + d_{\text{ff}}) = 12 n_{\text{layer}} d_{\text{model}}^2$, leaving out biases and other sub-leading terms; without the embeddings, they found, the scaling laws come out significantly cleaner. {#c-kaplan-n}
With $F = 4D$ the layers of eq-params hold $N = 12LD^2(1 + 13/(12D))$, so the approximation is short by 0.14% at width 768 and 0.0088% at 12,288. {#c-params-kaplan}
The shape contract gives the layers of GPT-3 175B as {shape:params_non_embedding@gpt3_175b}.
The embeddings are a large share of a small model and a small share of a large one: 31.7% of the smallest GPT-2 and 0.37% of GPT-3 175B. {#c-emb-share}
The derivation `parameter-count` proves these counts and checks them against the PyTorch layer.

## Computation and cost

The products of a forward pass over $B$ sequences of $n$ tokens cost

$$ C = B\Big[L\big(8nD^2 + 4n^2 D + 4nDF\big) + 2nDV\Big] $$ {#eq-flops}

FLOPs when the attention is computed in full, as dense code does, and $B[L(8nD^2 + 2n(n+1)D + 4nDF) + 2nDV]$ when only the places that the mask lets through are computed, counting a multiply-add as two FLOPs and leaving out the softmax, the normalizations and the GELU. {#c-flops}
The term $8nD^2$ is the four projections of the attention, $4n^2D$ the products $\mathbf{q}\mathbf{k}^\top$ and $A\mathbf{v}$, $4nDF$ the MLP and $2nDV$ the logits.
The example network on a batch of 2 costs {shape:flops_forward@example} FLOPs; GPT-3 175B costs {shape:flops_forward@gpt3_175b} for one full context of 2048 tokens.

**Twice the parameters per token.**
Kaplan et al. count a forward pass as roughly $C_{\text{forward}} \approx 2N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{model}}$ add-multiply operations per token, the factor of two from the multiply-accumulate of a matrix product. {#c-kaplan-c}
For $d_{\text{model}} > n_{\text{ctx}}/12$ the term that depends on the context is a small fraction, and with the backward pass at about twice the forward pass, a training token costs $C \approx 6N$. {#c-kaplan-6n}
The causal count per token is $L(8D^2 + 4DF) + 2DV + 2L(n+1)D$; for GPT-3 at $n = 2048$ it is between 1.014 times (175B) and 1.137 times (Small) twice the total number of parameters. {#c-flops-token}
PaLM states the same rule with FLOPs counted differently: the number of FLOPs per token is approximately equal to the number of parameters, since its models are standard dense Transformers. {#c-palm-flops}
That count takes a multiply-add as one operation; the two statements agree.

**The attention is a small part of the cost.**
In a layer with $F = 4D$, the share of the causal FLOPs spent in $\mathbf{q}\mathbf{k}^\top$ and $A\mathbf{v}$ is $(n+1)/(n+1+12D)$: at $n = 2048$ it is 18.2% for GPT-3 Small, 14.3% for Medium, 10.0% for Large, 7.7% for XL and 1.4% for 175B. {#c-attn-share}
Brown et al. ignored the attention operation when they computed the training compute, as it typically uses less than 10% of the total for the models they analyze. {#c-gpt3-attn-10}
That holds from XL up, which are the models that dominate the compute of the paper, but not for the three smallest.

**The compute of GPT-3.**
Six FLOPs per parameter per token over 300 billion tokens reproduce the training compute of every GPT-3 row of Table D.1 to within 0.5% with the counts of eq-params and to within 0.2% with the printed ones; for 175B it is $3.14 \times 10^{23}$ FLOPs, or 3,637.6 petaFLOP/s-days. {#c-d1-check}
The derivation `compute` counts the products of the NumPy reference through an instrumented array and proves these figures.

**Decoding and the cache of keys and values.**
At generation time a token is produced, appended and fed back, one at a time.
The keys and values of earlier tokens do not change when later tokens arrive, because of the mask, so each layer keeps them in a cache and the new token only computes its own and attends to the stored ones.
After $n$ tokens the cache holds $2LDn$ numbers per sequence, which is {shape:kv_cache@gpt3_175b} for a full context of GPT-3 175B: 2,359,296 numbers per token, 4.5 MiB at two bytes a number, and 9 GiB for 2048 tokens. {#c-kv}
Feeding $n$ tokens one at a time with the cache costs $nP + 2LDn(n+1)$ FLOPs, with $P = L(8D^2 + 4DF) + 2DV$, while running the whole prefix again for every new token costs $Pn(n+1)/2 + \tfrac{2}{3}LDn(n+1)(2n+1)$, 1,029 times more for GPT-3 175B at $n = 2048$. {#c-cache-flops}
The derivation `kv-cache` proves both counts, and its check counts the products of greedy decoding with and without the cache and finds the same tokens.

The cache moves the bottleneck from arithmetic to memory.
Incremental inference, where computation cannot be parallelized across the sequence, is often slow because of the memory-bandwidth cost of repeatedly loading the large keys and values. {#c-shazeer}
Shazeer finds that across $n$ calls the ratio of memory access to arithmetic is $\Theta(n/d + 1/b)$, and when $n \approx d$ or $b \approx 1$ the ratio is close to 1, so memory bandwidth becomes a major bottleneck on modern hardware. {#c-shazeer-ratio}

## Training

GPT was trained on BooksCorpus, over 7,000 unpublished books, chosen because it has long stretches of contiguous text from which the model can learn to condition on long-range information. {#c-gpt1-books}
It was trained with Adam at a maximum learning rate of $2.5 \times 10^{-4}$, warmed up linearly over 2000 updates and annealed to zero on a cosine schedule, for 100 epochs on minibatches of 64 contiguous sequences of 512 tokens. {#c-gpt1-train}
Because layer normalization is used throughout, an initialization of $\mathcal{N}(0, 0.02)$ was enough. {#c-gpt1-init}
When fine-tuning, the language-modelling loss was kept as an auxiliary objective, $L_3 = L_2 + \lambda L_1$, which improved generalization and sped up convergence. {#c-gpt1-aux}

GPT-2 was trained on WebText, the text of 45 million outbound links from Reddit that had at least 3 karma: slightly over 8 million documents and 40 GB of text after de-duplication and cleaning. {#c-gpt2-webtext}
With depth, the contributions of the residual layers add up, and GPT-2 scales the weights of the residual layers at initialization by $1/\sqrt{N}$, with $N$ the number of residual layers. {#c-gpt2-init}
In the reference this is `init(..., residual_scaled=True)`, which divides the two projections that write to the residual stream in each layer by $\sqrt{2L}$; `test_residual_scaled_init` checks it.
GPT-3's batch grew from 0.5 million tokens for the smallest models to 3.2 million for 175B, and its learning rate fell from $6.0 \times 10^{-4}$ to $0.6 \times 10^{-4}$. {#c-gpt3-batch}

**Why pre-norm.**
Xiong et al. prove that at initialization the gradients of the parameters near the output of a post-norm Transformer are large in expectation, while with the normalization inside the residual branches they are well behaved, so pre-norm models can be trained without a warm-up stage. {#c-xiong}

**How much data.**
Kaplan et al. trained mainly decoder-only Transformers, on WebText2 with a BPE vocabulary of 50,257 and the cross-entropy averaged over a 1024-token context. {#c-kaplan-decoder}
They found the loss to depend strongly on scale, the number of non-embedding parameters, the data and the compute, and very weakly, within reasonable limits, on the shape of the network. {#c-kaplan-shape}
Hoffmann et al. later found that for compute-optimal training the model and the data should grow together, doubling the training tokens for every doubling of model size. {#c-chinchilla}
The [[neural-scaling-laws|scaling laws]] page covers these fits.

## Inductive biases

**The mask fixes what each position may know.**
Eq-mask makes the state at position $t$ a function of the tokens at $1, \dots, t$ alone, in every layer.
Training gets a prediction at every position from one pass, and decoding can stop at any point and continue later from the cache.
The NumPy test `test_causality` changes the token at one position and checks that the logits before it do not change at all and those from it on do; the JAX test `test_causality_through_the_jacobian` checks that the Jacobian of the logits with respect to the embeddings is zero above the diagonal.

**Without position embeddings, one layer sees a bag.**
Attention is a weighted sum, and a sum does not depend on the order of its terms.
With no position embeddings and one layer, the logits at position $i$ depend only on the token at $i$ and the multiset of the tokens at positions $1, \dots, i$: reordering the earlier tokens does not change them, whatever the parameters. {#c-bag}
With two layers this no longer holds: there are parameters for which reordering the earlier tokens changes the logits at $i$. {#c-two-layers}
The reason is that the mask itself carries position.
If the first token is $a$ and the $i - 1$ after it are $b$, a head of the first layer gives $a$ the weight $e^{s_a}/(e^{s_a} + (i-1)e^{s_b})$ at position $i$, which is different at every position, so the states after the first layer differ by position even though no position was given. {#c-mask-position}
The derivation `permutation` proves the three statements, the tests `test_one_layer_without_positions_sees_a_bag_of_earlier_tokens` and `test_the_causal_mask_alone_reveals_position` check them on random networks, and the experiment below measures what they mean for training.

**Shape matters less than size.**
With the non-embedding parameters held fixed, Kaplan et al. found the loss to vary only a few percent across a wide range of shapes: the ratio of width to depth can change by a factor of 40 with little effect, and a model with 6 layers of width 4288 reaches a loss within 3% of the 48 layers of width 1600 of GPT-2. {#c-kaplan-aspect}
Brown et al. chose the shapes of GPT-3 for computational efficiency and load-balancing across GPUs, relying on this insensitivity. {#c-gpt3-shape}

**What the decoder-only form gives up.**
A token never sees the tokens after it, which a [[bert|BERT]]-style encoder trained by [[masked-language-modelling|masked language modelling]] does; for tasks where the whole input is given at once, such as classifying a sentence, a decoder-only model has to read it left to right and put its answer at the end.
In exchange, it is a generative model of the full text, and any task that can be written as text to be continued can be posed to it without changing the architecture.

## Evidence

**Pre-training, then fine-tuning.**
GPT, pre-trained on BooksCorpus and fine-tuned per task, improved on the previous results on 9 of 12 tasks, by 8.9% absolute on Story Cloze, 5.7% on RACE and 1.5% on MultiNLI. {#c-gpt1-abstract}
In the same framework a single-layer LSTM of 2048 units scored 5.6 points lower on average and was better on one dataset only, MRPC, and the Transformer without pre-training was 14.8% worse than the full model. {#c-gpt1-ablation}
Heuristics that used the pre-trained model with no fine-tuning improved steadily over pre-training, and those of the LSTM varied more, which the authors took as a sign that the inductive bias of the Transformer helps transfer. {#c-gpt1-zeroshot}

**Zero-shot.**
GPT-2, at 1.5 billion parameters, improved on the previous results on 7 of 8 language-modelling datasets with no fine-tuning, still underfitting WebText, and its performance on tasks grew log-linearly with capacity. {#c-gpt2-abstract}
Conditioned on a document and questions, it answered at 55 F1 on CoQA, matching or exceeding 3 of 4 baselines without the 127,000 training examples. {#c-gpt2-coqa}
All four sizes still underfit WebText, and their held-out perplexity was still improving with more training. {#c-gpt2-underfit}

**In context.**
GPT-3, 175 billion parameters and 10 times larger than any previous non-sparse language model, was applied to every task without gradient updates, with the task and a few demonstrations given as text, and few-shot performance improved greatly with scale. {#c-gpt3-abstract}
Brown et al. call this in-context learning: the model is conditioned on an instruction or a few demonstrations and expected to complete further instances of the task by predicting what comes next. {#c-gpt3-icl}
The [[few-shot-in-context-learning|in-context learning]] page covers it.

These are reported results; this page has not reproduced them.
The GPT-1 comparison with the LSTM is a single run of each, and the GPT-2 and GPT-3 results are on data that was not released.

## Limitations

**Cost grows with the context.**
The attention products grow with $n^2$ per sequence in training, and in decoding the cache grows with $n$ and must be read at every step; for long contexts, these terms dominate.
FlashAttention, an exact attention algorithm that is aware of the memory hierarchy, computes the same products in tiles to cut the reads and writes between the GPU's high-bandwidth memory and its on-chip SRAM. {#c-flash}
LLaMA uses an implementation of causal attention that does not store the attention weights and does not compute the scores that the mask removes. {#c-llama-attn}

**Memory bandwidth at decoding time.**
Autoregressive decoding is a severe bottleneck because of the memory bandwidth needed to load the weights and all the keys and values at every step. {#c-gqa-bottleneck}
Multi-query and grouped-query attention, below, shrink the cache for this reason.
[[speculative-decoding|Speculative decoding]] attacks it from the other side, scoring several proposed tokens in one pass.

**A fixed context.**
Learned position embeddings exist for $n_{\text{ctx}}$ places only; a longer input must be cut, and a model cannot be run beyond its context without a new position scheme and further training.

**One direction.**
The model predicts left to right and cannot revise a token once it is written, and the probability it assigns to a text is only as good as its probability for each next token.

## Variants and descendants

- **Pre-norm and RMSNorm.** LLaMA normalizes the input of each sub-layer with [[rms-normalization|RMSNorm]], uses SwiGLU in the MLP with a hidden width of $\tfrac{2}{3} 4d$, and replaces the absolute position embeddings by rotary embeddings at each layer. {#c-llama}
- **Rotary position embeddings.** [[rotary-position-embedding|RoPE]] encodes the absolute position with a rotation and makes the attention scores depend on relative position. {#c-rope}
- **Parallel layers and no biases.** PaLM is decoder-only with SwiGLU, parallel layers, multi-query attention, RoPE and shared input-output embeddings. {#c-palm} Its parallel layer, $y = x + \operatorname{MLP}(\operatorname{LN}(x)) + \operatorname{Attention}(\operatorname{LN}(x))$, trained about 15% faster at large scale, with a small loss of quality at 8B and none at 62B. {#c-palm-parallel} PaLM drops all biases, which its authors found to improve training stability for large models. {#c-palm-nobias}
- **Multi-query attention.** All heads share a single key and value. {#c-mqa} Shazeer reports much faster decoding with only minor loss of quality. {#c-mqa-result} PaLM found it neutral for quality and training speed and a large saving at decoding time. {#c-palm-mqa} If the heads share one key and value of width $d_{\text{head}}$, the cache holds $2L\,d_{\text{head}}\,n$ numbers, $H$ times fewer, 48 KiB per token for GPT-3 175B. {#c-kv-mqa}
- **Grouped-query attention.** Between one key-value head and one per query head; Ainslie et al. also convert multi-head checkpoints with 5% of the pre-training compute. {#c-gqa} They report quality close to multi-head attention at a speed comparable to multi-query attention. {#c-gqa-result}
- **Sparse attention.** GPT-3 used the architecture of GPT-2 with alternating dense and locally banded sparse attention in its layers, like the Sparse Transformer. {#c-gpt3-arch}
- **Mixture of experts.** A [[sparse-mixture-of-experts|sparse mixture of experts]] replaces the MLP of some layers by several MLPs and a router, adding parameters without adding compute per token.

## Notable models

- **GPT** (2018), 12 layers of width 768, trained on BooksCorpus.
- **GPT-2** (2019), up to 1.5 billion parameters, trained on WebText.
- **GPT-3** (2020), eight sizes from 125 million to 175 billion parameters, trained on 300 billion tokens.
- **PaLM** (2022), up to 540 billion parameters, with parallel layers and multi-query attention.
- **LLaMA** (2023), 7 to 65 billion parameters, with RMSNorm, SwiGLU and rotary embeddings.

## Implementations

The NumPy reference `impl/numpy/decoder_lm.py` implements eq-embed, eq-mask with eq-attn, eq-layer with the exact GELU, eq-logits with a tied or an untied output matrix, eq-loss and its gradient by hand through every layer, the cache of keys and values, greedy decoding with and without the cache, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/decoder_lm.py` is an `nn.Module` with the same parameters; its tests compare a layer with torch.nn.TransformerEncoderLayer under a causal mask and the attention with torch.nn.functional.scaled_dot_product_attention, run `torch.autograd.gradcheck`, and check that the tied embedding is one tensor.
The JAX version `impl/jax/decoder_lm.py` is a pure function of a parameter dictionary with a fixed-size cache, so that a step of decoding compiles once under `jax.jit`; its tests check the cache against the full pass and the gradients against the shared fixture.
All three load the fixture of `fixtures/`: the example network on two random sequences, its logits, loss and gradients, the logits computed in two pieces with the cache, and five greedy tokens after a prompt of three.

### Worked example

Take one layer of width 4 with one head, and set the qkv projection to zero, so that every score is 0.
After the mask, position $t$ gives the weight $1/t$ to each of the $t$ positions it may see: the rows of $A$ are $(1, 0, 0, 0)$, $(\tfrac12, \tfrac12, 0, 0)$, $(\tfrac13, \tfrac13, \tfrac13, 0)$ and $(\tfrac14, \tfrac14, \tfrac14, \tfrac14)$, the running averages of the values.
The first token sees only itself, whatever the scores, and the last sees the whole sequence.
For GPT-3 175B the cache holds $2 \times 96 \times 12{,}288 = 2{,}359{,}296$ numbers per token, $2{,}359{,}296 \times 2 / 2^{20} = 4.5$ MiB in two-byte numbers, and 9 GiB for 2048 tokens.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **Predicting the current token instead of the next.** The logits at position $t$ are scored against the token at $t + 1$: the loss uses logits[:, :-1] and targets ids[:, 1:]. Scoring them against the token at $t$ asks the model to copy its input, which the residual stream does easily, and the loss falls to near zero while the model learns nothing.

> [!pitfall]
> **Forgetting the mask.** Without it, the state at $t$ sees the token at $t + 1$, training loss falls far below what is possible, and the model fails at generation, where the future is not there. A low training loss on a language model is a reason to check the mask. `test_causality` changes one token and checks that nothing before it changes.

> [!pitfall]
> **A mask aligned wrong when using the cache.** When $n$ new tokens follow $t$ cached ones, query $i$ may see keys $1, \dots, t + i$, not $1, \dots, i$; a square mask of size $n$ is wrong for a rectangle of $n \times (t + n)$ scores. The position embeddings of the new tokens must start at $t$ too. `causal_mask(2, 3)` in the tests is this case, and `test_cache_of_keys_and_values` checks that the logits from the cache match the full pass.

> [!pitfall]
> **Running past the context.** Learned position embeddings exist for $n_{\text{ctx}}$ places; indexing beyond them fails or, with some indexing rules, wraps around silently. The reference raises an error, and `test_context_limit` checks it in all three implementations.

> [!pitfall]
> **Untying the embedding by accident.** Copying the token embedding into a new output matrix, instead of using the same tensor, gives a model with $VD$ more parameters whose two copies drift apart. In the tied model the gradient of $W_e$ is the sum of its gradient as an embedding and as an output layer; `test_tied_gradient_is_the_sum_of_both_uses` and `test_tied_embedding_is_one_tensor` check this.

> [!pitfall]
> **Counting FLOPs with different conventions.** "FLOPs per token $\approx 2N$" and "FLOPs per token $\approx N$" describe the same model; the first counts a multiply-add as two operations, the second as one. Check which convention, and whether $N$ includes the embeddings, before comparing.

## Experiments

The fixture script `fixtures/make.py` trains four small models with the NumPy reference on a copy task.
A sequence is a start token, six symbols drawn uniformly from 8, a separator, and the same six symbols again: 14 tokens.
The last six can be predicted exactly by looking seven places back, so the model has to know where it is.
Each model has width 32, 4 heads and an MLP of 64, and is trained for 1500 steps of Adam at a learning rate of 0.003 on batches of 32, with the residual-scaled initialization at scale 0.02.
The test set is 500 sequences; the loss is averaged over the 13 predictions of each sequence, and since the first six symbols are random, no model can go below $6 \ln 8 / 13 = 0.9597$ nats.
The copy accuracy is the share of the six copied symbols that the most likely next token gets right.

| model | test loss (nats) | copy accuracy |
|---|---|---|
| 2 layers, learned positions | 0.9610 | 100.00% |
| 2 layers, no positions | 0.9644 | 100.00% |
| 1 layer, learned positions | 0.9616 | 100.00% |
| 1 layer, no positions | 1.6062 | 37.43% |

Three models reach the floor to within 0.005 nats and copy every symbol.
The fourth, one layer without positions, cannot: by c-bag its prediction at each position depends only on the current token and the multiset of the tokens so far, which tells which symbols occur but not in what order.
It copies 37.4% of the symbols, above the 12.5% of a guess.
The model with two layers and no positions copies all of them, because the mask lets the first layer compute a signal that differs by position, as in c-mask-position, and the second layer can attend by it.
These are single runs of one seed, on a toy task, in the fixture image with NumPy 2.5.4 on Linux, where another platform's rounding can move the last digits; they show what the two results allow, not how real models use position.
The checks of the four derivations count the parameters, FLOPs and cache of the models of the papers, count the products of the reference, and permute the tokens of random networks.

## History

Language models predicted the next word long before the Transformer, from [[n-gram-language-model|n-gram counts]] to the [[neural-probabilistic-language-model|neural probabilistic language model]] and recurrent networks.
Vaswani et al.'s Transformer of 2017 had a masked decoder for translation, and in 2018 Radford et al. used a decoder stack alone as the language model of GPT, pre-trained on books and fine-tuned on each task.
Later the same year BERT used the encoder instead, with a masked-word objective, and for a time the two lines competed on fine-tuned benchmarks.
GPT-2, in 2019, scaled the decoder to 1.5 billion parameters on web text and evaluated it without fine-tuning; Kaplan et al., in January 2020, measured how the loss of such models falls with parameters, data and compute; and GPT-3, in May 2020, found that at 175 billion parameters a decoder could learn tasks from examples in its context.
From then on most large language models were decoder-only, with the changes of PaLM and LLaMA to the layer and with [[byte-pair-encoding|subword vocabularies]] of tens to hundreds of thousands of tokens.

## References

- Radford, A., Narasimhan, K., Salimans, T. and Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. and others (2020). Language Models are Few-Shot Learners. NeurIPS 2020. arXiv 2005.14165.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J. and Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv 2001.08361.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017. arXiv 1706.03762.
- Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv 1911.02150.
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F. and Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP 2023. arXiv 2305.13245.
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L. and Liu, T.-Y. (2020). On Layer Normalization in the Transformer Architecture. ICML 2020. arXiv 2002.04745.
- Press, O. and Wolf, L. (2017). Using the Output Embedding to Improve Language Models. EACL 2017.
- Hendrycks, D. and Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv 1606.08415.
- Touvron, H., Lavril, T., Izacard, G. and others (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv 2302.13971.
- Chowdhery, A., Narang, S., Devlin, J. and others (2022). PaLM: Scaling Language Modeling with Pathways. arXiv 2204.02311.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W. and Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568. arXiv 2104.09864.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. and Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv 2205.14135.
- Hoffmann, J., Borgeaud, S., Mensch, A. and others (2022). Training Compute-Optimal Large Language Models. arXiv 2203.15556.
