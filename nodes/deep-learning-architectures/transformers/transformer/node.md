---
tag: 4W3NTQ
kind: architecture
slug: transformer
name: Transformer
aliases: [transformer architecture, encoder-decoder transformer]
area: deep-learning-architectures/transformers
level: core
summary: >-
  A neural network for sequences built from attention and position-wise feedforward layers, with residual connections and layer normalization, and no recurrence or convolution.
uses: [dropout, layer-normalization@al, relu@al, embedding@al, dot-product@m2, matrix-multiplication@m2]
related_to: [label-smoothing@al, beam-search@al, learning-rate-schedule@al, cross-entropy-loss@al]
history:
  year: 2017
  people: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
  note: "Vaswani et al. (2017), arXiv 1706.03762, NeurIPS 30."
prereqs: [embedding@al, layer-normalization@al, softmax-regression@al, backpropagation@al, matrix-multiplication@m2]
same_as:
  wikipedia: Transformer_(deep_learning_architecture)
  arxiv: ["1706.03762"]
sources:
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "NeurIPS 30; arXiv 1706.03762v7, Sections 3 to 6 and Tables 1 to 3", note: "the original paper and the canonical reference"}
  - {paper: "Bahdanau, Cho and Bengio (2015)", title: "Neural Machine Translation by Jointly Learning to Align and Translate", locator: "ICLR 2015; arXiv 1409.0473, Section 3", note: "attention in an encoder-decoder network"}
  - {paper: "He, Zhang, Ren and Sun (2016)", title: "Deep Residual Learning for Image Recognition", locator: "CVPR 2016, Section 3.2", note: "the residual connection"}
  - {paper: "Ba, Kiros and Hinton (2016)", title: "Layer Normalization", locator: "arXiv 1607.06450, Section 3"}
  - {paper: "Radford, Wu, Child, Luan, Amodei and Sutskever (2019)", title: "Language Models are Unsupervised Multitask Learners", locator: "OpenAI technical report, Section 2.3", note: "pre-normalization and the scaled initialization of GPT-2"}
  - {paper: "Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang and Liu (2020)", title: "On Layer Normalization in the Transformer Architecture", locator: "ICML 2020, PMLR 119, Sections 3 and 4"}
  - {paper: "Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (2020)", title: "Scaling Laws for Neural Language Models", locator: "arXiv 2001.08361, Section 2.1 and Table 1", note: "the FLOP count per token"}
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "NAACL 2019, Section 3"}
  - {paper: "Radford, Narasimhan, Salimans and Sutskever (2018)", title: "Improving Language Understanding by Generative Pre-Training", locator: "OpenAI technical report, Section 3.1"}
  - {paper: "Dosovitskiy et al. (2021)", title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", locator: "ICLR 2021, Sections 3.1 and 4.2"}
  - {paper: "Press, Smith and Lewis (2022)", title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", locator: "ICLR 2022, Sections 2 and 3"}
  - {paper: "Su, Lu, Pan, Murtadha, Wen and Liu (2024)", title: "RoFormer: Enhanced Transformer with Rotary Position Embedding", locator: "Neurocomputing 568; arXiv 2104.09864, Section 3"}
  - {paper: "Shazeer (2019)", title: "Fast Transformer Decoding: One Write-Head is All You Need", locator: "arXiv 1911.02150, Sections 2 and 3"}
  - {paper: "Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón and Sanghai (2023)", title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", locator: "EMNLP 2023, Section 2"}
  - {paper: "Shazeer (2020)", title: "GLU Variants Improve Transformer", locator: "arXiv 2002.05202, Sections 2 and 3"}
  - {paper: "Zhang and Sennrich (2019)", title: "Root Mean Square Layer Normalization", locator: "NeurIPS 32, Section 4"}
  - {paper: "Dao, Fu, Ermon, Rudra and Ré (2022)", title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", locator: "NeurIPS 35, Section 3"}
  - {paper: "Jain and Wallace (2019)", title: "Attention is not Explanation", locator: "NAACL 2019, Sections 4 and 5"}
  - {paper: "Hahn (2020)", title: "Theoretical Limitations of Self-Attention in Neural Sequence Models", locator: "TACL 8, 156 to 171, Sections 5 and 6"}
  - {paper: "Szegedy, Vanhoucke, Ioffe, Shlens and Wojna (2016)", title: "Rethinking the Inception Architecture for Computer Vision", locator: "CVPR 2016, Section 7", note: "label smoothing"}
  - {paper: "Sennrich, Haddow and Birch (2016)", title: "Neural Machine Translation of Rare Words with Subword Units", locator: "ACL 2016, Section 3", note: "byte-pair encoding"}
  - {paper: "Press and Wolf (2017)", title: "Using the Output Embedding to Improve Language Models", locator: "EACL 2017, Section 3", note: "tied embeddings"}
---

The Transformer is a neural network for sequences that is built from attention and position-wise feedforward layers.
It has no recurrence and no convolution, so it processes all the positions of a sequence in parallel during training.
Vaswani et al. introduced it in 2017 for machine translation.
It is now the base of most language models, and of many models of images, speech, code and proteins.

## In plain words

Each word of a sentence starts as a vector.
In each layer, every word looks at all the other words, decides how much each one matters to it, and takes a weighted mix of them.
That step is attention.
Then each word passes through a small network of its own.
After a stack of such layers, the vector of each word carries information about the whole sentence.
A translation model has two stacks: one reads the source sentence, and the other writes the translation one word at a time while it looks back at the first stack.

## Overview

The original model is an encoder and a decoder.
The encoder maps the input tokens $x_1, \dots, x_T$ to vectors of width $d_{\text{model}}$ through a stack of $L$ identical layers.
Each encoder layer has two sublayers: multi-head self-attention, and a feedforward network applied to each position.
The decoder has $L$ layers with three sublayers: masked self-attention over the tokens written so far, attention over the output of the encoder, and the feedforward network.
Each sublayer has a residual connection and then layer normalization, so its output is $\mathrm{LayerNorm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$ (Vaswani et al. 2017, Section 3.1). {#c-post-ln}
A final linear layer and a softmax give the distribution of the next output token.
The base model has $L = 6$, $d_{\text{model}} = 512$, $h = 8$ heads and $d_{\text{ff}} = 2048$ (Section 3 and Table 3). {#c-base-config}

Many later models keep only one of the two stacks.
BERT is an encoder only (Devlin et al. 2019), and GPT is a decoder only, with no attention over an encoder (Radford et al. 2018). {#c-encoder-decoder-only}

## Components

The components are given in the order the data flows through them.
Let $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$ hold the vectors of the $T$ positions, one row per position.

**Embedding.**
A token with index $v$ in the vocabulary $\mathcal{V}$ becomes row $v$ of the embedding matrix $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$, multiplied by $\sqrt{d_{\text{model}}}$.
The original model shares one matrix between the encoder embedding, the decoder embedding and the output layer before the softmax (Section 3.4). {#c-tied-embeddings}
Press and Wolf (2017) study this tying in language models.

**Positional encoding.**
Attention treats its inputs as a set, so the model adds a vector that depends on the position $\mathrm{pos}$ to each embedding.
The original model uses fixed sinusoids, with $j$ the index of a pair of dimensions:

$$ \mathrm{PE}_{\mathrm{pos},\, 2j} = \sin\!\left(\frac{\mathrm{pos}}{10000^{2j/d_{\text{model}}}}\right), \qquad \mathrm{PE}_{\mathrm{pos},\, 2j+1} = \cos\!\left(\frac{\mathrm{pos}}{10000^{2j/d_{\text{model}}}}\right). $$ {#eq-pe}

Each pair of dimensions is a rotation at its own frequency.
For a fixed offset $k$, the pair at $\mathrm{pos} + k$ is a rotation of the pair at $\mathrm{pos}$ by an angle that depends only on $k$, so $\mathrm{PE}_{\mathrm{pos}+k}$ is a linear function of $\mathrm{PE}_{\mathrm{pos}}$ (Section 3.5). {#c-pe-linear-offset}
Learned position embeddings gave nearly the same results in the paper (Table 3, row E). {#c-learned-pe-same}

**Scaled dot-product attention.**
Given queries $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{T' \times d_k}$ and values $\mathbf{V} \in \mathbb{R}^{T' \times d_v}$,

$$ \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}, $$ {#eq-attention}

where the softmax is taken over each row, and $\mathbf{M}$ is a mask with entries $0$ or $-\infty$.
Row $t$ of the output is a weighted mean of the rows of $\mathbf{V}$, with weights that sum to 1.
In the decoder, $M_{ts} = -\infty$ for $s > t$, so a position cannot attend to later positions, and the model can be trained on a whole sequence at once while it still predicts each token from the tokens before it. {#c-causal-mask}
In self-attention, $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$ all come from the same $\mathbf{X}$, and $T' = T$.
In the attention over the encoder, $\mathbf{Q}$ comes from the decoder and $\mathbf{K}$ and $\mathbf{V}$ come from the output of the encoder.

**Why the scale.**
Suppose the entries of a query $\mathbf{q}$ and a key $\mathbf{k}$ are independent with mean 0 and variance 1.
Then $\mathbf{q}^\top \mathbf{k} = \sum_{c=1}^{d_k} q_c k_c$ is a sum of $d_k$ independent terms, each with mean 0 and variance $\mathbb{E}[q_c^2]\,\mathbb{E}[k_c^2] = 1$.
So the dot product has mean 0 and variance $d_k$ (Section 3.2.1, footnote 4). {#c-dot-variance}
Dividing by $\sqrt{d_k}$ brings the variance back to 1.
Without the scale, large dot products push the softmax into regions where its gradient is very small, which is the reason the paper gives for the scale. {#c-scale-reason}

**Worked example.**
Take $T = 4$ positions and $d_k = 2$, with no mask, and

$$ \mathbf{Q} = \mathbf{K} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}, \qquad \mathbf{V} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 2 \end{pmatrix}. $$

The scores $\mathbf{Q}\mathbf{K}^\top$ have the rows $(1, 0, 1, 0)$, $(0, 1, 1, 0)$, $(1, 1, 2, 0)$ and $(0, 0, 0, 0)$.
Divide by $\sqrt{2}$, so $1$ becomes $0.7071$ and $2$ becomes $1.4142$.
For the third row, $e^{0.7071} = 2.0281$, $e^{1.4142} = 4.1133$ and $e^0 = 1$, which sum to $9.1695$.
The weights are $(0.2212, 0.2212, 0.4486, 0.1091)$, and the output is $0.2212 \cdot (1, 0) + 0.2212 \cdot (0, 1) + 0.4486 \cdot (1, 1) + 0.1091 \cdot (0, 2) = (0.6698, 0.8879)$.
The fourth query is zero, so all its scores are equal, its weights are $1/4$ each, and its output is the mean of the values, $(0.5, 1)$.
With the causal mask, the first position can attend only to itself, so its output is $(1, 0)$, the first value.
The third position then attends only to the first three, with weights $(0.2483, 0.2483, 0.5035)$ and the output $(0.7517, 0.7517)$.

**Multi-head attention.**
The model runs $h$ attentions in parallel, each on its own projection of the input:

$$ \mathrm{head}_r = \mathrm{Attention}(\mathbf{X}\mathbf{W}^Q_r, \mathbf{X}\mathbf{W}^K_r, \mathbf{X}\mathbf{W}^V_r), \qquad \mathrm{MultiHead}(\mathbf{X}) = [\mathrm{head}_1, \dots, \mathrm{head}_h]\, \mathbf{W}^O, $$ {#eq-multihead}

with $\mathbf{W}^Q_r, \mathbf{W}^K_r \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $\mathbf{W}^V_r \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.
The base model uses $d_k = d_v = d_{\text{model}} / h = 64$, so multi-head attention costs about as much as one head of full width (Section 3.2.2). {#c-head-width}
With one head, the base model lost 0.9 BLEU on the development set (Table 3, row A). {#c-single-head-worse}

**Position-wise feedforward network.**
Each position goes through the same two-layer network:

$$ \mathrm{FFN}(\mathbf{x}) = \max(0,\, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\, \mathbf{W}_2 + \mathbf{b}_2, $$ {#eq-ffn}

with $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ (Section 3.3).
The weights are shared across positions and differ from layer to layer.

**Residual connection and layer normalization.**
The residual connection of He et al. (2016) adds the input of a sublayer to its output.
Layer normalization (Ba et al. 2016) normalizes each position by the mean and the variance of its $d_{\text{model}}$ features, then applies a learned scale and shift.
Where the normalization sits is the main difference between the original model and most later ones; see Training.

**Output.**
The decoder output at position $t$ goes through the output matrix and a softmax, which gives $p_\theta(y_{t+1} \mid y_{\le t}, \mathbf{x})$ over $\mathcal{V}$.

## Shapes and parameters

The tables below come from `shapes.yaml`, and so does every formula and count in this section.
Write $d_{\text{model}}$ for the width, $h$ for the number of heads, $d_{\text{ff}}$ for the width of the feedforward network, $L$ for the number of layers in the encoder and again in the decoder, and $|\mathcal{V}|$ for the size of the vocabulary.
The self-attention of an encoder layer has {shape:enc_attn.params} parameters, for the four projections with their biases.
Its feedforward network has {shape:enc_ffn.params}, and each of its two layer normalizations has {shape:norm.params}, a scale and a shift for each feature.
So an encoder layer has {shape:enc_layer.params} parameters.
A decoder layer adds the attention over the encoder and a third normalization, for {shape:dec_layer.params}.
The positional encoding has no parameters, and with tied embeddings the embedding matrix adds {shape:embed.params} once.
For the sizes of the base model in Table 3, and a vocabulary of 37,000 tokens, the formula gives {shape:params@base} parameters; for the big model it gives {shape:params@big}.
The paper reports 65 million parameters for the base model and 213 million for the big model (Table 3). {#c-param-counts}
It gives the shared vocabulary only as about 37,000 tokens (Section 5.1), so the count from the formula and the reported count need not agree exactly.

## Computation and cost

Count a multiply-add as 2 FLOPs, and leave out the biases, the softmax, the normalizations and the activations.
Take a batch of $B$ pairs, with $S$ tokens in each source and $T$ in each target.
The self-attention of an encoder layer costs {shape:enc_attn.flops_forward} FLOPs: the first term is the four projections, and the second is the scores $\mathbf{Q}\mathbf{K}^\top$ and the weighted sum of the values, over all heads.
Its feedforward network costs {shape:enc_ffn.flops_forward}.
In a decoder layer, the masked self-attention costs {shape:dec_attn.flops_forward}, and the attention over the encoder costs {shape:cross_attn.flops_forward}, because its keys and values are projections of the $S$ positions of the encoder.
The count takes the full $T \times T$ matrix of scores in the masked self-attention, as a dense matrix product computes it, though half of it is masked.
The output layer costs {shape:logits.flops_forward}, so the forward pass costs {shape:flops_forward} FLOPs in all.
With $S = T$, the terms of the projections and the feedforward networks come to $2N$ FLOPs per target token, for the $N =$ {shape:params_non_embedding} weights of the layers when their biases are left out; the output layer adds twice the size of the embedding, and the scores and the weighted sums add a term that grows with $T$.
Kaplan et al. (2020, Section 2.1) use the same count, $2N + 2 L T d_{\text{model}}$ per token for a decoder, and about 6 times $N$ per token for a step of training, because the backward pass costs about twice the forward pass. {#c-flops-per-token}

The cost in the length of the sequence is the main limit of the architecture.
The projections and the feedforward network grow linearly in $T$, but the scores grow as $T^2$, in FLOPs and in memory: each layer and each head forms a $T \times T$ matrix.
A self-attention layer costs $O(T^2 d_{\text{model}})$ against $O(T d_{\text{model}}^2)$ for a recurrent layer, needs $O(1)$ sequential operations against $O(T)$, and connects any two positions by a path of length $O(1)$ against $O(T)$ (Table 1). {#c-complexity-table}
Self-attention is cheaper than a recurrent layer when $T < d_{\text{model}}$, which was the case for the sentences of machine translation in 2017.

At inference, a decoder writes one token at a time.
It keeps the keys and the values of the earlier positions, so that each new token costs $O(T d_{\text{model}})$ in attention and not $O(T^2 d_{\text{model}})$.
This cache holds {shape:kv_cache} numbers per sequence, which is the main memory cost of serving long contexts.
Multi-query attention (Shazeer 2019) and grouped-query attention (Ainslie et al. 2023) share keys and values across heads to shrink it. {#c-mqa-cache}

## Training

The original model was trained on WMT 2014 English to German, about 4.5 million sentence pairs with a shared byte-pair vocabulary of about 37,000 tokens, and on WMT 2014 English to French, 36 million sentences with a vocabulary of 32,000 word pieces (Section 5.1). {#c-wmt-data}
Byte-pair encoding splits rare words into frequent pieces (Sennrich et al. 2016).
The loss is the cross-entropy of the next token, with label smoothing $\epsilon_{\text{ls}} = 0.1$ (Szegedy et al. 2016), which made the perplexity worse but improved accuracy and BLEU (Section 5.4). {#c-label-smoothing}
The optimizer was Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$, and the learning rate rose linearly for 4,000 warmup steps and then fell as the inverse square root of the step (Section 5.3):

$$ \eta_t = d_{\text{model}}^{-1/2} \cdot \min\!\left(t^{-1/2},\; t \cdot t_{\text{warmup}}^{-3/2}\right). $$ {#eq-schedule}

Dropout with $p = 0.1$ was applied to the output of each sublayer before the residual addition, and to the sums of the embeddings and the positional encodings (Section 5.4). {#c-residual-dropout}
On 8 NVIDIA P100 GPUs, the base model trained for 100,000 steps in 12 hours, and the big model for 300,000 steps in 3.5 days (Section 5.2). {#c-training-time}

**Where the normalization goes.**
The original model normalizes after the residual addition, which is called post-LN.
GPT-2 moved the normalization to the input of each sublayer and added one after the last layer (Radford et al. 2019, Section 2.3), which is called pre-LN: $\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))$.
Xiong et al. (2020) show that at initialization the gradients of the parameters near the output of a post-LN model are large, which is why it needs a warmup, and that a pre-LN model can be trained without warmup. {#c-pre-ln-warmup}
GPT-2 also scales the initial weights of the residual layers by $1/\sqrt{N_{\text{res}}}$, with $N_{\text{res}}$ the number of residual layers (Section 2.3). {#c-gpt2-init}

The modern recipe for language models trains a decoder only on next-token prediction over very large corpora.
It often replaces layer normalization by RMSNorm (Zhang and Sennrich 2019), the ReLU network by a gated one such as SwiGLU (Shazeer 2020), and the sinusoids by rotary position embeddings (Su et al. 2024).

## Inductive biases

Without positional information, self-attention and the feedforward network treat their input as a set.
Permuting the input positions permutes the output positions in the same way, because each output row depends on the other rows only through sums over all of them. {#c-permutation-equivariance}
So the only notion of order is the one the positional encoding gives.
The architecture also does not assume locality: any position can attend to any other in one layer.
Dosovitskiy et al. (2021, Section 3.1) note that a vision Transformer has much less image-specific inductive bias than a convolutional network, and they report that it does worse than ResNets when it is trained on ImageNet alone and better after pre-training on much larger datasets (Section 4.2). {#c-vit-data-hungry}
The causal mask of a decoder adds one bias: each prediction depends only on the past.

## Evidence

On WMT 2014, the big model reached 28.4 BLEU on English to German and 41.8 on English to French, better than the earlier single models and ensembles that the paper compares, at a fraction of their training cost (Table 2). {#c-wmt-bleu}
The base model reached 27.3 and 38.1. {#c-wmt-bleu-base}
The same architecture, with little task-specific tuning, gave 91.3 F1 on English constituency parsing trained on the Wall Street Journal part of the Penn Treebank only, and 92.7 in a semi-supervised setting (Section 6.3, Table 4). {#c-parsing}

## Limitations

The cost of attention is quadratic in the length of the sequence, in compute and in memory.
FlashAttention computes exact attention without storing the $T \times T$ matrix in the main memory of the GPU, which reduces the memory to linear in $T$ but not the FLOPs (Dao et al. 2022). {#c-flash-memory}
Models with sinusoidal positions do not keep their quality on sequences longer than the ones they were trained on, which Press et al. (2022) show and address with linear biases on the scores (ALiBi). {#c-length-extrapolation}
Hahn (2020) proves that self-attention with a fixed number of layers and heads cannot recognize some formal languages, such as parity and Dyck languages, for all lengths, under assumptions on the attention. {#c-formal-limits}
Attention weights are not a reliable explanation of a prediction: Jain and Wallace (2019) find attention distributions that differ a lot from the original ones but give the same predictions. {#c-attention-not-explanation}

## Variants and descendants

The lineage is generated from the edges `variant_of`.
The main lines are the encoder-only models such as BERT, the decoder-only language models such as GPT, the encoder-decoder models of translation and summarization, and the vision Transformer, which treats patches of an image as tokens.

## History

Attention in neural networks for translation came before the Transformer: Bahdanau et al. (2015) added it to a recurrent encoder-decoder, so the decoder could look at all the states of the encoder and not only the last one. {#c-bahdanau}
Vaswani et al. posted "Attention Is All You Need" to arXiv in June 2017 and presented it at NeurIPS 2017. {#c-published-2017}
The paper gives the affiliations Google Brain, Google Research and the University of Toronto. {#c-authors}
The paper says that the authors contributed equally and that the order of the names is random. {#c-equal-contribution}
Within two years, BERT and GPT used the architecture for pre-trained language models, and within four the vision Transformer used it for images.
