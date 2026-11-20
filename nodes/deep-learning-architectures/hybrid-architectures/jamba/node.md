---
tag: 2V1JK1
kind: architecture
slug: jamba
name: Jamba
aliases: [Jamba-v0.1, hybrid Transformer-Mamba, Attention-Mamba hybrid, Jamba block]
area: deep-learning-architectures/hybrid-architectures
level: core
summary: >-
  A language model that interleaves attention with Mamba layers, one attention layer in eight, and makes every other MLP a mixture of experts, so its cache stays small at long contexts.
uses: [matrix-multiplication@m2]
related_to: [mamba, transformer, sparse-mixture-of-experts, multi-head-attention, attention-mechanism, structured-state-space-model, rms-normalization, rotary-position-embedding, linear-attention, byte-pair-encoding, few-shot-in-context-learning, llama]
history:
  year: 2024
  people: [Opher Lieber, Barak Lenz, Yoav Shoham]
  note: "AI21 Labs posted Jamba to arXiv (2403.19887) in March 2024 and released the weights of Jamba-v0.1, 12B active and 52B total parameters, under the Apache 2.0 license. It followed Mamba (2023) and Mixtral (2023), and was the first hybrid of attention and a state space model trained at that scale."
prereqs: [mamba, transformer, sparse-mixture-of-experts]
same_as:
  arxiv: ["2403.19887"]
sources:
  - {paper: "Lieber et al. (2024)", title: "Jamba: A Hybrid Transformer-Mamba Language Model", locator: "arXiv 2403.19887v2, Sections 1 to 7, Tables 1 to 8 and Figures 1 to 9", note: "the architecture, the released configuration, the evaluations and the ablations"}
  - {paper: "AI21 Labs (2024)", title: "config.json of Jamba-v0.1", locator: "Hugging Face, revision 9efd115", note: "the sizes of the released model"}
family: hybrid
inputs: "a (B, T) array of token ids"
outputs: "a (B, T, V) array of next-token scores, causal: the score at position t depends on the tokens up to t only"
components: [mamba, multi-head-attention, sparse-mixture-of-experts, rms-normalization]
inductive_biases: ["causal; most layers are recurrent, with a state of fixed size, and one in l is attention over the whole past", "no explicit positional information: the Mamba layers before an attention layer are the only source of order it gets", "sparse capacity: every e-th MLP is a mixture of n experts of which each token uses K"]
params: "2 V D + D + L_b (P_attn + (l - 1) P_mamba + (l - l / e) 3 D F + (l / e)(D n + 3 n D F)) for L_b blocks of l layers, width D, MLP width F, n experts, vocabulary V, with the attention and Mamba layers of eq-params"
flops_per_example: "the products of eq-flops, 2 per multiply-add: about twice the active parameters per token, plus 4 H A T^2 for each attention layer"
---

Jamba is a decoder-only language model built from three parts that earlier models used separately: attention layers, Mamba layers, and mixtures of experts.
Lieber et al. present Jamba as a base large language model with a hybrid Transformer-Mamba mixture-of-experts architecture that interleaves blocks of Transformer and Mamba layers, with MoE added in some of these layers to increase model capacity while keeping active parameter usage manageable. {#c-def}
The released model is 7B-based, with 12B active and 52B total available parameters, and was designed to fit in a single 80GB GPU. {#c-release}
Its point is a trade: one layer in eight keeps the attention of a [[transformer|Transformer]], the other seven are [[mamba|Mamba]] layers whose state does not grow with the text, and the cache of keys and values shrinks by the same factor of eight.

## In plain words

A Transformer reads a text by letting every word look back at every earlier word, and to write the next word it keeps all the earlier ones in memory, which grows with the length of the text.
A recurrent model such as Mamba instead carries a summary of fixed size forward, which is cheap, but a summary can lose the detail that the next word needs.
Jamba stacks layers of both kinds: most are Mamba layers, which pass the summary along, and a few are attention layers, which can still look back at any word exactly.
On top, half of the layers replace their single feed-forward network by a team of 16, of which each word consults the 2 that a small router picks, so that the model has many parameters but uses few for each word.
The result fits in one large GPU with a context of over a hundred thousand words, where an all-attention model of the same size does not.

## Overview

Lieber et al. name two drawbacks of the Transformer: its high memory and compute requirements hinder the processing of long contexts, where the key-value (KV) cache becomes a limiting factor, and its lack of a single summary state entails slow inference and low throughput, since each generated token performs a computation on the entire context. {#c-transformer-drawbacks}
Older recurrent neural networks summarize the context in a single hidden state, but they are costly to train, since training cannot be parallelized across time steps, and struggle with long-distance relationships. {#c-rnn}
State space models such as Mamba are more efficient to train than RNNs and more capable at handling long-distance relationships, but still lag behind the performance of comparably sized Transformer language models. {#c-ssm-lag}
Jamba combines the two kinds of layers at a ratio, and varying the ratio of Transformer to Mamba layers balances memory usage, efficient training and long-context capabilities.

Earlier hybrids were smaller or weaker: H3 replaced the second and middle layers with self-attention at up to 2.7B parameters and 400B training tokens but lags pure Mamba, and StripedHyena interleaves attention and SSM layers in a 7B model but lags the attention-only Mistral-7B. {#c-prior-hybrids}
The authors call Jamba the first production-grade Attention-SSM hybrid model. {#c-first}
The mixture of experts adds a second trade: MoE layers increase the model capacity, the total number of available parameters, without increasing the compute requirements, the number of active parameters; more MoE layers and more experts per layer give more total parameters, and more experts used at each forward pass give more active parameters and more compute. {#c-moe-capacity}

## Components

### The Jamba block

A Jamba block contains $l$ layers mixed at a ratio of $a : m$, $a$ attention layers for every $m$ Mamba layers; each layer contains an attention or a Mamba module followed by an MLP, and every $e$ layers the MLP may be replaced by a mixture of $n$ experts with a router choosing the top $K$ at each token. {#c-block}
The five degrees of freedom are $l$, $a : m$, $e$, $n$ and $K$.
Every layer has the pre-norm residual form of a Transformer layer, with RMSNorm:

$$ r \leftarrow r + \operatorname{mixer}\big(\operatorname{RMSNorm}(r)\big), \qquad r \leftarrow r + \operatorname{ffn}\big(\operatorname{RMSNorm}(r)\big) . $$ {#eq-layer}

In Jamba-v0.1 each block has one attention layer and seven Mamba layers, and the released configuration places them by a period and an offset: layer $i$ has attention when $i \bmod 8 = 4$ and a mixture of experts when $i \bmod 2 = 1$. {#c-config-released}
For a block of $l$ layers and a mixture every $e$, the page writes this as

$$ \operatorname{mixer}_i = \begin{cases} \text{attention} & i \bmod l = l/2 \\ \text{Mamba} & \text{otherwise} \end{cases}, \qquad \operatorname{ffn}_i = \begin{cases} \text{MoE} & i \bmod e = e - 1 \\ \text{MLP} & \text{otherwise} \end{cases} $$ {#eq-layout}

so the attention layer, at an even index, always keeps a single MLP.
Figure 1 of the paper also draws an Attention MoE layer, which the architecture does not use but future variants could. {#c-attn-moe}

### Attention

The attention layers are causal self-attention with grouped-query attention (GQA): $H$ query heads of size $A$ share $G$ key-value heads, head $h$ reading the keys and values of group $\lfloor hG/H \rfloor$.
With $u = \operatorname{RMSNorm}(r)$ of shape $(T, D)$,

$$ \operatorname{head}_h = \operatorname{softmax}\!\Big(\frac{u W_Q^{(h)} \big(u W_K^{(g)}\big)^\top}{\sqrt{A}} + M\Big)\, u W_V^{(g)}, \qquad g = \lfloor hG/H \rfloor, $$ {#eq-attn}

with $M_{ts} = -\infty$ for $s > t$, and the heads are concatenated and multiplied by $W_O$.
The authors found that with the Mamba layers positional embeddings or mechanisms like RoPE are not necessary, and Jamba uses no explicit positional information. {#c-nope}
So $q$ and $k$ carry no position: an attention layer alone cannot tell the order of the tokens before it, and the order reaches it only through the Mamba layers below, whose recurrence is causal.
Apart from the grouping of the keys and values and the missing positions, each layer is the [[multi-head-attention|multi-head attention]] of the Transformer, with its full $T \times T$ scores.

### The Mamba layer

The Mamba layer is the block of [[mamba|Mamba]] with three normalizations added.
With $u = \operatorname{RMSNorm}(r)$, the input projection gives $x$ and the gate $z$; a causal depthwise convolution of width $W$ and SiLU give $x'$; a projection of $x'$ gives the low-rank input $\delta$ of the step size, and the input and output maps $B_t, C_t \in \mathbb{R}^N$; and

$$ \Delta_t = \operatorname{softplus}\big(\tilde{\delta}_t W_\Delta + b_\Delta\big), \quad h_t = e^{\Delta_t A} \odot h_{t-1} + \Delta_t\, \tilde{B}_t\, x'_t, \quad y_t = h_t \tilde{C}_t + D \odot x'_t, \quad \operatorname{out}_t = \big(y_t \odot \operatorname{SiLU}(z_t)\big) W_{\text{out}}, $$ {#eq-mamba}

where $\tilde{\delta}$, $\tilde{B}$ and $\tilde{C}$ are $\delta$, $B$ and $C$ after RMSNorm, $A$ is diagonal and negative, and $h_t \in \mathbb{R}^{I \times N}$ holds $N$ states for each of the $I = ED$ channels.
When scaling to the 7B-based model, the authors encountered large loss spikes, which they traced to large activation values in inner parts of the Mamba layers, and adding RMSNorm to internal activations stabilized training and prevented further spikes. {#c-rmsnorm}
The paper does not say which activations; the page normalizes $\delta$, $B$ and $C$, the three that the released code normalizes, and uses the discretization of Mamba's reference code, $\bar{B} = \Delta B$.

### The mixture of experts

Each MoE layer has $n$ SwiGLU experts $f_j(v) = \big(\operatorname{SiLU}(v W_{\text{gate}}^{(j)}) \odot v W_{\text{up}}^{(j)}\big) W_{\text{down}}^{(j)}$ and a linear router.
For a token $v$, let $p = \operatorname{softmax}(v W_r)$ and $\mathcal{T}(v)$ the $K$ experts of largest $p_j$; then

$$ \operatorname{MoE}(v) = \sum_{j \in \mathcal{T}(v)} p_j\, f_j(v), $$ {#eq-moe}

and only the $K$ experts in $\mathcal{T}(v)$ run on $v$.
The page weights the outputs by the router probabilities themselves, without renormalizing the $K$ of them to sum to 1; the paper does not specify this detail.
Jamba uses load balancing for the MoE, and other architecture details are standard: grouped-query attention, SwiGLU activations, a vocabulary of 64K from a BPE tokenizer in which each digit is a separate token, without the dummy space of the Llama and Mistral tokenizers. {#c-standard}
The page's balance term is that of the Switch Transformer: over the $M$ tokens of a batch, with $f_j$ the share of the $MK$ choices that go to expert $j$ and $P_j$ the mean of $p_j$,

$$ \mathcal{L}_{\text{aux}} = \alpha\, n \sum_{j=1}^{n} f_j P_j , $$ {#eq-aux}

which is $\alpha$ when both are uniform; the released configuration sets router_aux_loss_coef to 0.001. {#c-aux-coef}

## Shapes and parameters

Jamba-v0.1 is a sequence of 4 Jamba blocks, each with $l = 8$ layers, a ratio $a : m = 1 : 7$ of attention to Mamba layers, MoE every $e = 2$ layers, $n = 16$ experts and the top $K = 2$ at each token. {#c-config}
The 1:7 ratio was chosen as the most compute-efficient variant among the best performing ones in the ablations, and $n$ and $e$ were balanced to fit the model in a single 80GB GPU with int8 weights, an average of about 8 experts per layer. {#c-config-why}
The released config.json sets hidden_size 4096, intermediate_size 14336, num_hidden_layers 32, num_attention_heads 32, num_key_value_heads 8, num_experts 16, num_experts_per_tok 2, mamba_d_state 16, mamba_d_conv 4, mamba_expand 2, mamba_dt_rank 256 and vocab_size 65536. {#c-config-sizes}

Counting every array, with $P_{\text{attn}} = 2D^2 + 2DGA + 2D$, $P_{\text{mamba}} = 3ID + I(W + 1) + I(R + 2N) + R + 2N + (R + 1)I + IN + I + 2D$ (the two layer norms included in both), an MLP of $3DF$ and a mixture of $Dn + 3nDF$:

$$ P = 2VD + D + L_b\Big(P_{\text{attn}} + (l - 1)P_{\text{mamba}} + \big(l - \tfrac{l}{e}\big) 3DF + \tfrac{l}{e}\big(Dn + 3nDF\big)\Big), \qquad P_a = P - L_b \tfrac{l}{e} (n - K)\, 3DF . $$ {#eq-params}

With the released sizes, Jamba-v0.1 has 51,570,323,328 parameters, which round to the 52B of Table 1. {#c-params}
A token uses 12,110,311,296 of them, the 12B active parameters of Table 1: each of the 16 mixtures leaves 14 of its experts unused, 224 experts of $3DF = 176{,}160{,}768$ parameters. {#c-active}
The 256 experts hold 45.1B parameters, 87% of the model; the 28 Mamba layers hold 2.95B and the 4 attention layers 0.17B. {#c-experts-share}
The derivation `parameter-count` proves these numbers.
In the shape contract,

{shape:params}

and the released configuration has {shape:params@jamba_v0_1} parameters, {shape:active_params@jamba_v0_1} of them active; the example of the fixtures, one block of 4 layers of width 8 with 4 experts, has {shape:params@example}, {shape:active_params@example} active.

| Model | Available | Active | KV cache at 256K, 16-bit |
|---|---|---|---|
| Llama-2 | 6.7B | 6.7B | 128GB |
| Mistral | 7.2B | 7.2B | 32GB |
| Mixtral | 46.7B | 12.9B | 32GB |
| Jamba | 52B | 12B | 4GB |

The table is Table 1 of the paper. {#c-table1}

## Computation and cost

A forward pass over $B$ sequences of $T$ tokens costs, in the products,

$$ C = L_b\Big(C_{\text{attn}} + (l - 1)\, 2BT\big(2DI + I(R + 2N) + RI + IN + ID\big) + \big(l - \tfrac{l}{e}\big) 6BTDF + \tfrac{l}{e}\, 2BT(Dn + 3KDF)\Big) + 2BTDV $$ {#eq-flops}

FLOPs, a multiply-add counted as 2, with $C_{\text{attn}} = 2BT(2D^2 + 2DGA) + 4BHT^2A$; the norms, the activations, the convolution and the update of the scan are left out.
The contract gives {shape:flops_forward@example} for the example and {shape:flops_forward} in general.
All terms but one are linear in $T$; the $4BHT^2A$ of attention is quadratic, and with one attention layer in eight it is an eighth of what a Transformer of the same depth pays.
With short sequences attention takes a small fraction of the FLOPs, but with long sequences it takes most of the compute, while Mamba layers are more compute-efficient, so increasing the ratio of Mamba layers improves throughput, especially for long sequences. {#c-throughput-attn}

Generation keeps, per sequence, the keys and values of each attention layer and the convolution window and state of each Mamba layer:

$$ S(T) = n_a \cdot 2GA\,T + (L - n_a)\, I\,(W - 1 + N), $$ {#eq-cache}

for $n_a$ attention layers out of $L$.
With one attention layer in eight, Jamba's KV cache is 8 times smaller than that of a Transformer of the same depth with the same grouped-query heads. {#c-kv-8x}
At 256K tokens in 16-bit, 4 attention layers with 8 key-value heads of 128 hold exactly 4 GiB, and the same formula gives 32 GiB for 32 such layers and 128 GiB for 32 layers with 32 key-value heads, the three numbers of Table 1. {#c-table1-cache}
The 28 Mamba layers keep 4,358,144 numbers at any length, 0.2% of the cache at 256K tokens, and an attention layer outgrows a Mamba layer after 76 tokens. {#c-state-small}
The derivation `kv-cache` proves these numbers; the contract gives a KV cache of {shape:kv_cache@jamba_v0_1} numbers and a recurrent state of {shape:recurrent_state@jamba_v0_1}.

Jamba fits 2x the context length of Mixtral and 7x that of Llama-2-70B in a single A100 80GB GPU. {#c-context-fit}
On a single A100 80GB GPU with int8 weights, an 8K context and 512 output tokens, Jamba processes large batches with 3x the throughput of Mixtral, which does not fit a batch of 16; on 4 A100 GPUs with one sequence and no quantization the models are similar at small contexts, and at 128K tokens Jamba's throughput is 3x that of Mixtral. {#c-throughput}
The authors note that these results are without the optimizations that the community has developed for pure Transformer models.

## Training

The loss is the next-token cross-entropy plus the balance terms of the mixtures:

$$ \mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log \operatorname{softmax}\big(f(w_{b,1:t})\big)_{w_{b,t+1}} + \sum_{\text{MoE layers}} \mathcal{L}_{\text{aux}} . $$ {#eq-loss}

The model was trained on NVIDIA H100 GPUs with an in-house framework using FSDP and tensor, sequence and expert parallelism, on an in-house dataset of text from the Web, books and code last updated in March 2024. {#c-infra}
Jamba models were trained on context lengths of up to 1M tokens, and the released model supports up to 256K tokens. {#c-1m}
The released model is a pretrained base model that did not go through alignment or instruction tuning and has no moderation mechanisms. {#c-base}
The paper gives no learning rate, batch size or token count for the released model.

## Inductive biases

Seven layers in eight are recurrent, so the model holds most of its memory of the past in states of fixed size and pays for exact recall only in the attention layers.
The order of the tokens enters only through the Mamba layers: Lieber et al. compared Jamba with no positional information and with RoPE in the attention layers at 1.3B parameters and 250B tokens, found the results similar, and suggest that the Mamba layers, placed before the attention layers, provide implicit position information. {#c-nope-ablation}
The mixtures add capacity that a token reaches only through the router, so different tokens can use different parameters at the cost of one MLP's compute times $K$.

## Evidence

Table 2 compares Jamba-v0.1 with open models on standard benchmarks.
Jamba scores 87.1 on HellaSwag against 86.7 for Mixtral and 85.3 for Llama-2 70B, 59.9 on GSM8K against 60.4 and 55.3, 67.4 on MMLU against 70.6 and 69.8, and 29.3 on HumanEval against 34.8 and 29.9; the authors summarize that in most tasks it performs comparably to Llama-2 70B and Mixtral. {#c-bench}
Jamba has excellent performance in the needle-in-a-haystack evaluation, although it uses only 4 attention layers. {#c-needle}
On five long-context question-answering datasets of L-Eval with average inputs of 6K to 62K tokens in a 3-shot format, Jamba outperforms Mixtral on most datasets and on average, an F1 of 0.44 against 0.43. {#c-longqa}

The ablations explain the design.
At 1.3B parameters and 250B tokens the hybrid outperforms pure attention and pure Mamba, with an OLLM score of 37.2 against 36.4 and 36.1 and HellaSwag 65.1 against 62.4 and 62.6, and the ratios 1:3 and 1:7 differ virtually not at all. {#c-ablation-ratio}
At 7B parameters and 50B tokens, pure Mamba is quite competitive but lags slightly behind pure attention, an OLLM score of 35.3 against 36.1, and the hybrid without MoE outperforms both with 36.6. {#c-ablation-7b}
Pure Mamba did substantially worse than pure attention on IMDB, QuAC and NarrativeQA, 48.8 against 84.1 on IMDB, while the hybrid performed similarly to attention, 90.9; pure Mamba often did not follow the format of the answers, producing "Very Good" or "3/10" where the choices were "Positive" and "Negative". {#c-icl}
The authors hypothesize that this points to a difficulty of SSMs with in-context learning, which has been linked to induction heads, and note that the hybrid does in-context learning even when only 1 of 8 layers is attention. {#c-icl-hypothesis}
In a 1.3B hybrid they found 12 heads whose attention from the last token focuses on the labels of the few-shot examples, in all three attention layers, layers 4, 12 and 20. {#c-induction-heads}
At 7B parameters and 50B tokens, adding MoE with $n = 16$, $K = 2$ and $e = 2$ improves the hybrid, the OLLM score from 36.6 to 38.1 and NQ from 15.4 to 18.9. {#c-moe-helps}

## Limitations

Decreasing the share of attention layers reduces the memory for the KV cache and improves throughput, but might lower the model's capabilities; and a larger $n$ adds capacity at the cost of memory, a larger $K$ adds active parameters and compute, and a larger $e$ reduces capacity, compute and memory. {#c-tradeoffs}
The mixtures make the model large in memory even where it is cheap in compute: all 52B parameters must be stored, which is why the released model needs int8 weights to fit one 80GB GPU.
The paper's evaluations of the released model are its own and on a base model; the ablations are at 1.3B and 7B, and the paper does not test whether the 1:7 ratio stays best at larger scales.
The explanation of in-context learning by induction heads is a hypothesis, supported by examples of heads rather than by an intervention.
The Mamba layers need the inner normalizations to train stably at scale, so a Mamba block taken as it is from the original paper is not what Jamba uses.

## Variants and descendants

The five degrees of freedom of the block define a family: other ratios $a : m$, mixtures every $e$ layers, other $n$ and $K$, and the Attention MoE layer of Figure 1, which the released model does not use.
The ablation models are such variants: pure attention and pure Mamba, the latter with Mamba layers interleaved with MLP layers, and hybrids without MoE at ratios 1:3 and 1:7. {#c-pure-mamba}
[[linear-attention|Linear attention]] is the other way to a state of fixed size: it changes the attention itself into a recurrence, where Jamba keeps a few layers of exact softmax attention and makes the others recurrent.

## Notable models

- **Jamba-v0.1**, the released model: 4 blocks of 8 layers, 52B parameters of which 12B active, a context of up to 256K tokens, under Apache 2.0. {#c-weights}

## Implementations

The NumPy reference `impl/numpy/jamba.py` implements the model of eq-layer and eq-layout: grouped-query attention without positions (eq-attn), the Mamba layer with its three inner RMSNorms (eq-mamba), SwiGLU MLPs, and the mixture of eq-moe in which each expert runs only on the tokens that chose it, with the balance term of eq-aux.
It has the whole backward pass by hand, the router's included (the choice of experts is piecewise constant, so only the probabilities carry gradient); a recurrent `step` whose state is the keys and values of the attention layers and the windows and states of the Mamba layers; and the counts eq-params, eq-flops and eq-cache.
The PyTorch version `impl/pytorch/jamba.py` is an `nn.Module` with the same parameters, whose `kinds` argument builds the all-attention and all-Mamba models of the ablations.
The JAX version `impl/jax/jamba.py` runs the scan with `jax.lax.associative_scan` and dispatches the tokens to the experts through fixed-size buffers, as compiled code must; with a buffer per expert of $M$ slots nothing is dropped and it computes the reference, and with a capacity factor below that, as in GShard and the Switch Transformer, the choices that come after a full buffer are dropped and counted.
All three load the shared fixture: the example model of one block of 4 layers, Mamba, Mamba with MoE, attention, Mamba with MoE, on two sequences of 6 tokens, with its scores, loss, balance term, chosen experts and gradients.

### Worked example

The example configuration of the contract, one block of $l = 4$ layers of width 8 with 2 query heads of 4 sharing one key-value head, has 5,654 parameters, 4,502 of them active.
After 1,000 tokens of generation its attention layer holds $2 \cdot 4 \cdot 1000 = 8{,}000$ numbers and its three Mamba layers $3 \cdot 16 \cdot (2 + 4) = 288$, 8,288 in all; the same block with attention in all four layers would hold 32,000.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **The attention layer has no positions.** Adding RoPE or learned positions out of habit changes the model, and putting attention first in the block leaves it blind to order: an attention layer with no position and no layer below it is permutation-invariant over the past. `test_order_reaches_the_attention_through_the_mamba_layers` checks both halves.

> [!pitfall]
> **The cache is two kinds.** Generation must keep the growing keys and values of the attention layers and the fixed windows and states of the Mamba layers; dropping either gives outputs that differ from the parallel forward pass. `test_generation_with_a_cache_gives_the_same_scores` compares the two.

> [!pitfall]
> **Renormalizing the top-K weights.** Mixtral divides the chosen probabilities by their sum; this page does not. The two give different models from the same weights, so a port must match the convention of the checkpoint.

> [!pitfall]
> **Capacity drops.** An implementation with fixed buffers per expert silently drops tokens when the router is unbalanced, and the dropped tokens pass the layer with only part of their mixture. Count the drops, as the JAX `forward` does.

> [!pitfall]
> **Active is not resident.** 12B active parameters set the compute per token; the memory is set by all 52B. Table 1's "available" parameters are the ones that must be stored.

## Experiments

The PyTorch test `test_the_hybrid_recalls_where_pure_mamba_does_not` is a small version of the paper's ablation.
Each sequence lists 4 pairs of a key from 8 and a value from 8, then repeats one of the keys, and the model must output its value; chance is 1/8.
Three blocks of 4 layers of width 32, with 4 experts of which each token uses 2, are trained alike for 400 steps of Adam: the Jamba layout with attention at layer 2, the same with Mamba in all four layers, and the same with attention in all four, none with positional encoding.
On 1,000 held-out sequences, over the seeds 0, 1 and 2, the hybrid answered 1.000, 0.996 and 1.000, pure Mamba 0.798, 0.548 and 0.898, and pure attention 0.975, 0.949 and 0.947.
One attention layer is enough to make recall reliable, which pure Mamba of this size does not learn in 400 steps; the all-attention block, which has no Mamba layer to tell it the order, still learns the task through the causal mask but less well than the hybrid.

## History

Jamba was released by AI21 Labs, whose authors posted it to arXiv as 2403.19887 and made the weights public under the Apache 2.0 license at huggingface.co/ai21labs/Jamba-v0.1. {#c-history}
It came three months after Mamba and Mixtral, and combined the two: the recurrence of the first for most layers, and the sparse experts of the second in every other MLP.
Earlier hybrids of attention and state space layers, H3 and StripedHyena among them, had been smaller or had trailed attention-only models of their size.

## References

- Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., Abend, O., Alon, R., Asida, T., Bergman, A., Glozman, R., Gokhman, M., Manevich, A., Ratner, N., Rozen, N., Schwartz, E., Zusman, M. and Shoham, Y. (2024). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv 2403.19887.
- AI21 Labs (2024). config.json of Jamba-v0.1, revision 9efd115. Hugging Face.
