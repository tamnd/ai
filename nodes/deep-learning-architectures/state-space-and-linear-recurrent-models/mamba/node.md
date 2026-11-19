---
tag: 2WYA9X
kind: architecture
slug: mamba
name: Mamba
aliases: [Mamba block, selective state space model, selective SSM, S6, Mamba-1]
area: deep-learning-architectures/state-space-and-linear-recurrent-models
level: core
summary: >-
  A sequence model of stacked blocks around a selective state space model, whose step size and input and output maps depend on each token; it trains in linear time and generates with a fixed state.
uses: [matrix-exponential@m2, diagonal-matrix@m2, matrix-multiplication@m2]
related_to: [structured-state-space-model, transformer, multi-head-attention, recurrent-neural-network, long-short-term-memory, gated-recurrent-unit, rms-normalization, layer-normalization, residual-connection, neural-scaling-laws, gradient-checkpointing, adam, backpropagation-through-time@al]
history:
  year: 2023
  people: [Albert Gu, Tri Dao]
  note: "Gu and Dao posted Mamba to arXiv (2312.00752) in December 2023 with code and checkpoints of 130M to 2.8B parameters. It follows S4 (2021), its diagonal variants and the H3 and Hyena layers; Mamba-2 (2024) restricted the state matrix to connect selective SSMs with attention."
prereqs: [structured-state-space-model, recurrent-neural-network, residual-connection]
same_as:
  arxiv: ["2312.00752"]
sources:
  - {paper: "Gu and Dao (2023)", title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", locator: "arXiv 2312.00752v2, Sections 1 to 5, Tables 1 to 11 and Appendices D and E", note: "the selection mechanism, the selective scan, the block, the experiments and the ablations"}
  - {paper: "state-spaces/mamba contributors", title: "mamba_ssm/modules/mamba_simple.py, ops/selective_scan_interface.py and models/mixer_seq_simple.py", locator: "tag v2.3.2", note: "the reference code: the defaults, the initialization, the discretization and the language model"}
  - {paper: "state-spaces (2023)", title: "config.json of mamba-130m, mamba-370m, mamba-790m, mamba-1.4b and mamba-2.8b", locator: "Hugging Face, pinned revisions", note: "the sizes of the released models"}
family: state-space
inputs: "a (B, T) array of token ids, or a (B, T, D) array of features for a single block"
outputs: "a (B, T, V) array of next-token scores, causal: the score at position t depends on the tokens up to t only"
components: [structured-state-space-model, rms-normalization, residual-connection]
inductive_biases: ["causal and recurrent: everything the model knows about the past is a state of fixed size per layer, I (K - 1 + N) numbers", "selection: the step size Delta and the maps B and C depend on the current token, so the model can ignore a token, keep its state or reset it", "a local causal convolution of width K before the SSM, for short-range patterns", "no attention and no separate MLP: one homogeneous block, stacked"]
params: "V D + D + L (3 I D + I (K + 1) + I (R + 2 N) + (R + 1) I + I N + I + D) for L blocks of width D, inner width I = E D, state N, convolution K, Delta rank R and vocabulary V, with the output layer tied to the embedding"
flops_per_example: "2 T (L (2 D I + I (R + 2 N) + R I + I N + I D) + D V) for a sequence of T tokens, a multiply-add counted as 2, the matrix products; about 2 per parameter per token"
---

Mamba is a sequence model, most often a language model, built from one repeated block around a selective state space model.
A [[structured-state-space-model|structured state space model]] (SSM) carries a hidden state along the sequence by a linear recurrence whose matrices are the same at every position; Mamba lets the step size $\Delta$ and the maps $B$ and $C$ be computed from each token, so the recurrence can decide, token by token, what to write into its state and what to forget.
Gu and Dao let the parameters of a structured state space model be functions of the input, design a hardware-aware parallel algorithm in recurrent mode, and integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks, which they call Mamba. {#c-def}
They identify as a key weakness of subquadratic-time architectures such as linear attention, gated convolution, recurrent models and structured state space models their inability to perform content-based reasoning. {#c-weakness}
The authors report that Mamba has 5× higher throughput than Transformers at inference, scales linearly in sequence length, and that their Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation. {#c-abstract}

## In plain words

A [[transformer|Transformer]] keeps every earlier token at hand and looks back at all of them for each new one, which costs more the longer the text gets.
A [[recurrent-neural-network|recurrent network]] instead keeps a running summary of fixed size and updates it once per token, which costs the same at every step but forces it to decide what to keep.
Earlier state space models were recurrent networks of a special kind: linear, with the same update at every step, so that they could also be computed as one long convolution during training.
Because the update was the same for every token, they could not choose to skip a filler word or to remember a name more strongly than the word "the".
Mamba makes the update depend on the token: each token sets how much the summary should take it in and how much of the old summary to keep.
That loses the convolution, so Mamba computes the recurrence directly with a scan written for the GPU, and wraps it in a simple block that is stacked like the layers of a Transformer.

## Overview

Selective SSMs, and Mamba, are fully recurrent models: computation and memory scale linearly in sequence length during training, and unrolling the model autoregressively during inference requires only constant time per step since it does not require a cache of previous elements. {#c-properties}
The authors frame the problem as one of compression: attention does not compress context at all, so autoregressive inference stores the entire context in the KV cache, while recurrent models have a finite state, constant-time inference and linear-time training, but are limited by how well this state has compressed the context. {#c-compression}
They use two synthetic tasks to show what time-invariant models lack.
The Selective Copying task modifies the Copying task by varying the position of the tokens to memorize; the Induction Heads task requires context-aware reasoning to know when to produce the correct output in the appropriate context. {#c-tasks}
Global convolutions can solve the vanilla Copying task because it only requires time-awareness, but have difficulty with the Selective Copying task because of lack of content-awareness: the spacing between inputs and outputs is varying and cannot be modeled by static convolution kernels. {#c-lti-fail}
The answer is the selection mechanism, which makes the parameters that act on the sequence depend on the input, and an algorithm that computes the resulting time-varying recurrence efficiently.

## Components

### The state space model

S4 models are defined with four parameters $(\Delta, A, B, C)$ that define a sequence-to-sequence transformation in two stages: discretization, which transforms the continuous parameters $(\Delta, A, B)$ to discrete parameters by a rule such as the zero-order hold, and computation, as a linear recurrence or a global convolution. {#c-ssm}
For one channel with input $x_t$, state $h_t \in \mathbb{R}^N$ and output $y_t$,

$$ h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t), $$ {#eq-ssm}

and the zero-order hold of the paper with step $\Delta$ gives

$$ \bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}\big(\exp(\Delta A) - I\big)\, \Delta B, \qquad h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t . $$ {#eq-disc}

Commonly the model uses the convolutional mode for efficient parallelizable training, where the whole input sequence is seen ahead of time, and switches into recurrent mode for efficient autoregressive inference. {#c-modes}
In linear time invariance (LTI) the model's dynamics are constant through time: $(\Delta, A, B, C)$ are fixed for all time-steps; all structured SSMs before Mamba had been LTI because of fundamental efficiency constraints. {#c-lti}
Mamba uses a diagonal state matrix, so that $A$, $B$ and $C$ can each be represented by $N$ numbers; the SSM is applied independently to each of the $D$ channels, the total hidden state has dimension $DN$ per input, and computing it over the sequence length requires $O(BLDN)$ time and memory. {#c-diagonal}
The [[structured-state-space-model|structured state space model]] page covers the continuous model, its discretizations and the equivalence of the recurrence with a convolution.

### Selection

The selection mechanism makes the parameters $\Delta$, $B$ and $C$ functions of the input; they then have a length dimension $L$, the model changes from time-invariant to time-varying, and it loses the equivalence to convolutions. {#c-selection}
Mamba chooses $s_B(x) = \operatorname{Linear}_N(x)$, $s_C(x) = \operatorname{Linear}_N(x)$, $s_\Delta(x) = \operatorname{Broadcast}_D(\operatorname{Linear}_1(x))$ and $\tau_\Delta = \operatorname{softplus}$, where $\operatorname{Linear}_d$ is a parameterized projection to dimension $d$. {#c-choices}
The projection of $s_\Delta$ can be generalized from dimension 1 to a larger dimension $R$, a small fraction of $D$, giving the low-rank projection $s_\Delta(x) = \operatorname{Linear}_D(\operatorname{Linear}_R(x))$; the bias of $\Delta$ is initialized to $\tau_\Delta^{-1}(\operatorname{Uniform}([0.001, 0.1]))$. {#c-dt-rank}
With the inner width $I$ of the block in place of the paper's $D$, at each position $t$ of channel $i$,

$$ \Delta_{t,i} = \operatorname{softplus}\big((x_t W_{\Delta\downarrow} W_{\Delta\uparrow})_i + b_{\Delta,i}\big), \qquad B_t = x_t W_B, \qquad C_t = x_t W_C, $$ {#eq-selection}

with $W_{\Delta\downarrow} \in \mathbb{R}^{I \times R}$, $W_{\Delta\uparrow} \in \mathbb{R}^{R \times I}$ and $W_B, W_C \in \mathbb{R}^{I \times N}$; $B_t$ and $C_t$ are shared by the $I$ channels, $\Delta$ is per channel, and $A$ stays a learned constant.
In Algorithm 2 (S6), $B$ and $C$ of shape $(B, L, N)$ are computed from the input $x$ of shape $(B, L, D)$, $\Delta$ of shape $(B, L, D)$ is $\tau_\Delta$ of a parameter plus $s_\Delta(x)$, the discretized parameters have shape $(B, L, D, N)$, and the time-varying SSM is computed by a recurrence (scan) only. {#c-algorithm}
The paper abbreviates selective SSMs as S6 models, because they are S4 models with a selection mechanism and computed with a scan. {#c-s6}

What each parameter does:

- **$\Delta$.** $\Delta$ controls the balance between how much to focus or ignore the current input: a large $\Delta$ resets the state $h$ and focuses on the current input $x$, while a small $\Delta$ persists the state and ignores the current input. {#c-delta}
- **$A$.** $A$ could also be selective, but it affects the model only through its interaction with $\Delta$ in $\exp(\Delta A)$, so selectivity in $\Delta$ is enough to ensure selectivity in the discrete parameters; the authors leave a selective $A$ out for simplicity. {#c-a}
- **$B$ and $C$.** Making $B$ and $C$ selective allows finer-grained control over whether to let an input $x_t$ into the state $h_t$, or the state into the output $y_t$. {#c-bc}
- **Why $\Delta$ is broadcast.** If an input should be completely ignored, all $D$ channels should ignore it, which is why Mamba projects the input down to 1 dimension before broadcasting it to $\Delta$. {#c-broadcast}

Theorem 1 of Gu and Dao states that when $N = 1$, $A = -1$, $B = 1$, $s_\Delta = \operatorname{Linear}(x)$ and $\tau_\Delta = \operatorname{softplus}$, the selective SSM recurrence takes the form $g_t = \sigma(\operatorname{Linear}(x_t))$, $h_t = (1 - g_t) h_{t-1} + g_t x_t$. {#c-theorem1}
This is the gate of a [[gated-recurrent-unit|gated recurrent unit]] or of a [[long-short-term-memory|long short-term memory]], with one difference: the gate is the only nonlinearity, and the state enters linearly.
The derivation `selective-scan` of the structured state space model page proves the theorem, the loss of linearity and the associativity of the scan.

The paper names three effects of selection.
Selectivity allows filtering out irrelevant noise tokens that may occur between inputs of interest, such as language fillers like "um", because the model can filter out any particular input, for example in the gated case of Theorem 1 when $g_t \to 0$. {#c-spacing}
Many sequence models do not improve with longer context because they cannot effectively ignore irrelevant context; selective models can reset their state at any time to remove extraneous history, so their performance in principle improves monotonicly with context length. {#c-context}
Where independent sequences are stitched together, LTI models bleed information between the sequences, while selective SSMs can reset their state at boundaries, for example when $\Delta_t \to \infty$ or, in Theorem 1, when $g_t \to 1$. {#c-reset}
Gu and Dao find insufficient the explanation that architecture gating gives data-dependence, because such gating does not interact along the sequence axis and cannot affect the spacing between tokens. {#c-gating-not-selection}

### The selective scan

With a diagonal $A$, the recurrence for channel $i$ and state entry $n$ is a scalar one:

$$ h_{t,i,n} = \exp(\Delta_{t,i} A_{i,n})\, h_{t-1,i,n} + \bar{B}_{t,i,n}\, x_{t,i}, \qquad y_{t,i} = \sum_{n} C_{t,n} h_{t,i,n} + D_i x_{t,i} , $$ {#eq-scan}

where $D_i x_{t,i}$ is a skip connection with a learned weight per channel.
The recurrent mode would require computing and materializing the latent state $h$ with shape $(B, L, D, N)$, larger by a factor of $N$ than the input and output of shape $(B, L, D)$; the convolution mode of LTI models bypasses the state computation. {#c-materialize}
The naive recurrent computation uses $O(BLDN)$ FLOPs while the convolutional computation uses $O(BLD \log(L))$ FLOPs, and the former has a lower constant factor, so for long sequences and not-too-large state dimension $N$ the recurrent mode can use fewer FLOPs. {#c-recurrent-flops}
The selective scan therefore avoids the memory traffic rather than the FLOPs.
Instead of preparing the scan input of size $(B, L, D, N)$ in GPU HBM, it loads the SSM parameters $(\Delta, A, B, C)$ from HBM to fast SRAM, performs the discretization and recurrence in SRAM, and writes the final outputs of size $(B, L, D)$ back to HBM. {#c-fusion}
The recurrence is parallelized with a work-efficient parallel scan algorithm (Blelloch 1990). {#c-scan}
A scan applies to it because the pairs $(a, b)$ of a step $h \mapsto a h + b$ compose associatively, $(a_1, b_1) \circ (a_2, b_2) = (a_1 a_2, a_2 b_1 + b_2)$, so prefixes can be combined in a tree of depth $O(\log T)$.
The intermediate states are not stored but recomputed in the backward pass when the inputs are loaded from HBM to SRAM, so the fused selective scan layer has the same memory requirements as an optimized transformer implementation with FlashAttention. {#c-recompute}
This recomputation is [[gradient-checkpointing|gradient checkpointing]] applied inside one kernel.

The products of $\bar{A}$ along the sequence also give a closed form.
Because $\exp(\Delta_t A)$ multiply along the sequence into $\exp(A(S_t - S_s))$ with $S$ the running sum of $\Delta$, the selective SSM equals $y_t = \sum_{s \le t} C_t \exp(A(S_t - S_s)) \bar{B}_s x_s$, a lower-triangular $T \times T$ matrix per channel applied to the input, computed in $O(T^2 N)$ per channel instead of $O(TN)$ for the scan. {#c-quadratic}

$$ y = M x, \qquad M_{t,s} = \mathbb{1}[s \le t] \sum_{n} C_{t,n}\, e^{A_n (S_t - S_s)}\, \bar{B}_{s,n}, \qquad S_t = \sum_{r \le t} \Delta_r . $$ {#eq-quadratic}

It is the same masked matrix as causal attention with a different rule for the weights, the observation that Mamba-2 later built on; the PyTorch implementation uses it as a second scan, and the derivation `quadratic-form` proves it.

### The block

The Mamba block combines the H3 block, the basis of most SSM architectures, with the MLP block of Transformers into one block that is stacked homogenously, inspired by the gated attention unit (GAU); the block is repeated, interleaved with standard normalization and [[residual-connection|residual connections]]. {#c-block}
Compared to the H3 block, Mamba replaces the first multiplicative gate with an activation function; compared to the MLP block, Mamba adds an SSM to the main branch. {#c-h3-compare}
Mamba uses the SiLU / Swish activation function, so that the Gated MLP becomes the SwiGLU variant, and an optional normalization layer, LayerNorm, motivated by RetNet. {#c-silu}
For an input $u \in \mathbb{R}^{T \times D}$ of one sequence, with $\operatorname{conv}$ a causal depthwise convolution of width $K$ over the positions,

$$ \begin{aligned} (x, z) &= \operatorname{RMSNorm}(u)\, W_{\text{in}}, \qquad x, z \in \mathbb{R}^{T \times I}, \\ x' &= \operatorname{SiLU}(\operatorname{conv}(x)), \\ y &= \operatorname{SSM}_{\Delta(x'), B(x'), C(x')}(x'), \\ u' &= u + \big(y \odot \operatorname{SiLU}(z)\big)\, W_{\text{out}} . \end{aligned} $$ {#eq-block}

The reference code normalizes with [[rms-normalization|RMSNorm]] before each block, where the paper allows [[layer-normalization|LayerNorm]], and the convolution of width 4 gives each channel a short window of the past before the SSM.
The class Mamba of mamba_simple.py takes d_state=16, d_conv=4, expand=2, dt_rank=auto, which is math.ceil(self.d_model / 16), dt_min=0.001 and dt_max=0.1. {#c-code-defaults}
The language model of the reference code pads vocab_size to a multiple of pad_vocab_size_multiple, uses RMSNorm when rms_norm is set, and ties the weights of lm_head to the embedding. {#c-code-lm}

### Discretization in the code

In selective_scan_ref, deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A)) and deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u), and the state is updated as x = deltaA[:, :, i] * x + deltaB_u[:, :, i]. {#c-code-disc}
So the code keeps $\bar{A} = \exp(\Delta A)$ but takes $\bar{B} = \Delta B$, the Euler rule, not the zero-order hold of eq-disc.
The reference code discretizes $B$ as $\Delta B$ rather than by the zero-order hold of the paper's eq. 4; with a diagonal $A$ the two differ by the factor $(\exp(\Delta A) - 1)/(\Delta A) = 1 + \Delta A/2 + O(\Delta^2)$, so they agree for small steps, and for a large step the zero-order hold tends to $-B/A$ while $\Delta B$ grows without bound. {#c-disc-gap}
In the setting of Theorem 1 with the rule of the reference code, the recurrence is $h_t = (1 - g_t) h_{t-1} + \Delta_t x_t$ with $\Delta_t = -\log(1 - g_t)$, a weight on the input of at least $g_t$ that grows without bound as $g_t \to 1$. {#c-disc-gate}
The derivation `discretization` proves both statements.
The implementations of this page follow the code by default, `disc="simple"`, and offer `disc="zoh"` for the paper's rule; the tests check that the two agree to first order in $\Delta$.

## Shapes and parameters

The block expands the model dimension $D$ by a factor $E$; most of its parameters, $3ED^2$, are in the linear projections, $2ED^2$ for the input projections and $ED^2$ for the output projection; the authors fix $E = 2$ and use two stacks of the block to match the $12D^2$ parameters of a Transformer's interleaved MHA and MLP blocks. {#c-block-params}
Counting every array of the reference block, with $I = ED$:

$$ P = VD + D + L\big(3ID + I(K + 1) + I(R + 2N) + (R + 1)I + IN + I + D\big) . $$ {#eq-params}

A Mamba language model with $L$ blocks of width $D$, inner width $I = ED$, state size $N$, convolution width $K$, $\Delta$ rank $R$ and a vocabulary of $V$ tokens, with the output layer tied to the embedding, has this many parameters. {#c-params}
The terms are $W_{\text{in}}$ and $W_{\text{out}}$ ($3ID$), the convolution with its bias ($I(K+1)$), the projection of $x'$ to the low-rank $\Delta$, $B$ and $C$ ($I(R + 2N)$), the up-projection of $\Delta$ with its bias ($(R + 1)I$), $\log(-A)$ ($IN$), $D$ ($I$) and the norm ($D$).
In the shape contract this is

{shape:params}

and the example of the fixtures, 2 blocks of width 8 with a state of 4 and a vocabulary of 11, has {shape:params@example} parameters.

The released models share the vocabulary, the state, the convolution and the expansion and differ in width and depth.

- **Mamba-130M.** d_model is 768 and n_layer is 24, with vocab_size 50277 and pad_vocab_size_multiple 8. {#c-config-130m}
- **Mamba-370M.** d_model is 1024 and n_layer is 48. {#c-config-370m}
- **Mamba-790M.** d_model is 1536 and n_layer is 48. {#c-config-790m}
- **Mamba-1.4B.** d_model is 2048 and n_layer is 48. {#c-config-1-4b}
- **Mamba-2.8B.** d_model is 2560 and n_layer is 64. {#c-config-2-8b}

| Model | $D$ | $L$ | $R$ | Parameters | State per sequence |
|---|---|---|---|---|---|
| Mamba-130M | 768 | 24 | 48 | {shape:params@mamba_130m} | {shape:recurrent_state@mamba_130m} |
| Mamba-370M | 1024 | 48 | 64 | {shape:params@mamba_370m} | {shape:recurrent_state@mamba_370m} |
| Mamba-790M | 1536 | 48 | 96 | {shape:params@mamba_790m} | {shape:recurrent_state@mamba_790m} |
| Mamba-1.4B | 2048 | 48 | 128 | {shape:params@mamba_1_4b} | {shape:recurrent_state@mamba_1_4b} |
| Mamba-2.8B | 2560 | 64 | 160 | {shape:params@mamba_2_8b} | {shape:recurrent_state@mamba_2_8b} |

With the vocabulary of 50,277 padded to 50,280 and the default sizes of the reference code, the released models have 129,135,360 (Mamba-130M), 371,516,416 (Mamba-370M), 793,204,224 (Mamba-790M), 1,372,178,432 (Mamba-1.4B) and 2,768,345,600 (Mamba-2.8B) parameters. {#c-params-models}
The count of c-params reproduces the sizes of the paper's ablations on the 48 blocks of width 1024: 371.5M with $N = 16$ and $R = 64$, 367.1M with $N = 1$, 359.1M with $R = 1$ and 358.9M with no projection for $\Delta$. {#c-params-ablations}
The projections are {shape:projection_params@mamba_2_8b} of the 2.8B parameters; the SSM itself, $A$, $D$ and the projections to $\Delta$, $B$ and $C$, is a small part.
The derivation `parameter-and-flop-count` proves these counts.

## Computation and cost

A forward pass over $B$ sequences of $T$ tokens costs

$$ C = 2BT\Big(L\big(2DI + I(R + 2N) + RI + IN + ID\big) + DV\Big) $$ {#eq-flops}

FLOPs in its matrix products, linear in $T$, of which the readout of the scan by $C$ is $2BTLIN$; the convolution and the update of the state add $2BTLI(K + N)$ elementwise. {#c-flops}
For Mamba-2.8B a forward pass over 2048 tokens costs 11,329,066,762,240 FLOPs in its products by the count of c-flops, 5,531,770,880 per token, within 0.09% of twice the parameters; the convolution and the update of the state add 26,843,545,600, about 0.24%. {#c-flops-models}
The contract gives {shape:flops_forward@mamba_2_8b} for that pass and {shape:elementwise_flops@mamba_2_8b} for the elementwise part.
So the FLOPs of Mamba are those of its matrix products, like a Transformer at short context; what the selective scan saves is memory traffic, and what the recurrence saves is the cost that attention adds with every token of context.

In generation each Mamba block carries $I(K - 1 + N)$ numbers per sequence from one token to the next, the last $K - 1$ inputs of the convolution and the $I \times N$ state of the SSM, whatever the length; Mamba-2.8B carries 6,225,920. {#c-state}

$$ S_{\text{Mamba}} = L\, I\,(K - 1 + N), \qquad S_{\text{Transformer}}(T) = 2 \cdot \tfrac{L}{2} \cdot D \cdot T = LDT . $$ {#eq-state}

A Transformer with $L/2$ layers of width $D$, the parameter match of the paper, caches $LDT$ keys and values for $T$ tokens, which exceeds the $2LD(K - 1 + N)$ numbers of $L$ Mamba blocks with $E = 2$ once $T > 2(K - 1 + N)$, 38 tokens with the default $K = 4$ and $N = 16$. {#c-state-compare}
Past that length the Transformer's cache grows with every token and Mamba's state does not, which is where the throughput of generation comes from; the derivation `recurrent-state` counts both.
The cost of the fixed state is the compression of c-compression: whatever the model will need later must fit in it.

## Training

Mamba is trained like any causal language model, by next-token cross-entropy on batches of fixed length, with [[adam|Adam]]-type optimizers and gradients through the scan; the gradient of a recurrence is [[al:backpropagation-through-time|backpropagation through time]], which the fused kernel computes by a reverse scan with the states recomputed.
The reference code initializes A_log as the logarithm of 1 to d_state for every channel, D as ones, and the bias of dt_proj as the inverse of softplus of a step drawn between dt_min and dt_max, inv_dt = dt + torch.log(-torch.expm1(-dt)); A = -torch.exp(self.A_log). {#c-code-init}
The default initialization is S4D-Lin for the complex case and S4D-Real for the real case, which define the $n$-th element of $A$ as $-1/2 + ni$ and $-(n + 1)$ respectively. {#c-init}
Mamba uses real values by default, which work well for all but one of the paper's tasks; the authors hypothesize that complex numbers are helpful for continuous modalities such as audio and video but not discrete ones such as text and DNA. {#c-real}
Storing $\log(-A)$ keeps $A$ negative, so $\exp(\Delta A) \in (0, 1)$ and the recurrence is stable for every input; the code also excludes it, and $D$, from weight decay.
Transformer++ is the strongest Transformer recipe the authors know of, based on the PaLM and LLaMa architectures: rotary embedding, SwiGLU MLP, RMSNorm instead of LayerNorm, no linear bias, and higher learning rates. {#c-transformer-pp}
Pythia and RWKV were trained with the same tokenizer, dataset, and training length (300B tokens) as the Mamba models; Mamba and Pythia are trained with context length 2048. {#c-pythia-setting}

## Inductive biases

**A fixed state.**
Everything the model keeps about the past is $I(K - 1 + N)$ numbers per block; `test_state_size_does_not_grow_with_the_sequence` checks it.
This is the bias of a [[recurrent-neural-network|recurrent network]] and the opposite of attention, which keeps all of the past and compresses nothing.

**Content decides what is kept.**
Through $\Delta$, $B$ and $C$ each token chooses how much it writes and how much of the old state survives; `test_parameters_depend_on_the_input` checks that they vary along the sequence and `test_large_delta_resets_and_small_delta_ignores` the two extremes.
An LTI model can only weight positions by their distance, the same for every input.

**Decay, not lookup.**
The contribution of token $s$ to the state at $t$ is weighted by $\exp(A(S_t - S_s))$, which can only shrink; old information fades unless the steps between keep $\Delta$ small.
Retrieving one item from far back, which attention does by a direct comparison, needs the model to have decided to keep it when it went by.

**Locality before the SSM.**
The convolution of width $K$ mixes a few neighbouring tokens before selection, so $\Delta$, $B$ and $C$ see short n-grams, not single tokens.

## Evidence

On Selective Copying, S4 with no gate reached 18.3% accuracy and S6 with no gate 97.0%; in the H3 architecture S4 reached 57.0%, Hyena 30.1% and S6 99.7%; in the Mamba architecture S4 reached 56.4%, Hyena 28.4% and S6 99.8%. {#c-table1}
On Induction Heads, 2-layer models trained at sequence length 256 with a vocab size of 16 were tested on lengths from 64 to 1048576; Mamba solved the task perfectly and generalized to million-length sequences, 4000× longer than it saw during training, while no other method went beyond 2×. {#c-induction}
The induction heads models use a model dimension $D$ of 64 for Mamba and 128 for the other models, and [[multi-head-attention|multi-head attention]] with 8 heads and various positional encodings. {#c-induction-dims}
In the full table of the induction heads results, Mamba has 74K parameters and the attention models MHA-Abs, MHA-RoPE and MHA-xPos 137K each. {#c-induction-params}

Under the Chinchilla protocol of [[neural-scaling-laws|scaling laws]], on models trained on the Pile, Mamba is the first attention-free model to match the performance of a very strong Transformer recipe (Transformer++), particularly as the sequence length grows. {#c-scaling}
In the zero-shot evaluations, Mamba-2.8B averaged 63.3 against 59.1 for Pythia-2.8B and 61.7 for Pythia-6.9B, and Mamba-1.4B averaged 59.7 against 55.2 for Pythia-1.4B. {#c-zero-shot}
On the HG38 human genome, by the scaling curve of the paper, Mamba can match the Transformer++ and HyenaDNA models with roughly 3× to 4× fewer parameters. {#c-dna-size}
Pretrained on DNA sequences of lengths from 1024 to 1048576, Mamba's pretraining perplexity improves as the context increases up to sequences of length 1M, while the HyenaDNA model gets worse with sequence length. {#c-dna-length}
The authors' efficient scan is 40× faster than a standard implementation in training, and as a recurrent model Mamba can achieve 5× higher throughput than Transformers at inference. {#c-efficiency}

The ablations separate the parts.
In the ablations of architecture and SSM layer at about 350M parameters, among previous non-selective (LTI) SSMs, which are equivalent to global convolutions, performance is very similar, and replacing any of them with a selective SSM (S6) significantly improves performance. {#c-ablation-arch}
In Table 6, the Mamba architecture with S6 reached a perplexity of 8.69 and the H3 architecture with S6 8.95, while H3 with S4 (complex) reached 10.30 and with S4 (real) 10.34. {#c-ablation-table6}
With no selective parameter the perplexity was 10.93; with selective $B$ it was 10.15, with selective $C$ 9.98, with selective $\Delta$ 9.81, and with all three selective 8.71. {#c-ablation-table7}
With the complex S4D-Lin initialization the perplexity was 9.16; with real $A_n = -1/2$ it was 8.85, and with real $A_n = -(n + 1)$ and with a random real initialization 8.71. {#c-ablation-table8}
In Table 9 the parameters (M) are 358.9 with no projection for $\Delta$, with a perplexity of 9.12, 359.1 with a projection of size 1, and 371.5 with size 64, with a perplexity of 8.71. {#c-ablation-table9}
With the state dimension $N$ increased from 1 to 16, the perplexity went from 9.88 to 9.81 with constant $B$ and $C$ and from 9.73 to 8.71 with selective $B$ and $C$, and the parameters (M) from 367.1 to 371.5. {#c-ablation-table10}
A larger state helps only when the model can choose what to put in it: the 1.2% more parameters of $N = 16$ buy almost nothing for an LTI model and most of the gain for a selective one.

## Limitations

The authors name as limitations a continuous-discrete spectrum, on which the selection mechanism can impede performance on data that LTI SSMs excel on, the open question of downstream affordances such as fine-tuning and quantization, and an empirical evaluation limited to small model sizes below 7B parameters. {#c-limitations}

- **Recall from a fixed state.** A state of fixed size cannot hold an arbitrary amount of the past; a task that asks for one of many earlier items, chosen only after they have all gone by, needs as much state as there are items, where attention can look them up.
- **Continuous signals.** On audio and other continuous data the paper found that LTI SSMs can do better, since selection discards the smooth time invariance that suits them.
- **Hardware.** The speed comes from a fused GPU kernel; a scan written in framework operations materializes the state of c-materialize, and the quadratic form trades the recurrence for $O(T^2)$ work.
- **Scale.** The paper's models stop at 2.8B parameters trained on 300B tokens.

## Variants and descendants

- **Mamba-2** (Dao and Gu, 2024). Restricts $A$ to a scalar times the identity per head, so that the masked matrix of eq-quadratic becomes a structured form of attention, and computes it with matrix multiplications; it calls the correspondence state space duality.
- **Hybrids.** Models that interleave Mamba blocks with attention layers, such as Jamba (2024), keep a few layers that can look up the past exactly and use Mamba for the rest.
- **Vision and other modalities.** Mamba blocks scan the patches of an image in one or more orders, in place of the attention of a Vision Transformer.
- **Other selective recurrences.** Gated linear recurrences with input-dependent decay, in the line of the [[long-short-term-memory|LSTM]] forget gate, share the mechanism of Theorem 1 with different parameterizations.

## Notable models

- **Mamba-130M to Mamba-2.8B**, the models of the paper, trained on 300B tokens of the Pile, with the sizes of the table above.
- **Mamba-2.8B**, compared in the paper with Pythia models of 2.8B and 6.9B parameters.

## Implementations

The NumPy reference `impl/numpy/mamba.py` implements the language model of the reference code: the embedding, the blocks of eq-block with RMSNorm, the causal convolution, eq-selection, both discretizations and the scan of eq-scan, the tied head, the loss eq-loss and the whole backward pass by hand, including the reverse scan; it also has a recurrent `step` that carries the state of c-state, and the counts eq-params and eq-flops.
The PyTorch version `impl/pytorch/mamba.py` is an `nn.Module` with the same parameters and two scans, a loop over the positions and the quadratic form of eq-quadratic; `selective=False` gives the time-invariant ablation of Table 7.
The JAX version `impl/jax/mamba.py` is a pure function of a parameter dictionary with a scan by `jax.lax.associative_scan` over the operator of the selective scan, and one by `jax.lax.scan`, tested under `jax.jit` and `jax.grad`.
All three load the shared fixture of `fixtures/`: the example model of 2 blocks on two sequences of 8 tokens, with its logits, loss, last state and gradients.
None of them reproduces the fused GPU kernel; they compute the same function.

The training loss is the mean next-token cross-entropy over a batch of $B$ sequences,

$$ \mathcal{L} = -\frac{1}{B T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \operatorname{softmax}\big(f(w_{b,1:t})\big)_{w_{b,t+1}} . $$ {#eq-loss}

### Worked example

Take one channel with $N = 1$, $A = -1$, $B = C = 1$ and no skip, two inputs $x = (5, -2)$, and steps $\Delta_1 = \log 2$, $\Delta_2 = \log 4$, which softplus gives for $z = 0$ and $z = \log 3$.
With the zero-order hold, $\bar{A}_1 = 1/2$, $\bar{B}_1 = 1/2$, $\bar{A}_2 = 1/4$, $\bar{B}_2 = 3/4$: these are the gates $g = \sigma(z) = 1/2$ and $3/4$ of Theorem 1.
The scan gives $h_1 = 5/2$ and $h_2 = (1/4)(5/2) + (3/4)(-2) = -7/8$.
The quadratic form has $S = (\log 2, \log 8)$ and $M = \begin{pmatrix} 1/2 & 0 \\ 1/8 & 3/4 \end{pmatrix}$, since $M_{2,1} = e^{-(\log 8 - \log 2)} \cdot 1/2 = 1/8$, and $Mx = (5/2, -7/8)$, the same.
With the rule of the code, $\bar{B}_t = \Delta_t$: $h_1 = 5 \log 2 = 3.4657$ and $h_2 = 3.4657/4 - 2 \log 4 = -1.9062$; the second token, with a larger gate, weighs $\log 4 = 1.386$ instead of $3/4$.
The test `test_worked_example` checks these numbers.

### Pitfalls

> [!pitfall]
> **Euler for $B$, exponential for $A$.** The reference code takes $\bar{B} = \Delta B$, not the zero-order hold of the paper. For small $\Delta$ the two agree; for large $\Delta$ the code's input weight grows like $\Delta$ instead of tending to $-B/A$, so "reset to the current input" holds for $\bar{A} \to 0$ but the new state is scaled by $\Delta$. Porting weights between the two rules changes the model; the fixture uses the code's rule.

> [!pitfall]
> **Materializing the state.** A scan written as tensor operations that stores $h$ for every position uses $B \cdot T \cdot I \cdot N$ numbers, 16 times the activations with $N = 16$; for training at length 2048 that is the memory problem the fused kernel was written to avoid. Recompute it, or chunk the sequence.

> [!pitfall]
> **Overflow in the quadratic form.** $\exp(A(S_t - S_s))$ is at most 1 for $s \le t$, but above the diagonal the exponent is positive and can overflow before the mask zeroes it, giving NaN in the backward pass. Mask the exponent, not only the result, as `quadratic_scan` does.

> [!pitfall]
> **Padding the convolution on the wrong side.** The convolution must see only the past: pad $K - 1$ zeros on the left and cut the output to length $T$. Symmetric padding leaks the next tokens into the prediction; `test_causal` catches it.

> [!pitfall]
> **The state in generation is two parts.** A step needs the last $K - 1$ inputs of the convolution as well as $h$. Carrying $h$ alone gives outputs that differ from the parallel forward pass at every position; `test_step_mode_equals_the_parallel_forward` compares the two.

> [!pitfall]
> **Weight decay on $\log(-A)$ and $D$.** Decaying them pulls $A$ toward $-1$ for every state entry and the skip toward 0; the code exempts both.

## Experiments

This page has no experiment yet.
The PyTorch test `test_selection_solves_selective_copying` is a small one, the paper's Table 1 in miniature.
Each sequence has 3 data tokens from $\{1, \dots, 4\}$ at random positions among 16 noise tokens, followed by 3 markers, and at each marker the model must output the next data token in order; chance is 0.25.
Two models of 2 blocks of width 8 with a state of 4, one selective and one with $\Delta$, $B$ and $C$ fixed, the S4D ablation, are trained alike for 400 steps of Adam.
The selective model gets 0.85 of the tokens right and the time-invariant one 0.65; over the seeds 0 to 3 the first ranged from 0.75 to 0.85 and the second from 0.58 to 0.70.
The time-invariant model is not at chance, since with a convolution of width 4 and a decaying kernel it can still learn something about where tokens tend to be; but it cannot follow tokens whose positions move.

## History

Albert Gu of Carnegie Mellon University and Tri Dao of Princeton University introduced Mamba in arXiv 2312.00752, and open-sourced the model code and pre-trained checkpoints at github.com/state-spaces/mamba. {#c-history}
It came after two years of state space layers: S4 in 2021, its diagonal versions DSS and S4D in 2022, the H3 layer that put an SSM between gates, and Hyena with long implicit convolutions, all time-invariant.
Dao had written FlashAttention, whose way of keeping work in on-chip memory the selective scan follows.
Mamba-2 followed in 2024, and Mamba layers entered hybrid models with attention the same year.

## References

- Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2312.00752.
- state-spaces/mamba contributors. mamba_ssm, tag v2.3.2: modules/mamba_simple.py, ops/selective_scan_interface.py and models/mixer_seq_simple.py. GitHub.
- state-spaces (2023). config.json of mamba-130m, mamba-370m, mamba-790m, mamba-1.4b and mamba-2.8b. Hugging Face.
