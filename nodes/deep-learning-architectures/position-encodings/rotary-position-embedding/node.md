---
tag: 0V8E5T
kind: method
slug: rotary-position-embedding
name: Rotary position embedding
aliases: [RoPE, rotary embedding, rotary position embeddings, rotary positional embedding]
area: deep-learning-architectures/position-encodings
level: core
summary: >-
  Encode token positions by rotating pairs of query and key dimensions by angles proportional to the position, so each attention score depends on content and relative position only.
purpose: position encoding
uses: [block-diagonal-matrix@m2, linear-isometry@m2, dot-product@m2]
related_to: [transformer]
hyperparameters:
  - {name: b, meaning: "the base of the angles theta_j = b^(-2j/d)", typical: "10000 in the original paper; raised to extend the context, as with 1000000 in Code Llama"}
  - {name: d, meaning: "the number of rotated dimensions of each query and key head; must be even", typical: "the full head width"}
  - {name: s, meaning: "the scale factor L'/L of a context extension; 1 means no extension", typical: "1; 2 to 32 for position interpolation or YaRN"}
  - {name: alpha and beta, meaning: "the bounds of the ramp of NTK-by-parts and YaRN, in units of L over the wavelength", typical: "alpha = 1 and beta = 32 for the Llama family"}
  - {name: t, meaning: "the softmax temperature of YaRN", typical: "sqrt(1/t) = 0.1 ln s + 1 for LLaMA and Llama 2"}
cost:
  train: "two elementwise products and one sum per query and per key vector, with no parameters; the cosine and sine tables can be computed once"
  inference: "the same per new token; a cache of rotated keys must be rebuilt when the angles change, as in dynamic scaling"
history:
  year: 2021
  people: [Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu]
  note: "Su et al., arXiv 2104.09864 (2021), from Zhuiyi Technology; version 5 is dated 8 Nov 2023."
prereqs: [transformer, matrix-multiplication@m2, dot-product@m2, block-diagonal-matrix@m2]
same_as:
  arxiv: ["2104.09864"]
sources:
  - {paper: "Su, Lu, Pan, Murtadha, Wen and Liu (2021)", title: "RoFormer: Enhanced Transformer with Rotary Position Embedding", locator: "arXiv 2104.09864v5, Sections 3 and 4", note: "the method, its derivation and the first experiments"}
  - {paper: "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (2017)", title: "Attention Is All You Need", locator: "arXiv 1706.03762v7, Section 3.5", note: "the sinusoidal absolute encoding"}
  - {paper: "Chen, Wong, Chen and Tian (2023)", title: "Extending Context Window of Large Language Models via Position Interpolation", locator: "arXiv 2306.15595v2, Sections 2 and 3", note: "position interpolation"}
  - {paper: "Peng, Quesnelle, Fan and Shippole (2023)", title: "YaRN: Efficient Context Window Extension of Large Language Models", locator: "arXiv 2309.00071v2, Sections 2 to 4", note: "NTK-aware, NTK-by-parts, dynamic scaling and YaRN"}
  - {paper: "Press, Smith and Lewis (2022)", title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", locator: "ICLR 2022, arXiv 2108.12409v2, Sections 2 and 3", note: "a controlled comparison of extrapolation with sinusoidal, rotary and T5 positions"}
---

Rotary position embedding (RoPE) encodes the position of a token by rotating its query and key vectors in attention.
Each pair of dimensions turns by an angle proportional to the position, at its own frequency.
The dot product of a rotated query and a rotated key then depends on the two contents and on the distance between the tokens only.
Su et al. proposed it in 2021.
Position interpolation and YaRN later extended RoPE models to longer contexts.

## In plain words

Attention compares every query with every key through a dot product.
To know where tokens are, the model needs position in that comparison.
RoPE turns each query and key like the hands of many clocks, one clock for each pair of numbers.
A token at position 5 has every hand turned five steps; each clock has its own step size.
Turning two arrows by the same amount does not change the angle between them.
So only the difference of the two positions survives in the dot product.
Turning also keeps the length of each arrow, so the size of the vectors is untouched.
Fast clocks tell nearby tokens apart, and slow clocks track long distances.
Nothing is learned: the rotation is fixed, and it adds no parameters.

## Setting

The method applies to self-attention in a [[transformer]].
A sequence of tokens with positions $m = 0, 1, \dots$ gives for each token a query $\mathbf{q}_m = \mathbf{W}_q \mathbf{x}_m$ and a key $\mathbf{k}_n = \mathbf{W}_k \mathbf{x}_n$ in $\mathbb{R}^d$ per head, with $d$ even.
Attention compares them through the score $\mathbf{q}_m^\top \mathbf{k}_n$ before the softmax.
Without position information the score is the same wherever the two tokens are.

The original Transformer adds a sinusoidal vector to each input embedding ([[transformer#eq-pe|the positional encoding]]), with wavelengths in a geometric progression from $2\pi$ to $10000 \cdot 2\pi$ (Vaswani et al. 2017, Section 3.5). {#c-sinusoid}
Its authors chose it on the hypothesis that it lets the model attend by relative position and extrapolate to longer sequences. {#c-sinusoid-hypothesis}
That encoding is absolute and additive: the position enters the content before the projections $\mathbf{W}_q$ and $\mathbf{W}_k$.

## The method

RoPE leaves the embeddings alone and acts on the query and the key after the projections.
Split the $d$ dimensions into $d/2$ adjacent pairs $(2j, 2j+1)$, $j = 0, \dots, d/2 - 1$, and give pair $j$ the angle

$$ \theta_j = b^{-2j/d}, \qquad b = 10000 . $$ {#eq-theta}

Su et al. define it as $\theta_i = 10000^{-2(i-1)/d}$ for $i = 1, \dots, d/2$, the same set indexed from 1. {#c-theta-def}
Position $m$ rotates pair $j$ by $m\theta_j$, so the full map is the block-diagonal orthogonal matrix

$$ \mathbf{R}_m = \operatorname{diag}\big(\mathbf{B}(m\theta_0), \dots, \mathbf{B}(m\theta_{d/2-1})\big), \qquad \mathbf{B}(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix} . $$ {#eq-rope-matrix}

The rotated query is $\mathbf{R}_m \mathbf{q}_m$ and the rotated key is $\mathbf{R}_n \mathbf{k}_n$; values are not rotated.
In complex form, pair $j$ is the number $x_{2j} + i x_{2j+1}$, and the rotation multiplies it by $e^{im\theta_j}$:

$$ \big(\mathbf{R}_m \mathbf{x}\big)_{[j]} = (x_{2j} + i x_{2j+1})\, e^{i m \theta_j} . $$ {#eq-rope-complex}

The score of a query at $m$ and a key at $n$ becomes

$$ (\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m}\, \mathbf{k} . $$ {#eq-score}

The matrix in the middle depends on $n - m$ alone. {#c-relative-score}
RoPE thus encodes absolute position with a rotation matrix and brings an explicit relative position dependency into self-attention (Su et al. 2021, the abstract). {#c-rope-abstract}
Su et al. stress that the approach is multiplicative, unlike the additive encodings before it (Section 3.2.2). {#c-multiplicative}

$\mathbf{R}_m$ is sparse, so a dense product with it is wasteful (Section 3.2.2). {#c-matrix-inefficient}
The computation used in practice is elementwise, as given by Su et al. {eq:su-2024-roformer#34}

$$ \mathbf{R}_m \mathbf{x} = \mathbf{x} \odot \mathbf{c}_m + \tilde{\mathbf{x}} \odot \mathbf{s}_m, \qquad \tilde{\mathbf{x}} = (-x_1, x_0, -x_3, x_2, \dots, -x_{d-1}, x_{d-2}), $$ {#eq-rope-efficient}

where $\mathbf{c}_m$ holds $\cos m\theta_j$ in places $2j$ and $2j+1$, and $\mathbf{s}_m$ holds $\sin m\theta_j$ in the same places. {#c-eq34}

## Derivation

Su et al. obtain RoPE by asking for a map whose inner products depend on the two contents and on $m - n$ only (Section 3.1). {#c-goal}
In two dimensions they solve that equation and find that the radial part does not depend on position and the angle is linear in it, $\phi(m) = m\theta$ (Section 3.4.1). {#c-2d-solution}
The higher-dimensional form stacks $d/2$ such planes.

The facts below follow from that form.
Each block $\mathbf{B}(\phi)$ is orthogonal because $\cos^2\phi + \sin^2\phi = 1$, so $\mathbf{R}_m^\top \mathbf{R}_m = \mathbf{I}$. {#c-orthogonal}
So the rotation keeps the norm of every query and key, $\lVert \mathbf{R}_m \mathbf{x} \rVert = \lVert \mathbf{x} \rVert$. {#c-norm-preserved}
At $m = 0$ every angle is zero and $\mathbf{R}_0 = \mathbf{I}$, so the first token is left unchanged. {#c-position-zero}
The angle-sum formulas give $\mathbf{R}_m \mathbf{R}_n = \mathbf{R}_{m+n}$ and $\mathbf{R}_m^\top = \mathbf{R}_{-m}$. {#c-composition}
Then $(\mathbf{R}_m\mathbf{q})^\top(\mathbf{R}_n\mathbf{k}) = \mathbf{q}^\top \mathbf{R}_m^\top \mathbf{R}_n \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$, which is eq-score.
Su et al. write the same step as $\mathbf{R}^d_{\Theta, n-m} = (\mathbf{R}^d_{\Theta, m})^\top \mathbf{R}^d_{\Theta, n}$ in their Equation 16.
Multiplying complex numbers is rotating them, so the complex form and the real form give the same vector. {#c-complex-real}
Eq-rope-efficient writes out the two rows of each block, so it gives the same vector as the matrix. {#c-efficient-form}
It needs $2d$ multiplications and $d$ additions once the tables are known, while $\mathbf{R}_m$ has $2d$ non-zero entries of $d^2$. {#c-efficient-count}

Many code bases pair dimension $j$ with dimension $j + d/2$ instead of $2j + 1$.
That is the same method after a fixed permutation of the dimensions, so all the facts above hold for it. {#c-half-pairing}
The derivation `relative-position` gives every step, and its check script tests each fact on random vectors.

**Long-term decay.**
Su et al. write the score as $\mathrm{Re}\sum_j h_j e^{i(m-n)\theta_j}$, with $h_j$ the product of query pair $j$ and the conjugate of key pair $j$, and apply summation by parts (Section 3.4.3):

$$ \Big| \sum_{j=0}^{d/2-1} h_j e^{i r \theta_j} \Big| \le \Big( \max_{j} |h_{j+1} - h_j| \Big) \sum_{j=1}^{d/2} |S_j|, \qquad S_j = \sum_{i=0}^{j-1} e^{i r \theta_i}, \quad r = m - n . $$ {#eq-decay}

{#c-decay-bound}
The second factor depends on $r$ only, and Su et al. plot its average, which decays as the relative distance grows (Section 3.4.3 and Figure 2). {#c-decay-su}
They call this property long-term decay: the dependence between tokens fades with distance. {#c-decay-property}
My own computation for $d = 128$ and $b = 10000$ gives an average of $|S_j|$ of 32.5 at $r = 0$, 10.227 at $r = 100$, 4.471 at $r = 1000$ and 5.047 at $r = 2000$. {#c-decay-not-monotone}
So the bound falls on the whole but not at every step, and it bounds the size of the score rather than the score.
The derivation `long-term-decay` gives the steps and the check.

## Worked example

Take $d = 4$, so $\theta_0 = 10000^{0} = 1$ and $\theta_1 = 10000^{-1/2} = 0.01$.
Let $\mathbf{q} = (1, 2, 0, 1)$ and $\mathbf{k} = (2, 1, 1, 0)$.
At position $m = 1$ the first pair turns by 1 radian and the second by 0.01 radian.
The first pair becomes $(\cos 1 - 2\sin 1,\ \sin 1 + 2\cos 1) = (-1.1426, 1.9221)$, to four decimals.
The second pair becomes $(-\sin 0.01, \cos 0.01) = (-0.0100, 1.0000)$.
The squared norm of $\mathbf{R}_1\mathbf{q}$ is 6, the same as for $\mathbf{q}$.

The scores, computed with the NumPy reference, depend only on $m - n$:

| query position $m$ | key position $n$ | $m - n$ | score |
|---|---|---|---|
| 0 | 0 | 0 | 4.0000 |
| 5 | 5 | 0 | 4.0000 |
| 1 | 0 | 1 | -0.3732 |
| 2 | 1 | 1 | -0.3732 |
| 7 | 6 | 1 | -0.3732 |
| 0 | 1 | -1 | 4.6956 |
| 3 | 0 | 3 | -4.4133 |
| 10 | 7 | 3 | -4.4133 |

At $m = n$ the score is the plain dot product $\mathbf{q}^\top\mathbf{k} = 4$.
The scores at $m - n = 1$ and $m - n = -1$ differ, so RoPE knows the order of the two tokens and not only their distance.

## Hyperparameters

The base $b$ sets the slowest wavelength, $2\pi b^{(d-2)/d}$, and so the range of distances that the slow pairs can tell apart.
For $d = 128$ and $b = 10000$ my own computation gives wavelengths from $2\pi \approx 6.28$ to about 54,410 positions.
Code Llama raised the base to one million, which YaRN describes as NTK-aware scaling (Peng et al. 2023, Section 3.1). {#c-codellama}
The head width $d$ fixes the number of pairs.
The scale $s$ and the YaRN parameters $\alpha$, $\beta$ and $t$ matter only for context extension, in Variants.

## Cost

RoPE has no parameters.
Per query or key vector it costs $2d$ multiplications and $d$ additions, plus the tables $\mathbf{c}_m$ and $\mathbf{s}_m$, which depend only on the positions.
A dense matrix product with $\mathbf{R}_m$ would cost $d^2$ multiplications.
It does not change the cost of attention itself.
With linear attention, Su et al. multiply the rotation into the outputs of the feature maps and leave the normalizer unrotated, which keeps linear complexity (Section 3.3). {#c-linear-attention}

## Evidence

On WMT 2014 English to German, RoFormer reached a BLEU of 27.5 against 27.3 for Transformer-base (Su et al. 2021, Table 1). {#c-wmt}
In masked language model pre-training, the loss of RoFormer fell faster than that of BERT (Section 4.2.3). {#c-mlm-faster}
On GLUE, RoFormer beat BERT on three of six tasks (Section 4.3 and Table 2). {#c-glue}
On Enwik8, a Performer with RoPE converged faster and reached a lower loss than without it (Section 4.4.2). {#c-performer}
On the Chinese legal dataset CAIL2019-SCM, RoFormer at length 1024 scored 69.79% on the test set against 68.10% for WoBERT at 512 (Table 5). {#c-cail}
Chen et al. (2023, Section 2.1) report RoPE as the position encoding of the LLaMA models. {#c-llama-uses}

## Limitations and failure modes

Su et al. list two open questions: why RoPE converges faster, and why it does better on long texts (Section 4.5.5). {#c-limitations}
RoPE models do not extrapolate well past their trained length.
Press et al. found the same in a controlled comparison on WikiText-103: the rotary method improves over the sinusoidal one, but it still does not achieve satisfying results as the input grows past the training length, while their linear-bias method ALiBi does not degrade (Section 2 and Figure 1). {#c-press-extrap}
Chen et al. report perplexities above $10^3$ for LLaMA beyond its trained window without fine-tuning (Section 2.2). {#c-extrapolation-fails}
They argue that the decay bound of Su et al. can be too loose at large distances, so it does not protect against huge scores at unseen positions (Section 2.2). {#c-bound-loose}
Fine-tuning at the new length alone is slow: after more than 10,000 steps the effective window of LLaMA 7B grew from 2048 to only 2560 (Table 4). {#c-ft-slow}
The rotation acts on the key before it is cached.
If the angles change with the sequence length, a cache of rotated keys becomes wrong; YaRN advises caching keys before the rotation (Peng et al. 2023, Section 3.3). {#c-kv-cache}

## Variants

All the variants below keep the form of eq-rope-matrix and change the angles.
Let $L$ be the trained context length, $L'$ the target and $s = L'/L$.

**Position interpolation.**
Chen et al. scale the positions down so that the extended window falls inside the trained one:

$$ \mathbf{f}'(\mathbf{x}, m) = \mathbf{R}_{m/s}\, \mathbf{x} = \mathbf{R}_{m L / L'}\, \mathbf{x} . $$ {#eq-pi}

{#c-pi-def}
This equals RoPE with every $\theta_j$ divided by $s$, and every position below $L'$ maps into $[0, L)$. {#c-pi-fractional}
They prove an upper bound for interpolation that is at least about 600 times smaller than the bound for extrapolation (Section 2.3). {#c-pi-bound}
With it, LLaMA 7B to 65B reached windows of up to 32768 with fine-tuning within 1000 steps (the abstract). {#c-pi-result}
On Proof-pile the perplexity at the original window of 2048 rose by 0.01 to 0.05 (Section 3.2). {#c-pi-short-cost}

**NTK-aware scaling.**
Instead of shrinking every angle equally, the NTK-aware method raises the base (Peng et al. 2023, Definition 1):

$$ b' = b \cdot s^{d/(d-2)}, \qquad \theta'_j = b'^{-2j/d} = \theta_j\, s^{-2j/(d-2)} . $$ {#eq-ntk}

{#c-ntk-def}
The fastest pair keeps its angle, and the slowest is divided by $s$, as under interpolation. {#c-ntk-ratio}
Without fine-tuning it does much better than position interpolation; after fine-tuning it does worse, because some dimensions are extrapolated (Section 3.1). {#c-ntk-tradeoff}
For the same reason the nominal $s$ must be set higher than the target extension. {#c-ntk-s-higher}

**NTK-by-parts.**
Peng et al. sort the pairs by the ratio $r_j = L/\lambda_j$ of the context to the wavelength $\lambda_j = 2\pi/\theta_j$ (Section 3.2).
A ramp $\gamma(r)$ is 0 below $\alpha$, 1 above $\beta$ and linear between, and

$$ h(\theta_j) = \big(1 - \gamma(r_j)\big)\, \frac{\theta_j}{s} + \gamma(r_j)\, \theta_j . $$ {#eq-by-parts}

{#c-by-parts-def}
Pairs with long wavelengths are interpolated in full and pairs with short ones are left alone. {#c-by-parts-ends}
For the Llama family they found $\alpha = 1$ and $\beta = 32$ to work well. {#c-by-parts-values}

**Dynamic scaling.**
At inference the scale can follow the current length $l'$, $s = \max(1, l'/L)$, so the model degrades gradually past $L'$ (Section 3.3). {#c-dynamic}
Combined with NTK-aware scaling this is Dynamic NTK.

**YaRN.**
YaRN is NTK-by-parts plus a temperature $t$ on the attention logits (Section 3.4):

$$ \mathrm{softmax}\Big( \frac{\mathbf{q}_m^\top \mathbf{k}_n}{t \sqrt{d}} \Big), \qquad \sqrt{1/t} = 0.1 \ln s + 1 . $$ {#eq-yarn-temp}

{#c-yarn-def}
The value of $t$ was fitted on LLaMA models without fine-tuning and also fits Llama 2. {#c-yarn-fit}
Scaling both the rotated query and key by $\sqrt{1/t}$ divides the score by $t$, so the attention code does not change. {#c-yarn-scaling-trick}
The rule gives $\sqrt{1/t} = 1.2079$ at $s = 8$ and $1.2773$ at $s = 16$, in my own computation. {#c-yarn-numbers}
YaRN needs 10 times fewer tokens and 2.5 times fewer training steps than previous methods (the abstract). {#c-yarn-efficiency}
A model trained with $s = 32$ on 64k data extrapolated to a context of 128k (Section 4.2). {#c-yarn-128k}

## When to use it

Use RoPE in a decoder or encoder whose attention should depend on relative position and that adds no parameters for it.
It suits models that will later need a longer window, because interpolation, NTK scaling and YaRN extend it with little fine-tuning.
For a fixed short context, the sinusoidal or learned encodings of [[transformer]] remain an option; this page has no source that compares them at scale.

## Pitfalls

- **The pairing must match the weights.** Adjacent pairs and half pairs give the same scores only after a permutation of the rows of $\mathbf{W}_q$ and $\mathbf{W}_k$. Loading weights trained with one pairing into code with the other gives wrong scores and no error.
- **The index of $\theta$.** Su et al. use $10000^{-2(i-1)/d}$ in Section 3.2.2 and $10000^{-2i/d}$ in Section 3.3. With $i$ from 1, the second form shifts every frequency by one pair.
- **Rotate queries and keys, not values.** The relative form of eq-score comes from the query and the key alone; Su et al. leave the values unrotated (Equation 19).
- **Precision.** The fastest angle at position $m$ is $m$ radians, and float16 holds integers above 2048 only to a step of 2 or more. Compute the angles in float32 or float64.
- **Caching.** With dynamic scaling, cache keys before the rotation.
- **Interpolated positions are fractional.** Code that rounds $m/s$ to an integer loses the method.

## History

Su, Lu, Pan, Murtadha, Wen and Liu of Zhuiyi Technology posted RoFormer as arXiv 2104.09864 in 2021; its version 5 is dated 8 Nov 2023. {#c-hist-su}
RoFormer was integrated into the Hugging Face Transformers library (the abstract). {#c-hist-hf}
Press et al. note that the rotary method was popularized by GPT-J, an open source implementation of GPT-3, and describe it as multiplying the keys and queries of every attention layer by sinusoidal embeddings instead of adding sinusoidal embeddings at the bottom of the transformer (Section 3). {#c-hist-gptj}
In 2023 Chen et al. extended LLaMA with position interpolation, and they note the concurrent SuperHOT work by kaiokendev on the same idea (Section 1). {#c-hist-superhot}
NTK-aware scaling, Dynamic NTK and NTK-by-parts were unpublished works; Peng et al. gave a full account of them and added YaRN (Section 1). {#c-hist-yarn}
Code Llama uses NTK-aware scaling and Qwen 7B uses Dynamic NTK (Peng et al. 2023, Section 1). {#c-hist-adoption}
