---
tag: B6PWZA
kind: method
slug: qk-layernorm
name: QK-layernorm
aliases: [QK normalization, QK-norm, query-key normalization, qk-layernorm]
area: deep-learning-training/training-stability
level: advanced
summary: >-
  Apply LayerNorm to the queries and keys of each attention head before their dot product, so that every attention logit stays bounded however large the weights grow.
purpose: normalization
uses: [layer-normalization@al, cross-entropy-loss@al]
related_to: [transformer]
hyperparameters:
  - {name: eps, meaning: "the constant added to the variance inside the LayerNorm of the queries and keys", typical: "1e-6"}
  - {name: g_q and g_k, meaning: "the learned gains of the LayerNorm of the queries and keys, one per channel of a head", typical: "initialized at 1"}
  - {name: alpha, meaning: "the weight of the z-loss on the output logits", typical: "1e-4"}
  - {name: c_z, meaning: "the weight of the router z-loss of a mixture of experts", typical: "0.001"}
cost:
  train: "two LayerNorms over d_k channels per head and token, about T d_model operations per layer, small next to the T^2 d_model of attention; 2 d_k gains per layer"
  inference: "the same two LayerNorms per head and token; the z-loss costs nothing at inference"
history:
  year: 2023
  note: "Dehghani et al. (2023), arXiv 2302.05442, trained ViT-22B with LayerNorm on queries and keys; Henry et al. (2020), arXiv 2010.04245, had L2-normalized them; the z-loss is from PaLM (2022) and ST-MoE (2022)."
prereqs: [transformer, layer-normalization@al, cross-entropy-loss@al, shannon-entropy@m2, cauchy-schwarz-inequality@m2]
same_as:
  arxiv: ["2302.05442", "2309.14322"]
sources:
  - {paper: "Dehghani et al. (2023)", title: "Scaling Vision Transformers to 22 Billion Parameters", locator: "arXiv 2302.05442v1, pp. 2, 3 and 23", note: "the attention-logit divergence and LayerNorm on queries and keys"}
  - {paper: "Wortsman et al. (2023)", title: "Small-scale proxies for large-scale Transformer training instabilities", locator: "arXiv 2309.14322v2, pp. 1 to 9", note: "both instabilities at small scale, qk-layernorm and the z-loss, LR sensitivity"}
  - {paper: "Chowdhery et al. (2022)", title: "PaLM: Scaling Language Modeling with Pathways", locator: "arXiv 2204.02311v5, pp. 10 and 11", note: "the z-loss and the loss spikes"}
  - {paper: "Zoph et al. (2022)", title: "ST-MoE: Designing Stable and Transferable Sparse Expert Models", locator: "arXiv 2202.08906v2, pp. 7 to 9", note: "the router z-loss"}
  - {paper: "Henry et al. (2020)", title: "Query-Key Normalization for Transformers", locator: "arXiv 2010.04245v1, pp. 1 and 3", note: "QKNorm, the L2 form with a learned scale"}
---

QK-layernorm is a change to the attention layer of a Transformer: it applies layer normalization (LayerNorm) to the queries and to the keys before their dot product.
Every attention logit is then bounded by a quantity that does not depend on the weights.
This removes one cause of divergence in large Transformers and at high learning rates.
The page also covers the z-loss, a penalty on the log of the softmax normalizer, which stabilizes the output logits and the router of a mixture of experts.

## In plain words

Attention compares each query with each key and turns the scores into weights with a softmax.
If the queries and keys grow during training, the scores grow with the square of their size.
Large scores make the softmax pick one key and ignore the rest, and training can then go wrong.
QK-layernorm rescales each query and each key to a fixed length before the comparison.
The scores can no longer grow with the weights, only with the learned gains of the normalization.
The z-loss does a similar job for the final softmax: it adds a small penalty that keeps the log of the sum of exponentials near zero.

## Setting

The setting is the dot-product attention of a [[transformer]] with $h$ heads of width $d_k$ over a sequence of $T$ tokens.
For token $i$ and head width $d_k$, the attention logit with key $j$ is $z_{ij} = \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}$, where $\mathbf{q}_i = \mathbf{W}_Q^\top \mathbf{x}_i$ and $\mathbf{k}_j = \mathbf{W}_K^\top \mathbf{x}_j$.
The logit is bilinear in the two weight matrices, so scaling both by $c$ scales it by $c^2$. {#c-quadratic-growth}
Training at a high learning rate, or for a long time without weight decay, can let the norms of the queries and keys grow.

Dehghani et al. (2023, p. 2) met this when scaling Vision Transformers (ViT): models of around 8 billion parameters showed a divergent training loss after a few thousand steps. {#c-vit-divergence}
They traced it to extremely large attention logits, which gave almost one-hot attention weights with near-zero entropy. {#c-vit-cause}
At Adam learning rate $10^{-3}$, the logits of their 8B model grew past 50000 in magnitude, and the loss rose within 2000 steps (p. 23). {#c-vit-50000}
Wortsman et al. (2023, p. 4) find that the growth comes from the size of the queries and keys, not from their alignment, and they hypothesize a quadratic dependence on the parameter norms. {#c-ws-norm-growth}
They also note that Zhai et al. named the collapse to one-hot weights attention entropy collapse. {#c-ws-entropy-collapse-name}

A second instability lives in the output layer.
In a language model the output logits can drift to very negative values, and this output logit divergence happens towards the end of training, unlike attention logit growth (Wortsman et al. 2023, p. 5). {#c-ws-output-divergence}

## The method

LayerNorm over the $d_k$ channels of a vector $\mathbf{u}$ is

$$ \mathrm{LN}(\mathbf{u}) = \mathbf{g} \odot \frac{\mathbf{u} - \mu(\mathbf{u})\mathbf{1}}{\sqrt{s^2(\mathbf{u}) + \epsilon}}, \qquad \mu(\mathbf{u}) = \frac{1}{d_k}\sum_m u_m, \quad s^2(\mathbf{u}) = \frac{1}{d_k}\sum_m (u_m - \mu)^2 $$ {#eq-layernorm}

with a learned gain $\mathbf{g}$ and a small $\epsilon \ge 0$.
QK-layernorm normalizes each query and each key of each head:

$$ \mathbf{q}_i = \mathrm{LN}_Q(\mathbf{W}_Q^\top \mathbf{x}_i), \qquad \mathbf{k}_j = \mathrm{LN}_K(\mathbf{W}_K^\top \mathbf{x}_j) $$ {#eq-qk-layernorm}

and the attention is unchanged after that:

$$ \mathbf{A} = \mathrm{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big), \qquad \mathbf{O} = \mathbf{A}\mathbf{V} $$ {#eq-attention}

with the softmax over each row.
ViT-22B computes the weights as $\mathrm{softmax}(\mathrm{LN}(\mathbf{X}\mathbf{W}^Q)\,\mathrm{LN}(\mathbf{X}\mathbf{W}^K)^\top / \sqrt{d})$ (Dehghani et al. 2023, p. 2). {#c-vit-method}
The authors say they adopt the approach of Gilmer et al. (2023). {#c-vit-gilmer}
In ViT-22B the QKV projections have no bias, and all LayerNorms are applied without bias and centering (p. 3). {#c-vit-no-bias}
The values are not normalized.

The z-loss adds to the task loss a penalty on the log-normalizer of a softmax.
For a vector of logits $\mathbf{y}$ with $\log Z = \log \sum_j e^{y_j}$, averaged over the positions,

$$ \ell_z = \alpha \, (\log Z)^2 . $$ {#eq-z-loss}

PaLM uses $\alpha = 10^{-4}$ on the output softmax, to keep $\log Z$ close to 0, which its authors found increases stability (Chowdhery et al. 2022, p. 10). {#c-palm-z-loss}
Wortsman et al. (2023, p. 5) use the same loss against output logit divergence. {#c-ws-z-loss-def}
The router z-loss of ST-MoE applies it to the logits of the gating network of a mixture of experts, averaged over the tokens of the batch, to penalize large router logits (Zoph et al. 2022, p. 7). {#c-st-router-z-loss}

The diagnostic for both problems is the entropy of each row of attention weights,

$$ H(\mathbf{a}_i) = -\sum_j a_{ij} \log a_{ij} , $$ {#eq-entropy}

which is $\log T$ for uniform weights and 0 for one-hot weights, together with the largest absolute logit.

## Derivation

Three results explain the method; each has a full derivation with a numerical check.

**The logit bound** (derivation logit-bound).
With $\epsilon = 0$ and no bias, a query that is not constant has norm exactly $\sqrt{d_k}$ after LayerNorm, because the sum of its squared standardized entries is $d_k$. {#c-ln-norm}
A gain bounds each entry, so $\lVert \mathbf{g}_q \odot \hat{\mathbf{q}} \rVert \le \lVert \mathbf{g}_q \rVert_\infty \sqrt{d_k}$.
The Cauchy-Schwarz inequality then gives

$$ |z_{ij}| \le \sqrt{d_k}\, \lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty , $$ {#eq-bound}

and $|z_{ij}| \le \sqrt{d_k}$ with unit gains. {#c-logit-bound}
The bound is reached when the normalized key equals the normalized query, or its negative. {#c-bound-tight}
Multiplying $\mathbf{W}_Q$ and $\mathbf{W}_K$ by $c > 0$ changes neither the normalized vectors nor the logit, while the raw logit is multiplied by $c^2$. {#c-scale-invariance}
The same holds without centering: a nonzero vector divided by its root mean square also has norm $\sqrt{d_k}$, so the form of ViT-22B has the same bound. {#c-rms-norm}
With logits in $[-M, M]$ over $T$ keys, every weight lies between $1/(1 + (T-1)e^{2M})$ and $1/(1 + (T-1)e^{-2M})$, so no row is exactly one-hot. {#c-weight-floor}
Our own evaluation of this formula for $d_k = 64$, unit gains and $T = 512$ gives a floor of about $2.2 \times 10^{-10}$ and a ceiling of about $0.99994$.
The bound limits the logits, but at these values a row can still be close to one-hot.

**Entropy collapse** (derivation entropy-collapse).
For fixed logits $\mathbf{z}$ and a scale $c \ge 0$, the entropy of $\mathrm{softmax}(c\mathbf{z})$ has derivative $-c \, \mathrm{Var}_{\mathbf{p}}(\mathbf{z})$, so it never increases with $c$ and starts at $\log T$. {#c-entropy-monotone}
With a unique largest logit, the weights tend to one-hot and the entropy to 0 as $c \to \infty$. {#c-entropy-collapse}
With $k$ tied largest logits, the entropy tends to $\log k$. {#c-entropy-ties}
Since the raw logits grow as $c^2$ when the weights grow as $c$, weight growth alone drives the collapse.

**The z-loss gradient** (derivation z-loss-gradient).
Since $\partial \log Z / \partial y_j = \mathrm{softmax}(\mathbf{y})_j$,

$$ \frac{\partial \ell_z}{\partial \mathbf{y}} = 2\alpha \, \log Z \; \mathrm{softmax}(\mathbf{y}) , $$ {#eq-z-loss-grad}

which vanishes exactly when $\log Z = 0$. {#c-zloss-grad}
Adding a constant to all logits leaves the softmax cross-entropy unchanged and moves $\log Z$ by that constant; the z-loss picks the shift with $\log Z = 0$, where the logits are the log-probabilities. {#c-zloss-shift}
To first order, a gradient step of size $\eta$ on $\ell_z$ changes $\log Z$ by $-2\eta\alpha \log Z \sum_j p_j^2$, so it pulls $\log Z$ towards 0 from either side. {#c-zloss-pull}

## Worked example

Take $T = 3$ tokens with $d_k = 4$, one head, $\mathbf{W}_Q = \mathbf{W}_K = \mathbf{W}_V = \mathbf{I}$, $\epsilon = 0$ and unit gains:

$$ \mathbf{X} = \begin{pmatrix} 1 & 3 & 1 & 3 \\ 5 & 5 & 1 & 1 \\ 3 & 1 & 3 & 1 \end{pmatrix}. $$

Each row has mean 2 and standard deviation 1 or 2, so LayerNorm gives the rows $(-1, 1, -1, 1)$, $(1, 1, -1, -1)$ and $(1, -1, 1, -1)$, each of norm $2 = \sqrt{4}$.
The raw logits $\mathbf{X}\mathbf{X}^\top / 2$ are

$$ \begin{pmatrix} 10 & 12 & 6 \\ 12 & 26 & 12 \\ 6 & 12 & 10 \end{pmatrix}, \quad \text{and with QK-layernorm} \quad \begin{pmatrix} 2 & 0 & -2 \\ 0 & 2 & 0 \\ -2 & 0 & 2 \end{pmatrix}. $$

The normalized logits reach the bound $\sqrt{4} = 2$ on the diagonal, where the key is the query.
With QK-layernorm, the first row of weights is $(0.8668, 0.1173, 0.0159)$ and the second is $(0.1065, 0.7870, 0.1065)$.
The entropies of the three rows are $0.4411$, $0.6656$ and $0.4411$ nats, against $\log 3 \approx 1.0986$ for uniform weights.
Without the normalization, the first row has entropy $0.3801$.
Now multiply $\mathbf{W}_Q$ and $\mathbf{W}_K$ by 10.
The raw logits are multiplied by 100, every row puts all its weight on token 2 to machine precision, and the entropies are 0.
The normalized logits do not change.

For the z-loss, take the output logits $\mathbf{y} = \log(1, 2, 3, 2)$, so $Z = 8$ and $\mathrm{softmax}(\mathbf{y}) = (1, 2, 3, 2)/8$.
With $\alpha = 10^{-4}$, $\ell_z = 10^{-4} (\log 8)^2 \approx 4.324 \times 10^{-4}$, and the gradient is $2 \times 10^{-4} \log 8 \cdot (1, 2, 3, 2)/8$.
Shifting every logit by $-\log 8$ keeps the softmax and makes both the z-loss and its gradient 0.
All these numbers are from our own runs of the NumPy and PyTorch references, in `test_worked_example` and `test_z_loss_worked_example`.

## Hyperparameters

QK-layernorm adds the $\epsilon$ of the LayerNorm and the learned gains $\mathbf{g}_q$ and $\mathbf{g}_k$, usually initialized at 1.
A larger $\epsilon$ only lowers the norm below $\sqrt{d_k}$, so the bound still holds.
Wortsman et al. (2023, pp. 2 to 3) use qk-layernorm with a z-loss of weight $10^{-4}$, an independent weight decay of $10^{-4}$, and a LayerNorm $\epsilon$ of $10^{-6}$ by default. {#c-ws-defaults}
PaLM uses $\alpha = 10^{-4}$ for the output z-loss.
ST-MoE uses a router z-loss weight $c_z = 0.001$, chosen by the model quality after pre-training in a sweep (Zoph et al. 2022, p. 8). {#c-st-coefficient}
The bound grows with the gains, so a gain that training lets grow loosens it; weight decay on the gains, or no gains, keeps it fixed.
The learning rate is the hyperparameter that QK-layernorm makes less sensitive.

## Cost

Let $d_{\text{model}} = h d_k$.
QK-layernorm adds two LayerNorms of width $d_k$ per head and per token, about $T d_{\text{model}}$ operations per layer for each of the queries and the keys.
This is small next to the $T^2 d_{\text{model}}$ of the attention scores and the $T d_{\text{model}}^2$ of the projections.
The gains add $2 d_k$ parameters per layer when they are shared across heads, as in our references.
The normalization happens before the attention kernel, so fused kernels such as `torch.nn.functional.scaled_dot_product_attention` still apply; the PyTorch reference checks this in `test_agrees_with_scaled_dot_product_attention`.
The z-loss needs $\log Z$, which the softmax cross-entropy already computes, so it adds one square and one mean per position.

## Evidence

On an 8B parameter ViT, QK normalization prevented the divergence caused by attention logit growth (Dehghani et al. 2023, p. 2). {#c-vit-prevents}
ViT had lowered its learning rate with scale, from $10^{-3}$ to $4 \times 10^{-4}$ for ViT-H; with QK normalization, models up to ViT-L stayed stable at $10^{-3}$ (p. 23). {#c-vit-lr}

Wortsman et al. (2023) reproduce both instabilities in small Transformers at high learning rates, where the mitigations used at large scale are equally effective (p. 1). {#c-ws-small-scale}
They define the LR sensitivity of a model as the expected value, over a range of learning rates, of the final loss, capped at the loss at initialization, minus the loss of the best learning rate (p. 3). {#c-ws-lr-sensitivity}
With qk-layernorm and z-loss, models train across three orders of magnitude of learning rate (p. 2). {#c-ws-three-orders}
Without qk-layernorm, the learning rate at which models diverge falls as the models grow (p. 4). {#c-ws-diverge-smaller}
With it, a 1.2B parameter model trains at learning rate 0.3 (p. 4). {#c-ws-1-2b}
A fit of the largest attention logit across scales predicted that a 4.8B model without qk-layernorm would be unstable at learning rate $10^{-2}$, and it diverged; every run with attention logits above $10^4$ had diverged (p. 8). {#c-ws-predict}
Forcing the largest logit of a 10M parameter model to about $\kappa$ degrades the loss around $\kappa = 10^3$, and at $10^4$ the loss is worse than that of a zero-layer bigram model (p. 9). {#c-ws-kappa}
Output logit divergence appears without weight decay at every scale they tried; the z-loss resolves it, and weight decay also mitigates it for the larger models (p. 5). {#c-ws-no-weight-decay}

In ST-MoE, the router z-loss made 3 of 3 runs stable against 4 of 6 for the baseline, with no loss of quality, while tighter update clipping also stabilized training but at a catastrophic loss of quality (Zoph et al. 2022, pp. 7 to 8). {#c-st-stable}
Henry et al. (2020, p. 1) report a mean gain of 0.928 BLEU over earlier bilingual results on 5 low-resource translation pairs with their L2 form. {#c-henry-bleu}

## Limitations and failure modes

QK-layernorm lowers the LR sensitivity but does not remove its growth with scale: with or without it, LR sensitivity increases with model size (Wortsman et al. 2023, p. 4). {#c-ws-sensitivity-grows}
It targets slow divergence, not fast loss spikes, which Wortsman et al. leave aside (p. 2). {#c-ws-slow-divergence}
PaLM saw about 20 loss spikes in its largest model despite gradient clipping, at irregular intervals, and not in its smaller models (Chowdhery et al. 2022, p. 11). {#c-palm-spikes}
It handled them by restarting about 100 steps before each spike and skipping 200 to 500 batches, and attributed them to particular data batches meeting a particular parameter state. {#c-palm-restart}
The bound is on the logits, not on the attention pattern: with $d_k = 64$ a row can still put almost all its weight on one key, as the Derivation shows.
The learned gains can grow and loosen the bound.
The growth is not specific to the softmax: it also occurs with a pointwise squared-ReLU attention (Wortsman et al. 2023, pp. 8 and 24), so removing the softmax is not a fix. {#c-ws-not-softmax}
Neither $\mu$Param nor dropping the square root from the attention scale removes the need for qk-layernorm (pp. 7 and 18). {#c-ws-muparam}
Wortsman et al. report the largest logits in layer 0, so a diagnostic that reads only the last layer can miss the growth (p. 4). {#c-ws-layer0}

## Variants

- **RMS form.** ViT-22B normalizes without centering, dividing by the root mean square; the bound is the same (Derivation).
- **QKNorm.** Henry et al. (2020, p. 3) L2-normalize each query and key along the head dimension and replace $1/\sqrt{d_k}$ with a learned scalar $g$, initialized from the 97.5th percentile sequence length $L$ as $\log_2(L^2 - L)$. {#c-henry-qknorm}
  The logit is then a cosine similarity times $g$, bounded by $|g|$.
- **Attention z-loss.** Zoph et al. (2022, p. 8) report that a z-loss on the attention logits also improved stability without hurting quality. {#c-st-attention-z}
- **Router z-loss.** The same penalty on the router of a mixture of experts. Zoph et al. argue that it keeps the inputs of exponentials small, which reduces roundoff errors in bfloat16 (p. 8). {#c-st-roundoff}
- **Logit clipping.** Clipping the logits cannot replace the z-loss, because it acts after any roundoff error and creates larger discontinuities (Zoph et al. 2022, p. 9). {#c-st-not-clipping}

## When to use it

Use QK-layernorm when the attention logits or the attention entropy show growth, when a model is large, or when you want to train at a higher learning rate.
Use the z-loss on the output softmax when the output logits drift without weight decay, and the router z-loss in a mixture of experts.
With both in place, warm-up, weight decay and $\mu$Param change the LR sensitivity within the stable range but have little effect on that range (Wortsman et al. 2023, p. 2). {#c-ws-other-interventions}
A lower learning rate is the alternative, at the price of slower progress.
Track the largest attention logit and the attention entropy per layer in any case.

## Pitfalls

> [!pitfall]
> **Normalizing over the model width.** QK-layernorm normalizes each head over its own $d_k$ channels. A LayerNorm over the whole $d_{\text{model}}$ does not bound each head's logit by $\sqrt{d_k}$. The test `test_multi_head_is_per_head` checks that each head is normalized on its own.

> [!pitfall]
> **Forgetting the gains.** With learned gains the bound is $\sqrt{d_k} \lVert \mathbf{g}_q \rVert_\infty \lVert \mathbf{g}_k \rVert_\infty$, not $\sqrt{d_k}$. The test `test_logit_bound_values` checks the bound with gains.

> [!pitfall]
> **A naive log-sum-exp.** Computing $\log Z$ as `log(sum(exp(y)))` overflows for large logits, which are the case the z-loss is for. The test `test_softmax_and_logsumexp_are_stable` checks logits of $10^3$.

> [!pitfall]
> **Summing the z-loss over positions.** The z-loss is a mean over positions; a sum makes $\alpha$ depend on the batch size and the sequence length. The test `test_z_loss_is_a_mean_over_positions` checks it.

> [!pitfall]
> **Reading masked logits.** With a causal mask the masked logits are $-\infty$, and a largest-absolute-logit diagnostic that includes them reports infinity. The test `test_max_logit_ignores_masked_entries` checks that only finite logits count.

## History

Henry, Dachapally, Pawar and Chen (2020) proposed QKNorm for low-resource translation, with L2-normalized queries and keys and a learned scale.
Chowdhery et al. (2022) used the z-loss in PaLM, and Zoph et al. (2022) introduced the router z-loss for sparse expert models.
Dehghani et al. (2023) applied LayerNorm to the queries and keys of ViT-22B, following Gilmer et al. (2023), after attention logit growth made 8B parameter models diverge.
Wortsman et al. (2023) found that the same instabilities and mitigations appear in small models trained at high learning rates, which made the effect cheap to study.
