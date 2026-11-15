---
tag: NEW-sparse-mixture-of-experts
kind: method
slug: sparse-mixture-of-experts
name: Sparse mixture of experts
aliases: [sparsely-gated mixture of experts, sparse MoE, MoE layer, Switch layer, top-k routing]
area: deep-learning-architectures/mixture-of-experts
level: core
summary: >-
  A layer of many expert networks and a learned router that sends each token to only k of them, so parameters grow with the number of experts while compute per token stays near that of k experts.
purpose: conditional computation
uses: [matrix-multiplication@m2, cauchy-schwarz-inequality@m2, normal-distribution@m2, cumulative-distribution-function@m2]
related_to: [ensemble-learning@al, gaussian-mixture-model@al, mixed-precision-training@al, reinforce@al, mixture-model@m2]
hyperparameters:
  - {name: N, meaning: "the number of experts in the layer", typical: "8 to 2048; Switch Transformers use 128 for Switch-Base, ST-MoE reports diminishing returns past about 256"}
  - {name: k, meaning: "the number of experts each token is sent to", typical: "1 (Switch Transformers) or 2 (GShard, ST-MoE); Shazeer et al. used 4 in their language models"}
  - {name: capacity factor, meaning: "the slots per expert as a multiple of an even share of the tokens", typical: "1.0 to 1.25 in training, up to 2.0 at evaluation"}
  - {name: alpha, meaning: "the weight of the Switch load-balancing loss", typical: "0.01"}
  - {name: c_z, meaning: "the weight of the router z-loss", typical: "0.001"}
  - {name: placement, meaning: "which feed-forward layers become MoE layers", typical: "every other one (GShard, Switch Transformers)"}
cost:
  train: "per token, the router (2dN FLOPs) and k expert feed-forward passes, plus two all-to-all exchanges of token vectors when experts sit on different devices; memory for all N experts' weights and optimizer state"
  inference: "the same per-token compute as training without the auxiliary losses; all N experts' weights must stay in memory"
history:
  year: 2017
  note: "Shazeer et al. (2017), arXiv 1701.06538, made the gate sparse and trainable at scale; the mixture of experts itself goes back to Jacobs et al. (1991). GShard (2020) and Switch Transformers (JMLR 2022) brought it to the Transformer."
prereqs: [backpropagation@al, mixture-model@m2, normal-distribution@m2]
same_as:
  wikipedia: Mixture_of_experts
  arxiv: ["1701.06538", "2006.16668", "2101.03961", "2202.08906"]
sources:
  - {paper: "Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton and Dean (2017)", title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", locator: "arXiv 1701.06538v1, Sections 2 to 5 and Appendix A", note: "noisy top-k gating, the importance and load losses"}
  - {paper: "Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer and Chen (2020)", title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", locator: "arXiv 2006.16668v1, Section 2 and Algorithm 1", note: "top-2 gating, expert capacity, local groups and the auxiliary loss"}
  - {paper: "Fedus, Zoph and Shazeer (2022)", title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", locator: "JMLR 23 (2022), arXiv 2101.03961v3, Sections 2 to 4", note: "top-1 routing, the capacity factor and the load-balancing loss"}
  - {paper: "Zoph, Bello, Kumar, Du, Huang, Dean, Shazeer and Fedus (2022)", title: "ST-MoE: Designing Stable and Transferable Sparse Expert Models", locator: "arXiv 2202.08906v2, Sections 3 and 4", note: "the router z-loss and the capacity and routing ablations"}
---

A sparse mixture of experts is a layer made of many parallel sub-networks, the experts, and a small router that sends each token to only a few of them.
The layer holds the parameters of all the experts, but each token pays for the computation of only the ones it is sent to.
In Transformers the experts are copies, with separate weights, of the position-wise feed-forward network, and the MoE layer takes the place of every other feed-forward layer.
Making it work at scale needs three pieces besides the router: a fixed capacity per expert, a rule for tokens that overflow it, and an auxiliary loss that keeps the router from sending everything to a few experts.

## In plain words

A dense network uses all its weights on every input.
A mixture of experts keeps many specialist sub-networks and a gate that decides, for each input, which specialists to ask.
If only one or two of a hundred experts run for a given token, the model can have a hundred times more weights at about the same cost per token.
The catch is that the gate learns too: left alone, it tends to favour a few experts, which then get more training, which makes the gate favour them more.
So a sparse MoE adds a penalty for uneven use, and a hard cap on how many tokens one expert takes from a batch, with the overflow skipping the layer.

## Setting

Shazeer et al. start from the observation that the number of parameters limits how much information a neural network can absorb, and that conditional computation, in which parts of the network are active on a per-example basis, had been proposed as a way to raise capacity without a proportional rise in computation. {#c-sh-capacity-limit}
They note that when the whole model is active for every example, growing the model and the training set together makes training cost grow roughly quadratically. {#c-sh-quadratic}
They list what had kept conditional computation from paying off: GPUs are much faster at arithmetic than at branching, each active part sees a smaller batch, communication between devices can dominate, extra losses may be needed to control sparsity, and the image data sets of earlier work, of up to 600,000 images, were too small for billions of parameters. {#c-sh-challenges}
Four years later Fedus et al. describe the same trade: an MoE picks different parameters for each example, which gives a sparsely activated model with a very large parameter count at a constant computational cost, and its adoption had been held back by complexity, communication costs and training instability. {#c-sw-abstract}

The input is a batch of $T$ token vectors $x \in \mathbb{R}^d$, the output of the layer below.
The layer has $N$ experts $E_1, \dots, E_N$, each mapping $\mathbb{R}^d$ to $\mathbb{R}^d$, and a router with weights $W_r \in \mathbb{R}^{d \times N}$.
Training is ordinary back-propagation on the task loss plus auxiliary losses on the router.
The experts are usually spread over devices, one or a few per device, so routing is also a communication step.

## The method

### The layer

The layer of Shazeer et al. has $n$ experts and a gating network $G$ whose output is a sparse $n$-vector; its output is the gate-weighted sum of the expert outputs, and an expert whose gate is zero is not evaluated. {#c-sh-moe-def}

$$ y = \sum_{i=1}^{N} G(x)_i \, E_i(x) $$ {#eq-moe-output}

Any network with matching input and output sizes can be an expert, but Shazeer et al. use feed-forward networks of one architecture with separate weights. {#c-sh-experts-ffn}
GShard makes the Transformer sparse by turning every other feed-forward layer, in the encoder and the decoder, into a position-wise MoE layer with a variant of top-2 gating. {#c-gs-every-other}
Its experts are two-layer ReLU networks, $E_e(x) = W^{\text{out}}_e \, \mathrm{ReLU}(W^{\text{in}}_e x)$, each token goes to at most two of them, and the output combines the chosen experts' outputs with their gates. {#c-gs-expert-ffn}

### The gate

The plain, dense gate is a linear map followed by a softmax, as in Jordan and Jacobs (1994). {#c-sh-softmax-gate}

$$ G_\sigma(x) = \mathrm{softmax}(x W_g) $$ {#eq-softmax-gate}

Shazeer et al. make it sparse and noisy.
Gaussian noise, scaled per expert by a softplus of a second learned projection $W_{\text{noise}}$, is added to the logits; the $k$ largest are kept, the rest are set to $-\infty$, and the softmax makes the other gates exactly zero. {#c-sh-noisy-top-k}

$$ H(x)_i = (x W_g)_i + \varepsilon_i \cdot \mathrm{softplus}\big((x W_{\text{noise}})_i\big), \qquad \varepsilon_i \sim \mathcal{N}(0, 1) $$ {#eq-noisy-logits}

$$ G(x) = \mathrm{softmax}\big(\mathrm{KeepTopK}(H(x), k)\big), \qquad \mathrm{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is among the top } k, \\ -\infty & \text{otherwise.} \end{cases} $$ {#eq-noisy-top-k}

The softmax of the kept logits is the same as the $k$ largest softmax probabilities divided by their sum, because restricting a softmax to a subset only renormalizes it. {#c-topk-equivalence}
The derivation `top-k-gate` shows this; it is why GShard can take the top two probabilities and divide each by their sum (Algorithm 1), and why a top-1 gate after renormalizing would always be 1.
Shazeer et al. note that the top-k makes the gate discontinuous, say they had not seen this cause trouble in practice, and add the noise to help balance the load. {#c-sh-discontinuities}
The gate is trained by back-propagation with the rest of the model: with $k > 1$ the gates of the chosen experts depend smoothly on $W_g$, unlike the boolean gates trained with REINFORCE by Bengio et al. (2015). {#c-sh-backprop}

Switch Transformers write the router as logits $h(x) = W_r x$, probabilities $p_i(x) = \mathrm{softmax}(h(x))_i$, a top-$k$ set $\mathcal{T}$, and the output $\sum_{i \in \mathcal{T}} p_i(x) E_i(x)$. {#c-sw-router}
They recall that Shazeer et al. had conjectured that $k > 1$ was needed for the router to receive useful gradients, since learning to route seems to require comparing experts. {#c-sw-conjecture}
They route to one expert instead, calling this a Switch layer, and report that quality is kept while routing gets cheaper and the model performs better; the gate $p_i(x)$, which multiplies the expert output, is what carries the gradient to the router. {#c-sw-top-1}
They give three reasons for $k = 1$: less router computation, an expert capacity that can be at least halved, and a simpler implementation with less communication. {#c-sw-benefits}
With $k = 1$ the gate is not renormalized, or it would always be 1 and the router would get no gradient from the task loss.

### Capacity and dropped tokens

Lepikhin et al. warn that sending each token to its top-$k$ experts by softmax probability makes training unbalanced: a few experts get most of the tokens and large buffers while the rest barely train. {#c-gs-naive-top-k}
GShard therefore caps each expert at a capacity of order $N/E$ tokens for a batch of $N$ tokens and $E$ experts, counts the tokens sent to each expert, and treats a token whose chosen experts are all full as overflowed: its gates are zeroed and its vector reaches the next layer through the residual connection. {#c-gs-capacity}
The tokens of a batch are split into $G$ groups of $S$ tokens that are routed independently and in parallel, each with a capacity of $2N/(G E)$ tokens per expert. {#c-gs-groups}
Within a group, Algorithm 1 renormalizes the top two gates by their sum, places every token's first choice before any second choice, and keeps a choice only while the expert's count is below its capacity. {#c-gs-normalize}
With random routing, the second expert is used only with a probability that grows with its gate $g_2$, decided by $2 g_2 > u$ for a uniform $u$, to spare capacity when the second expert would add little. {#c-gs-random-routing}

Switch Transformers state the capacity with a capacity factor $c$.
TPUs need tensor shapes fixed at compile time, so each expert gets the tokens per batch divided by the number of experts, times $c$; a factor above 1 leaves room for uneven routing, and a token sent to a full expert is dropped, skipping the expert and passing on through the residual connection. {#c-sw-capacity}
Writing the rounding explicitly, and counting $k$ choices per token,

$$ C = \left\lfloor \frac{k \, T \, c}{N} \right\rfloor . $$ {#eq-capacity}

A larger $c$ means fewer dropped tokens but more computation and memory spent on empty slots, and Fedus et al. found that keeping the drop rate low mattered as sparse models grew. {#c-sw-capacity-tradeoff}

### Load-balancing losses

Shazeer et al. saw the gating network converge to large weights on the same few experts, an imbalance that feeds itself because the favoured experts train faster and are picked more. {#c-sh-self-reinforcing}
Their first remedy is the importance loss: the importance of an expert is the sum of its gates over a batch $X$, and the loss is a hand-set weight times the squared coefficient of variation of the importances. {#c-sh-importance}

$$ \mathrm{Importance}(X)_i = \sum_{x \in X} G(x)_i, \qquad L_{\text{importance}} = w_{\text{importance}} \cdot \mathrm{CV}\big(\mathrm{Importance}(X)\big)^2 $$ {#eq-importance}

Equal importance does not mean equal numbers of tokens: one expert can get a few tokens with large gates and another many with small ones, which is what overloads memory on a device, so they add a load loss. {#c-sh-load-motivation}
The count of tokens per expert is not differentiable, so they smooth it: $P(x, i)$ is the probability that expert $i$ stays in the top $k$ when only its own noise is drawn again, a normal CDF, the loads are the sums of $P(x, i)$ over the batch, and $L_{\text{load}}$ is a weight times their squared coefficient of variation. {#c-sh-load-def}

$$ P(x, i) = \Phi\!\left( \frac{(x W_g)_i - \mathrm{kth\_excluding}(H(x), k, i)}{\mathrm{softplus}\big((x W_{\text{noise}})_i\big)} \right), \qquad \mathrm{Load}(X)_i = \sum_{x \in X} P(x, i) $$ {#eq-load-prob}

Here $\mathrm{kth\_excluding}(H(x), k, i)$ is the $k$-th largest entry of $H(x)$ once entry $i$ is left out, the value that expert $i$'s noisy logit has to beat.
They also set $W_g$ and $W_{\text{noise}}$ to zero at the start, so the router begins with no signal and some noise, near an even load, before the losses take effect. {#c-sh-zero-init}

GShard uses one loss on each group of $S$ tokens. {#c-gs-aux}

$$ \ell_{\text{aux}} = \frac{1}{E} \sum_{e=1}^{E} \frac{c_e}{S} \, m_e, \qquad m_e = \frac{1}{S} \sum_{s=1}^{S} g_{s,e} $$ {#eq-gshard-aux}

Here $c_e$ is the number of tokens whose first choice is $e$ and $m_e$ the mean gate of expert $e$; the natural target would be the mean of $(c_e/S)^2$, but $c_e$ comes out of a top-$k$ and has no gradient, so $m_e$ replaces one of the factors.

Switch Transformers use the same form with a different scale.
For a batch of $T$ tokens, $f_i$ is the fraction whose highest-probability expert is $i$, $P_i$ is the mean of $p_i(x)$ over the batch, and the loss of each Switch layer is $\alpha N \sum_i f_i P_i$. {#c-sw-balance-def}

$$ L_{\text{balance}} = \alpha \, N \sum_{i=1}^{N} f_i \, P_i, \qquad f_i = \frac{1}{T} \sum_{x} \mathbb{1}\{\arg\max_j p_j(x) = i\}, \qquad P_i = \frac{1}{T} \sum_{x} p_i(x) $$ {#eq-switch-balance}

Fedus et al. present it as a simplification of the separate importance and load losses of Shazeer et al. (2017), in the line of Shazeer et al. (2018) and Lepikhin et al. (2020). {#c-sw-simplifies}
They say the loss is smallest under uniform routing, that only $P$ carries a gradient, and that the factor $N$ keeps the loss the same as the number of experts changes, since $\sum_i f_i P_i = 1/N$ when routing is uniform. {#c-sw-balance-claims}
They set $\alpha = 10^{-2}$ after trying powers of ten from $10^{-1}$ to $10^{-5}$, the value that balanced the load quickly without hurting the training loss. {#c-sw-alpha}

### The router z-loss

Zoph et al. found float32 routing not enough for stable training at their biggest scales and add a penalty on the size of the router logits, the mean over the $B$ tokens of the squared log-sum-exp of each token's logits. {#c-st-z-loss}

$$ L_z = \frac{1}{B} \sum_{i=1}^{B} \Big( \log \sum_{j=1}^{N} e^{h_j(x_i)} \Big)^2 $$ {#eq-z-loss}

The total loss of ST-MoE is the cross-entropy plus $c_B$ times the balance loss plus $c_z$ times the z-loss, with $c_z = 0.001$ picked by a sweep. {#c-st-total-loss}

## Derivation

Three derivations on this page support the main formulas; each lists its assumptions and has a numeric check.

**The balance loss** (`balance-loss-minimum`).
When routing is uniform, $f_i = P_i = 1/N$ and the loss is $\alpha N \cdot N \cdot (1/N^2) = \alpha$, for every $N$. {#c-balance-uniform}
Whether $\alpha$ is the minimum depends on how $P$ is tied to $f$.
If $P = f$, as when each token puts all its probability on one expert, the loss is $\alpha N \sum_i f_i^2$, and by Cauchy-Schwarz $(\sum_i f_i)^2 \le N \sum_i f_i^2$, so the loss is at least $\alpha$, with equality only at uniform $f$. {#c-balance-coupled-minimum}
In the other direction, $\sum_i f_i P_i \le \max_i P_i \le 1$, so the loss is at most $\alpha N$, the value when all tokens go to one expert with probability 1. {#c-balance-maximum}
Without the coupling the claim that uniform routing is the minimum is not exact.
A token's argmax expert gets at least probability $1/N$, so $P_i \ge f_i/N$ and the loss is at least $\alpha \sum_i f_i^2 \ge \alpha/N$; with two experts, three quarters of the tokens at $(1/2 + \varepsilon, 1/2 - \varepsilon)$ and a quarter at $(0, 1)$ give $f = (3/4, 1/4)$, $P \to (3/8, 5/8)$ and a loss of $2(9/32 + 5/32)\alpha = 0.875\,\alpha$. {#c-balance-below-alpha}
In practice the loss pushes toward balance through its gradient.
With $f$ held fixed, the derivative with respect to logit $j$ of token $x$ is $(\alpha N/T)\, p_j(x)\,\big(f_j - \sum_i f_i p_i(x)\big)$, so gradient descent lowers the logits of the experts that receive more than their weighted share and raises the others. {#c-balance-logit-gradient}

**Capacity** (`expert-capacity`).
With $C$ slots per expert and $n_i$ of the $kT$ choices sent to expert $i$, the layer keeps $\sum_i \min(n_i, C)$, drops $D = \sum_i \max(0, n_i - C)$, and a fixed-shape implementation computes $NC$ slots, of which $NC - kT + D$ are padding. {#c-capacity-count}
With perfectly even routing, $kT/N$ an integer and $c \ge 1$, nothing is dropped and at most a fraction $1 - 1/c$ of the slots is padding. {#c-capacity-balanced}
Whatever the routing, at most $NC$ choices fit, so $D \ge kT - NC \ge kT(1 - c)$: a capacity factor below 1 drops at least a fraction $1 - c$ of the choices, and with $k = 1$ and all tokens on one expert, $T - \min(T, C)$ tokens are dropped. {#c-capacity-forced-drops}
GShard's per-group capacity is this formula with $k = 2$, $S$ tokens per group, $E$ experts and $c = 1$. {#c-capacity-gshard-group}
Shazeer et al. make the matching count for their shrinking batch problem: with $k$ of $n$ experts chosen for each of $b$ examples, an expert sees about $kb/n$ examples, and combining the batches of $d$ synchronous data-parallel replicas raises this to about $kbd/n$. {#c-sh-shrinking-batch}

**Cost** (`sparse-cost`).
Take experts $d \to h \to d$ without biases and count a multiply-add as 2 FLOPs.
The layer has $2Ndh + dN$ weights and costs $4kdh + 2dN$ FLOPs per token if only the chosen experts run, or $4kc\,dh + 2dN$ on average if every padded slot is computed, against $2dh$ weights and $4dh$ FLOPs for one dense feed-forward layer of width $h$. {#c-cost-params-flops}
So the ratio of weights to per-token expert FLOPs is $N/(2k)$ against $1/2$ for the dense layer: for fixed $k$, adding experts adds weights and leaves the per-token compute alone, apart from the router's $2dN$. {#c-cost-ratio}
An expert does $2dh$ multiply-adds for each token and receives and returns $2d$ numbers, so its ratio of computation to communication is $h$. {#c-cost-intensity}
Shazeer et al. make the same point: for an expert with one hidden layer, computation per unit of input and output grows with the hidden size, so wider experts use the hardware better. {#c-sh-bandwidth}

## Worked example

Take $T = 8$ tokens, $N = 4$ experts and top-1 routing.
Say the router's argmax experts are $(0, 1, 0, 2, 0, 1, 3, 0)$ and its probabilities are one-hot, so the loads are $(4, 2, 1, 1)$.
The dispatch fractions are $f = (0.5, 0.25, 0.125, 0.125)$ and $P = f$, so the balance loss is $4\alpha(0.25 + 0.0625 + 0.015625 + 0.015625) = 1.375\,\alpha$, against $\alpha$ for an even split and $4\alpha$ if every token went to expert 0. {#c-balance-worked}
With $c = 1$, $C = \lfloor 8/4 \rfloor = 2$: expert 0 keeps tokens 0 and 2 and drops tokens 4 and 7, 6 of the 8 slots hold tokens, and the two dropped tokens leave the layer unchanged through the residual connection.
With $c = 1.5$, $C = 3$, one token is dropped and 7 of 12 slots are used; with $c = 0.5$, $C = 1$, four tokens are dropped and all 4 slots are used. {#c-capacity-worked}
The test `test_dispatch_worked_example` of both implementations runs this example.

For the cost, take $d = 768$, $h = 3072$ and $N = 128$.
One expert has 4,718,592 weights and the layer 604,078,080, about 128 times a dense layer; a token costs 9,633,792 FLOPs with $k = 1$, 11,993,088 with $k = 1$ and $c = 1.25$ counting padding, and 19,070,976 with $k = 2$, against 9,437,184 for the dense layer. {#c-cost-worked}
The router adds $2dN = 196{,}608$ FLOPs per token, about 2% of one expert.
Fedus et al.'s Table 9 shows the same pattern at full scale: T5-Base has 0.2B parameters and Switch-Base 7B with 128 experts, at the same 124B FLOPs per sequence, while Switch-C reaches 1571B parameters with 2048 experts at 890B FLOPs per sequence. {#c-sw-table9}

## Hyperparameters

The number of experts $N$ sets the parameter count.
Zoph et al. see the gain from more experts fall off quickly beyond about 256, under 1% of experts active, and since more experts add memory traffic but no computation, they advise at most one expert per core on their TPU systems. {#c-st-diminishing}
The number of choices $k$ is 1 in Switch Transformers and 2 in GShard and ST-MoE.
At a fixed capacity factor, Zoph et al. measured a small gain of 0.004 in negative log perplexity for top-2 over top-1 at a training capacity factor of 1.25, revising the earlier advice of Fedus et al., and attribute the difference to an eight times larger training scale. {#c-st-top-n}
The capacity factor $c$ trades dropped tokens for padding.
Raising it at training and evaluation improves quality, by 0.011 for top-1 going from 1.0 to 1.25, but the expert computation, activation memory and all-to-all traffic grow linearly with it. {#c-st-capacity-quality}
The balance weight $\alpha$ is $10^{-2}$ in Switch Transformers; the z-loss weight $c_z$ is $10^{-3}$ in ST-MoE; the importance and load weights of Shazeer et al. were 0.1 in their ablation.
GShard and Switch Transformers place an MoE layer at every other feed-forward layer.

## Cost

Per token the layer costs the router plus $k$ expert passes, from the derivation above; the weights grow by a factor of about $N$.
In memory, all $N$ experts and their optimizer state must be held somewhere, which is why experts are spread over devices.
With experts on different devices, each MoE layer needs two all-to-all exchanges, one to send tokens to their experts and one to bring the results back, each of about $k$ token vectors per token.
Fixed-shape implementations compute all $N \cdot C$ slots, so a capacity factor $c$ multiplies the expert computation by about $c$.
At inference the auxiliary losses are gone but the expert weights are not, so a sparse model needs the memory of its full parameter count even though it computes like a much smaller one.

## Evidence

On the 1-billion-word language modeling benchmark at about 8 million operations per timestep, a model with 4 always-active experts did about as well as the compute-matched baselines, and the largest, with 4096 experts, had 24% lower test perplexity. {#c-sh-4096}
On a corpus of about 100 billion words, test perplexity kept falling up to 65536 experts and 68 billion parameters, 39% below the compute-matched baseline, then rose at 131072 experts, perhaps from too much sparsity. {#c-sh-65536}
In the balance ablation of Shazeer et al., a model trained with neither loss reached perplexity 39.8 with its busiest expert at 17.80 times the mean load, and one with both losses at 0.1 reached 35.6 with a ratio of 1.14. {#c-sh-table6}
GShard trained a translation Transformer with more than 600 billion parameters, from 100 languages into English, on 2048 TPU v3 cores in 4 days, with far better quality than earlier work. {#c-gs-600b}
In Fedus et al.'s Table 1, at capacity factor 1.0 Switch-Base ran at 1000 examples per second and reached the quality threshold in 62.8 hours, against 860 examples per second and 80.1 hours for a top-2 MoE-Base; they conclude that top-1 routing does best at low capacity factors such as 1.0 and 1.25. {#c-sw-table1}
They report up to 7 times faster pre-training than T5-Base and T5-Large on the same resources, and 4 times faster than T5-XXL with models of up to a trillion parameters. {#c-sw-speedup}
With the auxiliary loss, the drop rate was typically below 1% of tokens and did not depend on the number of experts. {#c-sw-drop-rate}
Zoph et al. scaled a sparse encoder-decoder, ST-MoE-32B, to 269 billion parameters at about the cost of a 32-billion-parameter dense model, and report strong transfer results on tasks including SuperGLUE, XSum and Natural Questions. {#c-st-269b}

These results compare models at matched computation per token, not at matched parameter count or memory; a sparse model with many experts is compared with a dense model that has far fewer weights.

## Limitations and failure modes

**Training instability.**
Fedus et al. found the 1.6-trillion-parameter Switch-C, with 2048 experts, stable, while Switch-XXL, with nearly ten times the FLOPs per sequence, was sometimes unstable. {#c-sw-stability}
They compute the router in float32, casting its input up and its dispatch and combine tensors back to bfloat16, which kept float32 stability at close to bfloat16 speed. {#c-sw-selective-precision}
They also cut the default initialization scale of the Transformer by a factor of 10. {#c-sw-init}
In ST-MoE's stability study, 4 of 6 baseline runs were stable; clipping the updates at 0.1 made 3 of 3 stable but cut quality to −4.206, while the z-loss made 3 of 3 stable with quality −1.741 against −1.755 for the baseline. {#c-st-table4}

**Fine-tuning.**
Sparse models can overfit small downstream tasks.
Fedus et al. propose expert dropout, a much higher dropout rate inside the experts than elsewhere; 0.1 outside and 0.4 inside the experts helped on four small tasks, while raising the rate at every layer did worse. {#c-sw-expert-dropout}

**Dropped tokens.**
A dropped token skips the layer, so its prediction depends on which other tokens shared its batch or group.
The losses balance the batch on average, not each expert at each step, so a few tokens are still dropped at $c$ near 1.

**The balance loss is a proxy.**
It does not reach its minimum only at uniform routing, as the derivation shows, and its gradient flows only through $P$; it balances well in practice because, in the regime of a working router, $f$ and $P$ move together.

**Memory.**
Every expert must sit in device memory at inference, so the per-token compute advantage does not reduce the memory needed to serve the model.

## Variants

- **Dense MoE** keeps the softmax gate of the early work and runs every expert, so it has no sparsity (Jordan and Jacobs 1994, as cited by Shazeer et al.).
- **Noisy top-k gating** with the importance and load losses (Shazeer et al. 2017).
- **Hierarchical MoE**, a two-level tree of gates, which Shazeer et al. use for their larger models.
- **Top-2 gating with groups and random routing** (GShard), with the loss of eq-gshard-aux.
- **Switch routing**, $k = 1$ with the loss of eq-switch-balance (Fedus et al.).
- **Top-2 with the router z-loss** (ST-MoE), the loss of eq-z-loss on top of the balance loss.

The layer is related to several older ideas, and differs from each.
In ensemble learning every member sees every input and the outputs are averaged with fixed weights; here the weights depend on the input and most experts are not run at all.
A mixture model, such as a Gaussian mixture model, mixes probability densities with weights that do not depend on the input; the MoE gate mixes the outputs of functions, and its weights are a function of the token.
The float32 router inside a bfloat16 network is a targeted form of mixed precision training: only the router computation runs at the higher precision.

## When to use it

Use a sparse MoE when the model is limited by its parameter count and the hardware can hold many more weights than it can afford to multiply for every token, as in large-scale language modeling and translation.
It pays most when the experts are spread over many devices with fast all-to-all links, and when the batch is large enough that each expert receives a useful number of tokens.
For a model that has to fit on one small device, a dense model of the same memory is usually the better choice.
Start from $k = 1$ or $2$, a capacity factor of 1.25, $\alpha = 0.01$, $c_z = 0.001$, float32 routing, and an MoE layer at every other feed-forward position.

## Pitfalls

> [!pitfall]
> **Renormalizing a single gate.** With $k = 1$, dividing the gate by itself makes it 1 and cuts the task-loss gradient to the router. The test `test_top_k_route_matches_keep_top_k` checks that with $k = 1$ the gate is the router probability unless renormalization is asked for.

> [!pitfall]
> **Second choices taking the slots of first choices.** If the dispatch walks token by token through all $k$ choices, a token's second choice can fill the slot a later token needed for its first choice. The test `test_dispatch_first_choices_before_second` checks that every first choice is placed before any second choice, as in GShard's Algorithm 1.

> [!pitfall]
> **A gradient through f.** The dispatch fractions $f$ come from an argmax; code that builds them from a differentiable soft count changes the loss. The test `test_balance_loss_gradient_is_alpha_n_f_over_t` checks that the gradient with respect to the probabilities is $\alpha N f_i / T$.

> [!pitfall]
> **Router noise at evaluation.** The noise of eq-noisy-logits is a training device; leaving it on makes inference random. The test `test_noise_only_in_training` checks that evaluation is deterministic.

> [!pitfall]
> **Dropped tokens without the residual.** The layer returns zero for a dropped token; the caller must add the residual, or the token's representation is lost. The test `test_layer_drops_and_gradients` checks the zeros, and the docstring of `SparseMoE` says the residual is added outside.

## History

Shazeer et al. trace the mixture of experts to Jacobs et al. (1991) and Jordan and Jacobs (1994), more than two decades before their own work, and list later versions whose experts were SVMs, Gaussian processes, Dirichlet processes and deep networks. {#c-sh-jacobs}
In that work the mixture was the whole model; Eigen et al. (2013) stacked several mixtures, each with its own gate, inside a deep model, and Shazeer et al. take up this use of the MoE as a building block. {#c-sh-eigen}
Shazeer et al. (2017) put MoE layers of up to 137 billion parameters between stacked LSTM layers, applied at every position, for language modeling and translation, and report capacity gains of more than 1000 times with small losses in computational efficiency. {#c-sh-137b}
GShard (Lepikhin et al. 2020) moved the layer into the Transformer with top-2 gating, expert capacity and automatic sharding over 2048 TPU cores.
Switch Transformers (Fedus, Zoph and Shazeer, first posted in 2021 and published in JMLR in 2022) simplified routing to one expert and the losses to one.
ST-MoE (Zoph et al. 2022) added the router z-loss and studied stability and fine-tuning.
