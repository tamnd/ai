---
tag: NEW-gradient-checkpointing
kind: method
slug: gradient-checkpointing
name: Gradient checkpointing
aliases: [activation checkpointing, activation recomputation, rematerialization]
area: deep-learning-training/training-recipes
level: core
summary: >-
  Keep only some activations of the forward pass and recompute the rest in the backward pass: about one extra forward pass buys activation memory that grows like the square root of depth.
purpose: parallelism-and-scale
uses: [backpropagation@al]
related_to: [automatic-differentiation@al, backpropagation-through-time@al, mixed-precision-training@al]
hyperparameters:
  - {name: segments, meaning: "k, the number of segments of a chain of n layers; one checkpoint is kept at the end of each segment", typical: "about sqrt(n); one segment per transformer layer in large language models"}
  - {name: placement, meaning: "which activations are checkpoints when layers differ in size or cost", typical: "the outputs of expensive operations such as convolutions and matrix products; the cheap ones are recomputed"}
  - {name: what to recompute, meaning: "full recomputation of a block from its input, or selective recomputation of only the cheap, large activations", typical: "selective for the attention scores of a transformer"}
  - {name: memory budget, meaning: "B of Chen et al. or the number of slots m of Gruslys et al., when the schedule is planned for a given memory", typical: "the device memory minus the parameters, gradients and optimizer state"}
cost:
  train: "about one extra forward pass per step: 2n - k layer forwards instead of n for k segments; activation memory k + floor(n/k) instead of n + 1"
  inference: "none: inference keeps no activations for a backward pass"
history:
  year: 2016
  note: "Checkpointing for the reverse mode of automatic differentiation is older; Chen et al. cite Griewank and Walther's Revolve (ACM TOMS 26(1), 2000). Chen, Xu, Zhang and Guestrin brought it to deep networks in April 2016 (arXiv 1604.06174). Gruslys et al. gave the optimal policy for recurrent networks by dynamic programming in June 2016 (arXiv 1606.03401). Korthikanti et al. (2022, arXiv 2205.05198) made it selective inside transformer layers."
prereqs: [backpropagation@al, automatic-differentiation@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1604.06174", "1606.03401", "2205.05198"]
  al: gradient-checkpointing
sources:
  - {paper: "Chen, Xu, Zhang and Guestrin (2016)", title: "Training Deep Nets with Sublinear Memory Cost", locator: "the abstract, Sections 2, 4.2 to 4.4, 5.3 and 5.4", note: "the sqrt(n) segment schedule, the recursion, and the measurements"}
  - {paper: "Gruslys, Munos, Danihelka, Lanctot and Graves (2016)", title: "Memory-Efficient Backpropagation Through Time", locator: "the abstract, Sections 3.1, 3.5 and 4.1", note: "the dynamic program for a fixed memory budget"}
  - {paper: "Korthikanti, Casper, Lym, McAfee, Andersch, Shoeybi and Catanzaro (2022)", title: "Reducing Activation Recomputation in Large Transformer Models", locator: "the abstract, Sections 1, 4.1, 5 and 6.2", note: "activation memory per transformer layer and selective recomputation"}
---

Gradient checkpointing reduces the memory needed to train a neural network.
Backpropagation normally keeps every activation of the forward pass until the backward pass uses it.
Gradient checkpointing keeps only a few of them, the checkpoints, and recomputes the others from the nearest checkpoint when the backward pass needs them.
The gradients are exactly the same.
The price is extra forward computation, about one forward pass per step.
The method is usually called gradient checkpointing or activation recomputation. {#c-names}

## In plain words

To compute gradients, the backward pass needs the intermediate results of the forward pass.
Storing all of them is the usual way, and for deep networks or long sequences that memory can be larger than the model itself.
Gradient checkpointing stores a result only every few layers.
When the backward pass reaches a stretch of layers whose results were thrown away, it runs that stretch forward again from the last stored result, uses the results, and frees them.
You pay with compute and save memory, and the answer does not change.

## Setting

Take a chain of $n$ layers

$$ h_i = f_i(h_{i-1}; \theta_i), \qquad i = 1, \dots, n, $$ {#eq-layer}

with input $h_0$ and output $h_n$.
The backward pass computes $\partial L/\partial h_{i-1}$ and $\partial L/\partial \theta_i$ from $\partial L/\partial h_i$, and for that it needs the activations of layer $i$.
On this page, the backward of a layer needs its input $h_{i-1}$ and its output $h_i$.
For $\tanh$, for example, the derivative is $1 - h_i^2$.
If all activations have the same size, plain backpropagation holds $n + 1$ of them at the end of the forward pass.
That is linear in depth for a feedforward network and linear in sequence length for a recurrent network unrolled in time.
Parameters, gradients and optimizer state are not affected by checkpointing.
Gradient checkpointing applies when activations, not parameters, fill the memory.

## The method

Pick $k$ segments and cut the chain into $k$ runs of consecutive layers of sizes $\lceil n/k \rceil$ or $\lfloor n/k \rfloor$, the longer ones first.

1. **Forward.** Run the chain once. Keep $h_0$ and the output of each segment, and drop every other activation as soon as the next layer has used it.
2. **Backward.** Take the segments from last to first. For each one, run its layers forward again from its saved input, keep the activations inside it, do the usual backward through the segment, and free them.

Chen et al. (2016) divide the network into $k$ segments, with a memory cost of $O(n/k) + O(k)$, and choose $k = \sqrt{n}$ to get $O(2\sqrt{n})$. {#c-chen-segments}
They describe it as an algorithm that trains an $n$-layer network in $O(\sqrt{n})$ memory, for the price of one extra forward pass per mini-batch. {#c-chen-sqrt}

With equal layers, the exact count for this schedule is

$$ M(n, k) = k + \left\lfloor \frac{n}{k} \right\rfloor \ \text{activations at once}, \qquad \text{against } n + 1 \text{ with full storage}, $$ {#eq-segment-memory}

$$ F(n, k) = 2n - k \ \text{layer forwards}, \qquad \text{against } n. $$ {#eq-segment-compute}

Applied again inside each segment, the idea gives memory logarithmic in $n$.
Chen et al. store $k$ intermediate results and apply the same strategy inside each sub-path, which gives the recursion $g(n) = k + g(n/(k+1))$ with solution $g(n) = k \log_{k+1} n$. {#c-chen-recursion}
With $k = 1$, the chain is split in half again and again.
That needs $O(\log n)$ memory and $O(n \log n)$ forward computation. {#c-chen-log}
For $n = 2^p$, our bisection schedule runs

$$ F_{\text{rec}}(n) = n + \frac{n}{2} \log_2 n \ \text{layer forwards}. $$ {#eq-recursive-compute}

## Derivation

The derivation [segment-memory](derivations/segment-memory.md) counts the activations held while segment $j$ is processed in the backward pass.
These are $h_0$ and the outputs of segments 1 to $j$, which is $j + 1$ checkpoints, plus the $s_j - 1$ activations rebuilt inside segment $j$.
The largest of these sums is $k + \lfloor n/k \rfloor$, so the schedule never holds more than that. {#c-segment-memory}
Each segment rebuilds all of its layers except the last, because the last layer's output is a checkpoint.
So the backward pass adds $n - k$ forwards, at most one extra forward pass. {#c-segment-compute}
By the inequality of arithmetic and geometric means, $n/k + k \ge 2\sqrt{n}$, with equality at $k = \sqrt{n}$.
For $n = q^2$, the integer count $k + \lfloor n/k \rfloor$ is smallest at $k = q$, where it is $2q$. {#c-sqrt-optimum}
The check replays the schedule for every $n \le 80$ and every $k$, and confirms both counts.

The derivation [recursive-bisection](derivations/recursive-bisection.md) treats the case $k = 1$ at every level.
A range of $t$ layers is split at $\lfloor t/2 \rfloor$, and the call stack holds one midpoint per level.
So the schedule holds at most $2 + \lceil \log_2 n \rceil$ activations. {#c-recursive-memory}
The forward count follows $R(t) = \lfloor t/2 \rfloor + R(\lceil t/2 \rceil) + R(\lfloor t/2 \rfloor)$, which for $n = 2^p$ gives eq-recursive-compute. {#c-recursive-compute}

The derivation [transformer-memory](derivations/transformer-memory.md) counts, term by term, what a transformer layer stores for its backward pass.
Let $s$ be the sequence length, $b$ the micro-batch, $h$ the hidden size and $a$ the number of heads.
With 16-bit activations and 1-byte dropout masks, the attention block stores $11sbh + 5as^2b$ bytes, the MLP stores $19sbh$, and the two layer norms store $4sbh$.
Together this is

$$ \text{bytes per layer} = sbh\left(34 + \frac{5as}{h}\right). $$ {#eq-transformer-memory}

That is Equation 1 of Korthikanti et al. (2022). {#c-layer-memory}
The $5as^2b$ term is the softmax output, its dropout mask and the dropout output.
These are large but cheap to recompute.
For GPT-3 and MT-NLG, they are 70% and 65% of the activations. {#c-selective-share}

## Worked example

Take $n = 9$ layers and $k = 3$ segments, with checkpoints $h_0, h_3, h_6, h_9$.
For the last segment, the backward pass rebuilds $h_7$ and $h_8$ from $h_6$ with two forwards.
At that moment it holds $h_0, h_3, h_6, h_9, h_7, h_8$: six activations, against ten with full storage.
Each segment rebuilds two activations, so the step costs $9 + 6 = 15$ layer forwards against 9.
Our run of the NumPy reference gives exactly these counts, peak 6 and 15 forwards, and gradients equal bit for bit to full storage.

The integer count has wide ties.
For $n = 9$, our enumeration gives a peak of 6 for every $k$ from 2 to 5.
The forwards fall from 16 to 13 across that range, so $k = 5$ (segments of 2, 2, 2, 2, 1) is the better choice.
For $n = 100$, the peak is 20 for every $k$ from 8 to 13.
The reference picks the largest $k$ among ties, which recomputes the least.

Recursive bisection on $n = 16$ layers holds $2 + \log_2 16 = 6$ activations and runs $16 + 8 \cdot 4 = 48$ forwards, three times the forward cost.
On $n = 1024$, our simulation gives a peak of 12 against 1025, for 6144 forwards against 1024.

## Hyperparameters

The main choice is the number of segments $k$.
A small $k$ keeps few checkpoints but long segments; a large $k$ keeps many checkpoints and short segments.
With $k = 1$ or $k = n$, the peak is $n + 1$, the same as full storage.
Choose $k$ near $\sqrt{n}$, and among equal peaks take the largest $k$.

Real layers differ in size and in cost.
Chen et al. keep the convolution output in a Conv-BatchNorm-Activation pipeline, and drop the batch normalization, activation and pooling outputs, which are cheap to recompute. {#c-chen-cheap-ops}
When the layers differ in memory cost, they allocate greedily under a per-segment budget $B$ and search over $B$. {#c-chen-greedy}
In transformers, the usual checkpoint is the input of each layer.
The selective form keeps everything except the attention scores.

## Cost

With $k$ segments, the compute is eq-segment-compute: $2n - k$ layer forwards and one backward pass, against $n$ forwards.
If a backward costs about twice a forward, a full extra forward adds about a third to the work of a step (our arithmetic).
Measured overheads are close to that.
Chen et al. report that the doubled forward cost added 30% to the runtime. {#c-chen-runtime}
Korthikanti et al. observe a 30 to 40% execution-time overhead for full recomputation of transformer layers. {#c-full-overhead}
Checkpointing the input of each transformer layer and recomputing the rest cuts the activation memory to $2sbhL$ for $L$ layers. {#c-full-recompute-memory}
Selective recomputation costs much less compute: 2.7% extra FLOPs for GPT-3 and 1.6% for MT-NLG. {#c-selective-flops}

## Evidence

A 1,000-layer residual network on ImageNet needed 7G of memory instead of 48G with the sublinear plan (Chen et al. 2016). {#c-chen-resnet}
For a four-layer LSTM unrolled over long sequences, the reduction was more than 4x over the best plan without recomputation. {#c-chen-lstm}
For recurrent networks on sequences of length 1000, the dynamic-programming policy of Gruslys et al. (2016) saved 95% of memory for about one third more time per iteration. {#c-gruslys-95}
For one layer of a 22B transformer, the time overhead was 39% with full recomputation, 7% with selective recomputation, and 4% with sequence parallelism added (Korthikanti et al. 2022). {#c-layer-time}
For the 530B and 1T models, the overhead was 2%, against 36% for full recomputation. {#c-large-overhead}
A 530B model on 2240 A100 GPUs reached a model FLOPs utilization of 54.2%, against 42.1% with full recomputation. {#c-mfu}

## Limitations and failure modes

Gradient checkpointing saves only activation memory.
Parameters, gradients and optimizer state stay where they were, and for small batches or short sequences they dominate.
The extra forward pass is a real cost: full recomputation adds about a third or more to the step time.
So when the model fits without checkpointing, the method only slows training down.
The recomputed forward must give the same values as the first one.
Randomness such as dropout masks must be replayed, and in-place updates or state changed between the two passes break the gradient.
A uniform schedule ignores the structure of the network.
When the expensive and the memory-heavy operations are different, as in attention, a uniform $k$ wastes compute.
For large transformers, a schedule that only checkpoints whole layers is also coarse: with only a few layers per device, it cannot balance memory against compute finely (Korthikanti et al. 2022, Section 5).

## Variants

- **Recursive checkpointing** applies the segment idea inside each segment, for $O(\log n)$ memory and $O(n \log n)$ compute (Chen et al. 2016, Section 4.4).
- **Optimal policies for a fixed budget.** Gruslys et al. (2016) define $C(t, m)$, the least number of forward operations for backpropagation through time of length $t$ with $m$ hidden-state slots. {#c-hsm-recurrence}
  It satisfies

  $$ C(t, m) = \min_{1 \le y < t} \big[ y + C(t - y, m - 1) + C(y, m) \big], \quad C(t, 1) = \tfrac{t(t+1)}{2}, \quad C(t, m) = 2t - 1 \text{ for } m \ge t. $$ {#eq-hsm}

  They prove $C(t, m) \le m\,t^{1 + 1/m}$. {#c-hsm-bound}
  The strategies of Chen et al. are policies of the same dynamic program, so the optimum is at least as good under any budget. {#c-hsm-covers-chen}
  Our table gives $C(1000, 10) = 4636$ forwards for 1000 steps with 10 slots, against 1999 with full storage.
- **Selective recomputation** keeps the cheap-to-store, expensive-to-compute activations of a transformer layer and recomputes only the attention scores (Korthikanti et al. 2022, Section 5).
- **Checkpointing in automatic differentiation.** Revolve plans checkpoints for the reverse mode of general programs (Griewank and Walther 2000, cited by Chen et al.).

## When to use it

Use gradient checkpointing when activation memory stops you from training the model, the batch or the sequence length you want.
Korthikanti et al. (2022) use it in large transformers together with tensor, sequence and pipeline parallelism.
Try selective recomputation of the attention scores before full recomputation, because it costs a few percent instead of a third.
When the model fits without it, a larger batch or less parallelism usually gives more throughput.
Mixed precision training saves memory in a different way: it makes each stored activation smaller, while checkpointing stores fewer of them, and the two combine.
Other ways to save activation memory are offloading to host memory and reversible layers.

## Pitfalls

> [!pitfall]
> **The recomputed forward differs from the first one.** A dropout mask that is drawn again, or a buffer that is changed in place, gives a wrong gradient with no error. The tests `test_gradients_equal_full_storage` (NumPy) and `test_equal_to_full_storage` (PyTorch) assert that the gradients are bit for bit equal to those of full storage.

> [!pitfall]
> **The checkpoint at the end of a segment is counted twice or not at all.** The backward of a segment needs both its input checkpoint and its output, so an off-by-one count of memory or compute is easy. The test `test_counts_match_formulas` runs the reference with a memory meter for every $n \le 25$ and every $k$, and compares the peak and the forward count with eq-segment-memory and eq-segment-compute.

> [!pitfall]
> **Expecting the framework to skip the last layer of a segment.** In our runs, `torch.utils.checkpoint` with `use_reentrant=False` recomputes every layer of every segment, so a step costs $2n$ layer forwards, not $2n - k$. The PyTorch test `test_equal_to_full_storage` asserts the count $2n$.

> [!pitfall]
> **The checkpointed model still saves every activation.** Code that wraps the wrong function, or keeps a reference to the inner activations, gets the extra compute with no memory gain. The test `test_saved_activations` counts the tensors that autograd saves, $n + 1$ without checkpointing and $k$ with it.

> [!pitfall]
> **One segment, or one segment per layer.** Both give a peak of $n + 1$, no better than full storage. The test `test_optimal_k_by_enumeration` checks that the chosen $k$ reaches the least peak for every $n \le 300$.

## History

Checkpointing was known in automatic differentiation before deep learning took it up: Chen et al. (2016) note that dropping intermediate results is the gradient checkpointing technique of that literature. {#c-revolve-lineage}
For that literature they cite Revolve by Andreas Griewank and Andrea Walther (ACM Transactions on Mathematical Software, 2000).
Tianqi Chen, Bing Xu, Chiyuan Zhang and Carlos Guestrin put the $\sqrt{n}$ schedule for deep networks on arXiv in April 2016, together with its recursion and an implementation in MXNet.
Two months later, Audrūnas Gruslys, Rémi Munos, Ivo Danihelka, Marc Lanctot and Alex Graves gave a dynamic program for recurrent networks that fits a given memory budget with the least recomputation.
As language models grew, recomputing each transformer layer became the default, at a cost of about a third of the step time.
Korthikanti et al. (2022) counted the activation memory of a transformer layer term by term, and found that recomputing only the attention scores, together with sequence parallelism, removes most of that cost.
