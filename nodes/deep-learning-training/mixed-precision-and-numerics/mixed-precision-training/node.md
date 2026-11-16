---
tag: EPEFQ9
kind: method
slug: mixed-precision-training
name: Mixed precision training
aliases: [mixed-precision training, automatic mixed precision, AMP, FP16 mixed precision training, half-precision training]
area: deep-learning-training/mixed-precision-and-numerics
level: core
summary: >-
  Run the forward and backward passes in a 16-bit float, keep FP32 master weights for the update, and scale the loss so that small FP16 gradients do not round to zero.
purpose: efficiency
uses: [floating-point-arithmetic@pg, backpropagation@al]
related_to: [gradient-clipping@al, quantization@m2, batch-normalization@al, stochastic-gradient-descent@m2, adamw@al, weight-decay@al]
hyperparameters:
  - {name: half_dtype, meaning: "the 16-bit type of the passes: FP16, which needs loss scaling, or BF16, which has the exponent range of FP32 and usually does not", typical: "float16 or bfloat16"}
  - {name: init_scale, meaning: "the first loss scale S of the dynamic scaler, init_scale of torch.amp.GradScaler", typical: "2^16", default_in: {pytorch: 65536.0}}
  - {name: growth_interval, meaning: "the number N of steps in a row without an inf or NaN gradient after which the scale grows", typical: "2000", default_in: {pytorch: 2000}}
  - {name: growth_factor, meaning: "the factor that multiplies the scale after growth_interval good steps", typical: "2", default_in: {pytorch: 2.0}}
  - {name: backoff_factor, meaning: "the factor that multiplies the scale after a step with an inf or NaN gradient, which is skipped", typical: "0.5", default_in: {pytorch: 0.5}}
  - {name: constant_scale, meaning: "a fixed loss scale used instead of the dynamic one; Micikevicius et al. used values from 8 to 32K", typical: "8 to 32768"}
cost:
  train: "a scalar multiply of the loss, one pass over the gradients to unscale them and look for inf or NaN, and an FP32 copy of the weights, 50% more weight memory than FP32 training; the activations, which dominate memory, are halved"
  inference: "none; a model trained this way can be run in FP32 or in the 16-bit type"
history:
  year: 2017
  people: [Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu]
  note: "Posted to arXiv in October 2017 and published at ICLR 2018 by authors at Baidu Research and NVIDIA; Kalamkar et al. (2019) showed that BF16 trains without loss scaling, and Micikevicius et al. (2022) carried the scheme to the FP8 formats E4M3 and E5M2."
prereqs: [floating-point-arithmetic@pg, backpropagation@al, stochastic-gradient-descent@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1710.03740", "1905.12322", "2209.05433"]
  al: mixed-precision-training
sources:
  - {paper: "Micikevicius, Narang, Alben, Diamos, Elsen, Garcia, Ginsburg, Houston, Kuchaiev, Venkatesh and Wu (2018)", title: "Mixed Precision Training", locator: "pages 1 to 9", note: "FP32 master weights, loss scaling, FP32 accumulation, and the experiments on classification, detection, speech, translation, language modelling and GANs"}
  - {paper: "Kalamkar, Mudigere, Mellempudi et al. (2019)", title: "A Study of BFLOAT16 for Deep Learning Training", locator: "pages 1 to 8, Tables 1 to 5", note: "the BF16 format, its data flow with FP32 accumulation and master weights, and results without loss scaling"}
  - {paper: "Micikevicius, Stosic, Judd et al. (2022)", title: "FP8 Formats for Deep Learning", locator: "pages 1 to 7, Tables 1 to 5", note: "the E4M3 and E5M2 encodings, per-tensor scaling, and FP8 training and post-training quantization"}
---

Mixed precision training runs most of the arithmetic of a neural network in a 16-bit floating-point type while keeping the parts that need it in 32 bits.
Micikevicius et al. introduced it in 2017 with IEEE half precision, FP16, and three techniques: a 32-bit master copy of the weights for the update, a scale on the loss that lifts small gradients into the FP16 range, and dot products that accumulate in FP32.
With them, networks for image classification, detection, speech, translation and language modelling matched their FP32 accuracy without changes to the hyperparameters, while using about half the memory and faster half-precision hardware.
BF16, a 16-bit type with the exponent range of FP32, trains the same way without the loss scale, and the FP8 formats push the idea to 8 bits with a scale on every tensor.
The method is now the usual way to train large networks; its failures come from the narrow range and coarse spacing of the short formats, which the derivations below make exact.

## In plain words

A 16-bit number takes half the memory of a 32-bit one and, on suitable hardware, multiplies several times faster.
The price is fewer values: FP16 has no numbers between zero and about one in seventeen million, none above 65504, and only about three decimal digits.
Gradients are often tiny, so in FP16 many become zero, and tiny updates added to a much larger weight vanish in the rounding.
Mixed precision training keeps a 32-bit copy of the weights and adds the updates there.
It multiplies the loss by a large number before backpropagation, so every gradient grows by the same factor and stays above the FP16 floor, and divides it back out before the update.
It adds up long sums in 32 bits.
If a gradient overflows, the step is skipped and the number is halved; if nothing overflows for a while, the number is doubled.

## Setting

A network is trained by an optimizer step on its weights $w$ from the gradient $g = \nabla_w L$ of a loss $L$, computed by backpropagation.
Micikevicius et al. present the method as training with half-precision numbers without losing accuracy or changing hyperparameters, nearly halving memory and, on recent GPUs, speeding up arithmetic. {#c-mic-abstract}
They name three limits on the speed of any program, arithmetic bandwidth, memory bandwidth and latency, and point out that reduced precision eases the first two: fewer bits move through memory, and some processors compute faster with them. {#c-mic-limiters}
In their words, half-precision math throughput in recent GPUs was 2× to 8× that of single precision, and the smaller formats also cut the memory that training needs. {#c-mic-throughput}

**The formats.**
A binary floating-point format has a sign bit, $e$ exponent bits with bias $b = 2^{e-1} - 1$, and $m$ mantissa bits that follow an implicit leading 1.
Its four limits are

$$ \epsilon = 2^{-m}, \quad x_{\min} = 2^{2 - 2^{e-1}}, \quad x_{\mathrm{sub}} = 2^{2 - 2^{e-1} - m}, \quad x_{\max} = (2 - 2^{-m})\,2^{2^{e-1} - 1} , $$ {#eq-format-limits}

the machine epsilon, the smallest normal number, the smallest subnormal and the largest finite value. {#c-format-limits}

| format | $(e, m)$ | $\epsilon$ | smallest normal | smallest subnormal | largest |
|---|---|---|---|---|---|
| FP32 | (8, 23) | $2^{-23}$ | $2^{-126} \approx 1.18 \times 10^{-38}$ | $2^{-149} \approx 1.40 \times 10^{-45}$ | $\approx 3.4028 \times 10^{38}$ |
| FP16 | (5, 10) | $2^{-10}$ | $2^{-14} \approx 6.10 \times 10^{-5}$ | $2^{-24} \approx 5.96 \times 10^{-8}$ | 65504 |
| BF16 | (8, 7) | $2^{-7}$ | $2^{-126}$ | $2^{-133} \approx 9.18 \times 10^{-41}$ | $\approx 3.3895 \times 10^{38}$ |
| E5M2 | (5, 2) | $2^{-2}$ | $2^{-14}$ | $2^{-16}$ | 57344 |
| E4M3 | (4, 3) | $2^{-3}$ | $2^{-6}$ | $2^{-9}$ | 448 |

FP16 has normal exponents from $-14$ to $15$ and spans 40 binades from its smallest subnormal to its largest value. {#c-fp16-limits}
Micikevicius et al. note that the FP16 exponent bias centres the normal exponents on $[-14, 15]$, while gradients in practice are dominated by small magnitudes with negative exponents. {#c-mic-fp16-range}
BF16 keeps the 8 exponent bits of FP32 and cuts the mantissa to 7 bits, so it reaches down to the same $2^{-126}$ and up to within a factor of about $1 + 2^{-8}$ of the FP32 maximum. {#c-bf16-limits}
Kalamkar et al. value it for exactly this: its range is that of FP32 and conversion to and from FP32 is simple, and the range is what spares the hyperparameter tuning that FP16 needs. {#c-kal-same-range}
They describe BF16 values as truncated FP32 values with 8 bits of mantissa, counting the implicit bit, whose wide range holds small gradients without loss scaling. {#c-kal-8bit-mantissa}
Its multipliers need only 8-bit significands, which saves chip area and power while keeping the range of FP32. {#c-kal-fma}
Table 1 of Kalamkar et al. lists the three 16- and 32-bit formats with their bit layouts, limits and FP32 accumulation. {#c-kal-table1}
Two of its entries are cut rather than rounded, 3.38e38 for the BF16 maximum and 1.17e−38 for the smallest normal, and BF16 has subnormals although the table says N/A. {#c-bf16-table}

Micikevicius et al. (2022) define two 8-bit encodings: E4M3, which gives up infinities and keeps one NaN pattern to gain range, and E5M2, which follows the IEEE 754 rules for special values. {#c-fp8-encodings}
Their Table 1 gives E4M3 the bias 7 and limits 448, $2^{-6}$ and $2^{-9}$, and E5M2 the bias 15 and limits 57344, $2^{-14}$ and $2^{-16}$. {#c-fp8-table1}
The derivation number-formats gives the same numbers from the bit layouts, with 32 binades for E5M2. {#c-fp8-limits}
Without the reclaimed exponent E4M3 would stop at 240 in 17 binades; with it, the magnitudes 256 to 448 in steps of 32 and an 18th binade are added. {#c-e4m3-reclaim}
The authors give these numbers too, and explain that they kept two zeros and two NaNs, giving up the value 480, so that floating point values can still be compared and sorted as integers. {#c-fp8-reclaim}
E5M2 is FP16 with fewer mantissa bits, as BF16 is FP32 with fewer, so conversion between the two is direct, and its 32 binades are enough range for one tensor. {#c-fp8-e5m2-half}
Every E5M2 value is an FP16 value, so casting E5M2 to FP16 is exact. {#c-e5m2-in-fp16}

## The method

In the scheme of Micikevicius et al., weights, activations and gradients are stored in IEEE half precision, which has a narrower range than FP32, and three techniques keep critical information from being lost. {#c-mic-storage}
These are a single-precision copy of the weights that accumulates the gradients at each optimizer step and is rounded to half precision for the passes, loss scaling to keep small gradients, and half-precision arithmetic that accumulates into single-precision outputs before they are stored in half precision. {#c-mic-three}
None of the three changes the model or its hyperparameters, and every tensor and every operation of the forward and backward passes is in FP16. {#c-mic-no-hparams}

**FP32 master weights.**
An FP32 master copy of the weights is kept and updated with the weight gradient in the optimizer step, and each iteration runs the forward and backward passes with an FP16 copy of it. {#c-mic-master}
One step with a loss scale $S$ is

1. round the master weights $w_{32}$ to $w_{16}$;
2. run the forward pass in the 16-bit type and compute the loss $L$;
3. backpropagate $S \cdot L$, which gives the 16-bit gradients $S g$;
4. copy them to FP32 and multiply by $1/S$;
5. if any of them is inf or NaN, skip the update; otherwise apply the optimizer to $w_{32}$, clipping and weight decay included;
6. update $S$.

**Loss scaling.**
Scaling the loss computed in the forward pass before backpropagation moves the gradients into the FP16 range: by the chain rule every gradient is scaled by the same amount, and backpropagation needs no extra operations. {#c-mic-scale-loss}
The weight gradients must be divided by the scale before the update, so the updates keep their FP32 size; doing it right after the backward pass, before clipping or any other use of the gradients, leaves the clipping threshold and weight decay as they were. {#c-mic-unscale}
A constant scale can be chosen by trial, or, with gradient statistics, so that its product with the largest gradient stays below 65,504. {#c-mic-choose}
A large scale does no harm unless it overflows, and an overflow fills the weight gradients with infinities and NaNs that would irreversibly damage the weights at the update. {#c-mic-no-downside}
Overflows show up in the weight gradients, for instance when they are unscaled, and the step can then simply be skipped. {#c-mic-skip}
Micikevicius et al. left the choice of the scale to future work, suggesting a factor raised or lowered by watching the gradients for overflow and skipping the updates that overflow. {#c-mic-dynamic}

**Dynamic loss scaling.**
That is how frameworks now do it.
`torch.amp.GradScaler` starts at $S_0 = 2^{16}$; on a step with an inf or NaN gradient it skips the optimizer, multiplies the scale by 0.5 and resets its count of good steps; on a good step it adds one to the count, and when the count reaches 2000 it doubles the scale and resets the count.
The tests here check this rule against `GradScaler` on CPU, step by step, and against its update kernel `torch._amp_update_scale_`, which also leaves the scale unchanged when doubling it would overflow FP32.

**FP32 accumulation.**
Micikevicius et al. sort the arithmetic of a network into vector dot-products, reductions and point-wise operations, and treat each differently. {#c-mic-categories}
Some networks need the FP16 dot-products to accumulate their partial products into FP32, which is rounded to FP16 when written to memory; without it some FP16 models fell short of the baseline. {#c-mic-fp32-accum}
The Tensor Cores of NVIDIA Volta GPUs multiply FP16 matrices and accumulate into FP16 or FP32, where earlier GPUs had only an FP16 multiply-add. {#c-mic-tensor-cores}
Large reductions, mostly the statistics of batch normalization and the softmax, are done in FP32 while the tensors stay FP16 in memory, at no cost in speed because these layers are bound by memory bandwidth. {#c-mic-reductions}
Point-wise operations such as non-linearities and element-wise products are memory-bound too, so they can use FP16 or FP32 math. {#c-mic-pointwise}

**With BF16.**
Kalamkar et al. count three 16-bit formats used for mixed precision training, FP16, 16-bit integers and BF16, all with 16-bit inputs and 32-bit accumulators. {#c-kal-16bit}
In their data flow, GEMMs take BF16 inputs and accumulate into FP32 outputs that are rounded to BF16 for the next layer, the forward pass uses a BF16 copy of the FP32 weights, biases stay FP32, and the update is applied to the FP32 weights. {#c-kal-flow}
They emulate BF16 on FP32 hardware by keeping the top 16 bits of each FP32 number, rounded to nearest even on the bits dropped, which gives the precision and rounding of BF16 hardware. {#c-kal-quantlib}

**With FP8.**
FP8 operations are expected to produce their results in higher precision and round them to FP8 only when writing, as FP16 and BF16 matrix units produce FP32. {#c-fp8-higher-out}
Values are multiplied by a scale before the cast to FP8 to move them into its range, as loss scaling does for FP16; some networks need one scale per tensor, chosen so that the largest magnitude lands near the largest FP8 value, and overflows saturate to that value. {#c-fp8-scaling}
They are unscaled by the inverse factor after the conversion back or after the matrix product, once per dot-product, which costs little. {#c-fp8-unscale}
E4M3 is recommended for weights and activations and E5M2, with its wider range, for gradients; some networks train with one of them, others need both. {#c-fp8-recommend}

## Derivation

The four derivations state their assumptions and are checked by scripts; the results are these.

**Underflow.**
With round to nearest even, FP16 turns a value into zero exactly when its magnitude is at most $2^{-25}$, and rounds magnitudes between $2^{-25}$ and $2^{-24}$ up to $2^{-24}$. {#c-underflow-threshold}
This sharpens the statement of Micikevicius et al. that anything below $2^{-24}$ becomes zero in FP16, which they give as one reason for the master weights; in their Mandarin speech model about 5% of the weight gradients had exponents below $-24$. {#c-mic-underflow-reason}
The values that survive below $2^{-14}$ are subnormals, spaced $2^{-24}$ apart, so they carry a relative error of up to one half, where a normal number carries at most $2^{-11}$. {#c-subnormal-error}

**Lost updates.**
An update that is representable can still vanish when it is added to a much larger weight:

$$ \frac{|w|}{|u|} > 2^{m+2} \;\Rightarrow\; \mathrm{fl}(w + u) = w , \qquad \frac{|w|}{|u|} < 2^{m+1} \;\Rightarrow\; \mathrm{fl}(w + u) \ne w . $$ {#eq-swamping}

For FP16 an update is always lost when the weight is more than 4096 times larger, never when it is less than 2048 times larger, and in between it depends on where the weight sits in its binade. {#c-swamping-bounds}
Micikevicius et al. give the second reason for the master weights in the same terms: aligning the binary points shifts the update right, which can lose it when the weight is at least 2048 times larger, because FP16 has 10 mantissa bits, and loses it beyond recovery when the ratio is larger. {#c-mic-ratio-reason}
Taken exactly, 2048 is where losses can begin, and above 4096 they are certain; at $w = 1$ an update of $2^{-11}$ is a tie and is lost, so the threshold is reached already at 2048.
For BF16 the two ratios are 512 and 256, so BF16 weights lose updates at ratios eight times smaller than FP16 weights do, which is why BF16 training also keeps FP32 master weights. {#c-swamping-bf16}
A thousand updates of $2^{-12}$ to a weight of 1 leave an FP16 weight at exactly 1 and move an FP32 weight to $1.244140625$. {#c-master-example}

**Loss scaling.**
Backpropagating $S \cdot L$ multiplies every gradient by $S$, and for $S = 2^k$ a gradient $g$ is stored as a nonzero finite FP16 number exactly when

$$ \frac{2^{-25}}{S} < |g| < \frac{65520}{S} , $$ {#eq-scale-window}

since 65520 is the midpoint above 65504 where rounding reaches infinity: the scale slides the window down by $k$ binades. {#c-scale-window}
A power of two only changes the exponent, so FP16 backpropagation of the scaled loss gives exactly $S$ times the FP16 gradients of the unscaled loss, as long as nothing overflows or falls out of the normal range. {#c-scale-chain-rule}
Where $S|g|$ is in the normal range, the unscaled value is within a relative $2^{-11}$ of $g$ however small $g$ was. {#c-scale-precision}
Unscaling in FP32 is then exact: an FP16 value times $2^{-k}$ is an exact FP32 number for every $k \le 102$. {#c-unscale-exact}
Scaling by 8 lifts $[2^{-27}, 2^{-24})$ to $[2^{-24}, 2^{-21})$, which FP16 keeps, while unscaled everything in that range up to $2^{-25}$ would be zero. {#c-scale-by-8}
For a known largest gradient $G$ the largest safe power of two is $S = 2^{\lfloor \log_2(65504/G) \rfloor}$. {#c-largest-safe-scale}

**The dynamic scale.**
Under the halve-on-overflow, double-after-$N$ rule, the scale at step $t$ is

$$ S_t = S_0\, 2^{G_t - B_t} , $$ {#eq-dyn-scale}

with $B_t$ the skipped steps so far and $G_t$ the completed blocks of $N$ good steps within the runs between them. {#c-dyn-formula}
If every scale from some threshold $T$ up overflows, a start at $S_0 \ge T$ costs $\lfloor \log_2(S_0/T) \rfloor + 1$ skipped steps and ends at the largest $S_0 2^j$ below $T$, which is at least $T/2$. {#c-dyn-start}
After that the rule cycles through $N$ good steps and one skipped step, so it skips one step in $N + 1$. {#c-dyn-cycle}
Starting below $T$, it skips nothing for $N \lceil \log_2(T/S_0) \rceil$ steps while the scale doubles every $N$ steps. {#c-dyn-growth}
With PyTorch's $S_0 = 2^{16}$ and $N = 2000$ and a threshold of 1000, the first 7 steps are skipped, the scale settles at 512, and one step in 2001 is skipped after that. {#c-dyn-defaults}

## Worked example

Take a gradient $g = 10^{-8}$ in FP16, below the zero threshold $2^{-25} \approx 2.98 \times 10^{-8}$.

| loss scale $S$ | $S g$ | stored in FP16 | unscaled in FP32 | relative error |
|---|---|---|---|---|
| $1$ | $1.0 \times 10^{-8}$ | $0$ | $0$ | 1 |
| $2^{10}$ | $1.024 \times 10^{-5}$ | $1.0252 \times 10^{-5}$ (subnormal) | $1.00117 \times 10^{-8}$ | $1.2 \times 10^{-3}$ |
| $2^{16}$ | $6.5536 \times 10^{-4}$ | $6.5517 \times 10^{-4}$ (normal) | $0.999717 \times 10^{-8}$ | $2.8 \times 10^{-4}$ |

Without a scale the gradient is gone; $2^{10}$ brings it back as a subnormal with a coarse error, and $2^{16}$ puts it in the normal range, where the error is below $2^{-11} \approx 4.9 \times 10^{-4}$.
With $S = 2^{16}$ the window of eq-scale-window is $(4.5 \times 10^{-13}, 0.99976)$: any gradient of magnitude 1 or more would overflow, and the dynamic scaler would back off.

Now the update. A weight of 1 with an update of $10^{-4}$ has ratio $10^4 > 4096$, so in FP16 $1 + 10^{-4}$ rounds back to 1, while the FP32 master weight becomes $1.0001$.

Finally the scale itself, with a start of 4 and a growth interval of 3, over eleven steps of which the 5th, 9th and 10th overflow:

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overflow | | | | | yes | | | | yes | yes | |
| scale used | 4 | 4 | 4 | 8 | 8 | 4 | 4 | 4 | 8 | 4 | 2 |

The scale doubles after steps 1 to 3 and again after steps 6 to 8, halves at each overflow, and ends at 2; eq-dyn-scale gives $4 \cdot 2^{2 - 3} = 2$.
`torch.amp.GradScaler("cpu", init_scale=4, growth_interval=3)` uses the same scales, and both implementations here are tested against this table.

## Hyperparameters

The main choice is the 16-bit type.
FP16 needs a loss scale; BF16 has the range of FP32 and usually needs none, at the price of three fewer mantissa bits.
For FP16 the dynamic scale has four settings, with the defaults of `torch.amp.GradScaler`: `init_scale` $2^{16}$, `growth_factor` 2, `backoff_factor` 0.5 and `growth_interval` 2000.
The start matters little, since a scale that is too large costs only a few skipped steps, about $\log_2(S_0/T)$ by the derivation; a short growth interval wastes more steps on overflows, one in $N + 1$ at steady state.
Micikevicius et al. trained with constant scales between 8 and 32K, and many of their networks needed none. {#c-mic-constants}
FP8 replaces the single loss scale by one scale per tensor, set from the largest magnitude in the tensor.

## Cost

The FP32 copy of the weights adds 50% to the memory of the weights, but activations dominate the memory of training and are stored in 16 bits, so the total is roughly halved. {#c-mic-memory}
Each step adds a multiply of the loss, a pass over the gradients that unscales them and looks for inf or NaN, and the copy from the master weights to the 16-bit model.
With dynamic scaling a fraction of the steps, one in $N + 1$ at steady state, is computed and thrown away.
Speed depends on the hardware: the gains come from 16-bit matrix units and from moving half the bytes.

## Evidence

**Micikevicius et al. (2018).**
On ILSVRC12, every network matched or slightly beat its FP32 top-1 accuracy with the same hyperparameters, from 56.77% against 56.93% for AlexNet to 75.92% against 76.04% for ResNet-50. {#c-mic-ilsvrc}
These classifiers needed no loss scaling, only the FP32 master weights. {#c-mic-ilsvrc-noscale}
In detection, Faster R-CNN reached 69.1% mAP in FP32, 68.6% without and 69.7% with loss scaling, while SSD went from 76.9% in FP32 to divergence without loss scaling and 77.1% with it. {#c-mic-detection}
The activation gradients of SSD show why: much of the FP16 range went unused while many values fell below it and became zero. {#c-mic-ssd-hist}
In their histogram 2% of the values lie in $[2^{-34}, 2^{-32})$, 2% in $[2^{-24}, 2^{-23})$, and 67% are zero. {#c-mic-ssd-fig3}
A scale of 8 was enough for SSD, from which the authors conclude that gradients below $2^{-27}$ did not matter to this model but those in $[2^{-27}, 2^{-24})$ did. {#c-mic-ssd-scale}
For the master weights, a Mandarin speech model trained for 20 epochs on about 800 hours matched FP32 with an FP32 master copy and lost 80% relative accuracy when the FP16 weights were updated directly. {#c-mic-mandarin-master}
DeepSpeech 2 models of 115 and 215 million parameters were trained on 6000 hours of English and 2600 hours of Mandarin with Nesterov SGD and unchanged hyperparameters. {#c-mic-speech-setup}
Their character error rates fell from 2.20 to 1.99 and from 15.82 to 15.01 in mixed precision, which the authors read as a hint that half-precision storage may regularize. {#c-mic-speech-cer}
For English to French translation with LSTMs on WMT15, mixed precision matched FP32 with loss scaling and was slightly worse without it. {#c-mic-translation}
The bigLSTM language model on the 1 billion word benchmark diverged from FP32 after 300K iterations without loss scaling and matched it with a scale of 128. {#c-mic-biglstm}
A DCGAN for faces gave comparable images without loss scaling. {#c-mic-dcgan}
DeepBench operations on Volta ran 2 to 6 times faster than in FP32 when bound by memory or arithmetic bandwidth, less when bound by latency. {#c-mic-deepbench}

**Kalamkar et al. (2019).**
Across image classification, speech, translation, GANs and recommendation, emulated BF16 reached the FP32 results in the same number of iterations without changing hyperparameters. {#c-kal-abstract}
They contrast this with FP16, which needs a tuned loss scale, and 16-bit integers, which need block quantization with scale factors per block. {#c-kal-fp16-tuning}
FP16 lacks the range for error gradients, they argue, and while a constant scale works for some feed-forward networks, recurrent ones need a more elaborate and often iterative search for it. {#c-kal-rnn-scale}
BF16 avoids the need for loss scaling. {#c-kal-no-loss-scaling}
AlexNet reached 57.2% top-1 in BF16 against 57.4% in FP32, and ResNet-50 the same 74.7% in both, with 75.7% for the BF16 model using global statistics. {#c-kal-cnn}
Recurrent networks are known to be more demanding of range and more sensitive to half precision. {#c-kal-rnn-sensitive}
GNMT scored 29.3 BLEU in both formats on German to English, and 18.3 in BF16 against 17.1 in FP32 on Vietnamese to English. {#c-kal-gnmt}
DC-GAN scored an inception score of 2.06 in BF16 against 1.97 in FP32. {#c-kal-gans}
SR-GAN reached a PSNR of 26.1415 in BF16 against 26.1749 in FP32. {#c-kal-srgan}
For recommendation models, where a log-loss difference of 0.001 is too much, BF16 with an FP32 master copy and round to nearest matched FP32, while truncation was about 0.02% worse. {#c-kal-recsys}
The log losses were 0.44372 in both FP32 and rounded BF16 for the Deep & Cross Network, and 0.44393 with truncation. {#c-kal-table5}
On real BF16 instructions, emulated bit-accurately, ResNet-50 reached 75.62% top-1. {#c-kal-avx512}

**Micikevicius et al. (2022).**
FP8 training matched 16-bit training on image and language tasks with unchanged hyperparameters, for models up to 175B parameters. {#c-fp8-abstract}
These runs simulated FP8 by rounding the inputs of every matrix product to FP8, with scaling and saturation, and computing in 16 bits. {#c-fp8-simulated}
On ImageNet all networks but MobileNet v2 stayed within run-to-run variation, ResNet-50 at 76.76% against 76.71%; MobileNet v2 lost 0.6 points. {#c-fp8-imagenet}
Translation BLEU stayed within 0.2 of the baselines for GNMT and the Transformers. {#c-fp8-translation}
Language-model perplexities stayed within noise, 6.68 against 6.65 for GPT 175B at 75% of training. {#c-fp8-lm}
In post-training quantization, E4M3 kept BERT and GPT accuracy close to 16 bits where int8 lost much of it. {#c-fp8-ptq}

## Limitations and failure modes

**FP16 loses small gradients.**
Without a loss scale, everything at or below $2^{-25}$ becomes zero, and some networks, such as SSD and bigLSTM, then diverge, as the Evidence section reports.
The tests `test_fp16_without_scaling_loses_the_gradients` and `test_fp16_without_loss_scaling_does_not_learn_small_gradients` train a regression whose loss has a small weight: in FP16 it does not move, with a dynamic scale it learns.

**16-bit weights lose small updates.**
Updating the weights in FP16 or BF16 drops every update more than $2^{m+2}$ times smaller than the weight, so a long run of small learning-rate steps can leave a weight where it was; the FP32 master copy is what prevents this.

**Overflow and skipped steps.**
A dynamic scale is found by overflowing, so some steps are always thrown away, and a gradient that is inf or NaN for a reason other than the scale, a real divergence, looks the same: the scale keeps halving and the steps keep being skipped.
For FP8, Micikevicius et al. (2022) advise against skipping updates on overflow, because with its narrow range overflows would be so frequent that too many updates would be skipped. {#c-fp8-no-skip}

**One scale is not always enough.**
A single scale shifts every tensor by the same amount, so it cannot help when the important values of different tensors do not fit one window.
Micikevicius et al. (2022) found that casting the residual connections of a 1.3B GPT to E4M3 left no single exponent bias that worked, perplexity 12.59 at best against 10.19, while per-tensor scales gave 10.29 and 10.44. {#c-fp8-residual}

**Not everything should be 16-bit.**
Long sums, normalization statistics and softmax lose accuracy in 16 bits; frameworks keep lists of operations to run in FP32 for this reason.

## Variants

**BF16 mixed precision** keeps the master weights and FP32 accumulation and drops the loss scale, as Kalamkar et al. show.
`torch.autocast("cpu")` defaults to BF16, and the test `test_bf16_learns_without_scaling` trains the same regression in BF16 with no scale.

**Constant loss scaling** fixes $S$ in advance, from experience or from gradient statistics, as above.

**FP8 training** scales each tensor by its own factor; the exponent biases are kept at 7 and 15 and the scale is left to software, because a scale can be any real number while a programmable bias only gives powers of two. {#c-fp8-bias}
Converting inf or NaN to E4M3 should give NaN, so that the overflow checks of automatic mixed precision still see an overflow. {#c-fp8-special}

**Pseudo FP16** stores tensors in FP16 and computes in FP32, which Micikevicius et al. used on GPUs without Tensor Cores to emulate them.

Mixed precision changes the number formats, not the optimizer: stochastic gradient descent, Adam and AdamW run unchanged on the FP32 master weights.
AdamW keeps its two moment estimates in FP32 as well, so its optimizer state, twice the size of the weights, is not halved.
Gradient clipping is a separate step that bounds the norm of the gradient; under loss scaling it has to see the unscaled gradients.

## When to use it

Use it to train any network of real size on hardware with fast 16-bit units; it saves memory and time and, with the three techniques, costs no accuracy in the experiments above.
Prefer BF16 where the hardware supports it: it needs no loss scale, and its coarser precision rarely matters with FP32 master weights and accumulation.
With FP16, use a dynamic loss scale rather than a fixed one, and unscale before clipping.
Keep reductions, normalization statistics, softmax and the optimizer state in FP32.
FP8 is for hardware that has it, and needs per-tensor scales.

## Pitfalls

> [!pitfall]
> **FP16 master weights.** `GradScaler` expects FP32 parameters: given FP16 parameters, `unscale_` raises "Attempting to unscale FP16 gradients." Keep the model in FP32 under `autocast`, or keep a separate FP32 master copy as `MasterWeights` does. The test `test_grad_scaler_refuses_fp16_gradients` checks the error, and `test_master_weights_match_grad_scaler_step_for_step` checks that the hand-written loop agrees with `GradScaler` bit for bit, skipped steps included.

> [!pitfall]
> **Clipping scaled gradients.** Clipping before unscaling applies the threshold to $S$ times the gradients and clips them far too hard. Call `GradScaler.unscale_` first. The test `test_clip_after_unscaling` shows both orders.

> [!pitfall]
> **E4M3 casts that saturate.** On CPU, PyTorch casts inf and values above 448 to 448 in `float8_e4m3fn`, while Micikevicius et al. (2022) ask for NaN on infinities so that overflows stay visible. An overflow check after such a cast sees nothing. E5M2 keeps the infinity. The test `test_torch_e4m3_cast_saturates` checks this.

> [!pitfall]
> **A scale that grows forever.** With few overflows the scale doubles every $N$ steps. `GradScaler` leaves it unchanged when doubling would overflow FP32, and so do both implementations here, checked by `test_scaler_does_not_grow_to_inf` and `test_update_matches_the_kernel`.

> [!pitfall]
> **Reading "below $2^{-24}$ becomes zero" literally.** Values between $2^{-25}$ and $2^{-24}$ round up to $2^{-24}$; only those at or below $2^{-25}$ vanish. `test_fp16_underflow_threshold` and `test_underflow_fp16_and_bf16` check the threshold, and BF16 keeps all of them.

## History

Paulius Micikevicius and Sharan Narang, with coauthors at Baidu Research and NVIDIA, posted "Mixed Precision Training" as arXiv 1710.03740 in October 2017, and it appeared at ICLR 2018. {#c-hist-mic}
BF16 began in DistBelief and TensorFlow as a storage format that halved the traffic of weights and activations between machines. {#c-hist-bf16-origin}
Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi and coauthors at Intel Labs and Facebook posted "A Study of BFLOAT16 for Deep Learning Training", arXiv 1905.12322, in 2019. {#c-hist-kal}
Micikevicius and coauthors at NVIDIA, Arm and Intel posted "FP8 Formats for Deep Learning", arXiv 2209.05433, in 2022. {#c-hist-fp8}
