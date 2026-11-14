---
tag: NEW-lora
kind: method
slug: lora
name: LoRA
aliases: [low-rank adaptation, LoRA adapters, low-rank adapters]
area: post-training/parameter-efficient-fine-tuning
level: core
summary: >-
  Fine-tune a pretrained network by freezing its weights and learning, for chosen weight matrices, a low-rank update BA that is added to each matrix and can be merged into it after training.
purpose: parameter-efficient fine-tuning
uses: [matrix-multiplication@m2, rank-of-a-matrix@m2]
related_to: [intrinsic-dimension@al, singular-value-decomposition@m2, quantization@m2, knowledge-distillation@al, neural-network-pruning@al, weight-decay@al, adamw@al]
hyperparameters:
  - {name: r, meaning: "the rank of the update: the number of columns of B and of rows of A", typical: "1 to 64; 256 for large shifts such as code or math", default_in: {peft: 8}}
  - {name: alpha, meaning: "the numerator of the scale s = alpha / r of the update", typical: "equal to r, or 2r at high rank", default_in: {peft: 8}}
  - {name: target modules, meaning: "the weight matrices that get an adapter", typical: "Wq and Wv in the original paper; all attention and MLP projections in later practice", default_in: {peft: "the attention projections of known architectures"}}
  - {name: learning rate, meaning: "the step size of the optimizer for A and B", typical: "about ten times that of full fine-tuning of the same model"}
  - {name: dropout, meaning: "the probability of dropout on the input of the adapter branch", typical: "0 to 0.1", default_in: {peft: 0.0}}
cost:
  train: "two extra products of width r per adapted matrix in the forward pass and three in the backward pass; no gradient and no optimizer state for the frozen weights"
  inference: "none once merged; two thin products per adapted matrix if kept apart"
history:
  year: 2021
  note: "Hu et al. (2021), arXiv 2106.09685, from Microsoft; published at ICLR 2022."
prereqs: [matrix-multiplication@m2, rank-of-a-matrix@m2, singular-value-decomposition@m2, backpropagation@al]
same_as:
  arxiv: ["2106.09685"]
sources:
  - {paper: "Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen (2021)", title: "LoRA: Low-Rank Adaptation of Large Language Models", locator: "arXiv 2106.09685v2, Sections 4 and 7, Tables 1 and 4 to 6", note: "the method, the GPT-3 results and the analysis of the update"}
  - {paper: "Aghajanyan, Zettlemoyer and Gupta (2020)", title: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", locator: "the abstract", note: "the motivation: fine-tuning has a low intrinsic dimension"}
  - {paper: "Li, Farkhoor, Liu and Yosinski (2018)", title: "Measuring the Intrinsic Dimension of Objective Landscapes", locator: "the abstract"}
  - {paper: "Houlsby et al. (2019)", title: "Parameter-Efficient Transfer Learning for NLP", locator: "p. 1", note: "adapter modules, the main earlier method"}
  - {paper: "Dettmers, Pagnoni, Holtzman and Zettlemoyer (2023)", title: "QLoRA: Efficient Finetuning of Quantized LLMs", locator: "the abstract, Section 3"}
  - {paper: "Biderman et al. (2024)", title: "LoRA Learns Less and Forgets Less", locator: "the abstract, Section 4.7", note: "the learning-forgetting trade-off and practical settings"}
  - {paper: "Liu et al. (2024)", title: "DoRA: Weight-Decomposed Low-Rank Adaptation", locator: "the abstract"}
  - {paper: "Hayou, Ghosh and Yu (2024)", title: "LoRA+: Efficient Low Rank Adaptation of Large Models", locator: "p. 1"}
  - {paper: "Kalajdzievski (2023)", title: "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", locator: "the abstract"}
  - {paper: "Shuttleworth, Andreas, Torralba and Sharma (2024)", title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence", locator: "the abstract"}
  - {url: "https://huggingface.co/docs/peft/package_reference/lora", title: "PEFT LoraConfig", locator: "the parameters r, lora_alpha, lora_dropout, use_rslora and use_dora", note: "the defaults of a widely used implementation"}
---

LoRA, low-rank adaptation, is a method for fine-tuning a pretrained network that freezes the original weights and trains, for each chosen weight matrix, an update of low rank written as the product of two thin matrices.
It cuts the trainable parameters of GPT-3 175B by about 10,000 times and the training memory by about 3 times, and once the update is merged into the weights the model runs at its original speed.
Proposed by Hu et al. at Microsoft in 2021, it became a standard way to adapt large open models, often combined with a 4-bit base model as QLoRA.
Its cost is that it learns less of a distant new domain than full fine-tuning; its gain is that it forgets less of what the model knew.

## In plain words

Fine-tuning a large pretrained model normally changes every weight, and a copy of all the weights must be stored for every task.
LoRA keeps the pretrained weights fixed.
Next to each chosen weight matrix it trains a small correction, written as the product of two thin matrices.
The correction can only change the matrix in a few directions, but that is often enough to adapt the model to a new task.
Only the thin matrices are trained and saved, so a task costs megabytes instead of gigabytes, and training needs much less memory.
After training the correction can be added into the weight matrix, so the adapted model runs exactly as fast as the original.
The price is that LoRA tends to learn less of a new domain than full fine-tuning, and in exchange it forgets less of what the model knew.

## Setting

LoRA adapts a model that has already been pretrained, most often a Transformer language model, to a downstream task or domain.
Full fine-tuning updates all parameters with an optimizer such as Adam, which keeps two extra numbers per parameter, and gives a full-size model per task.
LoRA applies to any dense layer $h = W_0 x$ with $W_0 \in \mathbb{R}^{d \times k}$, and Hu et al. (2021) apply it to the projection matrices of self-attention.
In their Transformer experiments they adapt only the attention weights and freeze the MLP modules, treating each of $W_q$, $W_k$ and $W_v$ as a single $d_\text{model} \times d_\text{model}$ matrix even though its output is split into heads. {#c-attention-only}

The method rests on a guess about the update.
Aghajanyan et al. (2020) found that fine-tuning a pretrained language model in a random subspace of a few hundred dimensions recovers most of the effect of full fine-tuning; inspired by this, Hu et al. hypothesize that the updates to the weights also have a low "intrinsic rank" during adaptation. {#c-hypothesis}

## The method

Choose the matrices to adapt and a rank $r \ll \min(d, k)$.
For each such matrix, freeze $W_0$ and add two trainable matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, so that the weight in use is $W_0 + \Delta W$ with $\Delta W = s\,BA$. {#c-lora}
The layer gives

$$ h = W_0 x + s\, B A x $$ {#eq-lora-forward}

with the scale

$$ s = \frac{\alpha}{r} . $$ {#eq-lora-scale}

$A$ starts from random Gaussian values and $B$ from zero, so the update is zero at the beginning of training. {#c-init}
The constant $\alpha$ does not change with $r$; Hu et al. set it to the first $r$ they try and do not tune it, so that changing $r$ needs less retuning of the other hyperparameters. {#c-scale}
Training then runs the usual optimizer on $A$ and $B$ only; $W_0$ gets no gradient and no optimizer state.

After training, the adapter is folded into the weight:

$$ W = W_0 + s\, B A $$ {#eq-lora-merge}

and the model is served as a plain model.
To change task, subtract $sBA$ and add another adapter.
Hu et al. also argue that LoRA generalizes full fine-tuning: with adapters on all weight matrices, trained biases and $r$ equal to the rank of the pretrained matrices, it roughly recovers the expressiveness of full fine-tuning. {#c-generalizes-ft}

## Derivation

Three facts follow from the form of the layer.
First, with $B = 0$ the update $sBA$ is the zero matrix, so at the start the adapted network computes exactly the function of the pretrained one. {#c-zero-init-start}
Second, matrix products are associative and distribute over sums, so $W_0 x + s B (A x) = (W_0 + sBA)\, x$ for every input: the merged matrix has the shape of $W_0$ and the cost of the original layer, and subtracting $sBA$ gives $W_0$ back. {#c-merge}
Third, the rank of a product is at most the rank of each factor, and $B$ has $r$ columns, so $\operatorname{rank}(sBA) \le r$. {#c-rank-bound}
The derivation `merge-and-rank` writes these steps out.

The adapter is small.
$B$ has $dr$ entries and $A$ has $rk$, so

$$ |\Theta| = r\,(d + k) $$ {#eq-lora-count}

per adapted matrix, a fraction $r(1/d + 1/k)$ of its $dk$ entries. {#c-count}
For GPT-3 175B, with $d = k = 12{,}288$ and adapters on $W_q$ and $W_v$ in all 96 layers, this is $192 \times 24{,}576\, r = 4{,}718{,}592\, r$ parameters: 4.7M at $r = 1$, 18.9M at $r = 4$ and 37.7M at $r = 8$. {#c-gpt3-counts}
At $r = 4$ in FP16 that is $37{,}748{,}736$ bytes, about 36 MiB, and one parameter in about 9,300 of the model. {#c-checkpoint-size}
The derivation `parameter-count` gives the steps.

For the gradients, take a batch $X \in \mathbb{R}^{n \times k}$, the output $Y = X W_0^\top + s X A^\top B^\top$ and $G = \partial L / \partial Y$.
Writing $dL = \operatorname{tr}(G^\top dY)$ and moving factors around the trace gives

$$ \frac{\partial L}{\partial B} = s\, G^\top X A^\top, \qquad \frac{\partial L}{\partial A} = s\, B^\top G^\top X, \qquad \frac{\partial L}{\partial X} = G\,(W_0 + sBA) . $$ {#eq-lora-grads}

{#c-grads}
Let $G_W = G^\top X$ be the gradient that full fine-tuning would use for the weight.
The adapter sees it only through the products $G_W A^\top$ and $B^\top G_W$: it follows the full gradient projected onto the current row space of $A$ and column space of $B$. {#c-grad-projection}
Because $B = 0$ at the start, $\partial L / \partial A = 0$ at the first step, so the first step moves only $B$, and $A$ starts to move from the second step on. {#c-first-step}
The derivation `lora-gradients` gives every step and a finite-difference check.

## Worked example

Take $d = 3$, $k = 4$, $r = 1$ and $\alpha = 1$, so $s = 1$, with

$$ W_0 = \begin{pmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & 0 & -1 \\ 1 & 1 & 0 & 0 \end{pmatrix}, \quad A = \begin{pmatrix} 1 & 0 & -1 & 2 \end{pmatrix}, \quad B = \begin{pmatrix} 0.5 \\ 0 \\ -1 \end{pmatrix}, \quad x = \begin{pmatrix} 1 \\ 2 \\ 0 \\ 1 \end{pmatrix} . $$

The frozen path gives $W_0 x = (1, 1, 3)$.
The adapter first projects: $Ax = 1 + 0 + 0 + 2 = 3$, one number because $r = 1$; then $B \cdot 3 = (1.5, 0, -3)$.
The output is $h = (2.5, 1, 0)$.
The merged weight is $W_0 + BA$, whose rows are $(1.5, 0, 1.5, 1)$, $(0, 1, 0, -1)$ and $(0, 1, 1, -2)$, and $W x = (2.5, 1, 0)$, the same output.
The adapter has $r(d + k) = 7$ trainable entries against 12 in $W_0$; the saving grows with the size of the matrix.

With the output gradient $(1, 1, 1)$, the gradient for $B$ is $(Ax) \cdot (1, 1, 1) = (3, 3, 3)$, the gradient for $A$ is $(B^\top (1,1,1)) \, x^\top = -0.5 \cdot (1, 2, 0, 1) = (-0.5, -1, 0, -0.5)$, and the gradient for $x$ is $W^\top (1, 1, 1) = (1.5, 2, 2.5, -2)$.
Had $B$ been zero, as at initialization, the gradient for $A$ would be zero.

## Hyperparameters

The rank $r$ sets the size and the reach of the update.
On GPT-3 175B, a rank as small as one suffices for adapting both $W_q$ and $W_v$ on WikiSQL and MultiNLI, while adapting $W_q$ alone needs a larger $r$. {#c-rank-one}
Large shifts need more: Biderman et al. (2024) recommend rank 256 on all modules for code when memory allows.

Which matrices to adapt matters as much as the rank.
With a budget of 18M parameters on GPT-3, $W_q$ and $W_v$ at $r = 4$ give 73.7% on WikiSQL and 91.3% on MultiNLI, while the whole budget in $W_q$ or $W_k$ at $r = 8$ does significantly worse. {#c-which-weights}
For code with Llama-2-7B, adapting only attention underperforms adapting the MLP modules or all modules, and most of the gain of "all" comes from the MLP. {#c-mlp-targets}

The scale $\alpha$ interacts with the rank.
With $s = \alpha / r$ and a fixed $\alpha$, the update shrinks as $r$ grows; for a rank-256 adapter on Magicoder, $\alpha = 512$ does much better than 256 or 32 at every learning rate, and $\alpha = 2r$ is crucial at high rank. {#c-alpha-2r}

The learning rate should be higher than for full fine-tuning: for instruction fine-tuning LoRA is more sensitive to it, and its preferred values are about ten times higher, often between $5 \times 10^{-5}$ and $5 \times 10^{-4}$. {#c-lr-higher}
Hu et al. train the adapters with [[al:adamw|AdamW]], whose two moment estimates per parameter then exist only for $A$ and $B$; that is much of the memory saving below.

## Cost

Memory falls the most.
For GPT-3 175B, LoRA lowers the VRAM used in training from 1.2TB to 350GB, since the frozen weights need no optimizer state, and the checkpoint of $W_q$ and $W_v$ adapters at $r = 4$ is about 35MB instead of 350GB. {#c-vram}
Compared with GPT-3 fine-tuned with Adam, the abstract puts it as 10,000 times fewer trainable parameters and 3 times less GPU memory. {#c-abstract-10000}
Hu et al. also measure a 25% speedup of training on GPT-3, because the gradient of most parameters is not computed. {#c-speedup}
On a 7B model with a standard implementation and a fixed batch size, however, LoRA tends to train slower than full fine-tuning, since the adapter adds products to every adapted layer and the saved memory is not turned into a larger batch. {#c-slower}

Inference costs nothing extra once the adapter is merged.
Adapter modules, by contrast, add layers in series; in GPT-2 medium at batch size 1 and sequence length 128 they add 20.7% (AdapterL) and 30.3% (AdapterH) to the 19.8 ms of a forward pass. {#c-adapter-latency}

## Evidence

On GPT-3 175B, LoRA with 4.7M trainable parameters reaches 73.4% on WikiSQL and 91.7% on MultiNLI-matched, and with 37.7M reaches 74.0% and 91.6%, against 73.8% and 89.5% for full fine-tuning of all 175,255.8M parameters. {#c-gpt3-table4}
More broadly, Hu et al. report that LoRA performs on par with or better than fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3, with fewer trainable parameters, a higher training throughput and no added inference latency. {#c-on-par}
Those comparisons are on natural-language understanding and generation tasks close to pretraining; the evidence on harder shifts is in Limitations.

Two lines of work explain why so few parameters suffice.
Li et al. (2018) train networks in a random subspace of growing dimension, call the dimension at which solutions first appear the intrinsic dimension, and find it smaller than one might suspect for many problems. {#c-li-intrinsic}
For fine-tuning, 200 parameters projected randomly into the full space tune RoBERTa to 90% of full performance on MRPC, {#c-intrinsic-200} and larger pretrained models tend to have a lower intrinsic dimension after a fixed number of updates. {#c-larger-lower}

The learned update is not random.
In layer 48 of GPT-3, $\Delta W$ correlates with $W$ more than a random matrix does, amplifies directions that $W$ does not emphasize rather than its top singular directions, and does so by a factor of about 21.5 at $r = 4$. {#c-amplify}

## Limitations and failure modes

LoRA learns less.
Biderman et al. (2024) fine-tune Llama-2-7B on code and math, by continued pretraining on about 20B tokens and by instruction fine-tuning on about 100K examples; in the standard low-rank settings LoRA substantially underperforms full fine-tuning on the target domain, but keeps more of the base model's performance outside it. {#c-learns-less-forgets-less}
LoRA mitigates forgetting more than weight decay or dropout do, and keeps the generations more diverse. {#c-forgets-less-than-reg}
Full fine-tuning on these domains learns perturbations of rank 10 to 100 times that of typical LoRA settings, which may explain part of the gap. {#c-full-rank}
Hu et al. make the same point as a caution: a small $r$ need not work for every task, for example if the downstream task is in another language than pretraining.

LoRA and full fine-tuning can reach similar accuracy with different weights.
The difference shows in the [[m2:singular-value-decomposition|singular value decomposition]] of the fine-tuned matrices.
LoRA weights have new high-ranking singular vectors, intruder dimensions, that full fine-tuning does not create; scaling them down improves the model of the pretraining distribution with a small loss on the task. {#c-intruder}

Merging conflicts with batching: once $BA$ is folded into $W$, one forward pass cannot serve inputs of different tasks, so multi-task serving keeps the adapters apart and pays the extra products.

## Variants

QLoRA quantizes the frozen model to 4 bits and backpropagates through it into LoRA adapters, which fits fine-tuning of a 65B-parameter model on one 48GB GPU while keeping the task performance of 16-bit fine-tuning. {#c-qlora}
Its double quantization stores the quantization constants themselves in 8 bits, which cuts their cost from 0.5 to 0.127 bits per parameter for a block size of 64. {#c-double-quant}

DoRA splits each pretrained weight into a magnitude and a direction, trains the magnitude directly and the direction with LoRA, and outperforms LoRA on LLaMA, LLaVA and VL-BART tasks. {#c-dora}
LoRA+ gives $B$ a larger learning rate than $A$ by a fixed ratio, because the same rate for both does not allow efficient feature learning at large width; it reports gains of 1% to 2% and up to about twice the speed of fine-tuning at the same cost. {#c-lora-plus}
Rank-stabilized LoRA uses $s = \alpha / \sqrt{r}$; Kalajdzievski (2023) states a proof that this is the right scale and reports that dividing by $r$ slows learning at high rank. {#c-rslora}

Adapter modules (Houlsby et al., 2019) came first: small bottleneck layers inserted into each block, which almost match full fine-tuning of BERT on GLUE with 3% of task-specific parameters per task. {#c-houlsby}
They add latency that LoRA avoids.

## When to use it

Use LoRA when memory for training or storage per task is the limit: many tasks on one base model, a model too large to fine-tune in full on the hardware at hand, or adapters swapped at serving time.
Use it when the model should keep its general abilities, since it forgets less.
Prefer full fine-tuning, or a high rank on all modules, when the target domain is far from pretraining and accuracy there matters most, as with continued pretraining on code.
Combine it with 4-bit quantization of the base model when even the frozen weights do not fit in memory.
Do not use it to make a model cheaper to run: [[al:neural-network-pruning|pruning]] and [[al:knowledge-distillation|knowledge distillation]] give a smaller model for inference, while LoRA keeps the whole base model and only makes the trained update small.

## Pitfalls

- Keeping $\alpha$ fixed while raising $r$ shrinks the update, and a larger $r$ then seems not to help. Raise $\alpha$ with $r$, or use $\alpha/\sqrt{r}$.
- Initializing both $A$ and $B$ at random makes the model differ from the pretrained one at step 0. One of them must be zero.
- Reusing the learning rate of full fine-tuning is usually too small for LoRA.
- Merging an adapter into 4-bit weights can round much of the small update away; merge into a higher-precision copy of the base weights.
- The frozen parameters must have no gradient; if they are left trainable by mistake, the memory saving disappears and the model is fully fine-tuned.
- Comparing LoRA with full fine-tuning on target accuracy alone hides the forgetting that full fine-tuning causes; measure both.

## Experiments

Two experiments test the trade-off of Biderman et al. at a scale that runs on a CPU in about a minute.
A 784-128-128-10 MLP is pretrained on the MNIST digits 0 to 4 and then fine-tuned on the digits 5 to 9 in four ways: the output layer alone, LoRA of rank 4 on both hidden layers, full fine-tuning, and full fine-tuning at a ten times smaller learning rate.
The new error is on the test images of 5 to 9, and the old error on the test images of 0 to 4, read from the logits of 0 to 4.
Over 10 seeds, `mnist-split-1k`, with 1,000 fine-tuning images, gives a new error of 8.2% for LoRA and 6.3% for full fine-tuning, and an old error of 19.6% for LoRA and 78.5% for full fine-tuning, from 2.1% before fine-tuning.
`mnist-split-5k`, with 5,000 images, gives 5.3% and 4.4% on the new digits and 12.3% and 77.7% on the old ones.
Both hold their expectations: LoRA learns less and forgets less, with 5% of the trainable parameters.
At the smaller learning rate full fine-tuning still forgets 30% to 35%, and the output layer alone forgets least but learns least.

## History

Fine-tuning every weight of a pretrained network was the default for transfer learning in NLP after 2018.
Houlsby et al. posted adapter modules as a cheaper alternative, version 2 of arXiv 1902.00751 dated 13 Jun 2019. {#c-hist-houlsby}
Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen, at Microsoft, proposed LoRA in arXiv 2106.09685, whose version 2 is dated 16 Oct 2021, and it appeared at ICLR 2022. {#c-hist-lora}
LoRA spread with open models: QLoRA, posted on 23 May 2023, made the fine-tuning of 65B-parameter models possible on a single GPU, {#c-hist-qlora} and the Hugging Face PEFT library made LoRA a common way to adapt open language and diffusion models.
From 2024, studies such as Biderman et al. and Shuttleworth et al. measured what LoRA gives up, and variants such as DoRA, LoRA+ and rsLoRA changed its scale and parametrization.
