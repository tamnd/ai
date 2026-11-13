---
tag: NEW-neural-scaling-laws
kind: finding
slug: neural-scaling-laws
name: Neural scaling laws
aliases: [scaling laws, Kaplan scaling laws, Chinchilla scaling laws, compute-optimal scaling]
area: language-models/scaling-laws
level: core
summary: >-
  The test loss of a neural network falls as a power law in its parameters, its training data or its training compute, over several orders of magnitude, while the other two do not limit it.
history:
  year: 2020
  note: "Kaplan et al. (2020), arXiv 2001.08361, for language models; Hestness et al. (2017), arXiv 1712.00409, across four domains."
prereqs: [cross-entropy-loss@al, generalization-error@al, overfitting@al, linear-regression@al]
same_as:
  wikidata: Q119398888
  wikipedia: Neural_scaling_law
  arxiv: ["2001.08361", "2203.15556"]
sources:
  - {paper: "Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu and Amodei (2020)", title: "Scaling Laws for Neural Language Models", locator: "the abstract, Equation 1.1", note: "the laws for language models"}
  - {paper: "Hestness et al. (2017)", title: "Deep Learning Scaling is Predictable, Empirically", locator: "the abstract", note: "learning curves in four domains"}
  - {paper: "Henighan et al. (2020)", title: "Scaling Laws for Autoregressive Generative Modeling", locator: "the abstract"}
  - {paper: "Zhai, Kolesnikov, Houlsby and Beyer (2022)", title: "Scaling Vision Transformers", locator: "the abstract"}
  - {paper: "Hoffmann et al. (2022)", title: "Training Compute-Optimal Large Language Models", locator: "the abstract, p. 8", note: "the compute-optimal allocation and Chinchilla"}
  - {paper: "Besiroglu, Erdil, Barnett and You (2024)", title: "Chinchilla Scaling: A replication attempt", locator: "the abstract"}
  - {paper: "Pearce and Song (2024)", title: "Reconciling Kaplan and Chinchilla Scaling Laws", locator: "the abstract"}
  - {paper: "Porian, Wortsman, Jitsev, Schmidt and Carmon (2024)", title: "Resolving Discrepancies in Compute-Optimal Scaling of Language Models", locator: "the abstract"}
  - {paper: "Muennighoff et al. (2023)", title: "Scaling Data-Constrained Language Models", locator: "the abstract"}
  - {paper: "Sorscher, Geirhos, Shekhar, Ganguli and Morcos (2022)", title: "Beyond neural scaling laws: beating power law scaling via data pruning", locator: "the abstract"}
  - {paper: "Caballero, Gupta, Rish and Krueger (2023)", title: "Broken Neural Scaling Laws", locator: "the abstract"}
  - {paper: "Wei et al. (2022)", title: "Emergent Abilities of Large Language Models", locator: "the abstract"}
  - {paper: "Schaeffer, Miranda and Koyejo (2023)", title: "Are Emergent Abilities of Large Language Models a Mirage?", locator: "the abstract"}
  - {paper: "Sharma and Kaplan (2020)", title: "A Neural Scaling Law from the Dimension of the Data Manifold", locator: "the abstract"}
  - {paper: "Bahri, Dyer, Kaplan, Lee and Sharma (2024)", title: "Explaining Neural Scaling Laws", locator: "the abstract"}
claim: >-
  The test loss of a neural network falls as a power law in its parameters, its training data or its training compute, over several orders of magnitude, while the other two do not limit it.
first_reported: {work: hestness-2017-deep-learning-scaling-predictable, note: "power-law learning curves in machine translation, language modeling, image classification and speech recognition"}
replications:
  - {work: kaplan-2020-scaling-laws, note: "Transformer language models, in parameters, data and compute"}
  - {work: henighan-2020-scaling-laws-autoregressive-generative, note: "images, video, image-text and mathematics"}
  - {work: zhai-2022-scaling-vision-transformers, note: "Vision Transformers"}
  - {work: hoffmann-2022-training-compute-optimal, note: "language models, with a different compute-optimal allocation"}
scale_range:
  - {range: "language models of 70 million to over 16 billion parameters, trained on 5 to 500 billion tokens", work: hoffmann-2022-training-compute-optimal}
failures:
  - {work: besiroglu-2024-chinchilla-scaling-replication, note: "a refit of Approach 3 of Hoffmann et al. differs substantially from the reported fit"}
  - {work: sorscher-2022-beyond-neural-scaling-laws, note: "with a good pruning metric, error can fall faster than a power law in the pruned data size"}
  - {work: caballero-2023-broken-neural-scaling-laws, note: "a smoothly broken power law extrapolates better than a single power law"}
explanations:
  - {name: "the data manifold", summary: "A network that regresses on a data manifold of intrinsic dimension d gives exponents of about 4/d.", sources: [sharma-2020-neural-scaling-law-data-manifold]}
  - {name: "variance-limited and resolution-limited regimes", summary: "Scaling in data and in model size each has a variance-limited and a resolution-limited regime, four in all.", sources: [bahri-2024-explaining-neural-scaling-laws]}
---

Neural scaling laws are the empirical finding that the test loss of a neural network falls as a power law in its number of parameters, its training data and its training compute.
They were measured most closely for Transformer language models, and similar power laws appear for images, video, speech and mathematics.
The exponents depend on the data and on how scale is counted.
The laws guide how a compute budget is split between model size and data.

## In plain words

Train a family of models that differ only in size, and plot the test loss against the size on log-log axes.
The points fall close to a straight line.
The same happens when you vary the amount of data or of compute, as long as the other two are large enough not to be the limit.
A straight line on log-log axes is a power law: each tenfold step in scale multiplies the loss by the same factor.
So a lab can fit the line on small runs and predict the loss of a much larger run before it pays for it.
The line is a summary of measurements, not a law of nature.
It bends when the data run out or when the loss nears the noise floor of the data, and a pass-or-fail score can make smooth progress look like a sudden jump.

## The finding

The loss of trained deep networks often follows a power law in the size of the training set or in the number of parameters (Bahri et al. 2024). {#c-broad}
The central measurement of this page is that of Kaplan et al. (2020), for decoder-only Transformer language models trained on WebText2 with the cross-entropy loss.
For these models, the loss scales as a power law with model size, dataset size and training compute, and some trends span more than seven orders of magnitude. {#c-power-law}
The models had 768 to 1.5 billion parameters, not counting the embeddings, and the training sets had 22 million to 23 billion tokens.

Write $N$ for the number of non-embedding parameters, $D$ for the number of training tokens and $C_{\min}$ for the compute of compute-efficient training, in PF-days.
Kaplan et al. give three laws, each for the case where only one of the three quantities limits the loss:

$$ L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}} $$ {#eq-kaplan}

For models of limited size trained to convergence on large datasets, they fit $\alpha_N \approx 0.076$ (Equation 1.1). {#c-alpha-n}
For large models trained on a limited dataset with early stopping, they fit $\alpha_D \approx 0.095$ (Equation 1.2). {#c-alpha-d}
For a limited compute budget with an optimally sized model and a small enough batch, they fit $\alpha_C^{\min} \approx 0.050$ (Equation 1.3). {#c-alpha-c}
An exponent of 0.076 means that doubling $N$ multiplies the loss by $2^{-0.076} \approx 0.95$, and the constants $N_c$, $D_c$ and $C_c^{\min}$ depend on the vocabulary size and the tokenization, so they have no fundamental meaning. {#c-doubling}
Within a wide range, other details of the architecture, such as the width or the depth of the network, have minimal effects on the loss. {#c-shape}

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

Before the language model laws, Hestness et al. (2017) measured learning curves in machine translation, language modeling, image processing and speech recognition, and found that the generalization error falls as a power law in the size of the training set. {#c-hestness-domains}
They found that improvements to a model shift the error curve down but do not appear to change its exponent, and that the model size grows sublinearly with the data. {#c-hestness-shift}
Their exponents lay between -0.07 and -0.35, against -0.5 or -1 in earlier theory. {#c-hestness-range}

Henighan et al. (2020) found that the cross-entropy loss of autoregressive Transformers improves smoothly as model size and compute grow, following a power law plus a constant, in generative image modeling, video, multimodal image and text models, and mathematical problem solving. {#c-henighan}
In each of these domains, the optimal model size is a power law in the compute budget, with exponents that are nearly the same across domains. {#c-henighan-optimal-size}
Zhai et al. (2022) scaled Vision Transformers and their data up and down, and described how the error rate depends on data and compute. {#c-zhai}
Their largest model, ViT-G with two billion parameters, reached 90.45% top-1 accuracy on ImageNet, by their own report. {#c-vit-g}

Hoffmann et al. (2022) trained over 400 language models of 70 million to over 16 billion parameters on 5 to 500 billion tokens, and concluded that for compute-optimal training the model size and the number of training tokens should be scaled equally: for every doubling of model size, the number of tokens should also double. {#c-equal-scaling}
Their three estimation methods give the exponent $a$ of $N_{\mathrm{opt}} \propto C^a$, which relates the optimal number of parameters to the training compute, as 0.50, 0.49 and 0.46, where Kaplan et al. give 0.73 (Table 2). {#c-three-approaches}
The third method fits a loss with an irreducible term, $L(N, D) = E + A/N^{0.34} + B/D^{0.28}$, with $E = 1.69$, $A = 406.4$ and $B = 410.7$. {#c-parametric}

Besiroglu et al. (2024) rebuilt part of the data of Hoffmann et al. from their plots and refit the third method.
Their estimate differs substantially from the published one, and the published fit does not describe the rebuilt data and disagrees with the other two methods. {#c-besiroglu}
They also report that the equal-growth result was replicated by Anil et al. (2023), and that Bi et al. (2024) find the best ratio of tokens to parameters sensitive to the quality of the data. {#c-equal-replicated}
The equal-growth rule does not rest on the third method alone, since the first two give exponents of 0.50 and 0.49.

## Scope and conditions

**One limit at a time.**
Each law of Kaplan et al. holds only when the other quantities are not the limit.
Far beyond the sizes they studied, their laws for $L(C_{\min})$ and $L(D)$ contradict each other, and they expect the predictions to break down before that point. {#c-breakdown}
Hestness et al. describe three regions of a learning curve: a small-data region where the model does little better than a guess, a power-law region, and a region where the error saturates at an irreducible level. {#c-three-regions}
A power law fitted inside the middle region says nothing about the other two.

**How the scale is counted.**
Kaplan et al. found that the compute-optimal model size grows as $C^{0.73}$, and Hoffmann et al. found $C^{0.50}$.
Pearce and Song (2024) attribute much of this gap to two choices of the first study: it counted non-embedding parameters rather than all parameters, and it fit at small scale. {#c-pearce}
Porian et al. (2024) reproduced the Kaplan law on OpenWebText2 and RefinedWeb, traced the gap to three factors (the compute cost of the last layer, the length of the warmup, and optimizer tuning that depends on scale), and with these corrected found excellent agreement with the Chinchilla law. {#c-porian}
They also found that careful decay of the learning rate is not needed for the Chinchilla law to hold. {#c-lr-decay}
A fitted exponent therefore belongs to a protocol, not to an architecture alone: how the parameters are counted, the warmup and the tuning of the optimizer change it, and the tokenization changes the constants.

**Data that repeat.**
When unique data are scarce, Muennighoff et al. (2023) found that up to 4 epochs of repeated data change the loss very little compared with fresh data, but that with more repetition the value of added compute decays to zero. {#c-repeat}
So a law fitted on unique data does not carry over to a run that repeats a small corpus many times.

**Data that are chosen.**
Sorscher et al. (2022) show in theory that, with a good metric for which examples to drop, the error can fall faster than a power law in the size of the pruned dataset, and they observe faster than power-law scaling for ResNets on CIFAR-10, SVHN and ImageNet. {#c-pruning}
The power law in data size describes training sets drawn without such a selection.

**The form of the curve.**
Caballero et al. (2023) fit a smoothly broken power law, and report that it extrapolates considerably more accurately than other forms and can follow non-monotonic curves such as double descent and sharp bends such as those of arithmetic tasks. {#c-bnsl}

**The metric.**
The laws are about a smooth loss.
Wei et al. (2022) call an ability emergent when it is absent in smaller models and present in larger ones, so that it cannot be predicted by extrapolation from smaller models. {#c-emergent-def}
Schaeffer et al. (2023) argue that many such cases come from the metric: for a fixed task and model family, nonlinear or discontinuous metrics give apparent jumps, and linear or continuous metrics give smooth, predictable change. {#c-mirage}
The two papers disagree about whether such jumps in task scores are a property of the models or an effect of the metric, and this page does not take a side.

## Explanations

The sources of this page give two related theories of where the power laws come from and what sets their exponents.

**Regression on a data manifold.**
The claim: a network performs regression on a data manifold of intrinsic dimension $d$, and the loss falls as the network resolves that manifold more finely.
The evidence: under this view the exponent is about $4/d$ for cross-entropy and mean-squared error losses, and Sharma and Kaplan (2020) test it with teacher and student networks, with CNN image classifiers and with GPT-type language models. {#c-manifold}
The consequence: data of lower intrinsic dimension should give steeper curves.

**Four regimes.**
The claim: Bahri et al. (2024) separate a variance-limited and a resolution-limited regime, for both dataset size and model size, four regimes in total, and explain the resolution-limited regime by the same manifold picture. {#c-four-regimes}
The argument: in the variance-limited regime the exponent follows from the existence of a limit of infinite data or infinite width, and in the resolution-limited regime it depends on the data.
The consequence: a measured exponent depends on which of the four regimes the experiment is in.

## Implications

The implications below are those that the sources draw.
Kaplan et al. concluded that compute-efficient training uses very large models on a relatively modest amount of data and stops well before convergence. {#c-kaplan-allocation}
Hoffmann et al. reached a different conclusion about data: that the large language models of the time were significantly undertrained, because models had grown while the training data stayed the same. {#c-undertrained}
They tested this with Chinchilla, a 70-billion-parameter model trained on 1.4 trillion tokens with the compute budget of Gopher, which outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B) on a large range of tasks, by their own report. {#c-chinchilla}
Chinchilla reached 67.5% average accuracy on MMLU, more than 7% above Gopher. {#c-mmlu}
Hestness et al. wrote that predictable curves help to debug models, to set accuracy targets, to decide how much data to collect and to design computing systems. {#c-hestness-uses}

## History

Hestness and colleagues at Baidu Research posted "Deep Learning Scaling is Predictable, Empirically" to arXiv as 1712.00409 on 1 December 2017. {#c-hist-2017}
Jared Kaplan, Sam McCandlish and colleagues at OpenAI and Johns Hopkins University posted "Scaling Laws for Neural Language Models" as arXiv 2001.08361 on 23 January 2020. {#c-hist-2020}
Hoffmann and colleagues at DeepMind posted "Training Compute-Optimal Large Language Models", the Chinchilla paper, as arXiv 2203.15556 on 29 March 2022. {#c-hist-2022}
From 2022 the work turned to the conditions of the laws: data pruning, repeated data, the metric of emergent abilities, broken power laws, and from 2024 the reanalyses that reconcile the two compute-optimal rules.

## References
