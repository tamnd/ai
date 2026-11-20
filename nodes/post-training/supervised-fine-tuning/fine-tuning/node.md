---
tag: NEW-fine-tuning
kind: method
slug: fine-tuning
name: Fine-tuning
aliases: [supervised fine-tuning, SFT, full fine-tuning, linear probing, LP-FT, transfer learning by fine-tuning]
area: post-training/supervised-fine-tuning
level: core
summary: >-
  Adapting a pre-trained network to a new task by continuing gradient training on labelled task data, usually with a new output head, small learning rates and few epochs.
purpose: reusing what a model learned in pre-training on a large corpus to reach good accuracy on a task with far less labelled data and compute than training from scratch
uses: [adam, learning-rate-warmup, dropout, fisher-information@m2]
related_to: [lora, instruction-tuning, model-merging, bert, early-stopping, weight-decay, adamw@al]
hyperparameters:
  - {name: learning rate, meaning: "the peak step size of the optimizer; much smaller than in pre-training", typical: "BERT: 5e-5, 3e-5 or 2e-5 with Adam; ULMFiT: 0.01 for the top layer"}
  - {name: schedule, meaning: "how the rate changes over training", typical: "linear warmup over the first 10% of steps, then linear decay to 0; ULMFiT's slanted triangular schedule"}
  - {name: epochs, meaning: "passes over the task data", typical: "2 to 4 for BERT on large data; about 20 on small datasets (Mosbach et al.)"}
  - {name: which layers train, meaning: "all layers, the head only (linear probing), or an unfreezing schedule", typical: "all layers; the head first and then all (LP-FT) under distribution shift"}
  - {name: per-layer rate factor, meaning: "how much smaller each lower layer's rate is than the one above (discriminative fine-tuning)", typical: "2.6"}
  - {name: regularization, meaning: "dropout, weight decay, and any pull towards the pre-trained weights", typical: "dropout 0.1, weight decay 0.01; an EWC or L2 penalty to the pre-trained weights when forgetting matters"}
  - {name: seeds, meaning: "the random head initialization and data order", typical: "several restarts on small datasets, keeping the run that is best on validation data"}
cost:
  train: "a few epochs over the task data with gradients through the whole network: for BERT, at most an hour on one Cloud TPU, against days of pre-training; memory for the weights, gradients and two Adam moments of every parameter"
  inference: "the same as the pre-trained model plus a small head; one full copy of the weights per task"
history:
  year: 2014
  people: [Jason Yosinski, Jeremy Howard, Sebastian Ruder, Alec Radford, Jacob Devlin, Ananya Kumar, Percy Liang, Marius Mosbach, Tianyi Zhang, Jesse Dodge, James Kirkpatrick]
  note: "Fine-tuning ImageNet features for other vision tasks was standard by 2014, when Yosinski et al. measured how transferable each layer is. ULMFiT (2018) made it work for text classification with per-layer rates and gradual unfreezing; GPT and BERT (2018) fine-tuned entire transformers with a new head. Studies from 2020 to 2022 traced its instability to the optimizer and short training (Mosbach et al., Zhang et al., Dodge et al.) and its OOD losses to feature distortion (Kumar et al.)."
prereqs: [adam, learning-rate-warmup]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1801.06146", "1411.1792", "2202.10054", "2006.04884", "2006.05987", "2002.06305", "1612.00796", "1810.04805"]
  al: null
sources:
  - {paper: "Howard and Ruder (2018)", title: "Universal Language Model Fine-tuning for Text Classification", locator: "arXiv 1801.06146v5, pp. 1 to 5", note: "ULMFiT's three stages, discriminative fine-tuning, slanted triangular rates, gradual unfreezing, results"}
  - {paper: "Yosinski, Clune, Bengio and Lipson (2014)", title: "How transferable are features in deep neural networks?", locator: "arXiv 1411.1792v1, pp. 1 and 2", note: "general and specific features, freezing versus fine-tuning, co-adaptation, the lingering boost"}
  - {paper: "Kumar, Raghunathan, Jones, Ma and Liang (2022)", title: "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution", locator: "arXiv 2202.10054v1, pp. 1, 3 and 6", note: "linear probing versus fine-tuning ID and OOD, feature distortion, LP-FT"}
  - {paper: "Mosbach, Andriushchenko and Klakow (2021)", title: "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines", locator: "arXiv 2006.04884v3, pp. 1 to 3, 7 and 8", note: "the causes of instability, bias correction, the baseline recipe"}
  - {paper: "Zhang, Wu, Katiyar, Weinberger and Artzi (2021)", title: "Revisiting Few-sample BERT Fine-tuning", locator: "arXiv 2006.05987v3, pp. 1, 3 and 6", note: "BERTAdam's missing bias correction, re-initializing top layers"}
  - {paper: "Dodge, Ilharco, Schwartz, Farhadi, Hajishirzi and Smith (2020)", title: "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping", locator: "arXiv 2002.06305v1, pp. 1 and 2", note: "variance across seeds, its two sources, early stopping of bad runs"}
  - {paper: "Kirkpatrick, Pascanu, Rabinowitz, Veness and others (2017)", title: "Overcoming catastrophic forgetting in neural networks", locator: "arXiv 1612.00796v2, pp. 1 and 3", note: "elastic weight consolidation"}
  - {paper: "Devlin, Chang, Lee and Toutanova (2019)", title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", locator: "arXiv 1810.04805v2, pp. 3, 5, 13 and 14", note: "fine-tuning all parameters, cost, hyperparameter ranges"}
  - {paper: "Radford, Narasimhan, Salimans and Sutskever (2018)", title: "Improving Language Understanding by Generative Pre-Training", locator: "pp. 3 and 5", note: "the added linear head, the auxiliary language-modelling objective and its weight"}
---

Fine-tuning takes a network already trained on a large, general task and trains it further on a smaller task of interest, starting from the pre-trained weights instead of random ones.
Usually the old output layer is replaced by a new head for the task, and then either every weight is trained (full fine-tuning) or only the head is (linear probing), or something in between.
The pre-trained weights are a good starting point, so a few epochs at a small learning rate suffice, and the task needs far fewer labelled examples than training from scratch.
The difficulties are particular to the starting point: training too hard erases what pre-training learned, the new head starts random and disturbs the features while it learns, and small datasets make runs unstable from seed to seed.
Most of the recipe, from the learning rate schedule to which layers train first, exists to manage these.

## In plain words

A network trained on a lot of data has already learned features that are useful well beyond its original task: edges and textures for images, word meanings and syntax for text.
Fine-tuning keeps those features and teaches the network one more thing, using a much smaller dataset.
The new task usually needs a new output layer, which starts out knowing nothing; the rest of the network starts out knowing a lot.
So the training is gentle: small steps, few passes over the data, sometimes training the new layer alone first, sometimes letting the top layers change more than the bottom ones.
Push too hard and the network forgets its general knowledge and overfits the small dataset; push too softly and it never adapts.

## Setting

A pre-trained model has a body that maps inputs to features and an output layer for its pre-training task.
Fine-tuning attaches a new head for the target task to the body.
In GPT, the inputs are passed through the pre-trained model to obtain the final transformer block's activation, which is fed into an added linear output layer with parameters W_y to predict the label. {#c-gp-head}
In BERT, the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis. {#c-be-cls}
Then some or all of the weights are trained on the target task's labelled data with the usual loss, cross-entropy for classification.
Two popular methods for transferring a pretrained model to a downstream task are full fine-tuning, which updates all the model parameters, and linear probing, which updates only the last linear layer, the head. {#c-ku-defs}
For fine-tuning, the BERT model is first initialized with the pre-trained parameters and all of the parameters are fine-tuned using labeled data from the downstream task, so each downstream task has a separate fine-tuned model. {#c-be-all}
Compared to pre-training, fine-tuning BERT is relatively inexpensive: all of the paper's results can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the same pre-trained model. {#c-be-cheap}

Why the pre-trained features help is a question about which of them are general.
Networks trained on natural images learn first-layer features similar to Gabor filters and color blobs, which appear general rather than specific to a dataset or task, and features must eventually transition from general to specific by the last layer. {#c-yo-general}
If the target dataset is small and the number of parameters is large, fine-tuning may result in overfitting, so the features are often left frozen; if the target dataset is large or the number of parameters small, the base features can be fine-tuned to improve performance. {#c-yo-frozen}
Transferring features is negatively affected by the specialization of higher layer neurons to their original task and by optimization difficulties related to splitting networks between co-adapted neurons. {#c-yo-issues}
The second issue means a network cut between two layers and retrained on top can do worse even on the original task, because the layers above had adapted to the exact features below; fine-tuning the whole network lets them re-adapt.
The transferability of features decreases as the distance between the base task and target task increases, but transferring features even from distant tasks can be better than using random features. {#c-yo-distant}
Initializing a network with transferred features from almost any number of layers can produce a boost to generalization that lingers even after fine-tuning to the target dataset. {#c-yo-boost}

Supervised fine-tuning of a language model on instruction and response pairs, the first step of instruction tuning, is the same procedure with the language-model head kept and the loss restricted to response tokens; this page covers the procedure itself, and the instruction-tuning page covers the data.
Throughout, f(x) = vᵀBx is the two-layer linear network the theory uses: B the pre-trained features and v the head.

## The method

### The basic recipe

Replace the head, initialize it at random (or at zero), and train all parameters with Adam at a small learning rate, warming the rate up over the first steps and then decaying it linearly to zero, for a few epochs.
BERT's fine-tuning search uses learning rates (Adam) of 5e-5, 3e-5 and 2e-5 and 2, 3 or 4 epochs, and large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. {#c-be-hp}
For BERT fine-tuning, the dropout probability was always kept at 0.1. {#c-be-drop}
`warmup_linear` in the NumPy file computes the schedule, and `warmup_linear_lambda` in the PyTorch file turns it into a multiplier for `LambdaLR`.

### ULMFiT: per-layer rates and gradual unfreezing

Before ULMFiT, language models overfit to small datasets and suffered catastrophic forgetting when fine-tuned with a classifier. {#c-ul-problem}
Dai and Le also fine-tune a language model, but overfit with 10k labeled examples and require millions of in-domain documents for good performance. {#c-ul-dai}
Howard and Ruder kept the idea and changed how the fine-tuning is done.
ULMFiT has three stages: general-domain language model pretraining, target task language model fine-tuning, and target task classifier fine-tuning. {#c-ul-stages}
ULMFiT pretrains its language model on Wikitext-103, consisting of 28,595 preprocessed Wikipedia articles and 103 million words. {#c-ul-wiki}
The middle stage fine-tunes the language model itself on the target task's text, before any labels are used, so that the features match the target domain.
Three techniques control the last two stages.

**Discriminative fine-tuning.** Different layers capture different kinds of information, so they get different learning rates.
Howard and Ruder found it to work well to first choose the learning rate η^L of the last layer by fine-tuning only the last layer and using η^{l−1} = η^l/2.6 as the learning rate for lower layers. {#c-ul-discr}
`discriminative_lrs` in NumPy and `discriminative_param_groups` in PyTorch build these rates, the latter as optimizer parameter groups.

**Slanted triangular learning rates.**
Slanted triangular learning rates generally use cut_frac = 0.1, ratio = 32 and η_max = 0.01, modifying triangular learning rates with a short increase and a long decay period. {#c-ul-stlr}
The short increase moves the model quickly towards a region suited to the task; the long decay refines it.
`stlr` and `stlr_lambda` implement the schedule.

**Gradual unfreezing.**
Fine-tuning the target classifier is the most critical part: overly aggressive fine-tuning causes catastrophic forgetting, eliminating the benefit of language modeling, while too cautious fine-tuning leads to slow convergence and resultant overfitting. {#c-ul-aggr}
Gradual unfreezing first unfreezes the last layer, which contains the least general knowledge, and fine-tunes all unfrozen layers for one epoch, then unfreezes the next lower frozen layer and repeats. {#c-ul-unfreeze}
`gradual_unfreezing` lists the layers trained in each epoch and `GradualUnfreezer` sets `requires_grad` layer by layer.
The new head is trained for a full epoch before any pre-trained layer moves, so the pre-trained layers never see the large, noisy gradients of a random head.

### Linear probing, then fine-tuning

Gradual unfreezing has a two-stage special case: train the head alone with the body frozen, which is linear probing, and then train everything starting from the probed head.
On 10 distribution shift datasets, fine-tuning obtains on average 2% higher accuracy ID but 7% lower accuracy OOD than linear probing. {#c-ku-tradeoff}
Kumar et al. prove that the OOD error of fine-tuning is high when initializing with a fixed or random head, because while fine-tuning learns the head, the lower layers of the network change simultaneously and distort the pretrained features. {#c-ku-prove}
Fine-tuning updates the pretrained feature extractor B0 along the ID direction but not the perpendicular direction, which Kumar et al. call feature distortion. {#c-ku-dir}
The reason, derived below, is that in a linear network the gradient with respect to the features is a combination of the training inputs, so features of inputs unlike the training data never change while the head grows to fit the training data, and the head is shared by both.
Starting from a head that already fits the data removes most of the gradient and with it most of the distortion.
Linear probing then full fine-tuning (LP-FT) outperforms both fine-tuning and linear probing on the 10 datasets, 1% better ID and 10% better OOD than full fine-tuning. {#c-ku-lpft}
`lp_ft` in NumPy and `linear_probe_then_fine_tune` in PyTorch run both stages.

### Keeping the weights close: EWC

Instead of controlling how fast the weights move, a penalty can control how far.
Elastic weight consolidation remembers old tasks by selectively slowing down learning on the weights important for those tasks. {#c-ew-idea}
EWC minimizes L(θ) = L_B(θ) + Σ_i (λ/2) F_i(θ_i − θ*_A,i)², where L_B is the loss for task B only, F is the Fisher information and λ sets how important the old task is compared to the new one. {#c-ew-loss}
EWC approximates the posterior as a Gaussian distribution with mean given by the parameters θ*_A and a diagonal precision given by the diagonal of the Fisher information matrix. {#c-ew-post}
A weight with large Fisher information is one the old task's predictions depend on, so moving it is expensive; a weight the old task ignores is free to change.
For fine-tuning, task A is pre-training and θ*_A the pre-trained weights; with every F_i = 1 the penalty is plain L2 regularization towards the pre-trained weights rather than towards zero, which weight decay gives.
`ewc_penalty` and `ewc_grad` in NumPy, and the `EWC` class in PyTorch with an exact diagonal Fisher for a classifier, implement the penalty; the same diagonal Fisher weights the parameters in Fisher merging on the model-merging page.

### An auxiliary language-model loss

GPT kept the pre-training loss alive during fine-tuning: the objective is the task loss plus λ times the language-model loss on the same inputs, with λ = 0.5.
Including language modeling as an auxiliary objective to the fine-tuning helped learning by improving generalization of the supervised model and accelerating convergence. {#c-gp-aux}
`auxiliary_objective` computes the combination.

### Stabilizing small-data fine-tuning

On small datasets the outcome depends heavily on the random seed.
Even with the same hyperparameter values, distinct random seeds can lead to substantially different fine-tuning results. {#c-do-seeds}
Weight initialization and training data order contribute comparably to the variance of out-of-sample performance, and some weight initializations perform well across all tasks explored. {#c-do-wido}
Fine-tuning is an unstable process: training the same model with multiple random seeds can result in a large variance of the task performance. {#c-mo-var}
Mosbach et al. find that neither catastrophic forgetting nor small dataset size explains the fine-tuning instability, which is caused by optimization difficulties that lead to vanishing gradients. {#c-mo-cause}
Fine-tuning instability decomposes into optimization difficulties early in training, characterized by vanishing gradients, and differences in generalization late in training, a large variance of development set accuracy for runs with almost equivalent training loss. {#c-mo-two}
The first part traces to the optimizer.
BERTAdam differs from the original Adam algorithm in omitting a bias correction step; the change was introduced by Devlin et al. and made its way into common open source libraries, including the official implementation and huggingface's Transformers. {#c-zh-bertadam}
Bias correction reduces the original step size by a multiplicative factor √(1 − β2^t)/(1 − β1^t), which is significantly below 1 for the first iterations of training and approaches 1 as t increases. {#c-mo-bc}
Omitting the correction makes the early steps larger, several times larger at first (derived below), so the first few hundred updates are far bigger than the nominal learning rate suggests.
The setup Mosbach et al. reproduce uses dropout with probability 0.1, weight decay 0.01, 3 epochs, and, following Devlin et al., the AdamW optimizer without bias correction. {#c-mo-setup}
Mosbach et al. recommend small learning rates with bias correction to avoid vanishing gradients early in training, and increasing the number of iterations considerably to train to almost zero training loss. {#c-mo-guide}
Their baseline fine-tunes BERT using Adam with bias correction and a learning rate of 2e−5 for 20 epochs, the learning rate linearly increased for the first 10% of steps and linearly decayed to zero afterward. {#c-mo-base}
`adam` in the NumPy file takes `bias_correction` as a flag, and `bias_correction_ratio` gives the ratio of the two steps.
Zhang et al. also changed which layers start from pre-trained weights.
Zhang et al. re-initialize the pooler layers and the top L BERT Transformer blocks using the original BERT initialization, N(0, 0.02²), and find that Re-init leads to faster convergence. {#c-zh-reinit}
The top layers of BERT are specialized to its pre-training tasks, so starting them afresh can help; `reinit_top_layers` in PyTorch does this.
With the modified fine-tuning process, the impact of previously proposed methods for few-sample fine-tuning of BERT diminishes significantly. {#c-zh-diminish}

## Derivation

The full derivation is in `derivations/fine-tuning-arithmetic.md`, and `fine-tuning-arithmetic.check.py` checks every step numerically.

**Where the features move.**
For the two-layer linear network f(x) = vᵀBx and squared loss on inputs X, ∇_B L = v rᵀX/n with r the residuals, so every update to B lies in the row space of X and Bu stays at B₀u for every u orthogonal to the training inputs. {#c-ft-rowspace}
So in the linear model, fine-tuning changes only how the network represents the training inputs' span, and the head it learns is applied unchanged to every other direction.
If the linear probe fits the training data exactly, the residuals are zero, both gradients vanish, and fine-tuning started from the probed head leaves B and v unchanged. {#c-ft-lpfixed}

**A conserved quantity.**
Under gradient flow on vᵀBx with squared loss, d(BBᵀ)/dt = d(vvᵀ)/dt, so BBᵀ − vvᵀ is conserved from its value at initialization. {#c-ft-balance}
This ties the growth of the head to the growth of the features: whatever the head gains in size, the features along the data must gain too.

**The golden-ratio example.**
With B₀ = (1, 1), one input x = (1, 0) labelled 1 and the head started at 0, the second entry of B never moves, the conserved quantity gives b² − v² = 1 for the first entry b, and a zero-loss fit gives vb = 1, so b² = φ, the golden ratio: v = 1/√φ ≈ 0.786, and the prediction on x' = (0, 1), whose true label under the pretrained features is 1, is 0.786, while the linear probe predicts 1 exactly. {#c-ft-golden}
Fine-tuning fits the training point by enlarging both the feature and the head, and the enlarged head, applied to a feature that never moved, gets the unseen input wrong by about 21%, while the probe is exact.

**Schedules.**
With cut = ⌊T·cut_frac⌋, the slanted triangular rate η_max(1 + p(ratio − 1))/ratio, with p = t/cut while t < cut and p = 1 − (t − cut)/(cut(1/cut_frac − 1)) after, runs from η_max/ratio at t = 0 up to η_max at t = cut and back to η_max/ratio at t = T when T is a multiple of 1/cut_frac. {#c-ft-stlr}
With rates divided by 2.6 per layer going down, the bottom of L layers trains at η^L/2.6^{L−1}: about 17.6 times slower than the top for L = 4 and about 46 times slower for L = 5. {#c-ft-discr}
With linear warmup over the first fraction w of T steps and linear decay to 0, the peak rate is reached at step wT and the average rate over the run is half the peak. {#c-ft-warmup}

**Bias correction.**
Without bias correction, Adam's step is (1 − β1^t)/√(1 − β2^t) times the corrected one; with β1 = 0.9 and β2 = 0.999 this is √10 ≈ 3.16 at t = 1, peaks near 6.57 at t = 12, is still about 3.24 at t = 100 and 1.26 at t = 1000, and tends to 1. {#c-ft-bc}
The second moment is underestimated more than the first at small t because β₂ is closer to 1, so the uncorrected ratio m_t/√v_t is too large, not too small; the correction is a built-in warmup that BERTAdam removes.

**The EWC optimum.**
For one parameter with quadratic task loss (a/2)(θ − b)² and EWC penalty (λF/2)(θ − θ_old)², the minimum is θ = (ab + λFθ_old)/(a + λF): a precision-weighted average that tends to b as λF → 0 and to θ_old as λF → ∞. {#c-ft-ewc}

## Worked example

Take the golden-ratio example above with gradient descent instead of gradient flow.
The pre-trained features are B₀ = (1, 1), the only training input is x = (1, 0) with label 1, and the input x' = (0, 1) should also get 1.
Linear probing solves v·1 = 1, so v = 1, and predicts 1 on x', exactly right.
Full fine-tuning from v = 0 with learning rate 0.001 for 60,000 steps changes only the first entry of B, from 1 to 1.272, while v grows from 0 to 0.786; the training point is fitted, since 1.272 × 0.786 = 1, but x' gets 0.786.
LP-FT starts from v = 1, where the loss is already zero, and nothing moves.
The NumPy tests run all three.

For the schedules, with T = 1000 steps and ULMFiT's defaults, `stlr` gives 0.0003125 at step 0, 0.01 at step 100 and 0.0003125 again at step 1000; with four layers and a top rate of 0.01, discriminative fine-tuning gives the bottom layer 0.01/17.576 ≈ 0.00057.

## Hyperparameters

The learning rate matters most: too large and the pre-trained features are destroyed in the first steps, too small and the head never fits.
Ranges for BERT are given above; Mosbach et al.'s baseline uses the smallest of them for 20 epochs.
The number of epochs interacts with dataset size.
Across 450 BERT models fine-tuned on RTE, Mosbach et al. find that despite close to zero training loss, overfitting is not an issue during fine-tuning. {#c-mo-over}
On small datasets, training several seeds and keeping the one that does well on validation data is part of the recipe, and the run-to-run variation makes early stopping of poor runs worthwhile.
Worse performing fine-tuning runs can often be distinguished from better ones early in training, and a simple early stopping algorithm reduces the computational resources needed. {#c-do-early}
How many layers train, and in what order, is the choice between linear probing, full fine-tuning, LP-FT and gradual unfreezing; early stopping, weight decay and dropout are as in training from scratch.

## Cost

Full fine-tuning trains every parameter, so it needs the memory of training, not of inference: the weights, their gradients and two Adam moments, four copies of the model in all, plus activations.
It also stores one full copy of the model per task.
These two costs are what parameter-efficient methods such as LoRA reduce, by training a small number of added weights and keeping the pre-trained weights frozen and shared.
Linear probing needs one forward pass over the data to compute features and then the training of a linear model, far cheaper than either.
Time is small next to pre-training, as the BERT figures above illustrate.

## Evidence

**ULMFiT.**
On six text classification datasets, ULMFiT reduces the error of previous methods by 18-24% on the majority of datasets. {#c-ul-error}
On IMDb, with 100 labeled examples, ULMFiT matches the performance of training from scratch with 10× and, given 50k unlabeled examples, with 100× more data. {#c-ul-imdb}

**Feature distortion.**
An ImageNet pretrained ResNet-50 fine-tuned on CIFAR-10 gets 98% accuracy on CIFAR-10 (ID) but 82% on STL (OOD). {#c-ku-stl}
LP-FT changes both ID and OOD features 10× to 100× less than fine-tuning does. {#c-ku-change}

## Limitations and failure modes

**Catastrophic forgetting.** Training on the new task overwrites what the network knew; a fine-tuned model can be worse than the pre-trained one on everything except the target task, and on shifted versions of the target task.
EWC, small learning rates, early stopping and keeping the weights near their starting point all trade adaptation for retention.

**Feature distortion.** A random head fitted jointly with the body distorts the features for inputs unlike the training data, lowering accuracy out of distribution even as it raises it in distribution.

**Instability.** On a few thousand examples, runs that differ only in seed can range from strong scores to chance accuracy; some runs never leave their initial loss.

**Overfitting.** A model with hundreds of millions of parameters can memorize a small training set, although in the studies above training to near-zero loss did not hurt.

**One model per task.** Each fine-tuned model is a full copy, which is costly to store and serve for many tasks; model merging and adapters address this.

## Variants

- **Full fine-tuning** trains every weight.
- **Linear probing** trains only the head on frozen features; **feature extraction** in general trains any model on the frozen features.
- **LP-FT** probes first and then fine-tunes everything.
- **Gradual unfreezing** and **chain-thaw** train layers in stages from the top down.
- **Discriminative fine-tuning** gives lower layers smaller learning rates; layer-wise learning rate decay is the same idea.
- **Top-layer re-initialization** restarts the top blocks, which are the most specific to the pre-training task.
- **Regularized fine-tuning** adds a penalty towards the pre-trained weights, EWC or plain L2.
- **Intermediate-task fine-tuning** fine-tunes on a related labelled task before the target one.
- **Parameter-efficient fine-tuning**, such as LoRA and adapters, trains a few added parameters instead.
- **Supervised fine-tuning of language models** on demonstrations is the first stage of instruction tuning and of RLHF.

## When to use it

- When a pre-trained model for the input type exists and the target task has labelled data, starting from the pre-trained weights is the default over training from scratch.
- Prefer linear probing when the data is very small, the features are good, or robustness to distribution shift matters; prefer LP-FT when both in-distribution and shifted accuracy matter.
- Prefer full fine-tuning when the target domain differs from pre-training and there is enough data, and parameter-efficient methods when memory or the number of tasks is the constraint.
- On small datasets, use bias-corrected Adam, a small learning rate with warmup, more epochs than the defaults, and several seeds.

## Pitfalls

- Using an optimizer implementation that omits Adam's bias correction, as early BERT code did.
- Taking hyperparameters tuned on a large dataset to a small one, where three epochs may be far too few.
- Reporting a single seed on a small dataset.
- Using the pre-training learning rate, which can wipe out the pre-trained features in the first hundred steps.
- Leaving dropout or batch normalization in the wrong mode when computing frozen features for a linear probe.
- Comparing fine-tuned models only on the target distribution and missing their loss of robustness under shift.
- Fine-tuning a tokenizer-dependent model on text the tokenizer was not trained for, without checking how it is split.

## History

Starting a network from weights trained on another task is older than deep learning, but it became the default in computer vision once ImageNet-trained convolutional networks were available around 2013 and 2014, when features from them were reused for detection and other tasks.
Yosinski et al. (2014) measured layer by layer how general those features are and how much fine-tuning recovers.
In language, Dai and Le (2015) fine-tuned language models, and Howard and Ruder (2018), with ULMFiT, made fine-tuning a pre-trained language model work reliably for classification.
GPT (Radford et al. 2018) and BERT (Devlin et al. 2019) then fine-tuned whole transformers with a new head, which became the standard way to use pre-trained language models.
The instability of fine-tuning on small datasets was studied by Dodge et al. (2020), Mosbach et al. (2021) and Zhang et al. (2021), and its effect on robustness by Kumar et al. (2022).
As models grew, parameter-efficient methods took over much of fine-tuning's role, and supervised fine-tuning on demonstrations became the first step of turning a pre-trained language model into an assistant.
