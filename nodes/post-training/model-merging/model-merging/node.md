---
tag: NEW-model-merging
kind: method
slug: model-merging
name: Model merging
aliases: [weight averaging, model soups, task arithmetic, weight-space ensembling, parameter averaging]
area: post-training/model-merging
level: core
summary: >-
  Combining fine-tuned models that share an architecture and a starting point by arithmetic on their weights (averaging, task vectors, Fisher weighting) into one model with no extra inference cost.
purpose: getting much of the benefit of an ensemble, or of several separately fine-tuned skills, in a single model of the original size, without retraining on the combined data
uses: [mode-connectivity, instruction-tuning, fisher-information@m2, bernoulli-distribution@m2]
related_to: [mode-connectivity, lora, dropout, ensemble-learning@al, permutation-matrix@m2]
hyperparameters:
  - {name: which models, meaning: "the fine-tuned checkpoints to combine, all from one pre-trained model", typical: "all runs of a hyperparameter sweep (soups), or one checkpoint per task (task arithmetic, TIES, DARE)"}
  - {name: mixing coefficients, meaning: "the weights of the average, or the scale lambda of the summed task vectors", typical: "uniform 1/n; WiSE-FT alpha = 0.5; task arithmetic lambda tuned on validation data, about 0.4; TIES lambda = 1"}
  - {name: trim fraction, meaning: "the share k of each task vector's entries kept by magnitude in TIES-merging", typical: "top 20%"}
  - {name: drop rate, meaning: "the probability p of zeroing each delta entry in DARE, with the rest scaled by 1/(1 - p)", typical: "0.9; up to 0.99 for very large models; lower when deltas are large"}
  - {name: Fisher estimate, meaning: "how many examples estimate each diagonal Fisher, and whether the expectation over labels is exact", typical: "a few thousand examples; exact over the classes when there are few"}
  - {name: capture schedule, meaning: "for SWA, when snapshots of one run are averaged", typical: "once per epoch at a constant learning rate, or at each cycle's lowest learning rate"}
cost:
  train: "no gradient steps for averaging, task arithmetic, TIES and DARE: a few passes over the weights; Fisher merging adds about one training pass over N examples per model; greedy soups add one validation run per candidate"
  inference: "the same as one of the merged models, against n forward passes for an output ensemble"
history:
  year: 2018
  people: [Pavel Izmailov, Andrew Gordon Wilson, Michael Matena, Colin Raffel, Mitchell Wortsman, Ludwig Schmidt, Gabriel Ilharco, Prateek Yadav, Mohit Bansal, Le Yu, Samuel K. Ainsworth]
  note: "Izmailov et al. (2018) averaged the weights along one SGD trajectory. Matena and Raffel (2021) merged separately fine-tuned models with Fisher weights; WiSE-FT and model soups (2021, 2022) averaged fine-tuned CLIP models; task arithmetic (2022) treated the differences from the pre-trained model as vectors to add and negate; TIES-merging (2023) and DARE (2023) reduced the interference between them. Git Re-Basin (2022) aligned units so that independently trained models could be merged."
prereqs: [mode-connectivity, instruction-tuning, fisher-information@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["2203.05482", "2109.01903", "1803.05407", "2212.04089", "2111.09832", "2306.01708", "2311.03099", "2209.04836", "1912.05671"]
  al: null
sources:
  - {paper: "Wortsman, Ilharco, Gadre, Roelofs and others (2022)", title: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", locator: "arXiv 2203.05482v3, pp. 1 to 3, 6 and 8", note: "uniform, greedy and learned soups, the CLIP and ViT-G results, error barriers, calibration"}
  - {paper: "Wortsman, Ilharco, Kim, Li and others (2022)", title: "Robust fine-tuning of zero-shot models", locator: "arXiv 2109.01903v3, pp. 1 to 3 and 6", note: "WiSE-FT, its gains under distribution shift, the linear-classifier case, alpha = 0.5"}
  - {paper: "Izmailov, Podoprikhin, Garipov, Vetrov and Wilson (2018)", title: "Averaging Weights Leads to Wider Optima and Better Generalization", locator: "arXiv 1803.05407v3, pp. 1, 2, 4 and 5", note: "stochastic weight averaging, its running update and batch-norm pass, flat regions"}
  - {paper: "Ilharco, Ribeiro, Wortsman, Gururangan and others (2023)", title: "Editing Models with Task Arithmetic", locator: "arXiv 2212.04089v3, pp. 1 to 4, 7 and 8", note: "task vectors, negation, addition and analogies, orthogonality, limitations"}
  - {paper: "Matena and Raffel (2022)", title: "Merging Models with Fisher-Weighted Averaging", locator: "arXiv 2111.09832v2, pp. 2 to 5", note: "isotropic and Fisher merging, the diagonal Fisher estimate, numerical fallback, unmergeable heads"}
  - {paper: "Yadav, Tam, Choshen, Raffel and Bansal (2023)", title: "TIES-Merging: Resolving Interference When Merging Models", locator: "arXiv 2306.01708v2, pp. 1 and 3 to 8", note: "the two sources of interference, trim, elect sign and disjoint merge, results and default settings"}
  - {paper: "Yu, Yu, Yu, Huang and Li (2024)", title: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch", locator: "arXiv 2311.03099v3, pp. 1 to 3 and 5", note: "DARE, the rescaling argument, tolerated drop rates, merging decoder language models"}
  - {paper: "Ainsworth, Hayase and Srinivasa (2023)", title: "Git Re-Basin: Merging Models modulo Permutation Symmetries", locator: "pp. 3, 4 and 7", note: "permutation classes, the loss barrier, weight matching as a linear assignment problem"}
  - {paper: "Frankle, Dziugaite, Roy and Carbin (2020)", title: "Linear Mode Connectivity and the Lottery Ticket Hypothesis", locator: "p. 2", note: "stability to SGD noise early in training"}
---

Model merging builds one network out of several by doing arithmetic on their weights rather than on their outputs.
The models have the same architecture and, in nearly every method that works, were fine-tuned from the same pre-trained checkpoint.
The simplest merge averages them; task arithmetic adds their differences from the pre-trained model; Fisher merging weights each parameter by how much each model's predictions depend on it; TIES-merging and DARE prune and rescale the differences before adding them so that they interfere less.
The result runs at the cost of a single model, which is the point: an ensemble of n models costs n forward passes, a merged model costs one.
Merging works when the models lie in one low-loss region of weight space, which fine-tuning from a shared start usually ensures and training from scratch usually does not.

## In plain words

Two copies of a model fine-tuned from the same starting point usually end up close to each other in weight space, and the straight line between them stays good.
Averaging their weights then gives a model that is often better than either, much as averaging their predictions would, but without having to run both.
The same idea applies to skills: subtract the original weights from a model fine-tuned for a task and the difference, a task vector, can be added to another model to give it the skill, scaled down to give it less, or subtracted to remove a behaviour.
Adding several task vectors at once tends to work less well the more of them there are, because they pull some weights in opposite directions; the newer methods decide, weight by weight, which pull to follow.
None of this works for unrelated models trained from different random starts, whose hidden units are arranged in different orders, unless the units are first matched up.

## Setting

Merging, averaging the parameters of models, is generally only feasible when the models being averaged share a common architecture and initialization. {#c-mr-shared}
The reason is the geometry of fine-tuning.
Wortsman et al. observe that fine-tuned large pre-trained models often appear to lie in a single low error basin, and that averaging the weights of multiple models fine-tuned with different hyperparameter configurations often improves accuracy and robustness. {#c-ws-basin}
Neyshabur et al. found that fine-tuning the same model twice end to end yields two solutions connected by a linear path in weight space along which error remains low, known as linear mode connectivity. {#c-wf-lmc}
This is the property studied in mode connectivity: two networks are linearly connected when the loss does not rise along the straight line between their weights.
Even runs that share only the first part of training land in one basin.
Networks trained from one point with different samples of SGD noise become stable early in training, at 3% of training for ResNet-20 on CIFAR-10 and 20% for ResNet-50 on ImageNet, after which the outcome of optimization is determined to a linearly connected minimum. {#c-fr-stable}
Fine-tuning from a pre-trained model is the extreme case of a shared start, which is why merging is mostly a post-training technique, applied to models after instruction tuning or task-specific fine-tuning.
Independently trained networks are different.
For any trained weights there is an entire equivalence class of functionally equivalent weight assignments, obtained by permuting units, and which element training converges to is determined only by the random seed. {#c-ai-perm}
Two such networks can compute almost the same function with their hidden units in a different order, and averaging unit 3 of one with unit 3 of the other averages two unrelated features; the hidden units are matched by a permutation matrix, and only after aligning them does averaging make sense.

Throughout, θ_pre is the pre-trained weights, θ_1, ..., θ_n are fine-tuned models, and τ_t = θ_t − θ_pre is the task vector of model t.
Parameters that only some models have, such as a task-specific classifier, stay outside the merge.

## The method

### Averaging: soups, WiSE-FT and SWA

A model soup averages the weights of models from a fine-tuning hyperparameter sweep.
Unlike a conventional ensemble, a model soup averages many models without incurring any additional inference or memory costs. {#c-ws-cost}
The uniform soup averages all of them.
The greedy soup sorts the models in decreasing order of validation set accuracy and adds each model as a potential ingredient, keeping it only if performance on a held-out validation set improves, so on the held-out validation set the greedy soup can be no worse than any individual model. {#c-ws-greedy}
The learned soup optimizes model interpolation weights by gradient-based minibatch optimization and requires simultaneously loading all models in memory. {#c-ws-learned}
`greedy_soup` in the NumPy file implements the greedy recipe for any scoring function, and `uniform_soup` the uniform average.

WiSE-FT interpolates between the pre-trained model and one fine-tuned model.
WiSE-FT linearly interpolates between the zero-shot and fine-tuned models with a mixing coefficient α ∈ [0, 1], θ_α = (1 − α)·θ_zero-shot + α·θ_fine-tuned. {#c-wf-def}
When only a linear classifier is fine-tuned, WiSE-FT is equivalent to ensembling the outputs of the two models. {#c-wf-linear}
That second fact is the clean case of a general one derived below: when the output is linear in the weights being averaged, a weight average and an output average are the same, which places merging next to ensemble learning; through nonlinear layers it is only an approximation of an ensemble.

Stochastic weight averaging, SWA, averages snapshots of one training run rather than separate fine-tunings.
Stochastic Weight Averaging is an equally weighted average of the points traversed by SGD with a cyclical or high constant learning rate, which leads to better generalization than conventional training. {#c-iz-def}
SWA starts from a pretrained model, continues training with a cyclical or constant learning rate, captures the models at the minimum values of the learning rate or at each epoch for a constant rate, and averages their weights. {#c-iz-alg}
The SWA average is updated as w_SWA ← (w_SWA·n_models + w)/(n_models + 1), which only requires computing a weighted sum of the weights of two networks. {#c-iz-update}
If the network uses batch normalization, SWA runs one additional pass over the data to compute the running mean and standard deviation of the activations for the w_SWA weights, since these statistics are not collected during training. {#c-iz-bn}
`RunningAverage` in the NumPy file keeps this running mean; in PyTorch, `torch.optim.swa_utils.AveragedModel` does the same, and the PyTorch tests check that `uniform_soup` agrees with it.
The batch-norm pass is needed for any merge of networks with batch normalization: the stored running statistics belong to the individual models, not to their average.

### Task arithmetic

A task vector specifies a direction in the weight space of a pre-trained model, such that movement in that direction improves performance on the task, and is built by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task. {#c-il-def}
Ilharco et al. negate a task vector, τ_new = −τ; add task vectors, τ_new = Σ_i τ_i; and, when tasks form an analogy “A is to B as C is to D”, use τ_new = τ_C + (τ_B − τ_A); the edited weights are θ_new = θ + λτ_new, where λ is determined using held-out validation sets. {#c-il-ops}
In the files, `task_vector` and `apply_task_vectors` implement this.
Adding n task vectors with λ = 1/n is the uniform soup, and adding one with λ = α is WiSE-FT, so task arithmetic generalizes both (derived below); what it adds is λ > 1/n, which keeps more of each task's change, and negation, which removes a behaviour.

### Fisher merging

Averaging treats every parameter of every model as equally trustworthy.
Matena and Raffel cast merging as finding the parameters most probable under the models' posteriors.
If the posterior of each model is approximated by an isotropic Gaussian with mean set to the model's parameters, maximizing Σ_i log p(θ | θ_i, I) has the closed-form solution θ* = (1/M) Σ_i θ_i, an average of the model parameters. {#c-mr-iso}
Matena and Raffel add model-specific weights λ_i ≥ 0 with Σ_i λ_i = 1, which change the isotropic solution to θ* = Σ_i λ_i θ_i. {#c-mr-lam}
Fisher merging approximates each model's posterior by a Gaussian with mean θ_i and precision F_i, the diagonal approximate Fisher, and maximizes Σ_i λ_i log p(θ | θ_i, F_i), which sets each parameter to Σ_i λ_i F_i θ_i / Σ_i λ_i F_i. {#c-mr-merge}
A parameter that a model's predictions are sensitive to has a large Fisher value in that model and pulls the merge towards its own value; a parameter the model ignores has little say.
The Fisher information relates parameter changes to output changes: E_x[D_KL(p_θ(y|x) ‖ p_{θ+δ}(y|x))] ≈ ½ δᵀF_θδ as δ → 0. {#c-mr-kl}
The diagonal Fisher is estimated as the average over N training examples of E_{y∼p_θ(y|x_i)}(∇_θ log p_θ(y|x_i))², with the expectation over y sampled or computed exactly when the number of classes is small, at roughly the cost of training on N examples. {#c-mr-fisher}
Where the Fisher is close to zero across all models for a parameter, Matena and Raffel default to the parameter's value in a privileged target model, a choice that had little impact on performance. {#c-mr-num}
Parameters that do not appear in all models, such as task-specific classification heads, are left out of the merge and kept unchanged. {#c-mr-heads}
`fisher_merge` in both files implements the merge with the fallback to a target model, `softmax_regression_fisher` in NumPy computes the exact diagonal Fisher of a softmax classifier, and `diagonal_fisher` in PyTorch computes the exact expectation over classes for any classifier with one gradient per class.

### TIES-merging

Yadav et al. identify two sources of interference when merging: redundant parameter values and disagreement on the sign of a given parameter's values across models. {#c-ya-interf}
Redundant values are the many small entries of a task vector that fine-tuning changed by noise; summed over several tasks, they drift the merged model away from the pre-trained one for no benefit.
Sign disagreement is a parameter that one task moved up and another moved down; averaging cancels both.
TIES-merging trims each task vector to its top-k% values by magnitude, elects for each parameter the sign with the greater total movement, γ_m = sgn(Σ_t τ̂_t), and takes a disjoint mean over the models whose sign agrees with it, which always ignores the zero values; the merged model is θ_init + λτ_m. {#c-ya-steps}
`trim`, `elect_sign`, `disjoint_mean` and `ties_merge` in the NumPy file follow the three steps; the PyTorch `ties_merge` trims over all parameters of a state dict at once, as Yadav et al. do.

### DARE

DARE randomly drops delta parameters δ^t = θ_SFT − θ_PRE with a drop rate p, using a Bernoulli mask, and rescales the remaining ones by 1/(1 − p), then adds the result to θ_PRE. {#c-yu-def}
For a linear layer the expectation of each output after dropping with ratio p and rescaling by γ is h_PRE + (1 − p)·γ·Δh, so setting γ = 1/(1 − p) makes the expected output equal to the original. {#c-yu-proof}
This is dropout's rescaling applied to weights instead of activations, once, rather than to activations at every step.
DARE is a plug-and-play module that can be applied before any model merging method, such as Average Merging, Fisher Merging, RegMean and TIES-Merging. {#c-yu-plugin}
`dare` and `dare_merge` implement it; the PyTorch `dare` takes the fine-tuned and pre-trained state dicts and a random generator.

### Aligning independently trained models

For models that do not share a starting point, the units must first be matched.
Ainsworth et al. measure approximations to convexity by the loss barrier, max_{λ∈[0,1]} L((1 − λ)Θ_A + λΘ_B) − ½(L(Θ_A) + L(Θ_B)), which is non-negative. {#c-ai-barrier}
Matching the units of two networks by their weights or activations is a linear assignment problem per layer, for which efficient polynomial-time algorithms are known. {#c-ai-lap}
`loss_barrier` in the NumPy file measures the barrier along a line of weights; weight matching itself is implemented on the mode-connectivity page, and this page does not repeat it.

### Choosing a method

Averaging (uniform or greedy soups, SWA) suits models trained for the same task with different hyperparameters or seeds, where the aim is a better single model.
Task arithmetic, TIES-merging and DARE suit models trained for different tasks, where the aim is one model with all the skills.
Fisher merging suits either when a few thousand examples per model are available to estimate the Fisher.
Low-rank adapters such as LoRA produce task vectors directly, as the product of two small matrices, and can be merged with the same arithmetic after the product is formed.

## Derivation

The full derivation is in `derivations/model-merging-arithmetic.md`, and `model-merging-arithmetic.check.py` checks every step numerically.

**Task arithmetic contains the averages.**
With task vectors τ_t = θ_t − θ_pre, θ_pre + λ Σ_t τ_t = (1 − nλ)θ_pre + λ Σ_t θ_t: λ = 1/n gives the uniform average of the n fine-tuned models, and one task vector with λ = α gives WiSE-FT's (1 − α)θ_pre + αθ_1. {#c-mm-ta}
So the λ of task arithmetic is the position on the line from θ_pre through the uniform soup, and λ = 1 is the sum of all the changes, n times further out than the soup.

**Fisher merging is a per-coordinate weighted average.**
The gradient of Σ_i λ_i log N(θ; θ_i, diag(F_i)⁻¹) in θ_j is Σ_i λ_i F_ij(θ_ij − θ_j), which is zero at θ_j = Σ_i λ_i F_ij θ_ij / Σ_i λ_i F_ij; when all F_i are equal this is Σ_i λ_i θ_i, the isotropic merge. {#c-mm-fisher}
The objective is a sum of one-dimensional quadratics, one per parameter, so each parameter has its own weights; nothing ties coordinates together, which is what makes the diagonal Fisher cheap and also what it ignores.

**Weights and outputs.**
If a model's output is linear in the merged parameters, as for a linear classifier on fixed features, the output of a weighted average of weights equals the same weighted average of the outputs, so weight averaging and output ensembling coincide; through a nonlinearity they differ. {#c-mm-linear}
For f(x; w) = tanh(wx) at x = 1 with w_1 = 0 and w_2 = 4, the averaged weight gives tanh 2 ≈ 0.964 and the averaged output ½ tanh 4 ≈ 0.500.
Merging and ensembling are therefore close when the models are close, so that the network is nearly linear over the range of weights being averaged, and diverge when they are far apart.

**DARE is unbiased but noisy.**
Each DARE entry is δ/(1 − p) with probability 1 − p and 0 otherwise, so its mean is δ and its variance δ²p/(1 − p), which is 9δ² at p = 0.9 and 99δ² at p = 0.99. {#c-mm-dare}
Each kept entry is multiplied by 10 at p = 0.9, so DARE relies on the deltas being small and on each output summing many of them, so that the errors average out.

**SWA's running mean.**
The update w ← (w·n + w_new)/(n + 1) keeps the arithmetic mean of the n + 1 captured weights exactly, so SWA stores one extra copy of the weights however many it averages. {#c-mm-swa}

**The disjoint mean does not shrink updates.**
When the elected sign of a parameter is positive, the disjoint mean of its positive entries is at least the plain mean of all n entries, because negative entries and zeros can only lower the sum or raise the count; the same holds with the signs reversed. {#c-mm-shrink}
This is the shrinkage TIES removes: in a plain mean, a parameter that two tasks move up by 0.5 and a third moves down by 0.45 ends up moving by 0.15.

**The loss barrier.**
The loss barrier is at least ½|L(Θ_A) − L(Θ_B)|, because the maximum includes both endpoints, and it is zero when the loss is convex along the segment and equal at its ends. {#c-mm-barrier}
A positive barrier between two models of equal loss means the loss rises along the line between them, so their midpoint, the uniform soup of two, is worse than either.

**Permutations.**
Permuting the hidden units of a one-hidden-layer network leaves its function unchanged, but averaging the network with its permuted copy generally changes the function; undoing the permutation before averaging gives back the network. {#c-mm-perm}

## Worked example

Three task vectors over five parameters are merged by TIES-merging and by a plain mean.
For the task vectors (0.5, −0.2, 0.05, 0.8, −0.6), (−0.45, −0.4, 0.02, 0.6, 0.1) and (0.4, 0.1, −0.01, −0.05, −0.7) with k = 60%, TIES-merging gives τ_m = (0.45, −0.4, 0, 0.7, −0.65), against the plain mean (0.15, −0.167, 0.02, 0.45, −0.4). {#c-mm-we-ties}
Trimming to the top three entries of each zeroes 0.05, 0.02 and −0.01 in the third column, −0.2 in the second, 0.1 in the fifth and −0.05 in the fourth.
The elected signs, from the column sums (0.45, −0.3, 0, 1.4, −1.3), are (+, −, 0, +, −).
In the first column two models move the parameter up by 0.5 and 0.4 and one moves it down by 0.45; the plain mean is 0.15, TIES keeps the agreeing pair and gives 0.45.
In the fourth column the plain mean 0.45 is pulled down by −0.05, which TIES trims away, so TIES gives (0.8 + 0.6)/2 = 0.7.
In the third column all values are small and TIES leaves the parameter at its pre-trained value.
The NumPy tests run this example through `ties_merge`, and the PyTorch tests run it split across two tensors.

For Fisher merging, take two models with two parameters each.
For θ_1 = (1, 0) and θ_2 = (0, 1) with diagonal Fishers (3, 1) and (1, 1) and equal weights, Fisher merging gives (0.75, 0.5), against the plain average (0.5, 0.5). {#c-mm-we-fisher}
The first model is three times as sensitive to the first parameter, so it receives three quarters of the weight there; both are equally sensitive to the second, so it is averaged.

## Hyperparameters

The mixing coefficients matter most.
For WiSE-FT, α = 0.5 is a good default when there is nothing better to go on (see Evidence).
For task arithmetic λ is chosen on held-out data, and TIES-merging can do without it.
Keeping only the top 20% of each task vector's values by magnitude, resetting the rest to their initial value, delivers results comparable to retaining all parameters. {#c-ya-top20}
Without a validation set, Yadav et al. use a TIES-merging recipe that keeps the top-20% parameters in the task vector and sets λ = 1, and compare it with task arithmetic at the recommended λ = 0.4. {#c-ya-recipe}
The drop rate of DARE depends on the model and the size of the deltas.
The more parameters a language model has, the larger the drop rate p DARE can tolerate, and at 70 billion parameters DARE can eliminate up to 99% of delta parameters with minimal impact on performance. {#c-yu-scale}
For SWA the schedule, the learning rate after the switch, and the capture frequency are the choices; the averaging itself has no parameters.

## Cost

Averaging, task arithmetic, TIES and DARE only read and write weights: memory for two models at a time for a running average, or for all n plus the pre-trained model for TIES, and time linear in the number of parameters.
Task arithmetic has no extra cost at inference time in terms of memory or compute, since it only does element-wise operations on model weights. {#c-il-free}
A greedy soup also evaluates every candidate soup on a validation set, n evaluations in all.
Fisher merging needs the diagonal Fisher of every model, one pass with gradients over N examples each, plus one extra copy of the weights per model to hold it.
SWA adds the cost of the extra epochs it averages over and one forward pass to recompute batch-norm statistics.
The merged model costs exactly one model at inference.

## Evidence

**Soups.**
A model soup of ViT-G models achieves 90.94% on ImageNet, surpassing the 90.88% of CoAtNet while requiring 25% fewer FLOPs at inference time. {#c-ws-vitg}
For CLIP ViT-B/32 fine-tuned on ImageNet, the individual model chosen from the sweep by validation accuracy reaches 80.38% on ImageNet and 47.83% on distribution shifts, the uniform soup 79.97% and 51.45%, the greedy soup 81.03% and 50.75%, the learned soup 80.89% and 51.07%, and the ensemble 81.19% and 50.77%. {#c-ws-table}
From sweeps of 72 fine-tuned CLIP models and 12 fine-tuned ALIGN models, the greedy soup selects 5 models in both cases. {#c-ws-five}
In that table the uniform soup gains 3.6 points under distribution shift over the chosen individual model but loses 0.4 on ImageNet, the greedy soup gains on both, and the ensemble, at the inference cost of all its members, is 0.16 points ahead of the greedy soup on ImageNet.

**WiSE-FT.**
On ImageNet and five derived distribution shifts, WiSE-FT improves accuracy under distribution shift by 4 to 6 percentage points over prior work while increasing ImageNet accuracy by 1.6 pp, at no additional computational cost during fine-tuning or inference. {#c-wf-gain}
α = 0.5 yields close to optimal performance across a range of WiSE-FT experiments, and Wortsman et al. recommend α = 0.5 when no domain knowledge is available. {#c-wf-half}

**SWA.**
SGD generally converges to a point near the boundary of the wide flat region of optimal points, while SWA finds a point centred in this region, often with slightly worse train loss but substantially better test error. {#c-iz-flat}
Running SWA for just 10 epochs on ImageNet gives a 0.8% improvement for ResNet-50 and DenseNet-161. {#c-iz-imnet}

**Task arithmetic.**
Negating task vectors reduces the accuracy of a pre-trained ViT-L/14 by 45.8 percentage points on the target tasks, with little loss on the control task. {#c-il-neg}
For GPT-2 Large, negative task vectors decrease the amount of toxic generations by 6×, while resulting in a model with comparable perplexity on a control task. {#c-il-tox}
Adding two task vectors maintains 98.9% of the accuracy, and the average performance on the entire set of tasks increases as more task vectors are added. {#c-il-add}
Task vectors from different tasks are typically close to orthogonal, which Ilharco et al. speculate enables the combination of task vectors via addition with minimal interference. {#c-il-orth}

**Fisher merging.**
Merging fine-tuned BERT-Base checkpoints for RTE, MRPC and SST-2, Fisher merging significantly outperforms isotropic merging in all cases and attains comparable performance to prediction ensembling. {#c-mr-ens}

**TIES-merging.**
For in-domain evaluation, TIES-merging outperforms the strongest baseline by an average of 2.3% and 1.7% absolute in NLP and vision settings. {#c-ya-gain}
As the number of merged tasks increases the performance of all methods decreases; when merging two tasks TIES-merging and task arithmetic lose almost nothing while simple averaging suffers a 10% drop, and task arithmetic declines more rapidly than TIES-merging. {#c-ya-ntasks}
Sign conflicts occur even when merging only 2 models, and the likelihood of a sign conflict increases with the number of models being merged. {#c-ya-conflict}

**DARE.**
SFT delta parameter value ranges are typically small, within 0.002, and DARE can eliminate 90% or even 99% of them. {#c-yu-small}
Merging decoder language models with DARE and task arithmetic gives gains of 3.18% for LM & Math over Math on GSM8K and 19.57% for LM & Code over Code on MBPP. {#c-yu-merge}

**Alignment.**
After weight matching, wider models exhibit better linear mode connectivity, and wide ResNets on CIFAR-10 reach zero-barrier linear mode connectivity. {#c-ai-zero}

## Limitations and failure modes

**A barrier between the models.**
The uniform soup outperforms every individual model only when all individual models achieve high accuracy; there can be an error barrier between fine-tuned models, mainly when fine-tuning with high learning rates. {#c-ws-barrier}
Averaging across a barrier gives a model worse than its parts; the greedy soup guards against this by testing each addition.

**Calibration and diversity.**
While ensembles improve model calibration, model soups do not have the same effect. {#c-ws-calib}
For an ImageNet22k pre-trained model the greedy soup still provides improvements on ImageNet, but these are less substantial than when fine-tuning CLIP and ALIGN. {#c-ws-22k}

**Shared start required.**
Task vectors are restricted to models with the same architecture, since they depend on element-wise operations on model weights, and Ilharco et al. perform arithmetic only on models fine-tuned from the same pre-trained initialization. {#c-il-limit}
Since the Fisher information is a local property of a single parameter value, Fisher merging might be less performant for models whose parameters are far apart in parameter space. {#c-mr-far}
Models trained from different random initializations need permutation alignment first, and even then only wide networks reach low barriers.

**Many tasks.**
Merging more task vectors degrades every method (see Evidence), because more of them disagree on more parameters.

**DARE's assumptions.**
The rescale operation is crucial to DARE: dropping 30% or 40% of delta parameters without rescaling noticeably leads to worse results. {#c-yu-rescale}
Once models undergo continuous pre-training, the delta parameters can rapidly reach around 0.03, making DARE infeasible. {#c-yu-cpt}
Dropping only 10% of the fine-tuned parameters, the combination of pre-trained and delta parameters, leads to a catastrophic decrease in performance, even approaching zero. {#c-yu-ft}

## Variants

- **Uniform and greedy soups** average a hyperparameter sweep; the greedy soup keeps only additions that help on validation data.
- **Learned soups** optimize the mixing weights by gradient descent, with all models in memory.
- **WiSE-FT** interpolates a fine-tuned model back towards its zero-shot starting point to regain robustness.
- **SWA** averages snapshots of a single run; exponential moving averages of weights are a variant that weights recent snapshots more.
- **Task arithmetic** adds, negates and combines task vectors with a scale λ.
- **Fisher merging** weights each parameter by the models' diagonal Fisher.
- **TIES-merging** trims small entries, elects a sign per parameter and averages only agreeing entries.
- **DARE** randomly drops and rescales delta entries before any of the above.
- **Permutation alignment** (Git Re-Basin and related methods) matches units of independently trained models before averaging them.

## When to use it

- After a hyperparameter sweep, average the runs that are good on their own instead of keeping only the best; check with a greedy soup if some runs are poor.
- To regain robustness lost in fine-tuning, interpolate back towards the pre-trained model with WiSE-FT and α = 0.5.
- To combine skills from models fine-tuned on different tasks from one base, without the data or compute to train on all tasks together, use task arithmetic, preferably with TIES or DARE when there are more than two tasks.
- To remove a behaviour, such as toxic generation, subtract its task vector and check the model on a control task.
- Prefer an output ensemble instead when calibration matters most, or when the models were trained from different starting points.

## Pitfalls

- Merging models that do not share a pre-trained checkpoint, which averages unrelated units.
- Forgetting to recompute batch-norm statistics after merging networks that use batch normalization.
- Merging task-specific heads, which have no counterpart in the other models; keep them separate.
- Choosing λ for task arithmetic on the test set; it is chosen on held-out validation data.
- Applying DARE without the 1/(1 − p) rescaling, or to models whose deltas are large, as after continued pre-training.
- Merging checkpoints with different tokenizers or vocabularies, where the embedding rows do not correspond.
- Judging a merge only on the merged tasks; control tasks reveal what the merge broke.

## History

Weight averaging has a long history in optimization, where averaging the iterates of stochastic gradient descent is a classical way to reduce noise.
Izmailov et al. (2018) applied it to deep networks as SWA, averaging snapshots along one trajectory and arguing that the average lands in the centre of a flat region.
Work on mode connectivity (Garipov et al. 2018, Frankle et al. 2020) established when two networks are joined by a low-loss path, and found that networks that share the early part of training are joined by a straight line.
Matena and Raffel (2021) merged separately fine-tuned language models with Fisher weights.
With large pre-trained models, WiSE-FT and model soups (Wortsman et al. 2021, 2022) found that averaging fine-tuned CLIP models improves accuracy and robustness, and task arithmetic (Ilharco et al. 2022) turned the differences from the pre-trained model into editable vectors.
TIES-merging (Yadav et al. 2023) and DARE (Yu et al. 2023) addressed the interference that appears when many task vectors are added, and merging became a common way to combine open language models fine-tuned from the same base.
Git Re-Basin (Ainsworth et al. 2022) addressed the other case, independently trained models, by aligning their units before averaging.
