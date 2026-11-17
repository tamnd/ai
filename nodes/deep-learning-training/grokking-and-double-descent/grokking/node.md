---
tag: 52PVZH
kind: finding
slug: grokking
name: Grokking
aliases: [delayed generalization, generalization beyond overfitting, ungrokking, semi-grokking]
area: deep-learning-training/grokking-and-double-descent
level: core
summary: >-
  A network trained on a small algorithmic dataset can fit its training set early, stay at chance on held-out data for a long time, and then, with no change to the training, generalize almost perfectly.
history:
  year: 2022
  note: "Named by Power, Burda, Edwards, Babuschkin and Misra in arXiv 2201.02177 (January 2022) for small transformers trained on binary-operation tables."
prereqs: [overfitting@al, generalization-error@al, weight-decay@al, adamw@al, cross-entropy-loss@al]
see_also: [double-descent@al]
same_as:
  wikipedia: Grokking_(machine_learning)
  arxiv: ["2201.02177", "2301.05217", "2210.01117", "2309.02390"]
sources:
  - {paper: "Power et al. (2022)", title: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", locator: "pp. 1-4, 6-7", note: "the name, the first experiments, the ablations and the setup"}
  - {paper: "Nanda et al. (2023)", title: "Progress measures for grokking via mechanistic interpretability", locator: "pp. 2-9", note: "the Fourier algorithm in a one-layer transformer and the three phases of training"}
  - {paper: "Liu, Michaud and Tegmark (2022)", title: "Omnigrok: Grokking Beyond Algorithmic Data", locator: "pp. 1-9", note: "weight norm, the LU mechanism, grokking on MNIST, IMDb and QM9"}
  - {paper: "Varma et al. (2023)", title: "Explaining grokking through circuit efficiency", locator: "pp. 1, 3, 7, 9", note: "circuit efficiency, the critical dataset size, ungrokking and semi-grokking"}
claim: >-
  On small algorithmic tasks, a network can reach perfect training accuracy long before it generalizes, and then, after many more steps of the same training, move from chance to near-perfect accuracy on held-out data.
first_reported: {work: power-2022-grokking-generalization-beyond-overfitting, note: "named the effect and showed it for two-layer transformers on modular arithmetic, polynomials mod 97 and the group S5"}
replications:
  - {work: nanda-2023-progress-measures-grokking, note: "one-layer transformers on addition mod 113, over 5 seeds, with the learned algorithm reverse engineered"}
  - {work: liu-2022-omnigrok-grokking-beyond-algorithmic, note: "the same transformer setup, and induced grokking for MLPs on MNIST, LSTMs on IMDb and graph networks on QM9"}
  - {work: varma-2023-explaining-grokking-circuit-efficiency, note: "one-layer transformers on addition mod 113 and 9 other tasks, with ungrokking and semi-grokking"}
scale_range:
  - {range: "two-layer transformers of width 128, about 4 x 10^5 non-embedding parameters, up to 10^6 steps", work: power-2022-grokking-generalization-beyond-overfitting}
  - {range: "one-layer transformers with d = 128 and 512 MLP units on 30% of the 12,769 pairs mod 113", work: nanda-2023-progress-measures-grokking}
  - {range: "a depth-3 width-200 MLP on 1,000 MNIST images", work: liu-2022-omnigrok-grokking-beyond-algorithmic}
failures:
  - {work: nanda-2023-progress-measures-grokking, note: "no grokking on modular addition without weight decay or another regularizer; no gap between train and test loss with enough data"}
  - {work: liu-2022-omnigrok-grokking-beyond-algorithmic, note: "a small initialization with the weight norm held fixed almost removes grokking on modular addition"}
  - {work: power-2022-grokking-generalization-beyond-overfitting, note: "x^3 + xy^2 + y mod 97 did not generalize at any training fraction up to 95%"}
explanations:
  - {name: "memorize, form a circuit, clean up", summary: "The network memorizes first, builds the generalizing circuit gradually while test accuracy stays low, and weight decay then removes the memorizing parts; the jump in test accuracy comes at the removal.", sources: [nanda-2023-progress-measures-grokking]}
  - {name: "the LU mechanism", summary: "Against the weight norm, the training loss is L-shaped and the test loss U-shaped; a run that starts or drifts to a large norm overfits at once and generalizes only when regularization has slowly shrunk the norm.", sources: [liu-2022-omnigrok-grokking-beyond-algorithmic]}
  - {name: "circuit efficiency", summary: "The generalizing circuit produces the same logits with less parameter norm than the memorizing one but is learned more slowly, so weight decay moves norm from memorization to generalization over time.", sources: [varma-2023-explaining-grokking-circuit-efficiency]}
---

Grokking is the finding that a neural network can fit its training data early, stay at chance level on held-out data for a long time, and then, with nothing in the training changed, generalize almost perfectly.
Power and colleagues named it in 2022 for small transformers trained on tables of modular arithmetic and permutation products, where the jump in validation accuracy came about a thousand times later than the fit to the training set.
In modular addition the network that groks has been reverse engineered: it adds by rotating points on circles at a few frequencies, and it builds this circuit gradually, long before the jump.
The effect depends on the amount of data, on weight decay and on the weight norm, and three explanations, which overlap, account for it through weight norm and the efficiency of circuits.

## In plain words

Train a small network on a table such as "a plus b, wrapped around at 97", showing it half of the entries.
It soon gets every training entry right, by memorizing them, and it does no better than guessing on the hidden half.
Usual practice would stop here, since the model has overfit.
Keep training, and after a hundred or a thousand times as many steps, accuracy on the hidden entries climbs to nearly 100%.

In between, the network gradually builds a second way of computing the answer, one that uses the rule, alongside its lookup table.
Weight decay, which shrinks the weights a little at every step, favors the rule because it needs smaller weights than the table.
Once the table has shrunk away, the rule decides the answers on the hidden entries, and the test accuracy jumps.

## The finding

Power et al. (2022) trained small transformers on equations $a \circ b = c$, where each element is a separate token with no internal structure, on a random fraction of a binary-operation table, and measured accuracy on the rest.
They report that validation accuracy can sit at chance for a long time after the network has badly overfit, and then rise abruptly toward perfect generalization; they name this grokking. {#c-power-grokking}
Their model was a two-layer decoder-only transformer of width 128 with four attention heads and about $4 \cdot 10^5$ non-embedding parameters, trained with AdamW at learning rate $10^{-3}$, weight decay 1, $\beta_2 = 0.98$, minibatches of 512 and a budget of $10^5$ updates. {#c-power-setup}
In their headline example, division mod 97 trained on half of the equations, training accuracy is near perfect within $10^3$ steps; validation accuracy needs roughly $10^6$ steps to catch up and shows almost no generalization before $10^5$ steps. {#c-power-division}
That run used Adam with no weight decay and a budget of $10^6$ steps, chosen to emphasize how late generalization can begin. {#c-power-figure-one-adam}
Measured in steps, validation accuracy leaves chance only after 1000 times more optimization than training accuracy needs to get close to optimal. {#c-power-thousand-times}

The measured form is a time to generalize that grows steeply as the data shrinks.
For the product in the permutation group $S_5$, near 25-30% of the data, removing 1% of the training data raises the median number of steps to reach 99% validation accuracy by 40-50%. {#c-power-time-vs-data}
The steps to reach 99% training accuracy do not grow: they generally trend down as the dataset shrinks and stay in the range of $10^3$-$10^4$. {#c-power-train-time}
The late jump is typical for dataset sizes close to the smallest at which the network generalizes within the budget; with more data the training and validation curves track each other more closely. {#c-power-near-minimal}

## Evidence

The first report covered modular addition, subtraction and division, several polynomials mod 97 and three operations in $S_5$ (Power et al., 2022).
Nanda et al. (2023) replicated the effect on addition mod 113 with a one-layer ReLU transformer with $d = 128$, 4 heads and 512 hidden units, trained full-batch with AdamW at learning rate 0.001 and weight decay 1 on 30% of the $113 \times 113$ pairs for 40,000 epochs. {#c-nanda-setup}
Their networks quickly reach 100% training accuracy while test accuracy stays low, and after around 10,000 epochs test accuracy rises to near 100%. {#c-nanda-grokking}
Liu, Michaud and Tegmark (2022) and Varma et al. (2023) trained the same kind of model on the same task and report the same curves, and Varma et al. extend the experiments to 9 further tasks.
Outside algorithmic data, Liu, Michaud and Tegmark induced grokking on purpose.
With MNIST cut from 60k to 1k training images and the initial weights scaled up by a constant $\alpha > 1$, a depth-3 width-200 ReLU MLP trained with AdamW on the squared loss fits the training set quickly, and test accuracy improves much later. {#c-liu-mnist}
They saw weaker signals for an LSTM on IMDb reviews and a graph network on the QM9 molecules, where the jump is not as sharp as on algorithmic data. {#c-liu-imdb-qm9}

## Scope and conditions

**Data.**
Grokking appears near the smallest training set that allows generalization at all.
Some operations never generalized: for $x^3 + xy^2 + y \pmod{97}$ the models only memorized, at every training fraction up to 95%. {#c-power-no-pattern}
Operations that are symmetric in their operands, such as $x + y$ and $x^2 + y^2$, need less data than non-symmetric relatives such as $x - y$. {#c-power-symmetric}
With enough data there is no gap between train and test loss; both fall sharply some number of epochs into training. {#c-nanda-enough-data}

**Regularization.**
Weight decay has a very large effect on data efficiency, more than halving the samples needed compared to most other interventions, and decay toward the initialization helps but less than decay toward the origin. {#c-power-weight-decay}
In the one-layer setup, less weight decay makes grokking take significantly longer, and without weight decay or another regularizer the networks do not grok at all. {#c-nanda-weight-decay}
The headline run of Power et al. used no weight decay, while the full-batch runs of Nanda et al. needed it, so whether grokking needs an explicit regularizer depends on the setup.
Power et al. found that noise in the optimization, such as the gradient noise of minibatches or Gaussian noise on the weights, is beneficial for generalization. {#c-power-noise}
The learning rate had to be tuned within about one order of magnitude for generalization to happen. {#c-power-learning-rate}

**Initialization and weight norm.**
Liu, Michaud and Tegmark make the weight norm the controlling variable.
In a teacher-student MLP started at twice the standard scale, the steps to 95% training accuracy do not depend on the weight decay, while the steps to 95% test accuracy are inversely proportional to it. {#c-liu-teacher-student}
Holding the weight norm fixed at a small value removes most of the gap on modular addition: train and test accuracy rise together. {#c-liu-constrained}

**Dataset size, in both directions.**
Varma et al. find that a network that has grokked can "ungrok".
Trained to 100% test accuracy on the full table and then trained on a smaller subset, it shows a sharp transition from strong to near-zero test accuracy as the subset shrinks, independent of the weight decay. {#c-varma-ungrokking}
Near a critical dataset size, some runs "semi-grok": of 200 runs, at least 6 end with a delayed transition to middling rather than perfect test accuracy. {#c-varma-semi-grokking}

## Explanations

**The algorithm in the weights.**
Nanda et al. (2023) reverse engineered the mod-113 transformer.
The embedding maps $a$ and $b$ to sines and cosines at a sparse set of key frequencies, the attention and MLP layers combine them with trigonometric identities to compute the sine and cosine of $w_k(a+b)$, and the output layers read off the answer. {#c-nanda-algorithm}
With $w_k = 2\pi k / P$, the identities are

$$ \cos(w_k(a+b)) = \cos(w_k a)\cos(w_k b) - \sin(w_k a)\sin(w_k b), \qquad \sin(w_k(a+b)) = \sin(w_k a)\cos(w_k b) + \cos(w_k a)\sin(w_k b), $$ {#eq-sum-angle}

and the logit of each candidate answer $c$ is

$$ \ell(c) = \sum_k \alpha_k \cos(w_k(a+b-c)) = \sum_k \alpha_k \big[\cos(w_k(a+b))\cos(w_k c) + \sin(w_k(a+b))\sin(w_k c)\big]. $$ {#eq-read-off}

The embedding is sparse in the Fourier basis; five frequencies, $k \in \{14, 35, 41, 42, 52\}$, carry the computation. {#c-nanda-key-frequencies}
The neuron-to-logit map is matched by the sine and cosine directions of these frequencies up to a residual under 0.55% of its norm. {#c-nanda-wl}
A fit of the logits to $\sum_k \alpha_k \cos(w_k(a+b-c))$ explains 95% of their variance, and using the fit in place of the logits lowers the test loss from $2.4 \cdot 10^{-7}$ to $4.7 \cdot 10^{-8}$. {#c-nanda-logit-fit}
Of the 512 neurons, 433 (84.6%) are explained to over 85% by a degree-2 polynomial of one key frequency. {#c-nanda-neurons}
Ablating the key frequencies drops performance to chance, and ablating the other 95% of frequencies slightly improves it. {#c-nanda-ablation}

Why this rule is exact follows from the identity.
For a prime $p$, positive weights and any frequencies $k$ in $1, \dots, p-1$, the logit equals $\sum_k \alpha_k$ at $c = a + b \bmod p$ and is strictly smaller at every other $c$. {#c-fourier-peak}
With all frequencies $1, \dots, (p-1)/2$ and unit weights, every wrong answer gets exactly $-1/2$. {#c-fourier-all-frequencies}
Primality matters: for $p = 12$ and the single frequency $k = 3$, the wrong answer $a + b - 4$ ties with the right one. {#c-fourier-prime}
At $p = 113$ with the five key frequencies and unit weights, the right logit is 5 and the largest wrong logit is 2.5898, at $c = a + b + 16$, so the rule is right on all 12,769 pairs. {#c-fourier-113}
The derivation `fourier-identity` gives the steps, and the implementations check the last result on every pair.

**Three phases.**
From the algorithm, Nanda et al. built progress measures: the restricted loss keeps only the key frequencies of the logits, and the excluded loss removes them.
Training then splits into memorization (epochs 0k-1.4k), where the training and excluded losses fall; circuit formation (1.4k-9.4k), where the excluded loss rises, the sum of squared weights falls and the restricted loss starts to fall while train and test loss stay flat; and cleanup (9.4k-14k), where the test loss suddenly drops along with the sum of squared weights. {#c-nanda-phases}
So the jump in test accuracy comes during cleanup, after the generalizing circuit exists; grokking is the gradual strengthening of a structured mechanism followed by the removal of the memorizing parts. {#c-nanda-cleanup}

**Weight norm, the LU mechanism.**
Liu, Michaud and Tegmark minimize the training loss over the direction of the weights at each fixed norm.
Against the norm, the training loss looks like an L and the test loss like a U, and they attribute grokking to this mismatch. {#c-liu-lu}
A run that starts at a large norm overfits at once on the flat arm of the L and must wait for regularization to carry it back to the norm where the test loss is low.
If weight decay $\gamma$ is the only force and the training loss is negligible, $w(t) \approx e^{-\gamma t} w_0$, and reaching the generalizing norm $w_c$ takes $\ln(w_0/w_c)/\gamma$, so a small $\gamma$ means a long delay. {#c-liu-decay-time}
In AdamW the decay alone multiplies the weights by $1 - \eta\lambda$ per step:

$$ \theta_t = (1 - \eta\lambda)^t\,\theta_0 \quad \text{when the gradient is zero.} $$ {#eq-shrink}

With a zero data gradient and zero moments, each step scales the weights and their norm by $1 - \eta\lambda$. {#c-decay-geometric}
The steps to shrink the norm from $w_0$ to $w_c$ are $\lceil \ln(w_0/w_c) / (-\ln(1 - \eta\lambda)) \rceil$, at most $\lceil \ln(w_0/w_c)/(\eta\lambda) \rceil$, which grows as one over the weight decay. {#c-decay-time}
At the original settings, $\eta = 10^{-3}$ and $\lambda = 1$, decay alone multiplies the weights by 0.999 per step and by 0.3677 over 1,000 steps, and halves their norm in 693 steps. {#c-decay-numbers}
The derivation `weight-decay-shrinkage` gives the steps.
In the mod-113 transformer the weight norm rises during overfitting and falls below its initial value when the model generalizes, as this picture predicts. {#c-liu-norm-curve}
Liu, Michaud and Tegmark put the drama of algorithmic grokking down to representation learning: on MNIST a poor representation still gets about 95% test accuracy, while on algorithmic data it means a random guess. {#c-liu-representation}

**Circuit efficiency.**
Varma et al. (2023) state the idea of the cleanup phase as a theory.
Grokking happens when a task admits a memorizing and a generalizing solution, and the generalizing one is slower to learn but more efficient, producing larger logits with the same parameter norm. {#c-varma-efficiency}
In a constructed simulation, three properties suffice: the generalizing circuit generalizes and the memorizing one does not, the generalizing circuit is more efficient, and it is learned more slowly. {#c-varma-ingredients}
They measure that the efficiency of the generalizing circuit does not depend on the dataset size, while that of the memorizing circuit falls as the training set grows. {#c-varma-dataset-size}
So below a critical size memorization is the more efficient circuit, which predicts ungrokking and semi-grokking, and they then observed both.

The three accounts agree that weight decay, or another pressure toward small weights, does the work after the training loss is near zero, and that the generalizing solution has a smaller norm.
They differ in what they treat as the cause: the learned circuit and its phases, the shape of the loss against the norm, or the relative efficiency of two circuits.

## Implications

Nanda et al. present grokking as a case study for emergence: a capability that appears suddenly can rest on progress that is continuous when measured with the right quantity, and they found that quantity by reverse engineering the network.
They also note that the analysis took a lot of manual work and that their measures are specific to one small task.
Power et al. propose small algorithmic datasets as a testbed for theories of generalization, since the runs fit on one GPU and the effects are larger than on natural data.
Power et al. also found that networks trained on modular addition with weight decay show the circular structure of the task in their output embeddings, as a number line formed by adding 8 to each element. {#c-power-embeddings}
For training practice, the reports agree on one caution: on small datasets a flat test curve after the training loss reaches zero need not be final, and weight decay changes how long the wait is.

## History

Power, Burda, Edwards, Babuschkin and Misra posted the first report as arXiv 2201.02177 on 6 January 2022. {#c-hist-power}
Liu, Michaud and Tegmark posted Omnigrok in October 2022, and Nanda, Chan, Lieberum, Smith and Steinhardt posted their reverse engineering of the mod-113 transformer in January 2023; both papers appeared at ICLR 2023.
Varma, Shah, Kenton, Kramár and Kumar posted the circuit-efficiency account, with the first reports of ungrokking and semi-grokking, as arXiv 2309.02390 on 5 September 2023. {#c-hist-varma}

## References
