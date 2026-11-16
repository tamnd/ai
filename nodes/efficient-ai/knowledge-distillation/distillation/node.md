---
tag: JGNGHC
kind: method
slug: distillation
name: Knowledge distillation
aliases: [KD, distillation, teacher-student training, soft-target training, model distillation]
area: efficient-ai/knowledge-distillation
level: core
summary: >-
  Train a small student network to match the temperature-softened output distribution of a large teacher or ensemble, so it learns how the teacher generalizes and not only the labels.
purpose: compression
uses: [cross-entropy@m2, kullback-leibler-divergence@m2, mean-squared-error@m2]
related_to: [label-smoothing@al, dropout, ensemble-learning@al, neural-network-pruning@al, data-augmentation@al, overfitting@al, regularization@al, shannon-entropy@m2]
hyperparameters:
  - {name: temperature, meaning: "the temperature T that divides the logits of both teacher and student in the soft term; T = 1 is the ordinary softmax, larger T gives softer targets", typical: "2 to 20; 4 is a common choice"}
  - {name: alpha, meaning: "the weight of the soft term against the hard-label cross-entropy; conventions differ, as the Pitfalls say, and in the implementations here alpha weights the soft term", typical: "0.5 to 0.9 with labels, 1 without"}
  - {name: transfer set, meaning: "the inputs on which the student matches the teacher: the original training set, unlabeled data from the same distribution, or augmented data", typical: "the training set"}
cost:
  train: "a forward pass of the teacher on every transfer example, or a stored copy of its logits, and one more softmax and sum over the classes for each example"
  inference: "that of the student alone; the teacher is not needed after training"
history:
  year: 2015
  people: [Geoffrey Hinton, Oriol Vinyals, Jeff Dean]
  note: "Named, and given its temperature form, in Distilling the Knowledge in a Neural Network (arXiv, March 2015), which built on the model compression of Buciluǎ, Caruana and Niculescu-Mizil (KDD 2006) and the logit regression of Ba and Caruana (2014); FitNets (ICLR 2015) added hints from hidden layers, DistilBERT (2019) distilled a pre-trained language model, and Stanton et al. (NeurIPS 2021) separated fidelity to the teacher from generalization."
prereqs: [cross-entropy-loss@al, softmax-regression@al, kullback-leibler-divergence@m2, backpropagation@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1503.02531", "1312.6184", "1412.6550", "1910.01108", "2106.05945"]
  al: knowledge-distillation
sources:
  - {paper: "Hinton, Vinyals and Dean (2015)", title: "Distilling the Knowledge in a Neural Network", locator: "pages 1 to 5, 7 and 8", note: "the temperature, the weighted loss and its T² factor, the high-temperature limit, MNIST, speech and soft targets as regularizers"}
  - {paper: "Ba and Caruana (2014)", title: "Do Deep Nets Really Need to be Deep?", locator: "pages 2 to 9", note: "model compression by logit regression, TIMIT and CIFAR-10"}
  - {paper: "Romero, Ballas, Kahou, Chassang, Gatta and Bengio (2015)", title: "FitNets: Hints for Thin Deep Nets", locator: "pages 1, 3, 4, 5 and 7", note: "hints from a hidden layer, stage-wise training and deep thin students"}
  - {paper: "Sanh, Debut, Chaumond and Wolf (2019)", title: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", locator: "pages 1 to 4", note: "distillation during pre-training with a triple loss"}
  - {paper: "Stanton, Izmailov, Kirichenko, Alemi and Wilson (2021)", title: "Does Knowledge Distillation Really Work?", locator: "pages 1, 3, 4, 7, 9 and 10", note: "fidelity against generalization, and optimization as the obstacle"}
  - {paper: "Müller, Kornblith and Hinton (2019)", title: "When Does Label Smoothing Help?", locator: "pages 6 and 7", note: "label-smoothed teachers distill worse"}
---

Knowledge distillation trains a small network, the student, to reproduce the outputs of a large one, the teacher, which may be a single network or an ensemble.
The student learns from the teacher's full output distribution, softened by a temperature, and usually also from the true labels, through a weighted sum of two cross-entropies.
The probabilities the teacher gives the wrong classes carry much of what it has learned about how classes resemble each other, and the temperature makes them large enough to train on.
Hinton, Vinyals and Dean named the method in 2015, building on the model compression of Caruana and his collaborators, and reported that a single network could recover most of the gain of an ensemble of ten speech models.
It is now a standard way to make models cheaper to run, from image classifiers to DistilBERT, though a student often matches its teacher's predictions much less closely than the name suggests.

## In plain words

A large network, or a committee of networks, is often accurate but too slow to ship.
Distillation trains a smaller network to copy it.
The copy does not only learn the right answers; it learns the big model's whole opinion on each example, such as "this is a 2, but it looks a little like a 3 and hardly at all like a 7".
Those small preferences say a lot about how the big model sees the data, and they would be lost if the small model learned from the labels alone.
To bring them out, both models' scores are divided by a number called the temperature before they are turned into probabilities, which flattens the distributions so the small preferences become visible.
The small model is then trained to match these flattened distributions, usually alongside the ordinary labels, and used at normal temperature afterwards.

## Setting

Hinton et al. open with ensembles: almost any machine learning algorithm improves in performance if many different models are trained on the same data and their predictions averaged, but making predictions with the whole ensemble is cumbersome, and the computation may be too expensive for deployment to a large number of users. {#c-hi-ensemble-cumbersome}
They compare this to insects, whose larval form is optimized for extracting energy and nutrients and whose adult form for travelling and reproduction: in machine learning, training and deployment have very different requirements, yet usually use very similar models. {#c-hi-larva}
In their account the cumbersome model is either an ensemble of separately trained models or one very large model that a very strong regularizer such as dropout kept in check, and after it has been trained, a different kind of training they call distillation transfers its knowledge to a small model better suited to deployment. {#c-hi-cumbersome}
They credit Rich Caruana and his collaborators with pioneering a version of this strategy and with showing convincingly that a single small model can receive the knowledge transferred from a large ensemble. {#c-hi-caruana-pioneered}

**What knowledge is.**
Hinton et al. argue that identifying the knowledge of a model with its learned parameter values makes it hard to see how the form of a model could change while the knowledge stays the same, and propose instead to view the knowledge as a learned mapping from input vectors to output vectors. {#c-hi-knowledge-mapping}
A model trained to maximize the log probability of the correct answer also assigns probabilities to all the incorrect answers, and even when these are very small some are much larger than others; how the model tends to generalize is told largely by the relative probabilities of the incorrect answers. {#c-hi-relative-probabilities}
Their example is an image of a BMW: the chance of mistaking it for a garbage truck may be very small, yet that mistake is still many times more probable than taking it for a carrot. {#c-hi-bmw}
Models are normally trained to fit the training data when the real aim is to generalize to new data; when distilling, the small model can be trained to generalize in the same way as the large one, and if the large one generalizes well because it averages a large ensemble, a small model trained this way typically does much better on test data than one trained in the normal way on the same training set. {#c-hi-generalize-like}

**Model compression.**
Ba and Caruana describe model compression as training a compact model to approximate the function learned by a larger, more complex one, and cite Buciluǎ et al., whose small networks had 1000 times fewer parameters and were often just as accurate as the ensembles they mimicked. {#c-ba-compression}
Model compression passes unlabeled data through the large model to collect its scores and trains the smaller mimic model on this synthetically labeled data; the mimic is not trained on the original labels but on the function the larger model learned. {#c-ba-unlabeled}

## The method

A classifier with logits $z_1, \dots, z_N$ over $N$ classes produces probabilities with a softmax at temperature $T$,

$$ q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} , $$ {#eq-softmax-temperature}

where $T$ is normally set to 1, and a higher $T$ gives a softer distribution over the classes. {#c-hi-temperature}
Write $v$ for the teacher's logits and $p = \operatorname{softmax}(v/T)$ for its soft targets.
When the teacher is an ensemble, Hinton et al. use the arithmetic or the geometric mean of the members' predictive distributions as the soft targets, and use the class probabilities of the cumbersome model as soft targets for the small model, on the same training set or on a separate transfer set. {#c-hi-soft-targets}

**Raising the temperature.**
On a task like MNIST the teacher is almost always right and very confident, so much of what it has learned lies in the ratios of very small probabilities: one 2 may get a probability of $10^{-6}$ of being a 3 and $10^{-9}$ of being a 7, and another the other way round. {#c-hi-mnist-ratios}
With probabilities this close to zero, the similarity structure has very little influence on transfer through the cross-entropy. {#c-hi-little-influence}
Caruana and his collaborators avoid the problem by training the small model on the logits, which are what goes into the final softmax, minimizing the squared difference between the logits of the cumbersome and the small model. {#c-hi-caruana-logits}
Hinton et al. instead raise the temperature of the final softmax until the cumbersome model gives a suitably soft set of targets, then train the small model at that same high temperature to match them; matching logits turns out to be a special case of this. {#c-hi-raise-temperature}
Soft targets of high entropy carry much more information per training case than hard targets, and their gradients vary much less from case to case, so the small model can often be trained on much less data than the cumbersome one, with a much higher learning rate. {#c-hi-high-entropy}

**The loss.**
In the simplest form, the distilled model is trained on a transfer set with the teacher's soft targets at a high temperature, the same temperature is used in the student's softmax during training, and after training the student uses a temperature of 1. {#c-hi-simplest}
When the correct labels are known for some or all of the transfer set, Hinton et al. found it better to take a weighted average of two objectives than to modify the soft targets: the cross-entropy with the soft targets at the high temperature, and the cross-entropy with the correct labels at temperature 1 on the same logits, with the best results generally from a considerably lower weight on the second. {#c-hi-weighted-average}
The transfer set can be unlabeled data or the original training set; Hinton et al. found the original training set works well, especially with a small extra term that asks the student to predict the true targets, since a student that cannot match the soft targets exactly does better to err toward the correct answer. {#c-hi-transfer-set}
Because the gradients of the soft term scale as $1/T^2$, they multiply it by $T^2$ when hard and soft targets are used together, which keeps the relative contributions of the two roughly unchanged when the temperature is changed during tuning. {#c-hi-t2}
With $\alpha$ the weight of the soft term and $y$ the label, the loss of one example in the form used by the implementations here is

$$ L = \alpha\, T^2\, D_{\mathrm{KL}}\big(\operatorname{softmax}(v/T) \,\Vert\, \operatorname{softmax}(z/T)\big) + (1 - \alpha)\,\big(-\log \operatorname{softmax}(z)_y\big) . $$ {#eq-kd-loss}

The KL divergence and the cross-entropy $-\sum_i p_i \log q_i$ differ only by the teacher's entropy, which does not depend on the student, so either can be the soft term; the KL form is zero when the student matches the teacher.
In PyTorch the soft term is `F.kl_div(F.log_softmax(z / T, dim=1), F.log_softmax(v / T, dim=1), reduction="batchmean", log_target=True) * T * T`, with the teacher's logits computed without gradient; `distillation_loss` in both implementations computes eq-kd-loss.

## Derivation

**The gradient of the soft term.**
Hinton et al. write the gradient that each transfer case contributes to the student logit $z_i$, with teacher logits $v_i$ and soft target probabilities $p_i$ at temperature $T$, as $\partial C/\partial z_i = (q_i - p_i)/T$. {#c-hi-eq2}
Since $\partial\log q_k/\partial z_i = ([k = i] - q_i)/T$ and $\sum_k p_k = 1$,

$$ \frac{\partial C}{\partial z_i} = -\sum_k p_k\,\frac{[k = i] - q_i}{T} = \frac{1}{T}\,(q_i - p_i) , $$ {#eq-soft-gradient}

for the soft cross-entropy $C = -\sum_i p_i\log q_i$, as long as the teacher's distribution does not depend on $z$. {#c-soft-gradient}
The KL divergence $D_{\mathrm{KL}}(p\Vert q)$ equals $C - H(p)$, where the teacher entropy $H(p)$ does not depend on $z$, so it has the same gradient. {#c-kl-same-gradient}
$H(p)$ is the Shannon entropy of the soft targets, which is why the two forms of the loss report different values but train the student identically.
The entries of the gradient sum to zero, and adding the same constant to every student logit changes neither the loss nor the gradient. {#c-gradient-sums-to-zero}
With $T = 1$ and a one-hot target in place of $p$, the gradient is $q - y$, the gradient of the ordinary cross-entropy. {#c-hard-special-case}
So the gradient of eq-kd-loss is $\alpha T(q - p) + (1 - \alpha)(\operatorname{softmax}(z) - \delta_y)$, with $q$ and $p$ at temperature $T$. {#c-full-gradient}
The derivation `soft-target-gradient` gives the steps, and its check compares them with finite differences on random logits.

**Why $T^2$.**
For fixed logits and growing $T$, $\operatorname{softmax}(x/T)_i = 1/N + (x_i - \bar x)/(NT) + O(T^{-2})$, where $\bar x$ is the mean of the logits. {#c-softmax-expansion}
So $q - p$ is of order $1/T$ and the soft gradient $(q - p)/T$ of order $1/T^2$, while the gradient of the hard term does not depend on $T$; multiplying the soft term by $T^2$ keeps its gradient of order one, so changing $T$ does not silently change the balance between the two terms. {#c-t2-factor}

**The high-temperature limit is logit matching.**
Hinton et al. approximate the gradient for a temperature high compared to the size of the logits; with the logits of each transfer case zero-meaned separately, it is $\partial C/\partial z_i \approx (z_i - v_i)/(NT^2)$. {#c-hi-eq4}
In the high-temperature limit, then, distillation amounts to minimizing $\tfrac12(z_i - v_i)^2$, so long as each transfer case has its logits zero-meaned separately. {#c-hi-limit}
More precisely, with $\Delta_i = (z_i - \bar z) - (v_i - \bar v)$ the difference of the centred logits, $T^2\,\partial C/\partial z_i = \Delta_i/N + O(1/T)$. {#c-high-t-limit}
Under the zero-mean condition $\Delta_i = z_i - v_i$, and the limit is the gradient of a squared error on the logits:

$$ \lim_{T\to\infty} T^2\,\frac{\partial C}{\partial z_i} = \frac{z_i - v_i}{N} = \frac{\partial}{\partial z_i}\,\frac{1}{2N}\sum_k (z_k - v_k)^2 \qquad \Big(\sum_k z_k = \sum_k v_k = 0\Big) . $$ {#eq-logit-matching}

Without the condition, the limit is the gradient of $\frac{1}{2N}\sum_k\Delta_k^2$, which is zero exactly when $z - v$ is a constant vector, so the student matches the teacher's logits only up to a shift, which the softmax ignores. {#c-logit-matching}
The scaled loss has the matching limit $T^2 D_{\mathrm{KL}}(p\Vert q) \to \frac{1}{2N}\sum_i\Delta_i^2$. {#c-kl-limit}
Stanton et al. state the same limit for their $\tau^2$-scaled loss: as $\tau \to \infty$ its gradient approaches that of $\lVert z_t - z_s\rVert_2^2/2$, which gives equal weight to every class logit whatever its contribution to the predictive distribution. {#c-st-limit}
The derivation `high-temperature-limit` gives the steps, and its check measures the rate $O(1/T)$ of the convergence.

**Lower temperatures.**
Hinton et al. point out that at lower temperatures distillation pays much less attention to matching logits far more negative than the average, which can be an advantage, since the cost function that trained the cumbersome model leaves such logits almost completely unconstrained and they could be very noisy, yet such logits may also carry useful information about what the cumbersome model has learned; which effect dominates is an empirical question. {#c-hi-negative-logits}
They report that when the student is much too small to capture all of the teacher's knowledge, intermediate temperatures work best, which suggests that ignoring the large negative logits can help. {#c-hi-intermediate}

**Logit regression.**
Ba and Caruana train their mimic networks on the teacher's logits rather than its probabilities, with the regression loss $L(W, \beta) = \frac{1}{2T}\sum_t \lVert g(x^{(t)}; W, \beta) - z^{(t)}\rVert_2^2$, where $T$ there counts the training examples, not a temperature. {#c-ba-logit-loss}
They explain the choice with a teacher that predicts the three targets with probabilities $[2\cdot10^{-9}, 4\cdot10^{-5}, 0.9999]$: a cross-entropy on these lets the student ignore the first two, while the logits $[10, 20, 30]$ keep them and their relative difference, and the logits $[-10, 0, 10]$ give the same probabilities although the teacher clearly models the two cases differently. {#c-ba-logit-example}
By eq-logit-matching their loss is the limit of distillation at high temperature, applied to logits that are not centred.

## Worked example

Take $N = 3$ classes, teacher logits $v = (3, 1, -4)$, student logits $z = (1, 0.5, -1.5)$ and label $y = 0$; both logit vectors have mean 0.

| quantity | class 0 | class 1 | class 2 |
|---|---|---|---|
| teacher $p$ at $T = 1$ | 0.880090 | 0.119107 | 0.000803 |
| student $q$ at $T = 1$ | 0.592201 | 0.359188 | 0.048611 |
| teacher $p$ at $T = 4$ | 0.561702 | 0.340689 | 0.097609 |
| student $q$ at $T = 4$ | 0.413606 | 0.365006 | 0.221387 |
| gradient $(q - p)/T$ at $T = 1$ | −0.287889 | 0.240081 | 0.047808 |
| gradient $(q - p)/T$ at $T = 4$ | −0.037024 | 0.006079 | 0.030945 |
| scaled gradient $T(q - p)$ at $T = 4$ | −0.592381 | 0.097268 | 0.495113 |
| scaled gradient at $T = 100$ | −0.667294 | −0.155354 | 0.822648 |
| limit $(z - v)/3$ | −0.666667 | −0.166667 | 0.833333 |

At $T = 1$ the teacher gives class 2 less than a thousandth, and the soft gradient on $z_2$ is small; at $T = 4$ class 2 gets nearly a tenth, and its gradient is the second largest, though the whole gradient is about eight times smaller than at $T = 1$, the $1/T^2$ of the derivation at work.
With label 0, $\alpha = 0.9$ and $T = 4$, the soft term $T^2 D_{\mathrm{KL}} = 16 \cdot 0.068488 = 1.095802$ and the hard term $-\log\operatorname{softmax}(z)_0 = 0.523909$ give the loss $0.9 \cdot 1.095802 + 0.1 \cdot 0.523909 = 1.038612$, and the gradients in the rows above follow eq-soft-gradient. {#c-we-gradient}
As $T$ grows the scaled gradient approaches $(z - v)/3$, with an error of about $1.13/T$, and $T^2 D_{\mathrm{KL}}$ approaches $\frac16\sum_i (z_i - v_i)^2 = 1.75$, reaching $1.732200$ at $T = 100$. {#c-limit-worked}
`distillation_loss(Z, V, labels, 4.0, 0.9)` returns $1.038612$ in both implementations, and `test_worked_example_gradients` checks the rows.

## Hyperparameters

**Temperature.**
On MNIST, Hinton et al. found that once each of the student's two hidden layers had 300 or more units, any temperature above 8 gave fairly similar results, but with only 30 units per layer a temperature between 2.5 and 4 worked significantly better than higher or lower values. {#c-hi-temperature-range}
For speech they tried temperatures of 1, 2, 5 and 10, with a relative weight of 0.5 on the cross-entropy with the hard targets. {#c-hi-speech-settings}
Stanton et al. suggest a moderate temperature such as $\tau = 4$ when the student is much smaller than the teacher, so its capacity goes to matching the teacher's top predictions rather than the whole distribution. {#c-st-moderate}

**Weight of the hard term.**
Hinton et al. got their best results with a considerably lower weight on the hard term, as in The method, which in eq-kd-loss means $\alpha$ near 1; with $T^2$ in place, $\alpha$ and $T$ can be tuned separately.
FitNets weight the soft term by $\lambda$ and anneal it with a linear decay during training, so that examples on which the teacher is confident count most at first and harder ones later. {#c-fn-anneal}

**Transfer set.**
The transfer set matters as much as the loss; Ba and Caruana found that reusing the training points as unlabeled data costs accuracy, as the Evidence section reports, and Stanton et al. found that augmenting it changes fidelity, as the Limitations section reports.

## Cost

Training needs the teacher's logits on every transfer example: one forward pass of the teacher per example and epoch, or a single pass whose logits are stored, which is only possible without random augmentation.
The soft term itself is one more softmax and sum over the classes.
DistilBERT was trained on 8 16GB V100 GPUs for about 90 hours, against 1 day on 1024 32GB V100 GPUs for RoBERTa, the comparison its authors give. {#c-db-compute}
Ba and Caruana report that their deep models took 8 to 12 hours to train on GTX 580 GPUs, while the shallow mimic models trained much faster and reached similar accuracy in 1 to 2 hours. {#c-ba-train-time}
At inference only the student runs.

## Evidence

**MNIST.**
Hinton et al. trained a teacher with two hidden layers of 1200 rectified linear units on all 60,000 training cases, strongly regularized with dropout and weight constraints, on inputs jittered by up to two pixels. {#c-hi-mnist-teacher}
The teacher made 67 test errors and a smaller unregularized network with two hidden layers of 800 units made 146; regularized only by matching the teacher's soft targets at a temperature of 20, the smaller network made 74, evidence that soft targets transferred the knowledge about translations even though the transfer set contained none. {#c-hi-mnist}
When all examples of the digit 3 were left out of the transfer set, the distilled model made 206 test errors, 133 of them on the 1010 threes; raising the bias of class 3 by 3.5 brought this to 109 errors, 14 of them on threes, so it got 98.6% of the test threes right without having seen one. {#c-hi-no-threes}
With only the 7s and 8s of the training set as the transfer set, the distilled model made 47.3% test errors, which fell to 13.2% when the biases of 7 and 8 were reduced by 7.6. {#c-hi-only-78}

**Speech.**
The baseline acoustic model had 8 hidden layers of 2560 rectified linear units, a softmax over 14,000 labels and about 85M parameters, and was trained on about 2000 hours of spoken English, about 700M examples, reaching 58.9% frame accuracy and 10.9% word error rate on the development set. {#c-hi-speech-baseline}
An ensemble of 10 such models reached 61.1% and 10.7%, and a single model of the same size distilled from it reached 60.8% and 10.7%, so more than 80% of the ensemble's gain in frame accuracy transferred to the distilled model. {#c-hi-speech-table}
Hinton et al. contrast this with related work that distilled at a temperature of 1 with a large unlabeled set and closed only 28% of the gap between the error rates of the large and the small model. {#c-hi-li}

**Soft targets as regularizers.**
Trained on 3% of the speech data, about 20M examples, the 85M-parameter baseline overfit severely with hard targets and was stopped early at 44.5% test frame accuracy, while the same model trained with soft targets converged to 57.0%, without early stopping, against 58.9% for the baseline on all the data. {#c-hi-three-percent}

**Shallow mimics.**
On TIMIT, Ba and Caruana's shallow networks trained on the original data had phone error rates of 23.1% to 23.6%, against 21.9% for a deep network, 19.5% for a convolutional one and 18.5% for an ensemble of 9 convolutional networks with about 125M parameters. {#c-ba-timit-baselines}
Shallow mimics of that ensemble reached 21.6% with about 12M parameters and 20.0% with about 180M, as accurate as the deep network of similar size and close to the convolutional one, with one hidden layer and no convolution or pooling. {#c-ba-timit-mimic}
The shallow mimic never achieves the accuracy of the ensemble of CNNs it is trying to mimic, which Ba and Caruana put down to there not being enough unlabeled data, but it is able to match or exceed the accuracy of the deep nets. {#c-ba-never-reached}
With no unlabeled TIMIT data, they reused the 1.1M training points without their labels, which reduced the accuracy of the mimics; compression works best when the unlabeled set is much larger than the training set and its samples do not fall on training points where the teacher may have overfit. {#c-ba-transfer-size}
On CIFAR-10, with the first 1 million images of the 80 million tiny images dataset added to the 50,000 training images as the transfer set, shallow mimics with one convolutional layer reached 15.4% error mimicking a single convolutional teacher with 12.0% error, and 14.2% mimicking an ensemble with 11.0%, against 21.8% for a shallow network trained on the labels. {#c-ba-cifar}
Ba and Caruana find that logit regression worked better than their other mimic losses, the KL divergence from the teacher's to the student's probabilities and an L2 loss on the probabilities. {#c-ba-logits-better}
They offer three reasons a mimic can beat a network trained on the labels: the teacher may remove label errors, it may give simpler soft labels where the true function is hard to learn, and soft targets are non-zero for most outputs where hard labels are non-zero for one of 183. {#c-ba-why}
Ba and Caruana see the mechanisms of mimic training as forms of regularization that help prevent overfitting in the student, since shallow models trained on the original targets are more prone to overfitting than deep models, and increasing the accuracy of the teacher by forming an ensemble yielded a commensurate increase in the accuracy of the student. {#c-ba-regularizer}

**Hints.**
On CIFAR-10, a FitNet with about 2.5M parameters reached 91.61% accuracy against 90.18% for its teacher with about 9M, and well above the 85.8% of Ba and Caruana's best mimic with roughly 28 times fewer parameters. {#c-fn-cifar}
On MNIST the same deep thin student had 1.9% error with ordinary backpropagation, 0.65% with distillation and 0.51% with hints, slightly better than the teacher with 12 times fewer parameters. {#c-fn-mnist}

**Language models.**
In parameters counted in millions, DistilBERT has 66 against 110 for BERT-base, takes 410 seconds against 668 for a pass over the STS-B development set on a CPU, and scores 77.0 against 79.5 on the GLUE development sets, 97% of BERT's score with 40% fewer parameters. {#c-db-glue}
On IMDb it is 0.6 points behind BERT in test accuracy, and on SQuAD v1.1 within 3.9 points; on an iPhone 7 Plus it ran 71% faster than BERT for question answering, as a model of 207 MB. {#c-db-downstream}
In the ablation of DistilBERT, removing the distillation loss lowered the GLUE macro-score by 2.96 points, removing the cosine loss by 1.46 and removing the masked language modeling loss by 0.31, while a random initialization in place of the teacher's layers cost 3.69. {#c-db-ablation}

**Fidelity.**
Stanton et al. distinguish fidelity, the ability of a student to match its teacher's predictions, from generalization, its performance on unseen in-distribution data. {#c-st-definitions}
They measure fidelity by the average top-1 agreement between student and teacher and by the average KL divergence from the teacher's predictive distribution to the student's. {#c-st-metrics}
A LeNet-5 teacher distilled into an identical student on MNIST, with enough examples, learned to make the same predictions as the teacher, with over 99% top-1 test agreement. {#c-st-lenet}
In self-distillation a student can exceed its teacher only by failing to match it, while when distilling an ensemble of three ResNet-56 teachers, fidelity becomes positively correlated with generalization. {#c-st-self}
To the question "does knowledge distillation really work?" they answer yes, in the sense that it often improves student generalization, and no, in that it often fails to live up to its name, transferring very limited knowledge from teacher to student. {#c-st-yes-no}

**Label-smoothed teachers.**
On translated MNIST, Müller et al. found that a teacher trained with hard targets and dropout, at 0.67% test error, distilled into a student with 0.74%, while a teacher trained with label smoothing, at 0.59%, gave a much worse student, at 0.91%. {#c-mu-mnist-distill}
In their CIFAR-10 experiments a teacher with better accuracy was not necessarily the one that distilled better. {#c-mu-better-teacher}

## Limitations and failure modes

**The student rarely matches the teacher.**
Stanton et al. find that a surprisingly large difference often remains between the predictive distributions of teacher and student, even when the student has the capacity to match the teacher exactly, and identify optimization difficulty as a key reason. {#c-st-abstract}
Distilling a ResNet-56 teacher trained on CIFAR-100 into an identical student, enlarging the distillation data with samples from a GAN improved fidelity only modestly, nowhere near 99% test agreement. {#c-st-resnet}
Of the augmentation policies they tried, the best for fidelity, Mixup at $\tau = 4$, reached only 86% test agreement, and the baseline policy at $\tau = 4$ already reached 84.5%. {#c-st-augmentation}
For self-distillation of a ResNet-20 on CIFAR-100, training for 5k epochs instead of 300 raised train agreement only from 78.95% to 83.3%, and switching from SGD to Adam only reduced fidelity. {#c-st-optimizer}
A student initialized as a mix $\lambda\theta_t + (1 - \lambda)\theta_r$ of the teacher's weights and random ones converged to a sub-optimal distillation loss for $\lambda \le 0.25$, and to the optimum, with a sharp rise in agreement, from $\lambda = 0.375$. {#c-st-init}
They conclude that good student accuracy does not imply good fidelity, and that enlarging the distillation data beyond the teacher's training data helps identify the solution but makes an already difficult optimization harder. {#c-st-findings}

**Depth.**
FitNets report that as the student gets deeper than its teacher, distillation still suffers from the difficulty of optimizing deep networks. {#c-fn-kd-depth}
Under a budget of about 30M multiplications they could not train students with more than 5 layers by backpropagation, and with distillation only up to 7. {#c-fn-depth-budget}

**It needs data, or a very good teacher.**
Ba and Caruana call their method awkward, because it depends on a large unlabeled data set, on a teacher of very high accuracy, or on both. {#c-ba-awkward}

**A better teacher is not always a better source.**
The label-smoothing results under Evidence indicate that a teacher's accuracy does not predict its student's; Müller et al. found that teachers trained with label smoothing were no better, and sometimes worse, than training the student with label smoothing directly. {#c-mu-smoothed-teachers}
They add that distillation is only useful if it gives the student more than label smoothing alone, which is simpler because it needs no teacher. {#c-mu-only-useful}

**The low-temperature trade-off has no general answer.**
Whether to keep the very negative logits, which may be noise or may be knowledge, is left as an empirical question, as in the Derivation; the temperature has to be tuned.

## Variants

**Logit matching.**
Regressing the student's logits on the teacher's with a squared error, the method of Ba and Caruana, is the high-temperature limit of eq-logit-matching; `logit_matching_loss` in the NumPy implementation computes it.

**Hints.**
FitNets train a student that is deeper and thinner than its teacher, using the teacher's intermediate representations as hints as well as its outputs, and add parameters that map a student hidden layer to a prediction of the teacher's hidden layer. {#c-fn-abstract}
A hint is the output of a teacher hidden layer, a chosen guided layer of the student learns to predict it, and the loss is $\frac12\lVert u_h(x) - r(v_g(x))\rVert^2$, with $u_h$ and $v_g$ the teacher and student up to the hint and guided layers and $r$ a regressor on the guided layer. {#c-fn-hint-loss}
Hints regularize, so the deeper the guided layer the more the student risks being over-regularized; FitNets take the middle layers of both networks. {#c-fn-middle}
Training goes in two stages: first the student up to the guided layer is pre-trained to predict the hint, then the whole network is trained with distillation. {#c-fn-stagewise}
FitNets write the distillation loss as $H(y_{\text{true}}, P_S) + \lambda H(P_T^\tau, P_S^\tau)$, with $\lambda$ on the soft term and no $\tau^2$. {#c-fn-kd-loss}
`hint_loss` in both implementations computes the hint loss with a linear regressor.

**Distilling during pre-training.**
DistilBERT leverages knowledge distillation during the pre-training phase of a smaller general-purpose language representation model, where most prior work built task-specific models, and introduces a triple loss combining language modeling, distillation and cosine-distance losses. {#c-db-abstract}
Its distillation loss over the teacher's soft target probabilities is written $L_{ce} = \sum_i t_i \log(s_i)$, with the same temperature for teacher and student during training and $T = 1$ at inference, and it is combined linearly with the masked language modeling loss and a cosine loss that aligns the directions of the student's and teacher's hidden states. {#c-db-loss}
The printed $L_{ce}$ has no minus sign; minimized as written it would push the student away from the teacher, so the sign must be restored.
The student keeps BERT's general architecture with half the layers and without the token-type embeddings and the pooler, and is initialized from the teacher by taking one layer out of two. {#c-db-student}

**Ensembles and specialists.**
Stanton et al. combine the logits of an ensemble of $m$ teachers as $z_t = \log\big(\sum_i \sigma(z_i)/m\big)$, the log of the averaged predictive distribution. {#c-st-ensemble}
`ensemble_soft_targets` in the NumPy implementation computes this arithmetic mean and the geometric mean that Hinton et al. also allow.
Starting from a trained baseline network on JFT, Hinton et al. also trained 61 specialist models, each with 300 classes plus a dustbin class, and with them got a 4.4% relative improvement in test accuracy overall. {#c-hi-specialists}
They had not yet found a way to distill the knowledge in the specialists back into the single large net. {#c-hi-specialists-not-distilled}

## When to use it

Use distillation when a model must be cheaper to run than the best one you can train, and that best model, or an ensemble, already exists or can be trained.
Hinton et al. report that distilling works very well for transferring knowledge into a smaller, distilled model, from a large, highly regularized model or from an ensemble. {#c-hi-discussion}
Prefer a large transfer set from the same distribution, or the training set with the hard-label term, and a teacher clearly better than a student trained alone; a teacher no better than the student gives little to transfer.
Start from $T$ around 4 and $\alpha$ around 0.9 in eq-kd-loss, and tune $T$ lower for a much smaller student.
Do not train the teacher with label smoothing if it is meant to be distilled.
Do not expect the student to reproduce the teacher's predictions; measure agreement and predictive KL if fidelity matters, with `top1_agreement` and `predictive_kl`.
For deep thin students, add hints from a middle layer.
Distillation is the usual way to deploy what ensemble learning gains: the ensemble is trained once as the teacher and only the student runs.
Neural network pruning is the other common route to a cheaper network; it removes weights from the trained model itself, where distillation trains a separate, smaller one.
Data augmentation of the transfer set changes the inputs on which the student is asked to match the teacher, and the augmentation results under Limitations indicate that it does not by itself close the fidelity gap.

## Pitfalls

> [!pitfall]
> **Swapping the arguments of `F.kl_div`.** `F.kl_div(input, target)` computes $D_{\mathrm{KL}}(\text{target}\,\Vert\,\exp(\text{input}))$ and expects log-probabilities as `input`. The student's log-softmax goes first and the teacher's distribution second; swapping them computes $D_{\mathrm{KL}}(q\Vert p)$, and passing probabilities as `input` gives a meaningless number. The tests `test_kl_div_argument_order` and `test_probabilities_as_input_is_wrong` show both.

> [!pitfall]
> **`reduction="mean"` in `F.kl_div`.** It divides by the number of elements, batch times classes, not by the batch size, so the soft term comes out $N$ times too small. Use `reduction="batchmean"`. `test_kl_div_mean_reduction_divides_by_classes` checks the factor.

> [!pitfall]
> **Leaving out $T^2$.** Without it the soft gradient shrinks like $1/T^2$, so raising $T$ quietly turns the method into training on the hard labels. `test_without_t2_the_soft_term_fades` and `test_t2_keeps_the_gradient_scale` check the scale with and without the factor.

> [!pitfall]
> **Which term $\alpha$ weights.** Hinton et al. recommend a low weight on the hard term, FitNets put $\lambda$ on the soft term, and Stanton et al. put $\alpha$ on the negative log-likelihood, so a value copied from one paper to another code base can mean its complement. Here $\alpha$ weights the soft term, and `test_alpha_endpoints` checks that $\alpha = 0$ is plain cross-entropy and $\alpha = 1$ the soft term alone.

> [!pitfall]
> **Training the teacher by accident.** The teacher's logits must not carry gradient, or the loss pulls the teacher toward the student. Compute them under `torch.no_grad()` or detach them, and keep the teacher in evaluation mode so dropout and batch statistics do not change its targets. `test_teacher_receives_no_gradient` checks that the loss leaves the teacher's logits without gradient.

## History

Cristian Buciluǎ, Rich Caruana and Alexandru Niculescu-Mizil published "Model compression" in the proceedings of the 12th ACM SIGKDD conference, KDD 2006, on pages 535 to 541. {#c-hist-bucila}
Lei Jimmy Ba and Rich Caruana's "Do Deep Nets Really Need to be Deep?", arXiv 1312.6184, is marked in its arXiv version as a draft for NIPS 2014. {#c-hist-ba}
Geoffrey Hinton, Oriol Vinyals and Jeff Dean of Google posted "Distilling the Knowledge in a Neural Network" as arXiv 1503.02531 in March 2015. {#c-hist-hinton}
Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta and Yoshua Bengio published "FitNets: Hints for Thin Deep Nets" as a conference paper at ICLR 2015. {#c-hist-fitnets}
Victor Sanh, Lysandre Debut, Julien Chaumond and Thomas Wolf of Hugging Face presented DistilBERT at the 5th EMC² workshop, co-located with NeurIPS 2019. {#c-hist-distilbert}
Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander Alemi and Andrew Gordon Wilson published "Does Knowledge Distillation Really Work?" at NeurIPS 2021 in Sydney. {#c-hist-stanton}
