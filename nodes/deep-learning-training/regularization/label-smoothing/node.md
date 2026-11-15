---
tag: NEW-label-smoothing
kind: method
slug: label-smoothing
name: Label smoothing
aliases: [LSR, label-smoothing regularization, label smoothing regularization, uniform label smoothing, smoothed targets]
area: deep-learning-training/regularization
level: core
summary: >-
  Train a classifier on targets that give the true class 1 − ε and spread ε over all classes, so the cross-entropy stops rewarding ever larger logits.
purpose: regularization
uses: [cross-entropy@m2, kullback-leibler-divergence@m2, shannon-entropy@m2]
related_to: [dropout, weight-decay@al, knowledge-distillation@al, cross-entropy-loss@al, data-augmentation@al, beam-search@al, overfitting@al, regularization@al, early-stopping@al, mutual-information@m2, gibbs-inequality@m2]
hyperparameters:
  - {name: epsilon, meaning: "the smoothing weight ε, the label_smoothing of torch.nn.functional.cross_entropy and nn.CrossEntropyLoss, spread uniformly over all K classes so the label gets 1 − ε + ε/K", typical: "0.1", default_in: {pytorch: 0.0}}
  - {name: prior, meaning: "the distribution u that receives the mass ε; uniform by default, the unigram frequencies of the labels in unigram smoothing", typical: "uniform"}
cost:
  train: "one extra sum over the K log-probabilities of each example, and no extra memory"
  inference: "none"
history:
  year: 2016
  people: [Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna]
  note: "Introduced in Rethinking the Inception Architecture for Computer Vision, posted to arXiv in December 2015 and published at CVPR 2016; Pereyra et al. (2017) related it to the confidence penalty, and Müller, Kornblith and Hinton (NeurIPS 2019) studied what it does to representations, calibration and distillation."
prereqs: [cross-entropy@m2, cross-entropy-loss@al, softmax-regression@al, overfitting@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1512.00567", "1906.02629", "1701.06548", "1706.03762"]
  al: label-smoothing
sources:
  - {paper: "Szegedy, Vanhoucke, Ioffe, Shlens and Wojna (2016)", title: "Rethinking the Inception Architecture for Computer Vision", locator: "the title page, pages 6 and 7 (Section 7)", note: "the definition, the pair of losses, label dropout and the ImageNet result"}
  - {paper: "Müller, Kornblith and Hinton (2019)", title: "When Does Label Smoothing Help?", locator: "the abstract, Sections 1 to 6, Tables 1 to 3", note: "penultimate-layer clusters, calibration, translation and the harm to distillation"}
  - {paper: "Pereyra, Tucker, Chorowski, Kaiser and Hinton (2017)", title: "Regularizing Neural Networks by Penalizing Confident Output Distributions", locator: "the abstract, Sections 1, 3, 3.1, 3.2, 4.1.1 to 4.4.2, page 5", note: "the confidence penalty, the reversed KL divergence and six benchmarks"}
  - {paper: "Vaswani et al. (2017)", title: "Attention Is All You Need", locator: "Section 5.4", note: "label smoothing of 0.1 in the Transformer"}
---

Label smoothing trains a classifier against a softened copy of its labels.
Instead of asking for probability 1 on the true class, it asks for $1 - \varepsilon$ there and spreads $\varepsilon$ over all the classes, uniformly unless another prior is chosen.
The loss is then a weighted sum of the usual negative log-likelihood and the cross-entropy with the uniform distribution, so it stops rewarding the network for pushing the true logit ever further above the others.
Szegedy et al. introduced it in 2015 for the Inception networks on ImageNet, and it has since been a standard setting in image classification, speech recognition and machine translation, where the Transformer trains with $\varepsilon = 0.1$.
It tends to improve accuracy and calibration, but it makes the network forget how similar the wrong classes are to each other, which harms knowledge distillation from the smoothed network.

## In plain words

A classifier trained with cross-entropy on hard labels is told that the right answer has probability exactly 1 and every other answer exactly 0.
It can never reach that with finite numbers, so it keeps making the winning score larger, and becomes more and more sure of itself, even on examples it gets wrong.
Label smoothing tells it instead: the right answer has probability 0.9, say, and the remaining 0.1 is shared by all the classes.
That target can be reached exactly, so the network stops once its scores are far enough apart, and its confidence stays near what the target asks for.
The same loss is what one gets, on average, from relabelling a random tenth of the examples with a random class, without the noise of actually doing it.

## Setting

A network computes logits $z_1, \dots, z_K$ for an input $x$ and $K$ classes.
Szegedy et al. turn them into the probability of each label $k$ by the softmax $p(k \mid x) = \exp(z_k)/\sum_i \exp(z_i)$, where the logits are unnormalized log-probabilities. {#c-sz-softmax}
The loss of one example is the cross-entropy $\ell = -\sum_k \log p(k)\, q(k)$ against the ground-truth distribution $q(k \mid x)$, and minimizing it maximizes the expected log-likelihood of a label drawn from that distribution. {#c-sz-loss}
Its gradient in the logits is simply $\partial\ell/\partial z_k = p(k) - q(k)$, which lies between $-1$ and $1$. {#c-sz-gradient}
With a single label $y$, $q(k) = \delta_{k,y}$, and the log-likelihood is largest when $p$ equals this one-hot vector, which finite logits cannot reach but approach when the logit of $y$ is much larger than all the others. {#c-sz-unreachable}
Szegedy et al. see two problems in this: the network may overfit, because putting all probability on each training label does not guarantee generalization, and the gap between the largest logit and the rest grows, which together with the bounded gradient makes the model less able to adapt. {#c-sz-two-problems}
In their reading the network becomes too confident, and they propose a way to make it less so; this works against the log-likelihood of the training labels, but it regularizes the model and keeps it adaptable. {#c-sz-too-confident}
Pereyra et al. describe the same thing through entropy: a confident prediction is an output distribution with low entropy, and a network that puts all its probability on one class on the training set is overconfident, often a symptom of overfitting. {#c-pe-overconfident}

## The method

Label smoothing replaces the one-hot target of an example with label $y$ by

$$ q'(k \mid x) = (1 - \varepsilon)\,\delta_{k,y} + \varepsilon\, u(k) , \qquad \text{uniform: } q'(k) = (1 - \varepsilon)\,\delta_{k,y} + \frac{\varepsilon}{K} , $$ {#eq-smoothed-target}

and trains with the cross-entropy $H(q', p) = -\sum_k q'(k)\log p(k)$ in place of $H(q, p)$.
The new target mixes the ground-truth distribution with a fixed distribution $u$ that does not depend on the example, with weights $1 - \varepsilon$ and $\varepsilon$. {#c-sz-mixture}
Szegedy et al. suggest the prior distribution of the labels for $u$, and in their experiments use the uniform $u(k) = 1/K$; they call the change label-smoothing regularization, or LSR. {#c-sz-uniform}
Müller et al. write the same targets with $\alpha$ for $\varepsilon$, as $y_k^{LS} = y_k(1 - \alpha) + \alpha/K$, and minimize the cross-entropy between them and the outputs $p_k$. {#c-mu-ls-definition}
The target $q'$ is also the distribution of a noisy label: take $y$, and with probability $\varepsilon$ replace it by a draw from $u$. {#c-sz-sampling}
Szegedy et al. offer it as a regularizer of the classifier layer, an estimate of what label-dropout does to training once its randomness is averaged out. {#c-sz-label-dropout}
Every $q'(k)$ is at least $\varepsilon u(k) > 0$, so a prediction that puts nearly all its mass on one class has a large cross-entropy with $q'$; this keeps the largest logit from running far above the others. {#c-sz-lower-bound}

In PyTorch this is `torch.nn.functional.cross_entropy(logits, target, label_smoothing=eps)`, which uses the uniform $u$ over all $K$ classes, so the label gets $1 - \varepsilon + \varepsilon/K$.

## Derivation

**The pair of losses.**
Szegedy et al. expand the smoothed cross-entropy as $H(q', p) = (1 - \varepsilon)H(q, p) + \varepsilon H(u, p)$, so LSR replaces one cross-entropy loss with a pair of them, against the label and against the prior. {#c-sz-pair}
Because the cross-entropy is linear in its target, for any distribution $u$ and any $\varepsilon \in [0, 1]$,

$$ H(q', p) = (1 - \varepsilon)\,(-\log p_y) + \varepsilon\, H(u, p) = (1 - \varepsilon)\,(-\log p_y) + \varepsilon\, D_{\mathrm{KL}}(u \,\Vert\, p) + \varepsilon \log K , $$ {#eq-decomposition}

where the second form takes $u$ uniform. {#c-decomposition}
The second loss penalizes how far $p$ is from the prior, and Szegedy et al. note that the KL divergence measures the same thing, since $H(u, p) = D_{\mathrm{KL}}(u\Vert p) + H(u)$ and $H(u)$ is fixed. {#c-sz-kl}
So with uniform $u$, up to the constant $\varepsilon\log K$, label smoothing adds $\varepsilon D_{\mathrm{KL}}(u \Vert p)$ to $1 - \varepsilon$ times the negative log-likelihood. {#c-kl-form}
Pereyra et al. state it the same way: label smoothing estimates the marginalized effect of label noise, and with a uniform prior it adds the KL divergence from the uniform distribution to the prediction to the negative log-likelihood. {#c-pe-ls-kl}

**Label dropout.**
If the label is kept with probability $1 - \varepsilon$ and otherwise replaced by a draw from $u$, the noisy label has distribution $q'$, and since the cross-entropy is linear in the target, the expected loss over the noise is exactly $H(q', p)$. {#c-marginalized}
Label smoothing is label dropout with the noise averaged out.

**The gradient.**
Since $\partial\log p_k/\partial z_i = [k = i] - p_i$, the gradient of the smoothed loss in the logits is $p - q'$, with entries in $[-1, 1]$ that sum to 0; with $\varepsilon = 0$ it is the $p - q$ of hard targets. {#c-gradient}
The gradient pushes the label's logit up only while $p_y < 1 - \varepsilon + \varepsilon/K$, and pushes it down beyond that.

**The confidence penalty is the other direction.**
Szegedy et al. remark that with uniform $u$ the term $H(u, p)$ measures how far $p$ is from uniform, which the negative entropy $-H(p)$ would also measure, though not equivalently, and that they did not try it. {#c-sz-negentropy}
Pereyra et al. did: their confidence penalty adds $-\beta H(p_\theta(y \mid x))$ to the negative log-likelihood, with $\beta$ setting its strength. {#c-pe-penalty}
For uniform $u$, $-H(p) = D_{\mathrm{KL}}(p \Vert u) - \log K$, so up to a constant the confidence penalty is $\beta D_{\mathrm{KL}}(p\Vert u)$, while label smoothing penalizes $D_{\mathrm{KL}}(u \Vert p)$, the same divergence in the other direction. {#c-confidence-relation}
Pereyra et al. make this point: reversing the KL divergence of label smoothing gives the confidence penalty, which suggests regularizers with targets other than the uniform distribution. {#c-pe-reverse}
They give the gradient of the entropy in the $i$-th logit as $p_\theta(y_i \mid x)(-\log p_\theta(y_i \mid x) - H(p_\theta))$, a weighted deviation from the mean. {#c-pe-entropy-grad}
The two directions behave differently: $D_{\mathrm{KL}}(u\Vert p)$ grows without bound as any $p_k \to 0$, while $D_{\mathrm{KL}}(p\Vert u)$ never exceeds $\log K$.
The derivation `smoothed-cross-entropy` gives the steps; its check tests the identities on random logits, label dropout by Monte Carlo and the gradient by finite differences.

**Distillation looks similar.**
In knowledge distillation as Müller et al. write it, the loss is $(1 - \beta)H(y, p) + \beta H(p^t(T), p(T))$, with the teacher's and the student's outputs at temperature $T$. {#c-mu-distill-loss}
It has the form of eq-decomposition with the uniform $u$ replaced by the teacher's output for each example.

**Where training stops.**
For a fixed target $q'$, Gibbs' inequality says that the cross-entropy of one example is smallest exactly at $p = q'$, where it equals the entropy $H(q')$, which is positive for $\varepsilon > 0$. {#c-optimum}
Writing $p = q'$ in terms of logits, every wrong logit sits the same distance below the correct one:

$$ z_y - z_k = \log\frac{1 - \varepsilon + \varepsilon/K}{\varepsilon/K} = \log\Big(1 + \frac{K(1 - \varepsilon)}{\varepsilon}\Big) \qquad (k \neq y) , $$ {#eq-logit-gap}

while with $\varepsilon = 0$ no finite logits minimize the loss, and gradient descent widens the gap without bound. {#c-logit-gap}
For ImageNet with $K = 1000$ and $\varepsilon = 0.1$, the gap is $\log 9001 \approx 9.105$, and the largest probability the loss rewards is $0.9001$. {#c-gap-imagenet}
Some code instead gives $1 - \varepsilon$ to the label and $\varepsilon/(K-1)$ to each other class; that is uniform smoothing with $\varepsilon' = \varepsilon K/(K-1)$. {#c-convention}
The derivation `optimal-logits` gives the steps, and its check runs gradient descent on free logits to the gap.
Müller et al. describe the same effect: label smoothing pushes the gap between the correct logit and each incorrect one toward a constant that depends on $\alpha$, while hard targets let the correct logit grow far above the others and leave the incorrect logits free to differ. {#c-mu-constant-gap}

## Worked example

Take $K = 3$ classes, logits $z = (2, 0, 0)$, label $y = 0$ and $\varepsilon = 0.3$.

| quantity | class 0 | class 1 | class 2 |
|---|---|---|---|
| target $q'$ | 0.8 | 0.1 | 0.1 |
| $p = \operatorname{softmax}(z)$ | 0.786986 | 0.106507 | 0.106507 |
| $-\log p_k$ | 0.239545 | 2.239545 | 2.239545 |
| gradient $p - q'$ | −0.013014 | 0.006507 | 0.006507 |
| hard-target gradient $p - q$ | −0.213014 | 0.106507 | 0.106507 |

The negative log-likelihood is $0.239545$ and $H(u, p) = (0.239545 + 2 \cdot 2.239545)/3 = 1.572878$, so the smoothed loss is $0.7 \cdot 0.239545 + 0.3 \cdot 1.572878 = 0.639545$.
In the KL form, $D_{\mathrm{KL}}(u\Vert p) = 1.572878 - \log 3 = 0.474266$, and $0.7 \cdot 0.239545 + 0.3 \cdot 0.474266 + 0.3 \log 3$ gives the same $0.639545$.
`F.cross_entropy(torch.tensor([[2., 0., 0.]]), torch.tensor([0]), label_smoothing=0.3)` returns this number, and its gradient is the row above.
The optimal gap of eq-logit-gap is $\log(1 + 3 \cdot 0.7/0.3) = \log 8 = 2.0794$, a little above the current gap of 2, so the gradient still raises $z_0$, but only by a twentieth of what hard targets would ask.
At the optimum $p = q'$ the loss is $H(q') = 0.639032$.

## Hyperparameters

The one hyperparameter is $\varepsilon$, the `label_smoothing` of `torch.nn.functional.cross_entropy` and `nn.CrossEntropyLoss`, which defaults to 0.
The usual value is 0.1: Szegedy et al. used it on ImageNet, Vaswani et al. in the Transformer, and Pereyra et al. found it best on MNIST, CIFAR-10, Penn Treebank and WMT'14, as the Evidence section reports.
Müller et al. used 0.05 for ResNet-56 on CIFAR-100.
The prior $u$ is uniform unless the labels are strongly unbalanced, as under Variants.

## Cost

The uniform part of the loss is one more sum over the $K$ log-probabilities that the cross-entropy already computes, and its gradient one more subtraction.
Nothing changes at inference.

## Evidence

On ImageNet, with $K = 1000$, $u(k) = 1/1000$ and $\varepsilon = 0.1$, Szegedy et al. report a consistent gain of about 0.2% absolute in both top-1 and top-5 error on ILSVRC 2012. {#c-sz-imagenet}
Müller et al. collect earlier results: errors of 23.1 and 22.8 top-1 and 6.3 and 6.1 top-5 for Inception on ImageNet without and with smoothing, BLEU of 25.3 and 25.8 with perplexity 4.67 and 4.92 on English-German translation, and word error rates of 8.9 and 7.0/6.7 on WSJ. {#c-mu-table1}
Pereyra et al. compared the confidence penalty and label smoothing on six common benchmarks across image classification, language modeling, translation and speech recognition, and found that both improve strong models without changing their other hyperparameters. {#c-pe-abstract}

**What the outputs look like.**
On the MNIST validation set, a two-layer fully connected network with 1024 units and dropout gives softmax probabilities of 0 or 1, while label smoothing and the confidence penalty both give smoother output distributions. {#c-pe-fig1}
On permutation-invariant MNIST, smoothing values from 0.05 to 0.5 were tried and 0.1 was best, with test errors of 1.28 ± 0.06% for dropout, 1.23 ± 0.06% for label smoothing and 1.17 ± 0.06% for the confidence penalty. {#c-pe-mnist}
Both output regularizers had smaller gradient norms and converged faster than dropout, which Pereyra et al. explain by the large gradient a peaked output receives on a misclassified example. {#c-pe-gradnorm}

**Image, language and speech benchmarks of Pereyra et al.**
On CIFAR-10 without augmentation, a 40-layer DenseNet reached 7.04% test error with dropout, 6.89% with label smoothing added, where 0.1 was best, and 6.77% with the confidence penalty added; with augmentation neither the confidence penalty nor dropout helped. {#c-pe-cifar}
On word-level Penn Treebank the confidence penalty was clearly best, with test perplexities of 77.7 for label noise, 76.6 for label smoothing, 76.3 for unigram smoothing and 74.7 for the confidence penalty; 0.1 was the best smoothing value. {#c-pe-ptb}
On WMT'14 English-German, 0.1 was best for uniform and unigram smoothing, and label smoothing was a little ahead of the confidence penalty; each gave just over one BLEU point without dropout, and much less with it, from 23.41 to 23.52 test BLEU. {#c-pe-wmt}
On WSJ speech recognition all three helped, unigram smoothing the most and the least sensitive to its value: the test word error rate went from 14.2 to 11.3 with uniform and 11.0 with unigram smoothing. {#c-pe-wsj}

**Representations.**
Müller et al. read each logit $x^\top w_k$ as a measure of the squared distance between the penultimate activations $x$ and a class template $w_k$, because $\lVert x - w_k\rVert^2 = x^\top x - 2x^\top w_k + w_k^\top w_k$, the first term drops out of the softmax and the last is usually about the same for every class. {#c-mu-template}
In this reading, label smoothing pulls each example close to the template of its class and to equal distances from the templates of the other classes. {#c-mu-equidistant}
To see it, they take three classes, find an orthonormal basis of the plane through their three templates, and project the penultimate activations of examples of those classes onto it. {#c-mu-visualization}
For AlexNet on CIFAR-10, hard targets give broad but distinct clusters, and smoothing with 0.1 gives much tighter clusters arranged in a regular triangle, at nearly the same accuracy. {#c-mu-tight}
The networks of these pictures have top-1 accuracies without and with smoothing of 0.1 of 86.8 ± 0.2 and 86.7 ± 0.3 for AlexNet on CIFAR-10, 72.1 ± 0.3 and 72.7 ± 0.3 for ResNet-56 on CIFAR-100, and 80.9 and 80.9 for Inception-v4 on ImageNet. {#c-mu-table2}
With smoothing, the difference between two logits must stay bounded to hit the soft target, while hard targets let the projections reach much larger values, which are over-confident predictions. {#c-mu-scale}
For two similar ImageNet classes, toy poodle and miniature poodle, hard targets give nearby clusters with round spread, and label smoothing puts the two classes on an arc. {#c-mu-poodles}
With hard targets the projections change continuously between the tench cluster and the poodles, so one could read off how much a poodle resembles a tench; with smoothing that information is almost gone. {#c-mu-erased}

**Calibration.**
Müller et al. recall that Guo et al. found modern networks poorly calibrated and over-confident, measured this by the expected calibration error (ECE), and reduced it by temperature scaling, multiplying the logits by a scalar before the softmax. {#c-mu-guo}
They report that label smoothing reduces ECE as well, and can calibrate a network without temperature scaling. {#c-mu-calibrates}
Their ECE values are 0.150 with hard targets, 0.021 with temperature scaling at $T = 1.9$ and 0.024 with smoothing of 0.05 for ResNet-56 on CIFAR-100; 0.071, 0.022 at $T = 1.4$ and 0.035 with 0.1 for Inception-v4 on ImageNet; and 0.056, 0.018 at $T = 1.13$ and 0.019 with 0.1 for the Transformer on English-German. {#c-mu-ece}
On ImageNet they did not tune $\alpha$ much, and 0.1 already brought ECE to 0.035, better than the network trained with hard targets and no scaling. {#c-mu-imagenet-untuned}
They find it somewhat surprising that networks that squeeze their training examples into tiny clusters still generalize and are calibrated; on the validation set the tight clusters spread toward the center and cover the whole range of confidences. {#c-mu-surprising}
In translation the soft outputs feed beam search, an approximation of maximum likelihood sequence detection that depends on calibration, so a better calibrated model should translate better. {#c-mu-beam}
For the Transformer on English-German, label smoothing gave better calibration and better BLEU but worse negative log-likelihood than hard targets at every temperature, and temperature scaling of the hard-target model did not recover the BLEU gain. {#c-mu-nll}
Müller et al. conclude that the BLEU gain comes despite the worse NLL, is only partly explained by calibration, and that changing the temperature of the smoothed network makes both calibration and BLEU worse. {#c-mu-partly}

**Distillation.**
On translated MNIST, a fully connected teacher trained with hard targets and dropout has 0.67% test error and distills into a student with 0.74%; trained with label smoothing instead, the teacher trains much faster and reaches 0.59%, but its student is much worse, at 0.91%. {#c-mu-mnist-distill}
Since both methods train on soft targets, distillation only pays if the student gains more than label smoothing alone would give it, and smoothing is simpler because it needs no teacher. {#c-mu-distill-useful}
Distilling a ResNet-56 teacher into an AlexNet student on CIFAR-10 with $\beta = 1$, larger $\alpha$ improved the teacher up to 0.6 and slightly hurt the student baseline; every student distilled from a hard-target teacher beat the student trained with smoothing, while smoothed teachers were no better, and sometimes worse, than smoothing the student directly. {#c-mu-cifar-distill}
Müller et al. explain this by the tight, equally separated clusters: every example of a class is about as close to the examples of each other class, so the more accurate teacher is not necessarily the better one to distill from. {#c-mu-better-teacher}
They estimate the mutual information between the index of $N = 600$ training examples from two classes and the difference of their two logits; it rises fast early in training and then falls slowly, more so with label smoothing, staying a little above $\log 2$, the value for two collapsed clusters. {#c-mu-mi}
They conclude that label smoothing makes the network treat every incorrect class as equally likely, while hard targets impose less structure and leave the logits free to vary across classes and examples, and that smoothing lowers the mutual information between the input and the output logit. {#c-mu-conclusion}

**Practice.**
Vaswani et al. trained the Transformer with label smoothing $\varepsilon_{ls} = 0.1$, which made perplexity worse, as the model learns to be more unsure, but improved accuracy and BLEU. {#c-vaswani}
Müller et al. note that it has improved accuracy in image classification, speech recognition and machine translation, and that many leading image classifiers have trained with it since Szegedy et al. proposed it for Inception. {#c-mu-used}
In speech, Chorowski and Jaitly used it to lower the word error rate on WSJ, and in translation it gave Vaswani et al. a small but important BLEU gain. {#c-mu-chorowski}
Chorowski and Jaitly found that temperature scaling and label smoothing both improve word error rates after beam search, label smoothing the most. {#c-mu-chorowski-beam}

## Limitations and failure modes

**It erases similarity between classes.**
The clusters it produces keep little of how far an example is from each wrong class, which is what a distillation teacher passes on; the Distillation paragraphs above report students made worse by a smoothed teacher.
Müller et al. cite Kornblith et al. as showing that label smoothing hurts transfer learning, which likewise relies on information in the last layers that is not about the training classes. {#c-mu-kornblith}

**It makes likelihood worse.**
The loss deliberately stops short of the maximum likelihood fit, so negative log-likelihood and perplexity go up even when accuracy and BLEU improve, as Vaswani et al. and Müller et al. report; a model selected by validation perplexity may look worse with it.

**Its effect depends on the rest of training.**
On WMT'14 the gain from smoothing nearly disappeared once dropout was used, and on CIFAR-10 with data augmentation the confidence penalty did not help, so a gain in one setup does not carry over to another.
In speech, Pereyra et al. had to mask the outputs for tokens that never appear as labels, such as the start-of-sequence token, for uniform smoothing and the confidence penalty to work. {#c-pe-masking}

**It is still not fully explained.**
Müller et al. write that, for all its use, label smoothing is still poorly understood. {#c-mu-poorly-understood}

## Variants

**Unigram smoothing** uses the label frequencies as $u$; Müller et al. note that Pereyra et al. proposed such non-uniform priors, which help when the labels are unbalanced. {#c-mu-unigram}
`smooth_labels(..., prior=...)` in both implementations takes such a $u$.

**Label dropout**, or DisturbLabel, actually replaces some labels with random ones during training; Müller et al. call it label dropout, of which label smoothing is the marginalized version. {#c-mu-disturblabel}

**The confidence penalty** of Pereyra et al. subtracts $\beta H(p)$ instead, the reverse KL divergence from the uniform distribution; they suggest annealing it or applying it only when the output entropy falls below a threshold, through a hinge loss, since one wants fast convergence early and less overfitting late. {#c-pe-anneal}
Pereyra et al. see the output distribution as the knowledge of the network, so the probabilities it gives the wrong classes indicate how it generalizes. {#c-pe-functional}
Because it acts on the outputs, which have a natural scale, rather than on the weights, such output regularization does not depend on how the network is parameterized. {#c-pe-invariant}

**The $K - 1$ convention** gives $\varepsilon/(K-1)$ to each wrong class and $1 - \varepsilon$ to the label; it is uniform smoothing with $\varepsilon K/(K-1)$, as derived above.

**Knowledge distillation** replaces the uniform $u$ with a teacher's softened output for each example, as in the Distillation paragraphs.

## When to use it

Use $\varepsilon = 0.1$ for classifiers trained with cross-entropy on many classes, such as image classifiers and translation models, where accuracy matters more than likelihood; it costs nothing and usually helps a little.
Use it to get better calibrated probabilities without a separate temperature-scaling step.
Leave it off for a network that will be a distillation teacher, a feature extractor for transfer learning, or a model judged by log-likelihood or perplexity.
With strongly unbalanced labels, consider unigram smoothing.
It works on the targets, so it combines with weight decay, which pulls the weights toward zero, and with dropout, which adds noise to the activations; early stopping instead limits how long the network trains on hard targets and leaves the loss unchanged.

## Pitfalls

> [!pitfall]
> **A negative `label_smoothing` is ignored silently.** PyTorch raises an error for values above 1, but treats a negative value as no smoothing at all, so a sign error in a config turns the method off without a warning. The test `test_torch_range_check` checks both, and `test_eps_out_of_range` checks that the implementations here reject values outside $[0, 1]$.

> [!pitfall]
> **Mixing the two conventions.** PyTorch spreads $\varepsilon$ over all $K$ classes, so the label gets $1 - \varepsilon + \varepsilon/K$; code that gives $\varepsilon/(K-1)$ to the wrong classes uses $\varepsilon K/(K-1)$ in PyTorch's terms. The tests `test_torch_spreads_eps_over_all_classes` and `test_wrong_class_convention` check the conversion.

> [!pitfall]
> **Smoothing twice.** `F.cross_entropy` smooths soft targets too, so passing already smoothed targets with `label_smoothing` set smooths them again. `test_torch_spreads_eps_over_all_classes` checks that a one-hot target with `label_smoothing` equals the smoothed target without it.

> [!pitfall]
> **Class weights change the uniform part.** With `weight`, PyTorch weights each class's term in $H(u, p)$ and divides the mean by the summed weights of the labels, not of all classes. `test_matches_torch` checks the implementation here against PyTorch with weights, `ignore_index` and every reduction.

> [!pitfall]
> **Expecting confidence near 1.** A smoothed model is not meant to output probability 1: with the uniform convention its largest probability on a training example tops out near $1 - \varepsilon + \varepsilon/K$, so thresholds tuned on a hard-target model need retuning. `test_training_reaches_the_logit_gap` and `test_smoothing_lowers_confidence_of_a_trained_classifier` check where training stops, and `test_hard_targets_keep_growing` that without smoothing it does not.

## History

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe and Jonathon Shlens of Google and Zbigniew Wojna of University College London introduced label smoothing in "Rethinking the Inception Architecture for Computer Vision", arXiv 1512.00567. {#c-hist-szegedy}
As Müller et al. summarize it, Szegedy et al. improved accuracy by computing the cross-entropy against a weighted mixture of the hard targets and the uniform distribution. {#c-mu-intro}
Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser and Geoffrey Hinton posted "Regularizing Neural Networks by Penalizing Confident Output Distributions", arXiv 1701.06548, in January 2017, as a submission to ICLR 2017, connecting smoothing to the confidence penalty. {#c-hist-pereyra}
Rafael Müller, Simon Kornblith and Geoffrey Hinton of Google Brain in Toronto published "When Does Label Smoothing Help?" at NeurIPS 2019, the 33rd Conference on Neural Information Processing Systems in Vancouver. {#c-hist-muller}
