---
tag: NEW-mixup
kind: method
slug: mixup
name: Mixup
aliases: [mixup, input mixup, mixup training, mix-up]
area: deep-learning-training/data-augmentation
level: core
summary: >-
  Train on convex combinations of random pairs of examples and of their one-hot labels, with the weight drawn from Beta(α, α), so the network learns to behave linearly between training examples.
purpose: regularization
uses: [beta-distribution@m2, cross-entropy@m2, linearity-of-expectation@m2, variance@m2]
related_to: [dropout, label-smoothing@al, data-augmentation@al, empirical-risk-minimization@al, weight-decay@al, overfitting@al, regularization@al, adversarial-training@al, generalization-error@al, bias-variance-tradeoff@al]
hyperparameters:
  - {name: alpha, meaning: "the parameter of the symmetric Beta(α, α) from which the mixing weight λ is drawn; α → 0 recovers ERM, α = 1 draws λ uniformly, large α mixes nearly half and half", typical: "0.1 to 0.4 on ImageNet, 1 on CIFAR; 1 for CutMix"}
  - {name: layer, meaning: "where the mixing happens: the input for mixup and CutMix, a random hidden layer per minibatch for Manifold Mixup", typical: "the input"}
cost:
  train: "one weighted sum of the minibatch with a shuffled copy of itself and a second label term in the loss; no extra forward or backward pass"
  inference: "none"
history:
  year: 2018
  people: [Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz]
  note: "Posted to arXiv in October 2017 and published at ICLR 2018; Manifold Mixup (Verma et al., 2019) moved the mixing to hidden layers, CutMix (Yun et al., 2019) replaced the blend by a pasted patch, and Zhang et al. (2021) read the mixup loss as a regularized loss."
prereqs: [empirical-risk-minimization@al, data-augmentation@al, cross-entropy-loss@al, beta-distribution@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1710.09412", "1806.05236", "1905.04899", "2010.04819"]
  al: null
sources:
  - {paper: "Zhang, Cisse, Dauphin and Lopez-Paz (2018)", title: "mixup: Beyond Empirical Risk Minimization", locator: "the title page, Sections 1 to 5, Tables 1 and 2, Figures 2 to 4", note: "the method, the vicinal distribution, and the image, speech, tabular, corrupted-label, adversarial and GAN experiments"}
  - {paper: "Yun, Han, Oh, Chun, Choe and Yoo (2019)", title: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", locator: "Sections 1, 2, 3.1, 4.1.1, 4.1.3, 4.4, Tables 1, 3 and 5, Algorithm A1 on p. 12", note: "the pasted patch, the box sampling, the area-adjusted λ and comparisons with Mixup"}
  - {paper: "Verma, Lamb, Beckham, Najafi, Mitliagkas, Courville, Lopez-Paz and Bengio (2019)", title: "Manifold Mixup: Better Representations by Interpolating Hidden States", locator: "the abstract, Sections 2 and 3.1, Table 1", note: "mixing hidden states and the flattening theorem"}
  - {paper: "Zhang, Deng, Kawaguchi, Ghorbani and Zou (2021)", title: "How Does Mixup Help With Robustness and Generalization?", locator: "pages 1, 3 and 4", note: "the mixup loss as the standard loss plus regularization terms"}
---

Mixup trains a network on blends of its training data.
It draws two examples at random, takes a weighted average of their inputs and the same weighted average of their one-hot labels, and trains on that, with the weight $\lambda$ drawn from a symmetric Beta distribution.
Zhang, Cisse, Dauphin and Lopez-Paz introduced it in 2017 as a data-agnostic form of vicinal risk minimization, and it costs a few lines of code.
It improved image classifiers on ImageNet and CIFAR, made networks harder to fool with one-step adversarial examples and slower to memorize noisy labels.
Its close relatives Manifold Mixup and CutMix mix hidden states or paste a patch of one image into another.

## In plain words

A classifier trained the usual way only ever sees real examples with certain labels.
Between two training images, a cat and a dog, it may answer anything, and often answers with full confidence.
Mixup gives it a 70/30 blend of the cat and the dog and asks for the answer "70% cat, 30% dog".
Repeated over random pairs and random weights, this asks the network to change its prediction gradually, along a straight line, as the input moves from one example to another.
Nothing about the data is assumed beyond that, so the same trick works for images, speech spectrograms and tables of numbers.

## Setting

A network $f$ is trained on examples $(x_i, y_i)$, with $y_i$ a one-hot label over $K$ classes.
Empirical risk minimization (ERM) averages the loss over the $n$ training examples, which watches $f$ only at those $n$ points; a network with about as many parameters as examples can minimize it by memorizing the data, and then behaves badly away from it. {#c-zh-memorize}
Vicinal risk minimization instead trains on virtual examples drawn from a vicinity of each training example, which a person has to describe; Zhang et al. note that this kind of data augmentation depends on the dataset, needs expert knowledge, and assumes that the vicinity of an example has its class. {#c-zh-vrm}
Chapelle et al.'s Gaussian vicinities $\nu(\tilde x, \tilde y \mid x_i, y_i) = \mathcal N(\tilde x - x_i, \sigma^2)\,\delta(\tilde y = y_i)$ amount to adding Gaussian noise to the inputs. {#c-zh-gaussian}

## The method

Mixup builds each virtual example from two training examples drawn at random,

$$ \tilde x = \lambda x_i + (1 - \lambda)\, x_j , \qquad \tilde y = \lambda y_i + (1 - \lambda)\, y_j , \qquad \lambda \in [0, 1] , $$ {#eq-mix}

with raw input vectors $x$ and one-hot labels $y$. {#c-zh-virtual}
This builds into the training distribution the prior that linear interpolations of feature vectors should give linear interpolations of the targets, in a few lines of code and with little overhead. {#c-zh-prior}
In the language of vicinal risk, mixup is the vicinal distribution

$$ \mu(\tilde x, \tilde y \mid x_i, y_i) = \frac1n \sum_{j} \mathbb E_\lambda\big[\delta(\tilde x = \lambda x_i + (1 - \lambda) x_j,\ \tilde y = \lambda y_i + (1 - \lambda) y_j)\big] , \qquad \lambda \sim \mathrm{Beta}(\alpha, \alpha) . $$ {#eq-vicinal}

which averages the blends of $x_i$ with every training example over the random weight. {#c-zh-vicinal}
The hyperparameter $\alpha > 0$ sets how strongly pairs are interpolated, and $\alpha \to 0$ gives back ERM. {#c-zh-alpha}
The code of the paper draws `lam = numpy.random.beta(alpha, alpha)` once per minibatch and mixes the inputs and one-hot targets of two data loaders. {#c-zh-code}
In practice one loader suffices: the minibatch is mixed with a randomly shuffled copy of itself, which works as well and halves the I/O. {#c-zh-one-loader}
Mixing three or more examples with Dirichlet weights gave no further gain at more cost, and mixing only examples of the same label lost the gains of mixup. {#c-zh-alternatives}

Zhang et al. argue that the network is pushed to behave linearly between training examples, which reduces unwanted oscillations away from them, and that linearity is a good inductive bias by Occam's razor. {#c-zh-linear}
On CIFAR-10, at points $\lambda x_i + (1 - \lambda)x_j$ between training examples, a mixup model more rarely predicts a class outside $\{y_i, y_j\}$ and has smaller input gradients than an ERM model. {#c-zh-figure2}

**CutMix** keeps the label rule but replaces the blend by a patch: $\tilde x = M \odot x_A + (\mathbf 1 - M) \odot x_B$ and $\tilde y = \lambda y_A + (1 - \lambda) y_B$, with a binary mask $M$ and $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$, $\alpha = 1$ in all of Yun et al.'s experiments. {#c-cm-def}
The mask is 0 inside a box with $r_x \sim \mathrm{Unif}(0, W)$, $r_y \sim \mathrm{Unif}(0, H)$, $r_w = W\sqrt{1 - \lambda}$ and $r_h = H\sqrt{1 - \lambda}$, so the box covers the fraction $r_w r_h / (WH) = 1 - \lambda$ of the image, and 1 outside it. {#c-cm-box}
In their pseudo-code $(r_x, r_y)$ is the centre of the box, the corners are clipped to the image and rounded, and $\lambda$ is then reset to the exact area ratio,

$$ \lambda = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W H} . $$ {#eq-cutmix-lambda}

before the targets are mixed. {#c-cm-adjust}
A pasted patch gives a more locally natural image than a blend. {#c-cm-natural}

**Manifold Mixup** picks a layer $k$ at random from a set $S$ that may include the input, runs two minibatches up to layer $k$, mixes their hidden states and labels with $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$, finishes the forward pass and backpropagates through every layer. {#c-mm-def}
With $S = \{0\}$ it is mixup; it draws one $(k, \lambda)$ per minibatch, which performed as well as one per example, and mixes the minibatch with a shuffled copy of itself. {#c-mm-reduces}

## Derivation

**The mixed label is a mixed loss.**
Because the cross-entropy is linear in its target, for $p = \operatorname{softmax}(z)$ and any $\lambda \in [0, 1]$,

$$ -\sum_k \tilde y_k \log p_k = \lambda\,(-\log p_{y_i}) + (1 - \lambda)\,(-\log p_{y_j}) , \qquad \nabla_z = p - \tilde y . $$ {#eq-mixed-loss}

the two label losses weighted by $\lambda$ and $1 - \lambda$. {#c-mixed-loss}
So code can keep integer labels: with a permutation $\pi$ of the minibatch, $\lambda\,\mathrm{CE}(f(\tilde x), y) + (1 - \lambda)\,\mathrm{CE}(f(\tilde x), y_\pi)$ has the same value and gradient as the soft-target loss. {#c-mixed-batch}
Zhang et al. (2021) study exactly this kind of loss: their class $h(f_\theta(x)) - y f_\theta(x)$, linear in $y$, contains the losses of generalized linear models and the cross-entropy. {#c-zh21-class}
Their Lemma 3.1 expands the mixup loss into the standard loss plus three regularization terms and higher-order terms, which tie mixup to the first and second directional derivatives of $f_\theta$ at the training points. {#c-zh21-lemma}
From it they show that mixup approximately minimizes an upper bound of an adversarial loss, which accounts for its robustness to one-step attacks such as FGSM, and that it is a data-adaptive regularization that reduces overfitting. {#c-zh21-claims}
The derivation `mixed-cross-entropy` gives the steps; its check compares both forms and the gradient with finite differences.

**What α does.**
For $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$,

$$ \mathbb E[\lambda] = \tfrac12 , \qquad \operatorname{Var}[\lambda] = \frac{1}{4(2\alpha + 1)} , $$ {#eq-beta-moments}

so the variance runs from $1/4$, its largest possible value on $[0, 1]$, as $\alpha \to 0$, where $\lambda$ is 0 or 1 with probability $1/2$ each and every example is a real one, as in ERM, down to 0 as $\alpha \to \infty$, where every example is an even blend; $\alpha = 1$ is the uniform distribution. {#c-beta-moments}
The derivation `beta-moments` gives the steps, and its check samples $\lambda$ by Monte Carlo.

**What λ CutMix really uses.**
With the mask 0 on the clipped box, the share of pixels from $x_A$ is exactly eq-cutmix-lambda; before clipping it is $\lambda$, and clipping only removes area, so the adjusted $\lambda$ is never below the sampled one, up to rounding. {#c-cutmix-area}
Averaging over a uniform $\lambda$ and a uniform centre, the mean of the adjusted $\lambda$ is $163/240 \approx 0.68$ instead of $1/2$ for a large image. {#c-cutmix-mean}
The derivation `cutmix-area` counts pixels on a grid and integrates the mean.

## Worked example

Take $K = 3$ classes, $x_i = (2, 0)$ with label 0, $x_j = (0, 4)$ with label 2, and $\lambda = 0.7$.
The mixed input is $\tilde x = (1.4, 1.2)$ and the mixed label is $\tilde y = (0.7, 0, 0.3)$.
Suppose the network answers $\tilde x$ with logits $z = (1, 0.5, -0.5)$.

| quantity | class 0 | class 1 | class 2 |
|---|---|---|---|
| target $\tilde y$ | 0.7 | 0 | 0.3 |
| $p = \operatorname{softmax}(z)$ | 0.546549 | 0.331499 | 0.121952 |
| $-\log p_k$ | 0.604131 | 1.104131 | 2.104131 |
| gradient $p - \tilde y$ | −0.153451 | 0.331499 | −0.178048 |

The soft-target loss is $0.7 \cdot 0.604131 + 0.3 \cdot 2.104131 = 1.054131$, the same number as $0.7\,\mathrm{CE}(z, 0) + 0.3\,\mathrm{CE}(z, 2)$.
With $\alpha = 0.2$, as on ImageNet, $\operatorname{Var}[\lambda] = 1/5.6 = 0.178571$, a standard deviation of 0.4226, so most draws sit near 0 or 1; with $\alpha = 1$ it is $1/12$.
For CutMix on a $32 \times 32$ image with $\lambda = 0.75$, the box is $16 \times 16$; centred at $(4, 28)$ it is clipped to $x \in [0, 12)$, $y \in [20, 32)$, 144 pixels, so eq-cutmix-lambda gives $1 - 144/1024 = 0.859375$ instead of $0.75$.

## Hyperparameters

On ImageNet, $\alpha$ between 0.1 and 0.4 beat ERM, and larger values underfit. {#c-zh-imagenet-alpha}
On CIFAR, Zhang et al. used $\alpha = 1$, so $\lambda$ is uniform, with 200 epochs, minibatches of 128, weight decay $10^{-4}$ and no dropout. {#c-zh-cifar-setup}
CutMix improved on its CIFAR-100 baseline for every $\alpha$ from 0.1 to 4, was best at $\alpha = 1$, and was best applied to the input rather than to feature maps. {#c-cm-alpha}
On speech commands, five warm-up epochs on plain examples sped up early training. {#c-zh-warmup}
Because mixup regularizes, it may want less weight decay than ERM, as the ablation under Evidence finds.

## Cost

One weighted sum of the minibatch with a shuffled copy and one more label term in the loss; CutMix copies one rectangle.
There is no extra forward or backward pass and nothing changes at inference, though Zhang et al. and Yun et al. trained longer with it, 200 or 300 epochs.

## Evidence

**ImageNet.**
After 90 epochs, top-1 error went from 23.5 to 23.3 for ResNet-50 with $\alpha = 0.2$ and from 22.1 to 21.5 for ResNet-101; after 200 epochs from 23.6 to 22.1 and from 22.0 to 20.8. {#c-zh-imagenet}
Larger models and longer runs gained most: 0.5 to 0.6 points for ResNet-101 and ResNeXt-101 against 0.2 for ResNet-50 at 90 epochs, and a further 1.2 points for the mixup ResNet-50 at 200 epochs, where ERM did not move. {#c-zh-capacity}

**CIFAR.**
With $\alpha = 1$, CIFAR-10 error fell from 5.6 to 4.2 (PreAct ResNet-18), 3.8 to 2.7 (WideResNet-28-10) and 3.7 to 2.7 (DenseNet-BC-190), and CIFAR-100 error from 25.6 to 21.1, 19.4 to 17.5 and 19.0 to 16.8, at the same convergence speed. {#c-zh-cifar}

**Speech, tables and GANs.**
On Google commands, VGG-11 test error fell from 4.6 to 3.4 with $\alpha = 0.2$, but LeNet's rose from 10.3 to 11.3. {#c-zh-speech}
On six UCI datasets, small fully connected networks improved on four and were never worse. {#c-zh-uci}
On two toy datasets mixup steadied GAN training, which the authors attribute to a regularized discriminator gradient. {#c-zh-gan}

**Corrupted labels.**
With 20%, 50% or 80% random CIFAR-10 labels, mixup with $\alpha$ of 8 or 32 beat dropout at both the best and the last epoch, and mixup with dropout was best of all. {#c-zh-corrupt}
At 80% noise the best and last test errors were 36.5 and 73.9 for ERM, 30.9 and 35.1 for dropout, 25.3 and 30.9 for mixup and 24.0 and 24.8 for both. {#c-zh-corrupt-80}

**Adversarial examples.**
For ResNet-101 on ImageNet at $\varepsilon = 4$, mixup was 2.7 times more robust to white-box FGSM and 1.25 times to black-box FGSM in top-1 error, and about 40% more robust to black-box I-FGSM; neither model resisted white-box I-FGSM. {#c-zh-adversarial}

**Ablation.**
On CIFAR-10, mixup beat every alternative, the runner-up being input mixing with label smoothing; ERM preferred a large weight decay and mixup a small one, and label smoothing and Gaussian noise regularized little. {#c-zh-ablation}
Mixing deeper latent layers regularized less, and mixing random pairs from all classes regularized most among input methods. {#c-zh-ablation-layers}

**Against CutMix and Manifold Mixup.**
In Yun et al.'s ResNet-50 runs, ImageNet accuracy was 76.3% for the baseline, 77.4% for Mixup, 77.1% for Cutout and 78.6% for CutMix, and localization accuracy 46.3%, 45.8%, 46.7% and 47.3%. {#c-cm-table1}
Mixup and Cutout raised classification but lowered localization or detection, while CutMix improved all three. {#c-cm-loc}
ResNet-50 top-1 errors were 23.68% (baseline), 22.93% (Cutout), 22.58% (Mixup), 22.50% (Manifold Mixup) and 21.40% (CutMix). {#c-cm-imagenet}
On CIFAR-100 with PyramidNet-200 they were 16.45%, 15.63% for Mixup, 16.14% for Manifold Mixup and 14.47% for CutMix. {#c-cm-cifar100}
Verma et al. report CIFAR-10 errors for PreActResNet18 of 4.83% without mixing, 3.82% with input mixup and 2.95% with Manifold Mixup, and 24.01%, 22.11% and 20.34% on CIFAR-100. {#c-mm-table1}
These comparisons come from the authors of the competing methods.

## Limitations and failure modes

**The right α is found by trial.**
Larger $\alpha$ raises training error on real data and narrows the generalization gap, and there is no theory of the sweet spot; on CIFAR-10 training error stays low even as $\alpha \to \infty$, on ImageNet it rises sharply. {#c-zh-sweet-spot}
So $\alpha$ moves the model along the bias-variance tradeoff, trading fit on the real examples for a smaller gap.
Zhang et al. conjecture that more capacity makes training less sensitive to large $\alpha$. {#c-zh-conjecture}
The small LeNet on speech got worse with mixup.

**Blends are not natural inputs.**
Mixed images are locally ambiguous and unnatural, which confuses the model, especially for localization. {#c-cm-ambiguous}
Under occlusion mixup helped only marginally, and it made the network more over-confident, where CutMix made it less. {#c-cm-occlusion}

**Beyond classification it is not obvious.**
Regression is straightforward, but structured outputs such as segmentation are not. {#c-zh-structured}

**It does not defend against strong attacks.**
The robustness gain is for one-step attacks; against white-box I-FGSM, mixup and ERM both fail.
Adversarial training, which trains on adversarial examples generated during training, aims at such attacks directly, at the cost of extra gradient computations that mixup does not need.

## Variants

**Manifold Mixup** mixes at a random hidden layer; Verma et al. report smoother decision boundaries at several levels and class representations with fewer directions of variance. {#c-mm-flatten}
Their Theorem 1 proves that when the representation has at least $d - 1$ dimensions for $d$ classes the mixed loss can reach zero with a linear classifier, and Corollary 1 that the representations of each class then lie in a subspace of dimension $\dim H - d + 1$. {#c-mm-theorem}

**CutMix** pastes a box and mixes labels by its area; **Feature CutMix** does it at a random layer per minibatch, as Manifold Mixup does. {#c-cm-feature}

**Label smoothing** also trains on soft labels, but Zhang et al. contrast it: label smoothing and confidence penalties soften the target regardless of the input, while mixup ties the softened target to how the input was built. {#c-zh-related}

## When to use it

Use mixup when a large classifier overfits, when labels are noisy, or when one-step adversarial robustness and smoother predictions between examples matter, and the data has no obvious hand-designed augmentations.
Start from $\alpha = 0.2$ on large datasets and $\alpha = 1$ on small ones, train longer than for ERM and lower the weight decay.
For images where localization or detection transfer matters, prefer CutMix.
Skip it for small models that already underfit, and think twice for structured outputs.

## Pitfalls

> [!pitfall]
> **Shuffling inputs and labels differently.** The inputs and the labels must be mixed with the same permutation and the same $\lambda$; a second call to the shuffler pairs each input with someone else's label. `test_labels_follow_inputs` checks that every mixed target matches the mixed input.

> [!pitfall]
> **Using the sampled λ after clipping a CutMix box.** A box that runs off the image covers less than $1 - \lambda$ of it, so the label weight must be recomputed from the clipped box, as in eq-cutmix-lambda. `test_cutmix_lambda_is_pixel_fraction` counts the pixels, and `test_unadjusted_lambda_is_wrong_near_edges` measures how far the sampled value is off.

> [!pitfall]
> **Copying the CutMix pseudo-code literally.** Algorithm A1 writes $r_w = \sqrt{1 - \lambda}$ without the factor $W$, and clips $y_2$ with `min=H` where `max=H` is meant. `test_box_inside_image` and `test_box_area_before_clipping` check the box against Equation (2).

> [!pitfall]
> **Mixing integer images.** Blending `uint8` pixels in place truncates or wraps them. `test_integer_inputs_are_mixed_in_float` checks that the implementations here mix in floating point.

> [!pitfall]
> **Reading training accuracy off mixed batches.** A mixed example has no single correct class, so accuracy on mixed batches says little; evaluate on unmixed data. `test_eval_does_not_mix` checks that the PyTorch module leaves inputs alone in evaluation mode.

## History

Hongyi Zhang of MIT and Moustapha Cisse, Yann N. Dauphin and David Lopez-Paz of FAIR wrote "mixup: Beyond Empirical Risk Minimization", arXiv 1710.09412, published at ICLR 2018. {#c-hist-zhang}
Vikas Verma and seven coauthors, among them Lopez-Paz and Yoshua Bengio, extended it to hidden layers in "Manifold Mixup", arXiv 1806.05236. {#c-hist-verma}
Sangdoo Yun and five coauthors at Clova AI Research and Yonsei University proposed CutMix, arXiv 1905.04899. {#c-hist-yun}
Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani and James Zou analysed why mixup helps in "How Does Mixup Help With Robustness and Generalization?", ICLR 2021. {#c-hist-zhang21}
