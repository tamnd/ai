---
tag: 88YEPR
kind: method
slug: randaugment
name: RandAugment
aliases: [RandAugment, RA, rand augment, random augmentation policy]
area: deep-learning-training/data-augmentation
level: core
summary: >-
  Apply to each training image N transformations drawn uniformly from a fixed list, all at one magnitude M, so the augmentation policy is two integers tuned by grid search.
purpose: data-augmentation
uses: [data-augmentation@al, discrete-uniform-distribution@m2, probability-of-the-complement@m2, independence-of-random-variables@m2, linearity-of-expectation@m2]
related_to: [dropout, image-classification, imagenet, data-augmentation@al, hyperparameter-tuning@al, neural-architecture-search@al, ppo@al, regularization@al, overfitting@al, label-smoothing@al]
hyperparameters:
  - {name: N, meaning: "the number of transformations applied to each image, drawn uniformly with replacement from the list", typical: "1 to 3", default_in: {pytorch: 2}}
  - {name: M, meaning: "the one magnitude shared by all transformations, on an integer scale where the top value is the largest distortion of each", typical: "2 to 15 in the CIFAR and ResNet-50 grids of the paper, larger for larger models and datasets", default_in: {pytorch: 9}}
  - {name: magnitude bins, meaning: "the number of points of the magnitude scale, which fixes what M means; TrivialAugment draws its magnitude uniformly from them", typical: "11 (0 to 10) in the paper, 31 (0 to 30) in TrivialAugment", default_in: {pytorch: 31}}
cost:
  train: "N cheap image operations per image on the data-loading CPU, no extra forward or backward pass; tuning is a grid of a few (N, M) training runs, against thousands of GPU hours of policy search for AutoAugment"
  inference: "none"
history:
  year: 2019
  people: [Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le]
  note: "Cutout (DeVries and Taylor, 2017) masked a random square; AutoAugment (Cubuk et al., 2018) learned a policy of 25 sub-policies with a reinforcement-learning controller; RandAugment (arXiv 1909.13719, 2019) replaced the search by two integers; TrivialAugment (Müller and Hutter, 2021) removed even those."
prereqs: [data-augmentation@al, overfitting@al, hyperparameter-tuning@al, discrete-uniform-distribution@m2]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1909.13719", "1805.09501", "1708.04552", "2103.10158"]
  al: null
sources:
  - {paper: "Cubuk, Zoph, Shlens and Le (2019)", title: "RandAugment: Practical automated data augmentation with a reduced search space", locator: "arXiv 1909.13719v2: Table 1, Sections 3 and 4, Figures 2 and 3, Tables 2 to 6, Appendix pp. 12 and 13", note: "the method, the 14 transformations, the magnitude scale, and the CIFAR, SVHN, ImageNet and COCO results"}
  - {paper: "Cubuk, Zoph, Mané, Vasudevan and Le (2018)", title: "AutoAugment: Learning Augmentation Strategies from Data", locator: "arXiv 1805.09501v3: Section 3, p. 4, Section 4.2, Table 2", note: "the policy of sub-policies, the controller and the size of its search space"}
  - {paper: "DeVries and Taylor (2017)", title: "Improved Regularization of Convolutional Neural Networks with Cutout", locator: "arXiv 1708.04552v2: pp. 4 and 5, Table 1", note: "the random square mask that both methods use"}
  - {paper: "Müller and Hutter (2021)", title: "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation", locator: "arXiv 2103.10158v2: Section 3, Algorithm 1 on p. 4, Tables 1, 2 and 8, Section 4.2.2", note: "one operation at a random magnitude per image, and the operation ranges"}
  - {url: "https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandAugment.html", title: "torchvision.transforms.v2.RandAugment", locator: "the parameters num_ops, magnitude and num_magnitude_bins", note: "the defaults N = 2, M = 9 and 31 bins"}
  - {url: "https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.TrivialAugmentWide.html", title: "torchvision.transforms.v2.TrivialAugmentWide", locator: "the parameter num_magnitude_bins", note: "the default of 31 bins"}
---

RandAugment is a data augmentation policy for images with two integer knobs.
For every training image it draws $N$ transformations, such as a rotation or a change of contrast, uniformly at random from a fixed list of 14, and applies them in order, all at one shared magnitude $M$.
Ekin D. Cubuk, Barret Zoph, Jonathon Shlens and Quoc V. Le proposed it in 2019 to replace AutoAugment, which learns a policy with a reinforcement-learning search on a small proxy task, at a cost of thousands of GPU hours.
Tuning $N$ and $M$ by a small grid search matched AutoAugment on CIFAR-10 and ResNet-50 on ImageNet, and beat it on EfficientNet-B7.
TrivialAugment later removed even those two knobs.

## In plain words

Training a vision model on photos that are slightly rotated, shifted, recoloured or partly covered teaches it that those changes keep the label; the question is which changes, and how strong.
AutoAugment answered it by search: a controller network proposed recipes, a small model was trained with each, and the best recipes were kept.
RandAugment observed that most of the benefit came from variety, not from the exact recipe.
So it keeps the list of operations, picks a few of them at random for each image, and leaves only two numbers to choose: how many operations, and how strong.
Those two numbers are chosen the way a learning rate is, by ordinary [[al:hyperparameter-tuning]] on the real task.

## Setting

A model, usually a convolutional network for image classification or object detection, is trained by stochastic gradient descent on labelled images.
Each image passes through the data loader before it reaches the model, and the loader may change it in ways that keep the label.
Standard pipelines already flip and crop; on CIFAR the RandAugment experiments kept flips, pad-and-crop and Cutout in every column, baseline included.

AutoAugment, the method RandAugment replaces, defines a policy as 5 sub-policies; each sub-policy applies two image operations in sequence, and each operation has a probability of being applied and a magnitude. {#c-aa-policy}
The operations are 16 functions of the Python imaging library, from ShearX to Sample Pairing and Cutout, with the magnitude discretized into 10 values and the probability into 11. {#c-aa-ops}
That gives $(16 \cdot 10 \cdot 11)^{10} \approx 2.9 \times 10^{32}$ policies, eq-aa-space below. {#c-aa-space-count}
A recurrent controller trained with Proximal Policy Optimization emits the 30 choices of a policy, and about 15,000 policies are sampled and scored on each dataset by the validation accuracy of a small child model. {#c-aa-controller}
To keep the search affordable, the child models train on reduced CIFAR-10, 4,000 images. {#c-aa-reduced}
For ImageNet they train on a reduced subset of 120 classes and 6,000 images. {#c-aa-reduced-imagenet}
The search is [[al:neural-architecture-search]] applied to augmentation policies instead of architectures.
The sub-policies of the best 5 policies are then concatenated into one policy of 25 sub-policies, and for each image of each minibatch one of them is chosen at random. {#c-aa-final}

## The method

RandAugment uses the $K = 14$ transformations identity, autoContrast, equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x, shear-y, translate-x and translate-y. {#c-ra-ops}
Each has a magnitude on an integer scale from 0 to 10, where 10 is the strongest form of that transformation, and one global magnitude $M$ serves them all. {#c-ra-magnitude}
For each image, draw

$$ o_1, \dots, o_N \overset{\text{iid}}{\sim} \mathrm{Unif}\{1, \dots, K\}, \qquad x' = T_{o_N}^{M} \circ \cdots \circ T_{o_1}^{M}(x), $$ {#eq-randaugment}

where $T_k^{M}$ is transformation $k$ at magnitude $M$.
Each transformation is chosen with probability $1/K$, so $N$ draws can express $K^N$ ordered lists. {#c-ra-uniform}
The authors' own listing is three lines of NumPy: `np.random.choice(transforms, N)` draws the operations, with replacement by default, and the function returns the pairs `(op, M)`. {#c-ra-code}

The magnitude scale maps to an argument for each operation by a linear rule,

$$ a_k(m) = r_k \, \frac{m}{m_{\max}}, \qquad m \in \{0, 1, \dots, m_{\max}\}, $$ {#eq-magnitude}

where $r_k$ is the largest argument of operation $k$.
In the RandAugment space as tabulated by Müller and Hutter, rotate reaches $\pm 30°$, translate 10 pixels, posterize keeps between 8 and 4 bits, solarize lowers its threshold from 256 to 0, and cutout reaches 0.2 of the image side; their wider space goes to $\pm 135°$, 32 pixels and 2 bits. {#c-ta-ranges}
Rotations, shears and translations get a random sign, and Cutout a random centre.
The [NumPy](impl/numpy/randaugment.py) and [PyTorch](impl/pytorch/randaugment.py) implementations here cover identity, rotate, the two translations, posterize, solarize and cutout.

**Cutout.** Cutout sets a fixed-size square of the input to zero at a random place, anew at each epoch; the centre is a uniformly chosen pixel, so part of the square can fall outside the image. {#c-co-def}
DeVries and Taylor found that letting the square cross the border, instead of keeping it inside, was critical to its accuracy. {#c-co-border}
AutoAugment has Cutout among its operations, and RandAugment and TrivialAugment both apply a final Cutout after the policy on CIFAR.

**TrivialAugment.** For each image, draw one operation $a$ uniformly from the list and one strength uniformly from $\{0, \dots, 30\}$, and return $a(x, m)$: {#c-ta-def}

$$ a \sim \mathrm{Unif}(\mathcal A), \qquad m \sim \mathrm{Unif}\{0, \dots, 30\}, \qquad x' = a(x, m). $$ {#eq-trivialaugment}

It is not RandAugment with $N = 1$: RandAugment fixes one tuned strength for all images, while TrivialAugment draws the strength anew for each image. {#c-ta-not-ra}

## Derivation

**How much smaller the search is.**
An AutoAugment slot picks one of 16 operations, 10 magnitudes and 11 probabilities, and a policy has 10 slots, so by the rule of product

$$ |\mathcal S_{\mathrm{AA}}| = (16 \cdot 10 \cdot 11)^{2 \cdot 5} = 1760^{10} \approx 2.85 \times 10^{32} . $$ {#eq-aa-space}

A RandAugment policy is the pair $(N, M)$, and a grid search tries

$$ |\mathcal S_{\mathrm{RA}}| = |\mathcal N| \cdot |\mathcal M| $$ {#eq-ra-grid}

settings: 18 for the ImageNet ResNet-50 grid $N \in \{1, 2, 3\}$, $M \in \{5, 7, 9, 11, 13, 15\}$, and 33 for all of $N \le 3$ and $M \le 10$. {#c-ra-grid-count}
Cubuk et al. (2019) put the two at $10^{32}$ and $10^{2}$, with $10^{61}$ for Population Based Augmentation. {#c-ra-space}
The derivation [search-space](derivations/search-space.md) gives the steps and its check computes the counts exactly.
The two numbers count different things: the first is what a controller must search, the second is what a person tries; inside each RandAugment setting the sampler still draws one of $K^N$ lists per image.

**How often an operation is used.**
Since the $N$ draws are independent and uniform, a given operation is missed by all of them with probability $(1 - 1/K)^N$, so

$$ p_{\mathrm{hit}}(K, N) = 1 - \Big(1 - \frac{1}{K}\Big)^N . $$ {#eq-hit}

For $K = 14$ it is $1/14 \approx 0.071$ at $N = 1$, $27/196 \approx 0.138$ at $N = 2$ and $547/2744 \approx 0.199$ at $N = 3$. {#c-hit-probability}
With $N = 2$ the two draws coincide with probability $1/14$, and an image sees $27/14 \approx 1.93$ distinct operations on average. {#c-distinct-ops}
The derivation [hit-probability](derivations/hit-probability.md) proves this by enumeration and checks it by Monte Carlo.

**How much Cutout really hides.**
Because the centre is uniform and the square is clipped at the border, the masked area is less than $s^2$ on average.
For an $L \times L$ image and an even side $s \le 2L$, each axis keeps a mean length of $s - s^2/(4L)$, and the axes are independent, so

$$ \mathbb E[A] = \Big(s - \frac{s^2}{4L}\Big)^2 . $$ {#eq-cutout-area}

The formula is exact on the pixel grid. {#c-cutout-area}
For the 16 × 16 patch on 32 × 32 CIFAR images it gives 196 pixels, 0.191 of the image, against 256 pixels, 0.25, for a square that never crosses the border. {#c-cutout-16}
The derivation [cutout-area](derivations/cutout-area.md) counts every centre on grids up to 24 × 24.
The same mean side, $s - s^2/4$ on the unit square, appears for CutMix boxes on the [[mixup]] page.

## Worked example

Take the 4 × 4 grey image of the tests,

$$ x = \begin{pmatrix} 10 & 50 & 90 & 130 \\ 170 & 210 & 250 & 30 \\ 70 & 110 & 150 & 190 \\ 230 & 20 & 60 & 100 \end{pmatrix}, $$

and $N = 2$, $M = 9$ on the 0 to 10 scale.
By eq-magnitude, $M = 9$ means a rotation of $27°$, a shift of 9 pixels, a posterize to $8 - \operatorname{round}(3.6) = 4$ bits, and a solarize threshold of $256 - 230.4 = 25.6$.
Suppose the sampler draws posterize and then solarize.
Posterize to 4 bits clears the low 4 bits, so 50 becomes 48, 130 becomes 128 and 30 becomes 16.
A solarize at threshold 25.6 would then invert every value from 26 up, so 48 becomes 207.
At a weaker threshold of 128 instead, solarize maps 130 to 125 and 250 to 5, and leaves 10, 50 and 90 alone.
A Cutout of side 2 centred at row 0, column 3 masks rows 0 and the columns 2 and 3 up to the border, two pixels, where an unclipped square has four.
A translate-x by one pixel fills column 0 with zeros, and a rotation by $90°$ equals `np.rot90`; these values are the fixture `test_worked_example` of both implementations.

With $K = 14$ and $N = 2$, the chance that rotate is among the two draws is $27/196 \approx 0.138$; with TrivialAugment it is $1/14$, and its strength is uniform on 0 to 30 instead of fixed.

## Hyperparameters

The policy is $N$ and $M$, and the paper tunes them by grid search on held-out data.
On CIFAR-100 the grid was $N \in \{1, 2\}$ and $M \in \{2, 6, 10, 14\}$, and the best settings were $N = 1$, $M = 2$ for Wide-ResNet-28-2 and $N = 2$, $M = 14$ for Wide-ResNet-28-10. {#c-ra-cifar-grid}
On ImageNet with ResNet-50 the grid was $N \in \{1, 2, 3\}$ and $M \in \{5, 7, 9, 11, 13, 15\}$, and the best was $N = 2$, $M = 9$. {#c-ra-imagenet-grid}
So $M$ can exceed 10; the scale is linear and 10 marks the maximum of the AutoAugment ranges, not a hard limit.
The best magnitude grows with the size of the network and with the size of the training set, and smaller training sets gain more from augmentation, 3.0% against 1.5% in their sweep. {#c-ra-proxy-size}
This is the paper's main argument against a proxy task: a policy learned on a small model and a small subset fixes a magnitude that is wrong for the target.
Four schedules for $M$ over training, constant, random, linearly increasing and random with a growing upper bound, all gave between 97.2% and 97.3% on CIFAR-10, so the paper keeps $M$ constant. {#c-ra-schedules}
Tying all magnitudes to one $M$ cost little: tuning the magnitude of rotate alone moved accuracy by $0.19\% \pm 0.18\%$, and of translate-x by $0.07\% \pm 0.05\%$. {#c-ra-tied}
In torchvision, `RandAugment` defaults to `num_ops=2`, `magnitude=9` and `num_magnitude_bins=31`, so its $M = 9$ is 9/30 of the range, not 9/10; and `TrivialAugmentWide` defaults to 31 bins.

## Cost

Each image gets $N$ cheap operations on the CPU that loads the data, with no extra forward or backward pass and no state.
The cost that changes is the tuning.
AutoAugment spent about 15,000 GPU hours searching its COCO policy, while RandAugment was tuned on 6 values of its hyperparameters. {#c-ra-search-cost}
Müller and Hutter put the search overhead at 40 to 800× for AutoAugment, 4 to 80× for RandAugment's grid, 1× for Fast AutoAugment and 0× for TrivialAugment. {#c-ta-overhead}
Cubuk et al. note that earlier learned methods had 30 or more parameters, against 2 here. {#c-ra-30params}

## Evidence

On CIFAR-10, RandAugment reached 97.3% with Wide-ResNet-28-10, against 97.4% for AutoAugment and 96.1% for the baseline, and 98.5% with PyramidNet, the same as AutoAugment; on CIFAR-100 with Wide-ResNet-28-10 it reached 83.3% against 82.9%. {#c-ra-cifar}
For comparison, AutoAugment had lowered the CIFAR-10 error of Wide-ResNet-28-10 from 3.9% and 3.1% with Cutout to 2.6%, and of PyramidNet with ShakeDrop from 2.7% and 2.3% to 1.5%. {#c-aa-results}
On ImageNet, AutoAugment took ResNet-50 from 76.3% to 77.6% top-1. {#c-aa-imagenet}
RandAugment matched it at 77.6% on ResNet-50 and reached 85.0% with EfficientNet-B7, against 84.4% for AutoAugment and 84.0% for the baseline. {#c-ra-imagenet}
On the SVHN core set of 73K images, RandAugment on Wide-ResNet-28-2 helped more than adding the 531K extra images. {#c-ra-svhn}
On COCO detection with ResNet-101 it gave 40.1 mAP against 40.4 for AutoAugment and 38.8 without, close to AutoAugment but below it. {#c-ra-coco}
Cutout itself took WideResNet from 3.87% to 3.08% error on CIFAR-10, from 18.8% to 18.41% on CIFAR-100, and from 1.60% to 1.30% on SVHN. {#c-co-results}
TrivialAugment, in its wide space, reached 97.46% on CIFAR-10 and 84.33% on CIFAR-100 with Wide-ResNet-28-10 and 78.07% on ImageNet with ResNet-50, against 97.3, 83.3 and 77.6 listed for RandAugment. {#c-ta-results}
Each comparison is one paper against numbers copied from earlier papers, and gaps of a few tenths of a point are of the size of the 95% confidence intervals that TrivialAugment reports over repeated runs.

## Limitations and failure modes

The operation list is fixed by hand and its members are not equally useful.
In an ablation with $N = 3$, $M = 4$, adding rotate to a random subset raised accuracy by 1.3% on average, and adding posterize lowered it by 0.3%. {#c-ra-ablation}
Learning the selection probabilities by density matching helped a little on reduced CIFAR-10, 87.4% against 86.8% for Wide-ResNet-28-10, so uniform sampling leaves some accuracy behind on small data. {#c-ra-learned-probs}
TrivialAugment could drop 4 of its 14 operations and stay close to the full list. {#c-ta-drop}
An operation that helps on one dataset can hurt on another: Cutout improves CIFAR and SVHN but not ImageNet, and the AutoAugment ImageNet proxy covered only about 10% of the classes. {#c-ra-proxy-imagenet}
The operations are made for natural images; on digits or text, a large rotation can change the label.
One $M$ for all operations assumes the hand-set ranges make equal steps comparable.

## Variants

- **AutoAugment** learns 25 sub-policies with probabilities and magnitudes by reinforcement learning on a proxy task (Cubuk et al. 2018).
- **Fast AutoAugment** and **Population Based Augmentation** learn policies with much cheaper searches than AutoAugment; Cubuk et al. (2019) compare against both.
- **TrivialAugment** applies one operation per image at a uniformly random strength and has no hyperparameters (Müller and Hutter 2021).
- **RandAugment with learned probabilities** keeps $N$ and $M$ and learns the $K$ selection probabilities by a differentiable density-matching objective (Section 4.7 of the paper).
- **Cutout** alone is the one-operation case with a fixed size and probability 1 (DeVries and Taylor 2017).
- **Mixup and CutMix** mix two images and their labels instead of transforming one; see [[mixup]].

## When to use it

RandAugment is [[al:regularization]] against [[al:overfitting]] that acts on the inputs, where [[dropout]] acts on the units and [[al:label-smoothing]] on the targets; the three combine.
Use RandAugment, or TrivialAugment, when training an image classifier or detector from scratch and you want stronger augmentation than flips and crops without running a policy search.
Start from $N = 2$ and a middle magnitude, and raise $M$ for larger models and longer training; the paper's grids are a guide to the range.
If you cannot afford even a small grid, TrivialAugment has nothing to tune and matched or beat RandAugment on most benchmarks of its authors' comparison.
Keep the label-preserving assumption in mind: for data where orientation, colour or position carry the label, remove the operations that change them.

## Pitfalls

> [!pitfall]
> **Sampling without replacement.** `random.sample` draws distinct operations, but the paper's `np.random.choice(transforms, N)` draws with replacement, so the same operation can come twice. With $K = 14$, $N = 2$ that happens 1 time in 14. `test_randaugment_sample` checks the repeat rate against $1/K$.

> [!pitfall]
> **Comparing a uint8 image with 256.** Solarize at magnitude 0 has threshold 256 and must leave the image alone, but in PyTorch `img >= 256` on a `uint8` tensor wraps the threshold to 0 and inverts every pixel. `test_posterize_and_solarize` checks that a threshold of 256 is the identity.

> [!pitfall]
> **Reading the magnitude the wrong way round for posterize and solarize.** For these two, a larger magnitude means fewer bits and a lower threshold. A linear map that sends $M = 0$ to 0 bits destroys the image at the mildest setting. `test_magnitude_map` checks that $M = 0$ keeps 8 bits and threshold 256.

> [!pitfall]
> **Always rotating or shifting one way.** Signed operations need a random sign, or the augmented data are biased toward one direction. `test_signs_are_balanced` checks that left and right shifts are equally frequent.

> [!pitfall]
> **Keeping the Cutout square inside the image.** Drawing the centre so that the square never crosses the border changes the distribution DeVries and Taylor found to work, and makes the mean masked area $s^2$ instead of eq-cutout-area. `test_cutout_area_on_the_pixel_grid` checks the mean area over all centres, and `test_cutout_op_centre_and_size` checks that centres reach every row.

> [!pitfall]
> **Mistaking the magnitude scale between libraries.** The paper's $M$ is out of 10 and torchvision's default 31 bins put it out of 30, so copying $M = 9$ from one to the other changes the strength by three times. `test_magnitude_map` checks the map at $m_{\max} = 10$ and $m_{\max} = 30$.

## History

Terrance DeVries and Graham W. Taylor of the University of Guelph proposed Cutout in 2017, arXiv 1708.04552. {#c-hist-co}
They chose a 16 × 16 patch for CIFAR-10 and 8 × 8 for CIFAR-100, after accuracy rose and then fell with patch size. {#c-co-size}
Ekin D. Cubuk, Barret Zoph, Dandelion Mané, Vijay Vasudevan and Quoc V. Le of Google Brain posted AutoAugment in 2018, arXiv 1805.09501. {#c-hist-aa}
Cubuk, Zoph, Jonathon Shlens and Le of Google Research, Brain Team posted RandAugment in September 2019, arXiv 1909.13719. {#c-hist-ra}
Samuel G. Müller and Frank Hutter of the University of Freiburg posted TrivialAugment in 2021, arXiv 2103.10158, as the simplest baseline that still matched the learned methods. {#c-hist-ta}
torchvision ships RandAugment and TrivialAugmentWide as standard transforms.
