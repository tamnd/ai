---
tag: MTKC7P
kind: finding
slug: double-descent
name: Double descent
aliases: [double descent risk curve, deep double descent, model-wise double descent, epoch-wise double descent, sample-wise non-monotonicity, peaking phenomenon]
area: deep-learning-training/grokking-and-double-descent
level: core
summary: >-
  As a model grows past the size at which it can fit its training data exactly, its test error can first rise to a peak and then fall a second time, often below the best error of the smaller models.
history:
  year: 2018
  note: "Named by Belkin, Hsu, Ma and Mandal, arXiv 1812.11118; the peak in minimum-norm linear classifiers was reported by Vallet et al. (1989) and analysed by Opper et al. (1990)."
prereqs: [bias-variance-tradeoff@al, overfitting@al, generalization-error@al, linear-regression@al, ridge-regression@al]
same_as:
  wikipedia: Double_descent
  arxiv: ["1812.11118", "1912.02292"]
sources:
  - {paper: "Belkin, Hsu, Ma and Mandal (2018)", title: "Reconciling modern machine learning practice and the bias-variance trade-off", locator: "the abstract, Section 2, Figure 2", note: "the name and the first experiments across model families"}
  - {paper: "Nakkiran, Kaplun, Bansal, Yang, Barak and Sutskever (2019)", title: "Deep Double Descent: Where Bigger Models and More Data Hurt", locator: "the abstract, Section 2", note: "deep networks, epochs and samples; effective model complexity"}
  - {paper: "Loog, Viering, Mey, Krijthe and Tax (2020)", title: "A Brief Prehistory of Double Descent", locator: "p. 1", note: "the earlier reports"}
  - {paper: "Hastie, Montanari, Rosset and Tibshirani (2019)", title: "Surprises in High-Dimensional Ridgeless Least Squares Interpolation", locator: "Section 1.1", note: "the exact risk of minimum-norm least squares"}
  - {paper: "Advani and Saxe (2017)", title: "High-dimensional dynamics of generalization error in neural networks", locator: "the abstract"}
  - {paper: "Nakkiran, Venkat, Kakade and Ma (2020)", title: "Optimal Regularization Can Mitigate Double Descent", locator: "the abstract"}
  - {paper: "Schaeffer et al. (2023)", title: "Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle", locator: "Section 6"}
  - {paper: "Curth, Jeffares and van der Schaar (2023)", title: "A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning", locator: "the abstract, Section 6"}
claim: >-
  For many learning methods, test error as a function of model size first follows the classical U shape, peaks near the interpolation threshold where the model can just fit the training set, and falls again as the size grows further.
first_reported: {work: belkin-2018-reconciling-bias-variance, note: "named the curve and showed it for random Fourier features, two-layer networks, random forests and boosting; the peak itself was known earlier in minimum-norm linear classifiers"}
replications:
  - {work: nakkiran-2019-deep-double-descent, note: "CNNs, ResNets and Transformers on CIFAR-10, CIFAR-100 and IWSLT'14, as a function of width, epochs and samples"}
  - {work: hastie-2019-surprises-ridgeless-least-squares, note: "exact asymptotics for minimum-norm least squares, linear and random-feature models"}
  - {work: schaeffer-2023-double-descent-demystified, note: "ordinary linear regression on real data, with the three factors ablated"}
scale_range:
  - {range: "two-layer networks and random Fourier features on subsets of MNIST of 4,000 to 10,000 examples, up to 60,000 features", work: belkin-2018-reconciling-bias-variance}
  - {range: "ResNet18 of widths up to 64, five-layer CNNs and Transformers on IWSLT'14 German-to-English", work: nakkiran-2019-deep-double-descent}
failures:
  - {work: nakkiran-2020-optimal-regularization-double-descent, note: "optimally tuned l2 regularization gives monotone test risk in isotropic linear regression and mitigates the peak in networks"}
  - {work: curth-2023-u-turn-double-descent, note: "in trees, boosting and linear regression the second descent comes from a change of complexity axis, and an effective parameter count restores a U shape"}
  - {work: loog-2020-brief-prehistory-double-descent, note: "the linear SVM shows monotone feature curves where the pseudo-inverse classifier peaks"}
explanations:
  - {name: "the minimum-norm interpolant", summary: "Past the threshold many interpolants exist, and a larger class contains one of smaller norm, which generalizes better.", sources: [belkin-2018-reconciling-bias-variance, hastie-2019-surprises-ridgeless-least-squares]}
  - {name: "small singular values at the threshold", summary: "Near the threshold the training features have small but nonzero singular values, and the fit divides the residual noise by them.", sources: [schaeffer-2023-double-descent-demystified, advani-2017-high-dimensional-dynamics-generalization]}
  - {name: "two complexity axes", summary: "The raw parameter count mixes two ways of adding capacity, and the second descent appears where one gives way to the other.", sources: [curth-2023-u-turn-double-descent]}
---

Double descent is the finding that test error need not follow a single U-shaped curve as a model grows.
Past the size at which the model can fit its training data exactly, called the interpolation threshold, the error can peak and then fall a second time.
The curve was named in 2018 for random features, small networks and random forests, and it was then found in deep networks, along the number of training epochs, and along the number of training samples.
The peak is sharpest with noisy labels and no regularization, and tuned regularization can remove it.

## In plain words

The textbook picture says that a model that is too small misses the pattern, a model that is too large memorizes the noise, and the right size lies in between, so the test error traces a U.
Double descent adds a second half to that picture.
Keep making the model larger, past the point where it can fit every training example exactly, and the test error, which was rising, turns and falls again.
The worst model is the one that can only just fit the training set: it must bend to pass through every noisy point.
A much larger model has many ways to fit the same points, and the training method tends to pick a smooth one.
The peak depends on size relative to the data, so adding data to a fixed model can also make it worse.
The effect is strongest with noisy labels; good regularization flattens it.

## The finding

Belkin et al. (2018) proposed a "double descent" curve that subsumes the textbook U-shaped bias-variance curve, showing how increasing model capacity beyond the point of interpolation results in improved performance. {#c-belkin-curve}
Their evidence spans random Fourier features, fully connected networks, random forests and boosting.
They identify capacity with the number of parameters and, past the threshold, choose the interpolating solution of minimum norm.

The cleanest case is random Fourier features, a two-layer network with fixed random first-layer weights, trained with the squared loss on a subset of MNIST with $n = 10^4$ examples.
With $N$ features, the test risk shows the classical U shape for small $N$ and peaks at the interpolation threshold $N = n$; the model that interpolates with the fewest features is the least accurate, and as $N$ grows beyond $n$ the accuracy improves to exceed that of the bottom of the U. {#c-rff-mnist}
With $K$ outputs, each fitted by the squared loss, interpolation takes about $n \cdot K$ parameters: for a network with one hidden layer on a subset of MNIST with $K = 10$ classes, Belkin et al. observe the threshold at $n \cdot K$ parameters. {#c-threshold-nk}
They report the same shape before and after the threshold for boosting with decision trees and for random forests. {#c-trees}

Nakkiran et al. (2019) found the curve in modern deep learning: as model size grows, performance first gets worse and then gets better, and the same happens as a function of the number of training epochs. {#c-deep}
They report this model-wise double descent for CIFAR-10, CIFAR-100 and IWSLT'14 German-to-English with varying label noise, for CNNs, ResNets and Transformers, and for SGD and Adam. {#c-deep-range}
To place the peak, they define the effective model complexity of a training procedure as the largest number of samples on which it reaches, on average, close to zero training error. {#c-emc}
Their hypothesis is that when the effective complexity is well below or well above the number of samples, any change that raises it lowers the test error, and that when the two are close, a change that raises it may lower or raise the test error. {#c-hypothesis}

Write $n$ for the number of training samples and $\mathrm{EMC}(T)$ for the effective complexity of a procedure $T$.
The three regimes are

$$ \mathrm{EMC}(T) \ll n \ \text{(under-parameterized)}, \qquad \mathrm{EMC}(T) \approx n \ \text{(critical)}, \qquad \mathrm{EMC}(T) \gg n \ \text{(over-parameterized)}. $$ {#eq-regimes}

In the critical regime more data can hurt.
On IWSLT'14 German-to-English, Nakkiran et al. show cases in which increasing the number of samples by a factor of 4.5 gives worse test performance. {#c-more-data}
More data both lowers the curve of test error against model size and shifts it to the right, so for models near the threshold the two effects can cancel or reverse. {#c-shift}

## Evidence

The generated list gives each source with its label and its role.
The lines of evidence are these.

The peak in linear models is older than the name.
Loog et al. (2020) point out that Vallet et al. demonstrated double descent in 1989 for the learning curves of classifiers trained by minimum-norm linear regression, the pseudo-inverse solution, with artificial data, and that Opper et al. gave a first theoretical result. {#c-prehistory}
Duin was the first to show feature curves on real-world data quite similar to those of Belkin et al., with a pseudo-Fisher linear discriminant. {#c-duin}

For minimum-norm least squares the curve has an exact form.
Hastie et al. (2019) study the limit in which the number of features $p$ and samples $n$ grow with $p/n \to \gamma$.
For $\gamma < 1$ the minimum-norm estimator is ordinary least squares, its risk is pure variance, and the risk diverges as $\gamma$ approaches 1. {#c-diverge}
Their results cover both a linear model for the features and a nonlinear one, in which the features are the outputs of a random one-layer network. {#c-hastie-nonlinear}
In random features the peak sits where the number of features equals the number of samples, which is the case of Belkin et al.

Advani and Saxe (2017) analysed gradient descent on linear and shallow networks with random matrix theory.
They found overtraining worst at intermediate sizes, when the effective number of free parameters equals the number of samples, so that it can be reduced by making the network smaller or larger. {#c-advani}

Nakkiran et al. (2019) also report epoch-wise double descent: a large model trained longer first improves, then gets worse, then improves again, and many combinations of dataset and architecture show it in the presence of label noise. {#c-epoch-wise}
Schaeffer et al. (2023) reproduced the curve with ordinary linear regression on real data, then removed it by ablating each of three factors in turn (see Explanations). {#c-ablation}

## Scope and conditions

**Label noise.**
Nakkiran et al. observe all forms of double descent most strongly with label noise in the training set, and they also show realistic settings with a peak and no label noise. {#c-label-noise}
They believe the effect is not about the noise itself but about model mis-specification, and they view added label noise as merely a proxy for making the distribution harder. {#c-misspecification}
Schaeffer et al. state the same condition for linear regression: if some model in the class fits the training data with no error, no double descent occurs. {#c-no-residual}

**Regularization.**
Hastie et al. find that optimally tuned ridge regression dominates the minimum-norm estimator in risk for all values of $\gamma$ and of the signal-to-noise ratio. {#c-ridge-dominates}
Nakkiran et al. (2020) prove that for certain linear regression models with isotropic data, optimally tuned $\ell_2$ regularization gives test performance that is monotone in both the sample size and the model size, and show empirically that it mitigates double descent in more general models, including neural networks. {#c-optimal-reg}
Belkin et al. note that regularization of all forms can prevent interpolation and change the effective capacity of the function class, which attenuates or masks the peak. {#c-masking}
The peak is thus a property of fitting without regularization, or with too little, and not of size alone.

**Early stopping.**
Nakkiran et al. note that many of the phenomena they highlight often do not occur with optimal early stopping, which keeps the effective complexity below the number of samples. {#c-early-stopping}
They show at least one setting, ResNets on CIFAR-100 with no label noise, where model-wise double descent remains with optimal early stopping, and they have not observed more data hurting under optimal early stopping. {#c-early-stopping-exception}

**Where the second descent lands.**
The second descent does not always go below the bottom of the U.
In the well-specified isotropic linear model, Hastie et al. find that the risk of the minimum-norm estimator reaches its global minimum in the under-parameterized regime; in a misspecified model with a signal-to-noise ratio above 1 and strong enough approximation bias, the global minimum can lie in the over-parameterized regime. {#c-global-min}

**The x-axis.**
Curth et al. (2023) examined the non-deep cases.
In trees, boosting and linear regression they find that the raw parameter count grows along two distinct complexity axes, and that the second descent appears exactly where the growth passes from one axis to the other, so its location is not tied to the interpolation threshold $p = n$. {#c-axes}
In the tree experiment of Belkin et al., the parameter count grows first through the number of leaves of a single tree and then through the number of fully grown trees that are averaged, and along each axis alone the error has a convex U or L shape. {#c-tree-axes}
They limited this analysis to non-deep models and left deep double descent open. {#c-curth-scope}

**Other losses.**
Loog et al. report that some losses may not show double descent at all: in feature curves the linear SVM behaves monotonically, and the perceptron of optimal stability does the same in analytic results. {#c-svm}

## Explanations

**The minimum-norm interpolant.**
The claim: past the threshold the training set no longer pins down the fit, and the procedure picks the interpolant of smallest norm.
Belkin et al. argue that choosing the smoothest function that fits the data is a form of Occam's razor, and that larger function classes contain interpolating functions of smaller norm, so that increasing capacity improves performance. {#c-occam}
Hastie et al. make this exact for least squares: in the over-parameterized regime the variance falls as $\gamma$ grows, because the minimum-norm solution can spread its weight over more columns and so has a smaller norm, while the bias rises. {#c-variance-falls}
The consequence: the second descent depends on the implicit bias of the fitting procedure, not on the parameter count alone.

**Small singular values at the threshold.**
The claim: Schaeffer et al. identify three factors that together create double descent in ordinary linear regression: small but nonzero singular values in the training features, a test point that varies in directions the training data barely cover, and residual errors on the training data that no model in the class can remove. {#c-three-factors}
The evidence: removing any one factor prevents the curve, for example by ridge regression, which separates the smallest nonzero singular value from zero. {#c-ridge-gap}
Advani and Saxe describe a related picture for gradient descent in large networks: a frozen subspace of the weights in which no learning occurs, and input correlations that are better conditioned in high dimensions. {#c-frozen}
The consequence: the peak comes from fitting noise along directions the data barely determine, and it shrinks when those directions are damped.

**Two complexity axes.**
The claim: for the non-deep methods, Curth et al. argue that the curve is two U shapes joined on a composite axis.
The evidence: with a measure of the effective number of parameters that the model uses on unseen inputs, plotted in place of the raw count, the apparent double descent curves fold back into more traditional U shapes. {#c-fold-back}
The consequence: under this reading, double descent in these models refines what counts as complexity rather than overturning the bias-variance trade-off.

The three explanations overlap rather than compete: the first says why the far right of the curve is low, the second why the threshold is high, and the third how the axis is drawn.

## Implications

The implications below are those that the sources draw.
Belkin et al. argue that the peak was overlooked partly because it shows within a narrow range of parameters, so that a sweep that misses the range gives the impression that a larger network is simply better, and because early stopping makes the peak hard to observe. {#c-overlooked}
Nakkiran et al. advise that when a model and training procedure can only just fit the training set, small changes to the model or the procedure may have unexpected effects on test performance. {#c-practice}
Nakkiran et al. (2020) suggest studying how the test risk of a method scales with size under appropriately tuned regularization. {#c-tuned-scaling}
Curth et al. read the non-deep evidence as showing trade-offs between hyperparameter tuning and raw model size: one axis is sensitive to tuning, and along the other, bigger is better or at least not worse. {#c-tradeoff}

## History

Vallet, Cailton and Refregier reported the peak for the pseudo-inverse solution in Europhysics Letters in 1989, and Opper, Kinzel, Kleinz and Nehl gave a first theoretical result in 1990. {#c-hist-1989}
In pattern recognition, Raudys and Duin and Skurichina and Duin in 1998, and Duin in 2000, studied the same peak in the pseudo-Fisher linear discriminant, which is equivalent to minimum-norm linear regression. {#c-hist-pfld}
Advani and Saxe posted their analysis of overtraining at the threshold as arXiv 1710.03667 on 10 October 2017. {#c-hist-2017}
Belkin, Hsu, Ma and Mandal named the double descent curve in arXiv 1812.11118, posted in December 2018 and published in PNAS in 2019.
Nakkiran, Kaplun, Bansal, Yang, Barak and Sutskever posted "Deep Double Descent" as arXiv 1912.02292 on 4 December 2019. {#c-hist-2019}
Loog et al.'s letter on the prehistory followed in PNAS in 2020, and later work turned to explaining and removing the peak: optimal regularization in 2020, and the linear ablations and the parameter-counting account in 2023.

## References
