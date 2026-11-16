---
tag: NEW-early-stopping
kind: method
slug: early-stopping
name: Early stopping
aliases: [early-stopped gradient descent, patience, stopping rule, Landweber iteration]
area: deep-learning-training/regularization
level: core
summary: >-
  Stop training at the step where the loss on held-out data was lowest, and keep the weights from that step; on least squares this acts like ridge with penalty 1/(eta t).
purpose: regularization
uses: [stochastic-gradient-descent@m2, ridge-regression@al]
related_to: [dropout, weight-decay@al, overfitting@al, cross-validation@al, ridge-regression@m2, cross-validation@m2]
hyperparameters:
  - {name: patience, meaning: "the number of evaluations in a row without improvement after which training stops", typical: "a few evaluations; larger when the validation curve is noisy"}
  - {name: min_delta, meaning: "the amount by which an evaluation must beat the best so far to count as an improvement", typical: "0, or about the noise of the validation metric"}
  - {name: restore best, meaning: "whether to load the weights of the best evaluation when training stops, rather than keep the last ones", typical: "on"}
cost:
  train: "one pass over the validation set per evaluation, one copy of the parameters, and patience evaluations past the best"
  inference: "none"
history:
  year: 1989
  people: [Nelson Morgan, Hervé Bourlard]
  note: "Ali, Kolter and Tibshirani date early-stopped gradient descent for neural networks to Morgan and Bourlard (1989), and early-stopped Landweber iterations for inverse problems to Strand (1974)."
prereqs: [stochastic-gradient-descent@m2, overfitting@al, cross-validation@al]
same_as:
  wikidata: null
  wikipedia: null
  arxiv: ["1810.10082", "math/0508276"]
  al: early-stopping
sources:
  - {paper: "Ali, Kolter and Tibshirani (2019)", title: "A Continuous-Time View of Early Stopping for Least Squares Regression", locator: "Sections 1, 2.2, 3.1, 5 and 8, page 8", note: "the spectral maps, Lemmas 1 to 4 and 7, Theorems 1 and 3, and the numerical examples"}
  - {paper: "Yao, Rosasco and Caponnetto (2005)", title: "On Early Stopping in Gradient Descent Learning", locator: "the abstract, Sections 1 and 3.1", note: "the bias-variance view, the two filters, and saturation"}
  - {paper: "Zhang and Yu (2005)", title: "Boosting with Early Stopping: Convergence and Consistency", locator: "the abstract"}
  - {paper: "Caruana, Lawrence and Giles (2000)", title: "Overfitting in Neural Nets: Backpropagation, Conjugate Gradient, and Early Stopping", locator: "the abstract, Sections 4.1 and 4.2"}
---

Early stopping ends training before the training loss stops falling.
A held-out validation set is evaluated as training goes, and the run keeps the weights from the evaluation where the validation loss was lowest.
The usual rule waits a fixed number of evaluations, the patience, for a new best before stopping.
On least squares, gradient descent stopped after $t$ steps of size $\eta$ shrinks each eigen-direction of the data like ridge regression with penalty $1/(\eta t)$, so the number of steps acts as a regularization parameter.

## In plain words

A network trained for long enough fits its training data better and better, including the noise in it.
At first this also makes it better on new data; later it makes it worse.
Early stopping watches a separate set of examples that the network never trains on.
While the error there keeps falling, training goes on.
Once it has not fallen for a while, training stops, and the network goes back to the weights it had at its best moment.
Nothing in the loss or the network changes.
The only choice is when to stop, and the steps taken so far limit how far the weights can move from where they started, which is what makes it a regularizer.

## Setting

A model with weights $w$ is trained by an iterative optimizer from an initial $w_0$, and is evaluated every so often on a validation set.
Write $e_t$ for the validation loss at the $t$-th evaluation, with lower better.
Zhang and Yu describe early stopping, based on cross-validation or a test set, as an unusual regularization technique used in boosting. {#c-zy-unusual}
Yao, Rosasco and Caponnetto read the stopping time as a bias-variance trade-off: an early stop can cut the variance at the price of more bias, and a late stop the reverse. {#c-yao-bias-variance}
In their setting the iteration on the sample often converges to an overfitting function, and keeping it close to the iteration on the population acts as regularization. {#c-yao-paths}

For the analysis this page uses least squares, $\frac{1}{2n}\lVert y - Xw\rVert^2$, with $\hat\Sigma = X^\top X/n = V S V^\top$, eigenvalues $s_i \ge 0$, eigenvectors $v_i$, $b = X^\top y/n$ and $c_i = v_i^\top b$.
Gradient descent from $w_0 = 0$ takes

$$ w_k = w_{k-1} + \eta\, X^\top (y - X w_{k-1})/n . $$ {#eq-gd-step}

Ali, Kolter and Tibshirani compare this path with ridge regression, the minimizer of $\frac{1}{n}\lVert y - X\beta\rVert^2 + \lambda\lVert\beta\rVert^2$, which is $(X^\top X + n\lambda I)^{-1}X^\top y$. {#c-ali-ridge}

## The method

The patience rule with patience $p \ge 1$ and threshold $\delta \ge 0$ keeps the best value $B$, its epoch $b$ and its weights.
Evaluation $t$ improves when $e_t$ is not NaN and

$$ e_t < B - \delta , $$ {#eq-patience}

with $B = +\infty$ before the first evaluation.
On an improvement it sets $B = e_t$ and $b = t$ and copies the weights; otherwise it counts one more evaluation without improvement.
It stops when that count reaches $p$, and loads the weights of epoch $b$.
For a metric to maximize, such as accuracy, the comparison is $e_t > B + \delta$.

The patience rule stops exactly $p$ evaluations after its last improvement $b$, so every run trains $p$ evaluations past the weights it returns. {#c-patience-overshoot}
Every evaluation before $b$ is worse than $e_b$, and every one after it is at least $e_b - \delta$, so $e_b$ is within $\delta$ of the lowest value seen, and equal to it when $\delta = 0$. {#c-patience-best}
If the validation loss falls strictly to epoch $m$ and never falls after it, the rule with $\delta = 0$ returns epoch $m$ and stops at $m + p$. {#c-patience-unimodal}
The derivation `patience` proves these three facts, and its check runs them on 5000 random walks.

## Derivation

**Gradient descent is a spectral filter.**
In the eigenbasis eq-gd-step reads $a_{k,i} = (1 - \eta s_i) a_{k-1,i} + \eta c_i$, a geometric sum from $a_{0,i} = 0$.
After $t$ steps, gradient descent from zero is $w_t = \sum_{s_i > 0} \varphi_t(s_i)\,(c_i/s_i)\,v_i$, with the filter {#c-gd-filter}

$$ \varphi_t(s) = 1 - (1 - \eta s)^t . $$ {#eq-gd-filter}

Ridge with penalty $\lambda$ gives the same sum with the filter {#c-ridge-filter}

$$ r_\lambda(s) = \frac{s}{s + \lambda} . $$ {#eq-ridge-filter}

Directions with $s_i = 0$ get nothing from either, because $c_i = 0$ there.
Ali et al. write both paths on the scale of fitted values, with the shrinkage map $s/(s + \lambda)$ for ridge and $1 - \exp(-ts)$ for gradient flow; both shrink the low-variance directions more. {#c-ali-maps}
Yao et al. write Tikhonov regularization and their population iteration with the same two filters, and read both as low-pass filters. {#c-yao-filters}

**The matched penalty.**
Both filters vanish at $s = 0$, with slopes $\eta t$ and $1/\lambda$.
They agree to first order in $s$ exactly when {#c-matched-slope}

$$ \lambda = \frac{1}{\eta t} . $$ {#eq-matched-lambda}

Their second-order coefficients are then $-\eta^2 t(t-1)/2$ and $-\eta^2 t^2$, so no choice of $\lambda$ makes the two filters equal. {#c-not-identical}
Ali et al. say the same for gradient flow: no parametrization $\lambda = \phi(t)$ equates the two maps for all $t$, but $\phi(t) = 1/t$ makes them grossly similar. {#c-ali-no-param}
Yao et al. likewise relate the two roughly as $\lambda \sim 1/t$. {#c-yao-lambda-t}
With $\lambda = 1/(\eta t)$ and $0 \le \eta s \le 1$,

$$ r_\lambda(s) \le \varphi_t(s) \le \min(1, \eta t s) \le 2\, r_\lambda(s) , $$ {#eq-sandwich}

so the stopped weights have a norm between that of the matched ridge solution and twice it. {#c-sandwich}
The derivation `spectral-filters` proves this from $(1 - x)^t \le e^{-tx} \le 1/(1 + tx)$ and Bernoulli's inequality.
Its check runs gradient descent on two random designs; at $t = 200$ the stopped and matched-ridge weights differ by about 6% of the ridge norm in our runs.

**Gradient flow.**
As $\eta \to 0$ with $\tau = k\eta$ fixed, gradient descent becomes gradient flow $\dot w = X^\top(y - Xw)/n$.
Lemma 1 of Ali et al. gives its exact solution from zero, $(X^\top X)^+(I - \exp(-tX^\top X/n))X^\top y$. {#c-ali-lemma1}
Per direction this is the filter {#c-flow-filter}

$$ \psi_\tau(s) = 1 - e^{-s\tau} . $$ {#eq-flow-filter}

If $\eta s_{\max} \le 1$, then for every $k$

$$ \lVert w_k - w(k\eta)\rVert \le \frac{\eta}{2}\,\lVert X^\top y/n\rVert , $$ {#eq-flow-gap}

so gradient descent tracks the flow with an error of order $\eta$ that does not grow with time. {#c-flow-limit}
The derivation `gradient-flow` proves this from $|(1 - x)^k - e^{-kx}| \le \frac{x}{2}e^{x - 1}$ on $0 \le x \le 1$.
Lemma 2 of Ali et al. bounds the same gap for step sizes below $1/s_{\max}$ by a quantity that grows exponentially with the number of steps. {#c-ali-lemma2}
For the flow, $r_{1/\tau}(s) \le \psi_\tau(s) \le 1.2985\, r_{1/\tau}(s)$ for all $s \ge 0$, from $e^{-x} \le 1/(1 + x)$ and $1 - e^{-x} \le 1.2985\, x/(1 + x)$. {#c-flow-ridge-ratio}
These two inequalities are Lemma 7 of Ali et al. {#c-ali-lemma7}

**Generalized ridge.**
Lemma 3 of Ali et al. shows that each gradient descent iterate with step below $1/s_{\max}$ solves a least squares problem with a quadratic penalty $\beta^\top Q_k\beta$, where $Q_k = VS((I - \eta S)^{-k} - I)^{-1}V^\top$. {#c-ali-lemma3}
So stopping at step $k$ is ridge with a penalty that is heavier on the low-variance directions and relaxes as $k$ grows.

**Risk.**
The filters compare shrinkage; the paper compares risk.
Theorem 1 of Ali et al. states that under their data model, for every true coefficient vector and every $t \ge 0$, the estimation risk of gradient flow at time $t$ is at most 1.6862 times that of ridge at $\lambda = 1/t$. {#c-ali-theorem1}
The same holds for the Bayes risk under any prior, and for in-sample prediction risk. {#c-ali-theorem1-bc}
Because gradient flow at each $t > 0$ is a unique Bayes estimator and hence admissible, no universal constant below 1 can replace 1.6862. {#c-ali-admissible}
When both are tuned to their best, Theorem 3 bounds the ratio of the Bayes risk of gradient flow to that of ridge between 1 and 1.2147. {#c-ali-theorem3}

**Beyond ridge.**
Yao et al. find that early stopping does not suffer the saturation of Tikhonov regularization, whose rate stops improving once the target is smooth enough. {#c-yao-saturation}
For smooth targets their bounds for early stopping reach $O(t^{-r})$, while Tikhonov's cannot go faster than $t^{-1}$. {#c-yao-rates}
Their stopping rule, from a bias-variance trade-off, achieves the minimax rate $O(m^{-1/2})$ in some cases. {#c-yao-minimax}

## Worked example

Take three eigenvalues $s = 4, 1, 0.25$, step $\eta = 0.1$ and $t = 10$ steps, so the matched penalty is $\lambda = 1/(\eta t) = 1$ and the flow time is $\tau = 1$.

| $s$ | $\varphi_{10}(s)$, eq-gd-filter | $r_1(s)$, eq-ridge-filter | $\psi_1(s)$, eq-flow-filter |
|---|---|---|---|
| 4 | 0.9940 | 0.8 | 0.9817 |
| 1 | 0.6513 | 0.5 | 0.6321 |
| 0.25 | 0.2237 | 0.2 | 0.2212 |

Every gradient descent value lies between the ridge value and twice it, as eq-sandwich says, here with $\eta s = 0.4$ at most.
The small eigenvalue is shrunk about equally by all three; the large one is kept almost whole by gradient descent and shrunk to 0.8 by ridge.

Now the patience rule with $p = 2$ on the validation losses $1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74$.
Epochs 1, 2, 3 and 5 improve.
Epochs 6 and 7 do not, so the rule stops at epoch 7 and restores the weights of epoch 5, with loss 0.69.
With $\delta = 0.02$ epoch 4 does not improve, and neither does epoch 5, since $0.69 \not< 0.70 - 0.02$; the rule stops at epoch 5 and restores epoch 3.
The tests `test_worked_example`, `test_patience_worked_example` and `test_min_delta_worked_example` check these numbers.

## Hyperparameters

**Patience.**
A small patience stops soon after a minimum and wastes little compute, but stops at the first plateau.
A large one rides out noise and plateaus, at the cost of $p$ evaluations past the best on every run.
**Threshold.**
A positive $\delta$ ignores improvements smaller than the noise of the metric.
It can also end the run up to $\delta$ above the lowest loss seen, as in the worked example.
**Evaluation interval.**
Evaluating every epoch, or every few hundred steps, sets the grain of the stopping time and the cost of validation.
**Restore best.**
Restoring the best weights is what the guarantees above are about; the last weights are $p$ evaluations later and usually worse.

On least squares eq-matched-lambda turns the stopping time into a penalty: doubling $\eta$ or $t$ halves the matched $\lambda$.

## Cost

Each evaluation costs one forward pass over the validation set.
Each improvement copies the weights once, and the rule holds one extra copy of the parameters in memory.
A run trains $p$ evaluations beyond the epoch it returns, but stops long before a fixed long budget would.
Nothing changes at inference.

## Evidence

In Ali et al.'s numerical examples with Gaussian features, $n = 500$ and $p = 1000$, the largest ratio of gradient flow to ridge risk at $\lambda = 1/t$ was 1.2164, and the ratio of the minimum risks 1.0036. {#c-ali-experiment}
Calibrated by the $\ell_2$ norm of the estimators instead, the two risk curves agreed to a largest ratio of 1.0050. {#c-ali-norm}
Caruana, Lawrence and Giles trained one-hidden-layer networks of 2 to 800 hidden units on seven problems with backpropagation and early stopping. {#c-caruana-setup}
Only on three problems did nets that were too large generalize worse than smaller ones, and the loss was small. {#c-caruana-three}
On some tasks small nets overtrained considerably, so they conclude that early stopping matters for nets of all sizes. {#c-caruana-small}
They also found that conjugate gradient can generalize worse than backpropagation, because it overfits regions of low non-linearity. {#c-caruana-cg}

## Limitations and failure modes

The equivalence with ridge is proved for least squares from a zero start.
From another start $w_0$, each direction keeps $(1 - \eta s_i)^t v_i^\top w_0$ of it, so the stopped weights shrink toward $w_0$, not toward zero.
Ali et al. leave a continuum version of stochastic gradient descent, and losses beyond least squares such as logistic regression, to future work. {#c-ali-open}
For neural networks the link to an $\ell_2$ penalty is an analogy, not a theorem.

The rule itself guarantees little about the curve it has not seen.
For every patience $p$, a validation curve that stays flat for $p$ evaluations and then drops makes the rule stop before the drop. {#c-patience-plateau}
The kept validation loss is the minimum of noisy values, so it is biased low as an estimate of the error of the kept weights; on a flat noisy curve in our run the bias was 0.09 for noise of standard deviation 0.05 and $p = 10$. {#c-patience-bias}
Report the error of the stopped model on a separate test set, which the rule never reads.
The validation set also costs training data.

## Variants

**A fixed budget.**
Choose the number of steps $t$ by validation once, as one would choose $\lambda$, and train every later model for $t$ steps.
**Threshold criteria.**
Instead of patience, stop when the validation loss is a set fraction above its best so far.
With $E_{\text{opt}}(t) = \min_{t' \le t} e_{t'}$, the generalization loss is

$$ \mathrm{GL}(t) = 100\left(\frac{e_t}{E_{\text{opt}}(t)} - 1\right) , $$ {#eq-gl}

and $\mathrm{GL}_\alpha$ stops at the first $t$ with $\mathrm{GL}(t) > \alpha$.
The training progress over a strip of the last $k$ training losses $E_{\text{tr}}$ is

$$ P_k(t) = 1000\left(\frac{\sum_{t' = t-k+1}^{t} E_{\text{tr}}(t')}{k \min_{t-k < t' \le t} E_{\text{tr}}(t')} - 1\right) , $$ {#eq-progress}

and $\mathrm{PQ}_\alpha$ stops at the end of the first strip where $\mathrm{GL}/P_k > \alpha$.
$\mathrm{UP}_s$ stops after the validation loss rose over $s$ strips in a row.
These criteria are usually credited to Lutz Prechelt's chapter "Early Stopping, but when?"; no open copy of it could be fetched for this page, so the formulas above are this page's own and are not checked against it.
`gl_stop`, `pq_stop` and `up_stop` implement them.
**Boosting.**
Zhang and Yu find early stopping strategies under which boosting over the span of a family of basis functions is consistent, with bounds on its rate of convergence. {#c-zy-consistent}
Run to convergence instead, AdaBoost with step size going to 0 becomes an $\ell_1$-margin maximizer on separable data. {#c-zy-adaboost}

## When to use it

Use early stopping whenever a validation set is affordable and the validation loss can rise during training, which Caruana et al. report for nets of every size.
Combine it with [[dropout]] or weight decay; it costs no change to the model and saves the rest of the training budget.
Set the patience from the noise and plateaus of the validation curve, and always restore the best weights.
On linear least squares, read the stopping time as the ridge penalty $1/(\eta t)$, and prefer early stopping when the target is smooth, where Tikhonov regularization saturates.
Skip it when the training budget is fixed by a schedule that must run to its end, such as a learning rate decayed to zero, unless the stopping point is chosen within that schedule.

## Pitfalls

> [!pitfall]
> **Keeping the last weights.** The rule always stops $p$ evaluations after the best, so the last weights are not the best ones. The tests `test_patience_worked_example` and `test_patience_restores_the_best_module` check that the rule returns epoch 5 of the worked example, and loads it into a module.

> [!pitfall]
> **Keeping a reference instead of a copy.** `model.state_dict()` returns the live tensors, so a kept dictionary changes with every later step. The tests `test_restore_is_a_copy` and `test_kept_state_is_a_clone` check that the kept state does not move when training goes on.

> [!pitfall]
> **A NaN validation loss.** Any comparison with NaN is false, so a NaN must never count as an improvement or reset the count. The test `test_nan_never_improves` checks this.

> [!pitfall]
> **Watching a metric to maximize with a rule that minimizes.** Accuracy must improve upward. The test `test_mode_max` checks the rule in mode "max", and `test_ties_do_not_improve` checks that ties never improve.

> [!pitfall]
> **Too little patience on a plateau.** A flat stretch as long as the patience ends the run before a later drop. The test `test_plateau_defeats_any_patience` builds such a curve for each patience.

> [!pitfall]
> **Matching ridge from a nonzero start.** The filters of eq-gd-filter hold from $w_0 = 0$; from another start the weights shrink toward $w_0$. The test `test_gradient_descent_from_w0` checks the extra term $(1 - \eta s_i)^t v_i^\top w_0$.

## History

Ali et al. date early-stopped gradient descent for training neural networks back to Morgan and Bourlard (1989). {#c-hist-morgan}
They date the same idea in applied mathematics, as early-stopped Landweber iterations for ill-posed linear inverse problems, to Strand (1974). {#c-hist-strand}
Yao et al. trace the low-pass filter reading in statistics to Wahba (1987), who derived generalized cross-validation criteria for data-dependent stopping rules. {#c-hist-wahba}
They list systematic studies of early stopping in boosting: Jiang (2004) for AdaBoost, Bühlmann and Yu (2002) for L2 Boost, and Zhang and Yu for general convex losses. {#c-hist-boosting}
Caruana, Lawrence and Giles found that with early stopping large networks pass through models similar to those of smaller nets, so excess capacity hurts little. {#c-caruana-sequence}
Alnur Ali, J. Zico Kolter and Ryan J. Tibshirani compared the whole gradient flow path with ridge regression in a paper at AISTATS 2019. {#c-hist-ali}
