---
id: teacher-ema
derives: [c-teacher-ema]
kind: limit
assumes: ["the data are a point mass at xi", "the model is f_theta(x, sigma) = (sigma_min / sigma) x + (1 - sigma_min / sigma) theta, with one scalar parameter", "the squared l2 metric and lambda = 1", "the levels sigma_min = sigma_1 < ... < sigma_N fill a fixed interval and their largest gap tends to 0 as N grows"]
uses: [loss-function@m2, expected-value@m2, stochastic-gradient-descent@m2]
status: full
check: teacher-ema.check.py
lean: null
sources:
  - {work: song-2023-improved-techniques-training-consistency, locator: "pp. 5 to 6, Section 3.2, Proposition 1 and Eqs. 6 to 7", note: "the toy of a point mass, the limit of the loss when the teacher differs from the student, and the scaled gradient when they are equal"}
---

### Statement

Let the data be a point mass at $\xi$, so that the consistency function is $f(x, \sigma) = (\sigma_{\min}/\sigma)\,x + (1 - \sigma_{\min}/\sigma)\,\xi$, and model it by $f_\theta(x, \sigma) = (\sigma_{\min}/\sigma)\,x + (1 - \sigma_{\min}/\sigma)\,\theta$. For the training pair $(\xi + \sigma_{i+1} z, \xi + \sigma_i z)$ the difference of the student's and the teacher's outputs is

$$ \Delta_i = \sigma_{\min}\,\xi\Bigl(\frac{1}{\sigma_{i+1}} - \frac{1}{\sigma_i}\Bigr) + \Bigl(1 - \frac{\sigma_{\min}}{\sigma_{i+1}}\Bigr)\theta - \Bigl(1 - \frac{\sigma_{\min}}{\sigma_i}\Bigr)\theta^- , $$

which does not depend on the noise $z$. Hence, as the gaps $\Delta\sigma$ between levels shrink:

1. if the teacher's parameter $\theta^-$ differs from $\theta$, the loss tends to $\mathbb E[(1 - \sigma_{\min}/\sigma_i)^2]\,(\theta - \theta^-)^2$, which does not involve $\xi$: it pulls the student towards the teacher and carries no information about the data;
2. if $\theta^- = \theta$ (the teacher is the student with its gradient stopped), $\Delta_i = \sigma_{\min}(1/\sigma_i - 1/\sigma_{i+1})(\theta - \xi)$ and the gradient in the student's $\theta$, divided by $\Delta\sigma$, tends to $\mathbb E[2(\sigma_{\min}/\sigma_i^2)(1 - \sigma_{\min}/\sigma_i)]\,(\theta - \xi)$, which is zero only at $\theta = \xi$, the true consistency function.

### Assumptions

1. A point mass and the one-parameter model, which contains the true consistency function at $\theta = \xi$. Used in step 1.
2. Squared $\ell_2$ metric with $\lambda = 1$. Used in steps 2 and 3.
3. The gaps shrink to 0 on a fixed interval of levels. Used in steps 2 and 3.

### Steps

1. Substituting the pair into the two models, the terms in $z$ are $(\sigma_{\min}/\sigma_{i+1})\,\sigma_{i+1} z - (\sigma_{\min}/\sigma_i)\,\sigma_i z = 0$, which leaves $\Delta_i$.
2. As $\sigma_{i+1} - \sigma_i \to 0$, the first term of $\Delta_i$ tends to 0 and $\Delta_i \to (1 - \sigma_{\min}/\sigma_i)(\theta - \theta^-)$; squaring and averaging over the levels gives the limit of item 1.
3. With $\theta^- = \theta$, $\Delta_i = \sigma_{\min}\,\xi\,(1/\sigma_{i+1} - 1/\sigma_i) + \sigma_{\min}(1/\sigma_i - 1/\sigma_{i+1})\,\theta = \sigma_{\min}(1/\sigma_i - 1/\sigma_{i+1})(\theta - \xi)$, and $1/\sigma_i - 1/\sigma_{i+1} = \Delta\sigma/(\sigma_i\sigma_{i+1})$.
4. The gradient through the student only is $\mathbb E[2\Delta_i\,(1 - \sigma_{\min}/\sigma_{i+1})]$; dividing by $\Delta\sigma$ and letting it go to 0 gives item 2.

### Result and check

A teacher kept as an exponential moving average of the student, $\theta^- \ne \theta$, makes the consistency-training loss in the limit of many levels a pure regression of the student onto the teacher, blind to the data; only the stop-gradient teacher, $\mu = 0$, gives a gradient that points at the data.
That is Song and Dhariwal's reason for dropping the EMA from the teacher in consistency training, while distillation can keep it, because there the teacher's step comes from a pre-trained score model and carries information about the data at any $N$.
With finitely many levels the EMA teacher does carry the data through the first term of $\Delta_i$, which is how the original consistency training of Song and colleagues worked, but with a weight that vanishes as $N$ grows.
The check `teacher-ema.check.py` computes the loss and the gradient of the toy on 101 and 4001 levels and 20 noises, and compares them with the exact finite-$N$ forms and with the limits of items 1 and 2.
